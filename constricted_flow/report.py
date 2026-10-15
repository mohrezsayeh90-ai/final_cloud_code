"""Fluid-only diagnostics and figures. Solid regions are excluded, not painted over a rectangular solve."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from parcel_pinn.figures import Gallery, PALETTE
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from parcel_pinn.physics import derivative
from .model import steady_terms


def dump(path, value):
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False))


def field_data(model, points, chunk=256):
    data = []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points)/chunk)))):
        xy = torch.tensor(batch, dtype=torch.float64, requires_grad=True)
        values = model(xy)
        momentum, div, grad = steady_terms(values, xy, 1/model.case['re'])
        omega = grad[1][:, 0]-grad[0][:, 1]
        data.append(torch.cat([values, omega[:, None], div[:, None], momentum], 1).detach().numpy())
    return np.concatenate(data)


def evaluate(model, folder, state):
    cfg, geo = model.cfg, model.geometry
    x, y = np.linspace(0, cfg.length, 161), np.linspace(0, 1, 65)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()], 1)
    valid = geo.fluid(xy)
    data = field_data(model, xy[valid])
    cube = np.full((len(xy), 7), np.nan); cube[valid] = data
    cube = cube.reshape(*xx.shape, 7)
    np.savez_compressed(folder/'fields.npz', x=x, y=y, mask=valid.reshape(xx.shape),
                        u=cube[..., 0], v=cube[..., 1], pressure=cube[..., 2], vorticity=cube[..., 3],
                        divergence=cube[..., 4], momentum_x=cube[..., 5], momentum_y=cube[..., 6])
    pd.DataFrame(np.column_stack([xy[valid], data]), columns=['x', 'y', 'u', 'v', 'p', 'vorticity',
                 'divergence', 'momentum_x_residual', 'momentum_y_residual']).to_csv(folder/'fields.csv', index=False)
    generator = torch.Generator().manual_seed(29032)
    independent = geo.sample(2048, generator).numpy()
    held = field_data(model, independent)
    momentum_rms = float(np.sqrt(np.mean(held[:, 5:7]**2)))
    continuity_rms = float(np.sqrt(np.mean(held[:, 4]**2)))
    boundary = torch.tensor(geo.walls(191), dtype=torch.float64)
    inlet = torch.tensor(np.column_stack([np.zeros(131), np.linspace(0, 1, 131)]), dtype=torch.float64)
    outlet = inlet.clone(); outlet[:, 0] = cfg.length
    with torch.no_grad():
        wall_error = float(model(boundary)[:, :2].abs().max())
        exact_inlet = torch.stack([6*inlet[:, 1]*(1-inlet[:, 1]), torch.zeros(131)], 1)
        inlet_error = float((model(inlet)[:, :2]-exact_inlet).abs().max())
    outlet_rms = float(model.outlet(outlet).square().mean().sqrt().detach())
    sections = torch.linspace(0, cfg.length, 151, dtype=torch.float64)
    sections = torch.unique(torch.cat([sections, sections.new_tensor([geo.xc])]), sorted=True)
    with torch.no_grad():
        flux = model.flux(sections, order=96).numpy()
        center = model(torch.stack([sections, torch.full_like(sections, .5)], 1)).numpy()
    lo, hi = geo.bounds(sections.numpy())
    table = pd.DataFrame(dict(x=sections.numpy(), lower_wall=lo, upper_wall=hi, flux=flux,
                             relative_flux_error=np.abs(flux-1), center_u=center[:, 0],
                             center_v=center[:, 1], center_pressure=center[:, 2]))
    table.to_csv(folder/'sections.csv', index=False)
    # Wall traction is evaluated using the symmetric Cauchy stress, independently
    # of the documented Laplacian do-nothing condition at the open outlet.
    rows = []
    for side in ['lower', 'upper']:
        wx = np.linspace(0, cfg.length, 201)
        lower, upper = geo.bounds(wx); wy = lower if side == 'lower' else upper
        wall = torch.tensor(np.column_stack([wx, wy]), dtype=torch.float64, requires_grad=True)
        values = model(wall)
        du, dv = derivative(values[:, 0], wall), derivative(values[:, 1], wall)
        normal = np.tile([0., -1. if side == 'lower' else 1.], (len(wx), 1))
        arc = np.abs(wx-geo.xc) < geo.radius
        if geo.radius:
            normal[arc, 0] = -(wx[arc]-geo.xc)/geo.radius
            normal[arc, 1] = (-wy[arc]/geo.radius if side == 'lower' else (1-wy[arc])/geo.radius)
        n = torch.tensor(normal, dtype=torch.float64)
        tangent = torch.stack([-n[:, 1], n[:, 0]], 1)
        tx = (2*du[:, 0]*n[:, 0]+(du[:, 1]+dv[:, 0])*n[:, 1])/model.case['re']
        ty = ((du[:, 1]+dv[:, 0])*n[:, 0]+2*dv[:, 1]*n[:, 1])/model.case['re']
        shear = (tx*tangent[:, 0]+ty*tangent[:, 1]).abs().detach().numpy()
        rows.extend(dict(side=side, x=a, y=b, pressure=p, shear_magnitude=s)
                    for a, b, p, s in zip(wx, wy, values[:, 2].detach().numpy(), shear))
    pd.DataFrame(rows).to_csv(folder/'walls.csv', index=False)
    analytic = None
    if geo.radius == 0:
        pts = independent
        reference = np.column_stack([6*pts[:, 1]*(1-pts[:, 1]), np.zeros(len(pts)),
                                     12/model.case['re']*(cfg.length-pts[:, 0])])
        analytic = dict(velocity_relative_l2=float(np.linalg.norm(held[:, :2]-reference[:, :2])/np.linalg.norm(reference[:, :2])),
                        pressure_relative_l2=float(np.linalg.norm(held[:, 2]-reference[:, 2])/np.linalg.norm(reference[:, 2])))
        pd.DataFrame(dict(x=pts[:, 0], y=pts[:, 1], u_reference=reference[:, 0], u_pinn=held[:, 0],
                          p_reference=reference[:, 2], p_pinn=held[:, 2])).to_csv(folder/'analytic_comparison.csv', index=False)
    flux_error = float(np.max(np.abs(flux-1)))
    screening = (momentum_rms < cfg.residual_tolerance and continuity_rms < cfg.residual_tolerance
                 and outlet_rms < cfg.outlet_tolerance and flux_error < cfg.flux_tolerance
                 and max(wall_error, inlet_error) < 1e-6 and state['best_step'] > 0
                 and state['termination'] in ['time_budget_exhausted', 'budget_exhausted'])
    result = dict(case=model.case, directory=str(folder.name), training=state,
                  heldout_momentum_rms=momentum_rms, heldout_continuity_rms=continuity_rms,
                  wall_velocity_max=wall_error, inlet_velocity_max_error=inlet_error,
                  outlet_residual_rms=outlet_rms, max_relative_flux_error=flux_error,
                  centerline_pressure_drop=float(center[0, 2]-center[-1, 2]),
                  max_speed=float(np.linalg.norm(data[:, :2], axis=1).max()),
                  screening_met=bool(screening), analytic_straight_control=analytic,
                  literature_validation_cases_run=0)
    dump(folder/'case_status.json', result)
    return result, (xx, yy, cube), table


def domain(ax, geo):
    x = np.unique(np.r_[np.linspace(0, geo.length, 801), geo.xc-geo.radius, geo.xc+geo.radius])
    lo, hi = geo.bounds(x)
    ax.fill_between(x, 0, lo, color='#334155', zorder=5)
    ax.fill_between(x, hi, 1, color='#334155', zorder=5)
    ax.plot(x, lo, color='#172638', lw=1.1, zorder=6)
    ax.plot(x, hi, color='#172638', lw=1.1, zorder=6)
    ax.set(xlim=(0, geo.length), ylim=(0, 1), xlabel='x / H', ylabel='y / H', aspect='equal')


class FluidGallery(Gallery):
    def geometry(self, geo):
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        domain(ax, geo)
        ax.set_facecolor('#edf6fb')
        ax.annotate('', (geo.xc, 1-geo.radius), (geo.xc, geo.radius),
                    arrowprops=dict(arrowstyle='<->', color='#c47730', lw=2))
        ax.text(geo.xc+.05, .5, f'Gap = {1-2*geo.radius:g} H', fontsize=10, color='#9b551b')
        for y in [.2, .5, .8]:
            ax.annotate('', (.18, y), (.01, y), arrowprops=dict(arrowstyle='->', color='#2563a6'))
        ax.set(title='Channel geometry')
        ax.text(.01, 1.06, f'L/H = {geo.length:g}   r/H = {geo.radius:g}   x₀/L = {geo.xc/geo.length:g}', transform=ax.transAxes)
        self.save(fig, 'geometry', 'Channel geometry',
                  'Fluid domain excludes two semicircles centered on the lower and upper wall. Inlet mean speed is one; no particles are modeled.')

    def architecture(self):
        fig, ax = plt.subplots(figsize=(12, 5.6), layout='constrained')
        ax.set(xlim=(0, 12), ylim=(0, 5.6)); ax.axis('off')
        positions = [.7, 2.4, 4., 5.6, 7.2, 8.9]
        counts = [4]+[6]*self.cfg.depth+[2]
        # Width is illustrated by six representative neurons, not six actual neurons.
        if self.cfg.depth != 4:
            positions = np.linspace(.7, 8.9, self.cfg.depth+2)
        layers = []
        for i, (x, n) in enumerate(zip(positions, counts)):
            ys = np.linspace(1.6, 4.2, n)
            for y in ys:
                ax.add_patch(Circle((x, y), .12, facecolor='#bcd8ed' if i else '#b5dbd0', edgecolor='white', zorder=3))
            layers.append((x, ys))
            label = 'Features' if i == 0 else ('ψ, p' if i == len(counts)-1 else f'Layer {i}')
            ax.text(x, 1.1, label, ha='center', fontsize=9)
        for (x, ya), (xx, yb) in zip(layers[:-1], layers[1:]):
            for a in ya:
                for b in yb: ax.plot([x+.12, xx-.12], [a, b], color='#a9bdce', alpha=.22, lw=.7)
        box = FancyBboxPatch((9.65, 2.1), 1.95, 1.8, boxstyle='round,pad=.15', facecolor='#e0e8db', edgecolor='white')
        ax.add_patch(box); ax.text(10.62, 3., 'Momentum\nOpen outlet\nFlux quadrature', ha='center', va='center')
        ax.add_patch(FancyArrowPatch((9.1, 2.9), (9.5, 2.9), arrowstyle='-|>', mutation_scale=15, color='#60748a'))
        ax.text(6, 5.05, 'Streamfunction PINN', fontsize=19, weight='bold', ha='center', color='#203449')
        ax.text(6, .45, f'{self.cfg.depth} hidden layers × {self.cfg.hidden} neurons · tanh · hard continuity, unit flux, inlet and walls', ha='center', color='#60748a')
        self.save(fig, 'architecture', 'Flow PINN',
                  f'Streamfunction–pressure PINN, {self.cfg.depth} hidden layers of width {self.cfg.hidden}; hidden neurons are represented schematically. '
                  'Velocity is (psi_y,-psi_x), enforcing continuity. Actual circle level sets enforce walls; psi=0/1 on the two walls fixes unit flux. Reflection symmetry assumes a symmetric steady solution.')

    def sampling(self, geo):
        points = geo.sample(1200, torch.Generator().manual_seed(440), focused=True).numpy()
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        ax.scatter(*points.T, s=3, c='#268a91', alpha=.5); domain(ax, geo); ax.set(title='Collocation points')
        self.save(fig, 'sampling', 'Collocation points', 'Representative sampling: uniform coverage plus throat enrichment. All PDE points are in the fluid; held-out diagnostics use an independent uniform sample.')

    def fields(self, model, mesh, folder):
        xx, yy, f = mesh
        speed = np.linalg.norm(f[..., :2], axis=-1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout='constrained')
        for ax, (values, title, cmap) in zip(axes.ravel(), [(speed, 'Speed', 'viridis'), (f[..., 2], 'Pressure', 'coolwarm'),
                                                       (f[..., 3], 'Vorticity', 'RdBu_r'), (f[..., 4], 'Divergence', 'RdBu_r')]):
            im = ax.pcolormesh(xx, yy, np.ma.masked_invalid(values), shading='auto', cmap=cmap, rasterized=True)
            domain(ax, model.geometry); ax.set(title=title); fig.colorbar(im, ax=ax, shrink=.75)
        self.save(fig, 'fields', 'Flow fields', f"{model.case['name']}: steady PINN predictions evaluated only in the actual fluid domain; no extrapolated flow is drawn inside the solids.", folder)
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        im = ax.pcolormesh(xx, yy, np.ma.masked_invalid(speed), cmap='viridis', shading='auto', rasterized=True)
        ax.streamplot(xx[0], yy[:, 0], np.ma.masked_invalid(f[..., 0]), np.ma.masked_invalid(f[..., 1]),
                      density=1.15, color='white', linewidth=.65, arrowsize=.65)
        domain(ax, model.geometry); ax.set(title='Streamlines'); fig.colorbar(im, ax=ax, label='Speed / U', shrink=.8)
        self.save(fig, 'streamlines', 'Streamlines', 'Streamlines of the predicted steady fluid velocity; these are not particle trajectories.', folder)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout='constrained')
        for ax, values, title in [(axes[0], np.linalg.norm(f[..., 5:7], axis=-1), 'Momentum residual'),
                                  (axes[1], np.abs(f[..., 4]), 'Continuity residual')]:
            im = ax.pcolormesh(xx, yy, np.ma.masked_invalid(values), shading='auto', cmap='magma', rasterized=True)
            domain(ax, model.geometry); ax.set(title=title); fig.colorbar(im, ax=ax, shrink=.8)
        self.save(fig, 'residual_maps', 'Equation residuals', 'Direct automatic-differentiation residual maps. Held-out RMS values are reported separately from these plotting locations.', folder)

    def profiles(self, model, sections, folder):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
        for position in [0., model.geometry.xc, .7*self.cfg.length, self.cfg.length]:
            lo, hi = model.geometry.bounds(np.array([position]))
            y = np.linspace(lo[0], hi[0], 151)
            with torch.no_grad():
                f = model(torch.tensor(np.column_stack([np.full(len(y), position), y]), dtype=torch.float64)).numpy()
            axes[0].plot(f[:, 0], y, label=f'x/H={position:g}')
        axes[0].set(xlabel='u / U', ylabel='y / H', title='Velocity profiles'); axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(sections.x, sections.flux); axes[1].axhline(1, color='#777777', ls='--', label='Inlet flux')
        axes[1].set(xlabel='x / H', ylabel='Q / UH', title='Flow rate'); axes[1].legend(frameon=False)
        axes[2].plot(sections.x, sections.center_pressure)
        axes[2].set(xlabel='x / H', ylabel='p / ρU²', title='Center pressure')
        for ax in axes: ax.grid(alpha=.16)
        self.save(fig, 'profiles', 'Section diagnostics', 'Exact semicircle limits and 96-point Gauss quadrature are used for held-out section flux; inlet target Q=1. No throat-speed rescaling is applied to predictions.', folder)
        walls = pd.read_csv(folder/'walls.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        for side, group in walls.groupby('side'):
            axes[0].plot(group.x, group.pressure, label=side)
            axes[1].plot(group.x, group.shear_magnitude, label=side)
        axes[0].set(xlabel='x / H', ylabel='p / ρU²', title='Wall pressure')
        axes[1].set(xlabel='x / H', ylabel='|τw| / ρU²', title='Wall shear')
        for ax in axes: ax.legend(frameon=False); ax.grid(alpha=.16)
        self.save(fig, 'walls', 'Wall diagnostics', 'Shear magnitude uses symmetric viscous Cauchy stress on the actual curved walls. Corner values are not a resolved singular-stress prediction.', folder)

    def surfaces(self, model, mesh, folder):
        xx, yy, f = mesh
        fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111, projection='3d')
        im = ax.plot_surface(xx, yy, np.linalg.norm(f[..., :2], axis=-1), cmap='viridis', linewidth=0, rstride=2, cstride=2)
        ax.set(xlabel='x / H', ylabel='y / H', zlabel='Speed / U', title='Speed surface'); ax.view_init(30, -60)
        fig.colorbar(im, ax=ax, shrink=.5, pad=.1)
        self.save(fig, 'speed_surface', 'Speed surface', 'Surface height encodes speed of the two-dimensional solution; it is not a third solved spatial direction.', folder)
        fig = plt.figure(figsize=(10, 5.2)); ax = fig.add_axes([.02, .12, .84, .8], projection='3d')
        x = np.linspace(0, self.cfg.length, 401); z = np.linspace(-.18, .18, 5)
        X, Z = np.meshgrid(x, z); lo, hi = model.geometry.bounds(X)
        for wall in [lo, hi]: ax.plot_surface(X, Z, wall, color='#8ba6b9', alpha=.5, linewidth=0, rstride=1, cstride=1)
        for y in [.2, .5, .8]:
            ax.plot([0, .15], [0, 0], [y, y], color='#2563a6')
        ax.set(xlabel='x / H', ylabel='', zlabel='', title='Channel view', yticks=[-.15, 0, .15])
        ax.text2D(.74, .05, 'Display depth', transform=ax.transAxes, fontsize=9)
        ax.text2D(.97, .50, 'y / H', transform=ax.transAxes, fontsize=9, rotation=90)
        ax.tick_params(labelsize=8, pad=1)
        ax.set_box_aspect((self.cfg.length, .6, 1)); ax.view_init(23, -65)
        self.save(fig, 'channel_view', 'Channel view', 'Extruded view of the actual semicircular geometry. Fluid physics is solved in x–y only; no particles are included.', folder)

    def analytic(self, model, folder):
        if model.geometry.radius: return
        data = pd.read_csv(folder/'analytic_comparison.csv')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
        for ax, variable, label in zip(axes, ['u', 'p'], ['Velocity', 'Pressure']):
            ax.scatter(data[variable+'_reference'], data[variable+'_pinn'], s=6, alpha=.3)
            a, b = data[variable+'_reference'].min(), data[variable+'_reference'].max()
            ax.plot([a, b], [a, b], '--', c='#555555'); ax.set(xlabel='Analytical', ylabel='PINN', title=label)
        self.save(fig, 'analytic_control', 'Straight-channel check', 'Poiseuille comparison for the separate straight-limit implementation check. The inlet lifting contains the Poiseuille baseline; no pointwise labels enter optimization. This is not independent discovery of that profile or validation of the constricted cases.', folder)

    def section_scan(self, model, mesh, folder):
        xx, yy, f = mesh
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout='constrained', gridspec_kw={'height_ratios': [1.4, 1]})
        axes[0].pcolormesh(xx, yy, np.ma.masked_invalid(np.linalg.norm(f[..., :2], axis=-1)), cmap='viridis', shading='auto')
        domain(axes[0], model.geometry); axes[0].set(title='Section scan')
        cursor = axes[0].axvline(0, c='white', lw=1.2)
        line, = axes[1].plot([], [], c='#2563a6'); text = axes[1].text(.02, .9, '', transform=axes[1].transAxes)
        axes[1].set(xlim=(0, max(1.6, float(np.nanmax(f[..., 0]))*1.1)), ylim=(0, 1), xlabel='u / U', ylabel='y / H')
        scans = []
        for x in np.linspace(0, self.cfg.length, self.cfg.frames):
            lo, hi = model.geometry.bounds(np.array([x])); y = np.linspace(lo[0], hi[0], 131)
            with torch.no_grad(): u = model(torch.tensor(np.column_stack([np.full(len(y), x), y]), dtype=torch.float64))[:, 0].numpy()
            scans.append((x, y, u))
        def update(k):
            x, y, u = scans[k]; cursor.set_xdata([x, x]); line.set_data(u, y); text.set_text(f'x/H = {x:.2f}')
            return cursor, line, text
        anim = FuncAnimation(fig, update, frames=len(scans), interval=150, blit=False)
        path = folder/'section_scan.gif'; anim.save(path, writer=PillowWriter(fps=8), dpi=100)
        item = dict(title='Section scan', caption='A moving diagnostic section through one steady field. Frame index is position, not physical time; no particle motion or transient fluid evolution is simulated.', files=[str(path.relative_to(self.root))])
        self.items.append(item)
        if FFMpegWriter.isAvailable():
            try:
                video = folder/'section_scan.mp4'
                anim.save(video, writer=FFMpegWriter(fps=8, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']), dpi=100)
                item['files'].append(str(video.relative_to(self.root)))
            except Exception as exc: self.errors.append(dict(figure='optional MP4', error=str(exc)))
        plt.close(fig)

    def comparison(self, cases):
        if not cases: return
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
        for item in cases:
            data = pd.read_csv(self.root/'cases'/item['directory']/'sections.csv')
            label = item['case']['name'].replace('_', ' ')
            axes[0].plot(data.x, data.center_u, label=label)
            axes[1].plot(data.x, data.center_pressure, label=label)
            axes[2].plot(data.x, data.relative_flux_error, label=label)
        for ax, label, title in zip(axes, ['u / U', 'p / ρU²', '|Q−1|'], ['Center speed', 'Center pressure', 'Flux error']):
            ax.set(xlabel='x / H', ylabel=label, title=title); ax.grid(alpha=.16)
        axes[0].legend(frameon=False, fontsize=7)
        self.save(fig, 'case_comparison', 'Case comparison', 'Each curve is an independently trained steady case. The straight-channel control has r=0; all other curves share the requested bilateral constriction.')


def report(out, cfg, gallery, cases, stages, errors, seconds):
    gallery.write_manifest()
    required_errors = [e for e in gallery.errors if e['figure'] != 'optional MP4']
    complete = len(cases) == len(cfg.cases()) and not errors and not required_errors
    payload = dict(workflow_complete=complete, research_quality_criteria_met=complete and all(c['screening_met'] for c in cases),
                   seconds=seconds, model='steady 2D incompressible Navier–Stokes streamfunction PINN', particles_modeled=False,
                   continuity_enforced_hard=True, unit_flux_enforced_hard=True,
                   physical_time_modeled=False, physical_validation_cases_run=0, cases=cases, errors=errors,
                   plot_errors=gallery.errors, stages=stages,
                   limitations=['Symmetric steady branch; transient or asymmetric stability is not evaluated.',
                                'No Gaussian parcels, particle forces, trajectories or collisions in this fluid-only extension.',
                                'Straight Poiseuille control is analytical verification, not constriction literature validation.',
                                'Budget completion is not proof of accuracy; compare held-out residuals and mass flux.',
                                'Open outlet uses nu*d_n(u)-p*n=0 for the Laplacian form, not zero symmetric Cauchy traction.',
                                'One seed; no mesh/collocation independence or statistical uncertainty study.'])
    dump(out/'summary.json', payload)
    rows = [dict(case=c['case']['name'], Re=c['case']['re'], radius=c['case']['radius'],
                 momentum_rms=c['heldout_momentum_rms'], continuity_rms=c['heldout_continuity_rms'],
                 max_flux_error=c['max_relative_flux_error'], outlet_rms=c['outlet_residual_rms'],
                 pressure_drop=c['centerline_pressure_drop'], max_speed=c['max_speed'], screening_met=c['screening_met']) for c in cases]
    table = pd.DataFrame(rows); table.to_csv(out/'paper_metrics.csv', index=False)
    if rows: table.to_latex(out/'paper_metrics.tex', index=False, float_format=lambda x: f'{x:.4g}', escape=True)
    notes = ['# Bilateral constriction flow', '', f'Workflow complete: {complete}. Screening criteria: {payload["research_quality_criteria_met"]}.',
             '', 'Steady fluid only; no particles or physical time are modeled. Radius and length are nondimensionalized by H.',
             'No literature validation of the constricted geometry is claimed. Straight-channel analytical errors are reported separately.', '',
             '## Assumptions', '']+['- '+x for x in payload['limitations']]
    notes += ['', '## Figures', '']
    cards = []
    for item in gallery.items:
        first = item['files'][0]
        notes += [f'### {item["title"]}', '', item['caption'], '', f'![{item["title"]}]({first})', '']
        links = ' · '.join(f'<a href="{html.escape(p)}">{Path(p).suffix.upper()[1:]}</a>' for p in item['files'])
        cards.append(f'<article><h2>{html.escape(item["title"])}</h2><img src="{html.escape(first)}" loading="lazy"><p>{html.escape(item["caption"])}</p>{links}</article>')
    (out/'REPORT.md').write_text('\n'.join(notes))
    (out/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Constriction flow</title>'
        '<style>body{font:16px/1.6 system-ui;background:#edf2f6;color:#203449;margin:30px auto;max-width:1100px}article{background:white;border-radius:14px;padding:24px;margin:20px 0}img{max-width:100%}a{color:#2563a6}</style>'
        f'<h1>Constriction flow</h1><p>Steady fluid only · no particles · no transient simulation.</p><p>Workflow complete: {complete}. '
        f'Screening criteria: {payload["research_quality_criteria_met"]}. Literature validation: 0.</p>'
        '<p><a href="summary.json">Status</a> · <a href="paper_metrics.csv">CSV</a> · <a href="paper_metrics.tex">LaTeX</a> · <a href="config.json">Configuration</a> · <a href="REPORT.md">Report</a></p>'+''.join(cards)+'</html>')
    return payload
