from pathlib import Path
import numpy as np
import torch
from parcel_pinn.figures import Gallery, PALETTE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from parcel_pinn.physics import gaussian_raw, derivative
from .model import Geometry


def domain(ax, geo, limits=None):
    a, b = limits or (0, geo.length)
    x = np.linspace(a, b, 1201); lo, hi = geo.bounds(x)
    ax.fill_between(x, 0, lo, color='#334155', zorder=5)
    ax.fill_between(x, hi, 1, color='#334155', zorder=5)
    ax.plot(x, lo, c='#172638', lw=1, zorder=6); ax.plot(x, hi, c='#172638', lw=1, zorder=6)


class ConstrictedGallery(Gallery):
    def __init__(self, root, cfg, geometry):
        super().__init__(root, cfg); self.geometry = Geometry(cfg.lengths[0], geometry)

    def save(self, fig, name, title, caption, folder=None):
        if name == 'paths': domain(fig.axes[0], self.geometry, fig.axes[0].get_xlim())
        if name == 'channel_view':
            ax = fig.axes[0]
            for collection in list(ax.collections):
                if isinstance(collection, Poly3DCollection): collection.remove()
            x = np.linspace(0, self.geometry.length, 401); z = np.linspace(-.25, .25, 5)
            X, Z = np.meshgrid(x, z); lo, hi = self.geometry.bounds(X)
            for wall in [lo, hi]: ax.plot_surface(X, Z, wall, color='#a9bdce', alpha=.4, linewidth=0, rstride=1, cstride=1)
            ax.set_yticks([-.2, 0, .2])
        return super().save(fig, name, title, caption, folder)

    def geometry_plot(self):
        geo = self.geometry
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained'); domain(ax, geo)
        ax.set(xlim=(0, geo.length), ylim=(0, 1), aspect='equal', xlabel='x / H', ylabel='y / H', title='Channel geometry')
        ax.annotate('', (geo.xc, geo.radius), (geo.xc, 1-geo.radius), arrowprops=dict(arrowstyle='<->', color='#c47730', lw=2))
        ax.text(geo.xc+.06, .5, f'Gap = {1-2*geo.radius:g} H', color='#9b551b')
        ax.set_title('Channel geometry', pad=30)
        ax.text(.01, 1.03, f'L/H={geo.length:g} · r/H={geo.radius:g} · periodic x', transform=ax.transAxes, fontsize=9)
        self.save(fig, 'geometry', 'Channel geometry', 'Original transient Gaussian-parcel problem with two added semicircular wall constrictions. Same periodic x boundary, body forcing, Stokes numbers, loadings and gravity.')

    def predictions(self, model, time, x=None, y=None, gradients=False):
        x = np.linspace(0, self.geometry.length, 121) if x is None else x
        y = np.linspace(0, 1, 61) if y is None else y
        xx, yy = np.meshgrid(x, y); xy = np.stack([xx.ravel(), yy.ravel()], 1)
        valid = self.geometry.fluid(xy)
        output = np.full((len(xy), 5), np.nan)
        points = torch.tensor(np.column_stack([xy[valid], np.full(sum(valid), time)]), dtype=torch.float64, requires_grad=gradients)
        if gradients:
            pieces = []
            for chunk in points.split(512):
                field = model.field(chunk)
                du, dv = derivative(field[:, 0], chunk), derivative(field[:, 1], chunk)
                pieces.append(torch.cat([field, (dv[:, 0]-du[:, 1])[:, None]], 1).detach().numpy())
            values = np.concatenate(pieces)
            output[valid, :4] = values
        else:
            with torch.no_grad(): output[valid, :3] = model.field(points).numpy()
        with torch.no_grad():
            q, _, _, _, Z, _ = model.coupling(torch.tensor([[time]], dtype=torch.float64))
            raw = gaussian_raw(torch.tensor(xy[valid][None], dtype=torch.float64), q, model.cfg.width, model.cfg.lengths)/Z
            output[valid, 4] = raw.sum(-1)[0].numpy()*model.physical['represented_mass']/model.cfg.density_ratio
        return xx, yy, output.reshape(*xx.shape, 5)

    def fields(self, model, name, folder):
        xx, yy, f = self.predictions(model, model.cfg.final_time, gradients=True)
        speed = np.linalg.norm(f[..., :2], axis=-1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
        for ax, values, title, cmap in zip(axes.ravel(), [speed, f[..., 2], f[..., 3], f[..., 4]],
                                          ['Speed', 'Pressure', 'Vorticity', 'Volume fraction'], ['viridis', 'coolwarm', 'RdBu_r', 'magma']):
            im = ax.pcolormesh(xx, yy, np.ma.masked_invalid(values), shading='auto', cmap=cmap, rasterized=True)
            domain(ax, self.geometry); ax.set(xlim=(0, self.geometry.length), ylim=(0, 1), aspect='equal', xlabel='x / H', ylabel='y / H', title=title)
            fig.colorbar(im, ax=ax, shrink=.8)
        axes[0, 0].streamplot(xx[0], yy[:, 0], np.ma.masked_invalid(f[..., 0]), np.ma.masked_invalid(f[..., 1]),
                              color='white', density=.7, linewidth=.5, arrowsize=.6)
        self.save(fig, 'fields', 'Flow fields', f'{name}: transient fluid and Gaussian volume fraction at t={model.cfg.final_time:g}; only fluid-domain points are evaluated.', folder)
        np.savez_compressed(folder/'plot_fields.npz', x=xx[0], y=yy[:, 0], velocity_pressure=f[..., :3], vorticity=f[..., 3], volume_fraction=f[..., 4])
        fig, ax = plt.subplots(figsize=(6.5, 3.8), layout='constrained')
        position = model.geometry.xc; lo, hi = model.geometry.bounds(np.array([position])); y = np.linspace(lo[0], hi[0], 131)
        for t, label in [(0., 'Initial field'), (model.cfg.final_time, 'Final PINN')]:
            with torch.no_grad(): field = model.field(torch.tensor(np.column_stack([np.full(len(y), position), y, np.full(len(y), t)]), dtype=torch.float64)).numpy()
            ax.plot(field[:, 0], y, label=label)
        ax.set(xlabel='u / U', ylabel='y / H', title='Throat profile'); ax.legend(frameon=False); ax.grid(alpha=.16)
        self.save(fig, 'profile', 'Throat profile', 'Initial geometry-compatible field and final trained prediction at the throat. This is not a reference validation.', folder)
        fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, speed, cmap='viridis', rstride=2, cstride=2, linewidth=0)
        ax.set(xlabel='x / H', ylabel='y / H', zlabel='Speed / U', title='Speed surface'); ax.view_init(28, -65)
        fig.colorbar(surf, ax=ax, shrink=.5, pad=.1)
        self.save(fig, 'speed_surface', 'Speed surface', 'Height encodes the speed of the 2D solution; this is not a third solved spatial direction.', folder)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.5), layout='constrained')
        for ax, t in zip(axes.ravel(), np.linspace(0, model.cfg.final_time, 4)):
            X, Y, a = self.predictions(model, t)
            ax.pcolormesh(X, Y, np.ma.masked_invalid(np.linalg.norm(a[..., :2], axis=-1)), cmap='viridis', shading='auto', vmin=0, vmax=max(1.5, float(np.nanmax(speed))))
            with torch.no_grad(): q, _ = model.particle_state(torch.tensor([[t]], dtype=torch.float64)); q = q[0].numpy()
            domain(ax, self.geometry)
            ax.scatter(q[:, 0] % self.geometry.length, q[:, 1], s=25, c=[PALETTE[i % len(PALETTE)] for i in range(len(q))], edgecolors='white', zorder=8)
            ax.set(xlim=(0, self.geometry.length), ylim=(0, 1), aspect='equal', xlabel='x / H', ylabel='y / H', title=f't U/H = {t:.2f}')
        self.save(fig, 'snapshots', 'Parcel transport', 'Actual time-dependent fluid and parcel PINN predictions, with periodic wrapping in x. Invalid solid penetration is reported, not corrected by projection.', folder)

    def animations(self, model, particles, name, folder):
        c = self.cfg; times = np.sort(particles.time.unique())
        x, y = np.linspace(0, self.geometry.length, 81), np.linspace(0, 1, 33)
        states = [self.predictions(model, t, x, y)[2] for t in times]
        concentrations = [f[..., 4] for f in states]
        max_alpha = max(float(np.nanmax(a)) for a in concentrations) or 1.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.5), layout='constrained')
        for ax, k in zip(axes.ravel(), np.linspace(0, len(times)-1, 4).astype(int)):
            im = ax.imshow(concentrations[k], extent=(0, self.geometry.length, 0, 1), origin='lower', cmap='magma', vmin=0, vmax=max_alpha)
            domain(ax, self.geometry); ax.set(xlim=(0, self.geometry.length), ylim=(0, 1), xlabel='x / H', ylabel='y / H', title=f't U/H = {times[k]:.2f}')
            centers = particles[particles.time == times[k]]
            ax.scatter(centers.x, centers.y, s=12, c='cyan', edgecolors='white', linewidths=.4, zorder=9)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.85, label='Volume fraction')
        self.save(fig, 'gaussian_snapshots', 'Gaussian transport', 'Normalized Gaussian clouds follow the actual parcel network. The kernel is integrated only over fluid quadrature; width remains the original fixed filter width.', folder)
        for key in ['transport', 'gaussian_transport']:
            data = concentrations if key == 'gaussian_transport' else [np.linalg.norm(f[..., :2], axis=-1) for f in states]
            maximum = max(float(np.nanmax(a)) for a in data) or 1.
            fig, ax = plt.subplots(figsize=(9, 3.6), layout='constrained')
            image = ax.imshow(data[0], extent=(0, self.geometry.length, 0, 1), origin='lower', cmap='magma' if key == 'gaussian_transport' else 'viridis', vmin=0, vmax=maximum)
            domain(ax, self.geometry); ax.set(xlim=(0, self.geometry.length), ylim=(0, 1), xlabel='x / H', ylabel='y / H', title='Gaussian transport' if key == 'gaussian_transport' else 'Parcel transport')
            dots = ax.scatter([], [], s=50, c='cyan', edgecolors='white', zorder=9)
            label = ax.text(.02, .92, '', transform=ax.transAxes, color='white')
            def update(k):
                table = particles[particles.time == times[k]]
                image.set_data(data[k]); dots.set_offsets(table[['x', 'y']].to_numpy())
                dots.set_edgecolors(['red' if bad else 'white' for bad in table.inside_solid])
                label.set_text(f't U/H = {times[k]:.2f}')
                return image, dots, label
            anim = FuncAnimation(fig, update, frames=len(times), interval=90, blit=False)
            path = folder/f'{key}.gif'; anim.save(path, writer=PillowWriter(fps=12), dpi=100)
            item = dict(title='Gaussian transport' if key == 'gaussian_transport' else 'Parcel transport',
                        caption=f'{name}: actual predicted time evolution of the coupled fluid–Gaussian-parcel model. Red outlines flag centers inside solids; no clipping or artificial rebound is applied.', files=[str(path.relative_to(self.root))])
            self.items.append(item)
            if FFMpegWriter.isAvailable():
                try:
                    video = folder/f'{key}.mp4'; anim.save(video, writer=FFMpegWriter(fps=12, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']), dpi=100)
                    item['files'].append(str(video.relative_to(self.root)))
                except Exception as exc: self.errors.append(dict(figure='optional MP4', error=str(exc)))
            plt.close(fig)
