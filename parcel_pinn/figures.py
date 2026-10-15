import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(),'parcel-pinn-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(tempfile.gettempdir(),'parcel-pinn-cache'))
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import pandas as pd
import torch

PALETTE = ['#2563a6', '#d97725', '#218c74', '#984c89', '#59677a', '#b64848']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 12, 'axes.labelsize': 10,
                     'axes.prop_cycle': matplotlib.cycler(color=PALETTE),
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white',
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


class Gallery:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root), cfg
        self.items, self.errors = [], []

    def save(self, fig, name, title, caption, folder=None):
        folder = Path(folder or self.root/'figures')
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in ['png', 'pdf', 'svg']:
            p = folder/f'{name}.{ext}'
            fig.savefig(p, dpi=self.cfg.dpi, bbox_inches='tight')
            paths.append(str(p.relative_to(self.root)))
        plt.close(fig)
        self.items.append(dict(title=title, caption=caption, files=paths))

    def safe(self, method, *args, **kwargs):
        try:
            getattr(self, method)(*args, **kwargs)
        except Exception as exc:
            plt.close('all')
            self.errors.append(dict(figure=method, error=f'{type(exc).__name__}: {exc}'))

    def verification_plot(self, mode, data, cols, kind, out):
        fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 3.3), squeeze=False, layout='constrained')
        for ax, col in zip(axes[0], cols):
            r, p = data[col+'_reference'], data[col+'_pinn']
            if mode == 'taylor':
                ax.scatter(r, p, s=10, alpha=.45, edgecolors='none')
                limits = [min(r.min(), p.min()), max(r.max(), p.max())]
                ax.plot(limits, limits, '--', color='#444444', lw=1, label='Identity')
                ax.set(xlabel='Analytical target', ylabel='PINN prediction', title=col)
            else:
                ax.plot(data.time, r, color='#202e3d', lw=2, label='Analytical target')
                ax.plot(data.time, p, '--', color=PALETTE[1], lw=1.8,
                        label='Quadrature' if mode == 'overlap' else 'PINN')
                ax.set(xlabel='r' if mode == 'overlap' else ('t / τ' if kind == 'V01' else 't'),
                       ylabel=col.replace('_', ' '))
                ax.legend(frameon=False)
            ax.grid(alpha=.16)
        title = {'V01': 'Stokes response', 'V02': 'Momentum exchange', 'V04': 'Gaussian overlap', 'V10': 'Taylor–Green flow'}[kind]
        fig.suptitle(title)
        self.save(fig, kind, title,
                  f'{kind}: independent reduced-model verification. Analytical targets were excluded from PINN training. '
                  'Errors and acceptance criteria are in verification.json; a plot is not a pass verdict.', out)

    def architecture(self):
        fig, ax = plt.subplots(figsize=(12.5, 6.2), layout='constrained')
        ax.set(xlim=(0, 12.5), ylim=(0, 6.2)); ax.axis('off')
        def box(x, y, w, h, text, color):
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15,rounding_size=0.14',
                               facecolor=color, edgecolor='white', linewidth=1.6)
            ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)
        def arrow(a, b, label=None):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, color='#61758a', lw=1.5))
            if label: ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.12, label, ha='center', fontsize=8)
        box(.2, 4.1, 1.5, 1.2, '(x, y, t)\nPeriodic features' if self.cfg.dimension == 2 else '(x, y, z, t)\nPeriodic features', '#e0edf8')
        box(2.4, 3.9, 2.2, 1.6, f'Fluid MLP\n{self.cfg.depth} × {self.cfg.hidden} · tanh\nHard walls and IC', '#c1dbed')
        box(5.3, 4.1, 1.6, 1.2, '(u, v, p)' if self.cfg.dimension == 2 else '(u, v, w, p)', '#e0edf8')
        box(.2, .9, 1.5, 1.2, 't\nInitial states', '#f8e9dc')
        box(2.4, .7, 2.2, 1.6, f'Parcel MLP\n{self.cfg.depth} × {self.cfg.hidden} · tanh\nHard initial states', '#f1d0b2')
        box(5.3, .9, 1.6, 1.2, 'qᵢ(t), vᵢ(t)', '#f8e9dc')
        box(7.6, 2.5, 2.1, 1.25, 'Gaussian transfer\nSame quadrature\nInterpolation ↔ force', '#d4e8dc')
        box(10.4, 2.5, 1.7, 1.25, 'NS + parcel ODE\nAutodifferentiation\nWeighted residuals', '#dfe3ee')
        for a, b in [((1.9,4.7),(2.2,4.7)), ((4.8,4.7),(5.1,4.7)),
                     ((1.9,1.5),(2.2,1.5)), ((4.8,1.5),(5.1,1.5)),
                     ((7.0,4.7),(8.5,3.9)), ((7.0,1.5),(8.5,2.3)),
                     ((9.9,3.1),(10.2,3.1))]: arrow(a,b)
        ax.text(6.2, 5.95, 'Coupled PINN', ha='center', fontsize=18, weight='bold', color='#203449')
        ax.text(6.2, .15, 'Physics-only training  •  held-out reference data  •  independent residual audit', ha='center', fontsize=10, color='#61758a')
        self.save(fig, 'architecture', 'Coupled PINN',
                  f'Two separate fully connected tanh networks, both with {self.cfg.depth} hidden layers of width {self.cfg.hidden}. '
                  'Trainable outputs are coupled through normalized Gaussian quadrature; reference arrays are used only after training.')

    def training(self, history, name, out):
        fig, ax = plt.subplots(figsize=(6.8, 3.6), layout='constrained')
        if len(history):
            ax.semilogy(history.step, history.loss.clip(lower=1e-16), alpha=.65, label='Training batch')
            ax.semilogy(history.step, history.audit_loss.clip(lower=1e-16), lw=2, label='Fixed audit batch')
        ax.set(xlabel='Optimizer step', ylabel='Weighted residual loss', title='Training history')
        ax.legend(frameon=False); ax.grid(alpha=.16)
        self.save(fig, 'training', 'Training history',
                  f'{name}: no smoothing or loss truncation beyond the positive logarithmic floor. Checkpoint uses minimum fixed-audit loss.', out)

    def fields(self, model, name, out):
        c = self.cfg
        x, y = np.linspace(0,c.lengths[0],121), np.linspace(0,1,61)
        xx, yy = np.meshgrid(x,y)
        coords = [xx.ravel(), yy.ravel()]
        if c.dimension == 3: coords.append(np.full(xx.size, c.lengths[2]/2))
        points = np.stack(coords+[np.full(xx.size,c.final_time)],1)
        with torch.no_grad():
            fields = model.field(torch.tensor(points,dtype=torch.float64)).numpy()
            q, v, _, W, Z, force = model.coupling(torch.tensor([[c.final_time]],dtype=torch.float64))
            from .physics import gaussian_raw
            g = gaussian_raw(torch.tensor(points[None,:,:c.dimension],dtype=torch.float64),q,c.width,c.lengths)/Z
            alpha = g[0].sum(1).numpy()*model.physical['represented_mass']/c.density_ratio
        u, vfluid = fields[:,0].reshape(xx.shape), fields[:,1].reshape(xx.shape)
        du_dx = np.gradient(u,x,axis=1); dv_dy = np.gradient(vfluid,y,axis=0)
        curl = np.gradient(vfluid,x,axis=1)-np.gradient(u,y,axis=0)
        fig, axes = plt.subplots(2,2,figsize=(11,6),layout='constrained')
        panels = [(np.hypot(u,vfluid),'Speed','viridis'),(fields[:,-1].reshape(xx.shape),'Pressure','coolwarm'),
                  (curl,'Vorticity','RdBu_r'),(alpha.reshape(xx.shape),'Volume fraction','magma')]
        for ax,(a,title,cmap) in zip(axes.ravel(),panels):
            im=ax.pcolormesh(xx,yy,a,shading='auto',cmap=cmap,rasterized=True)
            fig.colorbar(im,ax=ax,shrink=.82)
            ax.set(xlabel='x / H',ylabel='y / H',title=title,aspect='equal')
        axes[0,0].streamplot(x,y,u,vfluid,color='white',density=.7,linewidth=.5,arrowsize=.6)
        self.save(fig,'fields','Flow fields',f'{name}: direct PINN fields at t={c.final_time:g}. Vorticity computed by plotting-grid differences. '
                  'Uncertainty and residuals must be checked before quantitative interpretation.',out)
        fig, ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
        xquery=1.5/c.lengths[0]*c.lengths[0]
        query=np.stack([np.full(len(y),min(xquery,c.lengths[0]/2)),y]+([np.full(len(y),c.lengths[2]/2)] if c.dimension==3 else [])+[np.full(len(y),c.final_time)],1)
        with torch.no_grad(): profile=model.field(torch.tensor(query,dtype=torch.float64)).numpy()[:,0]
        ax.plot(6*y*(1-y),y,'--',color='#777777',label='Unladen initial profile')
        ax.plot(profile,y,label='Particle-laden PINN')
        ax.set(xlabel='u / U',ylabel='y / H',title='Channel profile');ax.legend(frameon=False);ax.grid(alpha=.16)
        self.save(fig,'profile','Channel profile',f'{name}: final profile at channel midpoint. The dashed initial unladen profile is not a two-way validation target.',out)
        fig=plt.figure(figsize=(9,5.3));ax=fig.add_axes([.02,.08,.75,.84],projection='3d')
        speed=np.hypot(u,vfluid)
        surf=ax.plot_surface(xx,yy,speed,cmap='viridis',rstride=2,cstride=2,linewidth=0,antialiased=True)
        ax.set(xlabel='x / H',ylabel='y / H',zlabel='Speed / U',title='Speed surface')
        ax.tick_params(labelsize=8);ax.zaxis.labelpad=7
        ax.view_init(27,-65);ax.set_box_aspect((3,1,1))
        color_axis=fig.add_axes([.88,.24,.018,.49])
        fig.colorbar(surf,cax=color_axis)
        self.save(fig,'speed_surface','Speed surface',f'{name}: height encodes speed over the solved x–y plane; the vertical axis is a field value, not a third spatial coordinate.',out)
        fig,axes=plt.subplots(2,2,figsize=(10.5,5.5),layout='constrained')
        with torch.no_grad():
            for ax,tt in zip(axes.ravel(),np.linspace(0,c.final_time,4)):
                query=points.copy();query[:,-1]=tt
                f=model.field(torch.tensor(query,dtype=torch.float64)).numpy()
                im=ax.pcolormesh(xx,yy,np.linalg.norm(f[:,:c.dimension],axis=1).reshape(xx.shape),cmap='viridis',shading='auto',vmin=0,vmax=max(1.5,float(speed.max())),rasterized=True)
                qp,_=model.particle_state(torch.tensor([[tt]],dtype=torch.float64));qp=qp[0].numpy()
                ax.scatter(qp[:,0]%c.lengths[0],qp[:,1],s=34,c=[PALETTE[k%len(PALETTE)] for k in range(c.nparcels)],edgecolors='white',linewidths=.8)
                ax.set(xlabel='x / H',ylabel='y / H',title=f't U/H = {tt:.2f}',aspect='equal')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,label='Speed / U')
        self.save(fig,'snapshots','Transport snapshots',f'{name}: four direct evaluations of PINN fluid and parcel states. Marker radii are illustrative. Colors use one common speed scale.',out)

    def trajectories(self, table, name, out):
        c=self.cfg
        fig, ax=plt.subplots(figsize=(9,3.6),layout='constrained')
        for i,g in table.groupby('particle'):
            ax.plot(g.x_unwrapped,g.y,lw=1.8,color=PALETTE[i%len(PALETTE)])
            ax.scatter(g.x_unwrapped.iloc[0],g.y.iloc[0],s=24,facecolors='white',edgecolors=PALETTE[i%len(PALETTE)],zorder=5)
            ax.scatter(g.x_unwrapped.iloc[-1],g.y.iloc[-1],s=25,color=PALETTE[i%len(PALETTE)],zorder=5)
        lo=min(0,table.x_unwrapped.min());hi=max(c.lengths[0],table.x_unwrapped.max())
        ax.axhline(0,color='#26384a');ax.axhline(1,color='#26384a')
        ax.set(xlim=(lo,hi),ylim=(-.06,1.06),aspect='equal',xlabel='Unwrapped x / H',ylabel='y / H',title='Parcel paths')
        self.save(fig,'paths','Parcel paths',f'{name}: direct PINN parcel trajectories; open markers denote initial states and filled markers denote final states. No boundary clipping is applied.',out)
        fig=plt.figure(figsize=(9,5.3));ax=fig.add_axes([.01,.10,.83,.82],projection='3d')
        for i,g in table.groupby('particle'):
            z=g.z_unwrapped if c.dimension==3 else np.zeros(len(g))
            ax.plot(g.x_unwrapped,z,g.y,color=PALETTE[i%len(PALETTE)],lw=2)
            ax.scatter(g.x_unwrapped.iloc[-1],float(z.iloc[-1]) if c.dimension==3 else 0.,g.y.iloc[-1],color=PALETTE[i%len(PALETTE)],s=22)
        X,Z=np.meshgrid(np.linspace(0,c.lengths[0],3),np.linspace(-.25,.25,3) if c.dimension==2 else np.linspace(0,c.lengths[2],3))
        for yy in [0,1]:ax.plot_surface(X,Z,np.full_like(X,yy),color='#bfcedb',alpha=.2,linewidth=.3,edgecolor='#a9bac9')
        ax.set(xlabel='x / H',ylabel='',zlabel='',title='Channel view')
        ax.text2D(.76,.04,'Display depth' if c.dimension==2 else 'z / H',transform=ax.transAxes,fontsize=9)
        ax.text2D(.97,.49,'y / H',transform=ax.transAxes,fontsize=9,rotation=90)
        ax.tick_params(labelsize=8)
        ax.set_box_aspect((3,1,1));ax.view_init(23,-64)
        self.save(fig,'channel_view','Channel view',f'{name}: '+('extruded illustration of a 2D solution; every particle has z=0. Display depth contains no solved physics.' if c.dimension==2 else '3D solution trajectories and channel walls.'),out)
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        for i,g in table.groupby('particle'):
            axes[0].plot(g.time,g.vx,lw=1.3)
            axes[1].plot(g.time,g.vy,lw=1.3)
        for ax,component in zip(axes,['vx / U','vy / U']):
            ax.set(xlabel='t U / H',ylabel=component);ax.grid(alpha=.16)
        fig.suptitle('Parcel velocities')
        self.save(fig,'velocities','Parcel velocities',f'{name}: PINN velocity outputs. Kinematic consistency with position derivatives is audited separately.',out)

    def budgets(self, data, name, out):
        fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
        axes[0].plot(data.time,data.energy);axes[0].set(title='Kinetic energy',ylabel='E')
        axes[1].plot(data.time,data.momentum_balance_residual,label='Momentum')
        axes[1].plot(data.time,data.energy_balance_residual,label='Energy');axes[1].set(title='Balance residuals',ylabel='Absolute residual');axes[1].legend(frameon=False)
        axes[2].plot(data.time,data.drag_dissipation);axes[2].set(title='Drag dissipation',ylabel='D')
        for ax in axes:ax.set(xlabel='t U / H');ax.grid(alpha=.16)
        self.save(fig,'budgets','Conservation diagnostics',f'{name}: driven channel includes wall viscous and pressure tractions, body force and particle gravity. '
                  'Total momentum and energy need not be constant. Quadrature errors enter global balances. Contact cases omit unsupported global balances.',out)

    def animation(self, table, name, out):
        c=self.cfg
        fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
        times=np.sort(table.time.unique());ids=np.sort(table.particle.unique())
        ax.set(xlim=(0,c.lengths[0]),ylim=(-.04,1.04),aspect='equal',xlabel='x / H',ylabel='y / H',title='Parcel transport')
        ax.axhline(0,color='#26384a');ax.axhline(1,color='#26384a')
        dots=ax.scatter(np.zeros(len(ids)),np.zeros(len(ids)),s=65,c=[PALETTE[int(i)%len(PALETTE)] for i in ids],edgecolors='white',linewidths=.8,zorder=5)
        label=ax.text(.02,.94,'',transform=ax.transAxes)
        tails=[ax.plot([],[],lw=1.3,color=PALETTE[int(i)%len(PALETTE)],alpha=.7)[0] for i in ids]
        def update(k):
            current=table[table.time==times[k]].sort_values('particle')
            dots.set_offsets(current[['x','y']].to_numpy());label.set_text(f't U/H = {times[k]:.2f}')
            for line,i in zip(tails,ids):
                g=table[(table.particle==i)&(table.time<=times[k])].tail(14)
                x=g.x.to_numpy().copy();y=g.y.to_numpy().copy()
                jumps=np.where(np.abs(np.diff(x))>c.lengths[0]/2)[0]+1
                x[jumps]=np.nan;y[jumps]=np.nan
                line.set_data(x,y)
            return [dots,label]+tails
        anim=FuncAnimation(fig,update,frames=len(times),interval=90,blit=False)
        p=Path(out)/'transport.gif';anim.save(p,writer=PillowWriter(fps=12),dpi=min(c.dpi,140))
        self.items.append(dict(title='Parcel transport',caption=f'{name}: PINN positions, periodic wrapping in x. Marker sizes are illustrative, not physical radii.',files=[str(p.relative_to(self.root))]))
        if FFMpegWriter.isAvailable():
            mp=Path(out)/'transport.mp4'
            try:
                anim.save(mp,writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=120)
                self.items[-1]['files'].append(str(mp.relative_to(self.root)))
            except Exception as exc:
                self.errors.append(dict(figure='optional MP4',error=str(exc)))
        plt.close(fig)

    def sweep(self, cases):
        if not cases:return
        fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
        # Each line is one case/seed. An ensemble band is plotted only for >=2 seeds.
        groups={}
        for case in cases:
            df=pd.read_csv(self.root/case['directory']/'particles.csv')
            averaged=df.groupby('time')[['vx','vy']].mean()
            groups.setdefault(case['case'],[]).append(averaged)
        for color,(name,frames) in zip(PALETTE,groups.items()):
            for ax,variable in zip(axes,['vx','vy']):
                values=np.array([f[variable].to_numpy() for f in frames]);t=frames[0].index.to_numpy()
                mean=values.mean(0)
                ax.plot(t,mean,label=name.replace('_',' '),color=color)
                if len(frames)>1:
                    sd=values.std(0,ddof=1)
                    ax.fill_between(t,mean-sd,mean+sd,color=color,alpha=.13)
        for ax,v in zip(axes,['Mean vx / U','Mean vy / U']):
            ax.set(xlabel='t U / H',ylabel=v);ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8)
        fig.suptitle('Parameter effects')
        self.save(fig,'parameter_effects','Parameter effects','Parcel-mean velocities by case. Multiple-seed shading, when present, is ±1 sample standard deviation; it is not a confidence bound or a discretization error estimate.')

    def verification_summary(self, records):
        fig,ax=plt.subplots(figsize=(9,3.3),layout='constrained')
        labels=[r['case_id'] for r in records]
        colors=[{'passed':'#218c74','failed':'#d97725','error':'#b64848'}.get(r['outcome'],'#59677a') for r in records]
        ax.bar(labels,[1]*len(labels),color=colors,width=.65)
        for k,r in enumerate(records):ax.text(k,.5,r['outcome'],ha='center',va='center',rotation=90,color='white',fontsize=10)
        ax.set(ylim=(0,1.05),yticks=[],title='Verification overview',xlabel='Atlas check')
        self.save(fig,'verification_overview','Verification overview','Pass/fail classifications are implementation checks using declared project tolerances. They do not certify the 51 literature benchmarks.')

    def write_manifest(self):
        (self.root/'figure_manifest.json').write_text(json.dumps(self.items,indent=2))
        (self.root/'plot_errors.json').write_text(json.dumps(self.errors,indent=2))
