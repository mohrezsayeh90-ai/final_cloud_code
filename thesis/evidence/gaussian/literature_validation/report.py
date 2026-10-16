"""Direct paper comparison, field checks, short-titled export figures and report."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import html
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
from parcel_pinn.figures import plt,PALETTE
from matplotlib.patches import Circle,FancyBboxPatch,FancyArrowPatch
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
from skfem import MeshTri,Basis,ElementVector,ElementTriP2
from .couette import Settings,CylinderPINN,VelocityPINN,assess,sample_points


def load_model(folder):
    folder=Path(folder)
    cfg=Settings(**json.loads((folder/'settings.json').read_text()))
    model=CylinderPINN(cfg) if cfg.formulation=='streamfunction' else VelocityPINN(cfg)
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True))
    model.eval()
    return model


def fem_interpolator(folder):
    with np.load(Path(folder)/'fem_solution.npz') as z:
        mesh=MeshTri(z['vertices'],z['triangles'])
        basis=Basis(mesh,ElementVector(ElementTriP2()))
        return basis.interpolator(z['velocity'].copy())


def digitize_reference(pack,out):
    source=Path(pack)/'figures/garcia2023_fig2_couette.png'
    a=np.asarray(Image.open(source).convert('RGB'));rows=[]
    for x in range(530,882):
        ys=np.where((a[43:321,x,2]>180)&(a[43:321,x,0]<80)&(a[43:321,x,1]<100))[0]+43
        if len(ys):
            y=float(np.median(ys))
            rows.append(dict(time_deltaU_over_H=(x-529)/(882-529)*30,
                             omega_magnitude=.55-(y-42)/(321-42)*.25,
                             pixel_x=x,pixel_y=y,half_pixel_uncertainty=.5*.25/(321-42)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'digitized_CO01_rotation.csv',index=False)
    (out/'digitization_metadata.json').write_text(json.dumps(dict(
        source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        panel='Figure 2(b), blue Rep=1 curve',axis_pixels=dict(x0=529,x1=882,y0=321,y1=42),
        axis_values=dict(time=[0,30],omega=[.3,.55]),method='Median blue-pixel row per image column; no smoothing or phase alignment.',
        limitations='Only a visual plateau check. Numeric table is the primary reference. Pixel uncertainty excludes source and calibration uncertainty.'),indent=2))
    fig,ax=plt.subplots(figsize=(12,5.3),layout='constrained');ax.imshow(a)
    ax.scatter(frame.pixel_x.iloc[::8],frame.pixel_y.iloc[::8],s=18,facecolors='none',edgecolors='#e38635',linewidths=.8)
    ax.set_axis_off();ax.set_title('Reference extraction')
    fig.savefig(out/'reference_extraction.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return frame


def generate_report(out,records,pack,dpi=220):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    items=[];rows=[];models={};field_data={}
    def save(fig,name,title,caption):
        files=[]
        for ext in ['png','pdf','svg']:
            file=out/f'{name}.{ext}';fig.savefig(file,dpi=dpi,bbox_inches='tight');files.append(file.name)
        plt.close(fig);items.append(dict(title=title,caption=caption,files=files))
    fig,ax=plt.subplots(figsize=(12,5.4),layout='constrained');ax.set(xlim=(0,12),ylim=(0,5.4));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.14',facecolor=color,edgecolor='white'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#61758a',lw=1.4))
    box(.2,3.4,2,1.2,'Geometry and Rep\nNo-slip boundaries\nFree angular velocity','#dceaf5')
    box(3,3.4,2.2,1.2,'Independent FEM\nP2/P1 flow\nZero-torque equation','#d5e9dd')
    box(6.1,3.4,2.2,1.2,'Internal velocity\nanchor points\nNo paper targets','#d5e9dd')
    box(3,.8,2.5,1.4,'PINN: u, v, p\nTrainable ω\nNS + continuity + torque','#f4dfcd')
    box(6.5,.8,2,1.4,'Held-out PDE\nBoundary checks\nSpin prediction','#dfe3ef')
    box(9.4,.8,2.2,1.4,'Article comparison\nEquilibrium table\nReference plateau','#e0eaf2')
    for a,b in [((2.4,4),(2.8,4)),((5.4,4),(5.9,4)),((1.2,3.2),(3.2,2.4)),
                ((7.2,3.2),(5.0,2.4)),((5.7,1.5),(6.3,1.5)),((8.7,1.5),(9.2,1.5))]:arrow(a,b)
    ax.text(6,5.05,'Resolved-cylinder PINN',ha='center',fontsize=17,weight='bold',color='#203449')
    save(fig,'workflow','Resolved-cylinder PINN','The field network and a trainable angular velocity are constrained by Navier–Stokes, continuity, hard no-slip and zero torque. FEM-generated velocity data assist training. Published spin values are used only in the final comparison.')
    ref_file=Path(pack)/'data/couette_equilibrium.csv';ref=pd.read_csv(ref_file)
    for record in records:
        model=load_model(record['pinn']);models[record['case_id']]=model
        metrics=json.loads((Path(record['pinn'])/'metrics.json').read_text())
        fine=json.loads((Path(record['fine'])/'fem_metrics.json').read_text())
        coarse=json.loads((Path(record['coarse'])/'fem_metrics.json').read_text()) if record.get('coarse') else None
        truth=float(ref[np.isclose(ref.Re,model.settings.particle_re)].iloc[0].omega_magnitude_H_over_dU)
        metrics.update(reference_omega_magnitude=truth,predicted_omega_magnitude=abs(metrics['omega']),
                       relative_error=abs(abs(metrics['omega'])-truth)/truth,
                       comparison_within_tolerance=abs(abs(metrics['omega'])-truth)/truth<=model.settings.comparison_relative_tolerance)
        (Path(record['pinn'])/'metrics.json').write_text(json.dumps(metrics,indent=2))
        interpolation=fem_interpolator(record['fine'])
        # Independent velocity samples, not the 1024 FEM training anchors.
        points=sample_points(1200,torch.Generator().manual_seed(453226))
        uref=interpolation(points.numpy().T).T
        upred=model(points.requires_grad_(True))[:,:2].detach().numpy()
        relative_velocity_error=float(np.linalg.norm(upred-uref)/np.linalg.norm(uref))
        row=dict(case_id=record['case_id'],Rep=model.settings.particle_re,
                 reference_spin=truth,FEM_spin=abs(fine['omega']),PINN_spin=abs(metrics['omega']),
                 FEM_error_percent=100*abs(abs(fine['omega'])-truth)/truth,
                 PINN_error_percent=100*abs(abs(metrics['omega'])-truth)/truth,
                 FEM_mesh_change_percent=100*abs(fine['omega']-coarse['omega'])/abs(fine['omega']) if coarse else None,
                 PINN_velocity_relative_L2=relative_velocity_error,
                 PINN_momentum_rms=metrics['momentum_rms'],PINN_continuity_rms=metrics['continuity_rms'],
                 PINN_torque_normalized=metrics['torque_normalized'],
                 wall_error=metrics['wall_boundary_max_error'],surface_error=metrics['cylinder_boundary_max_error'],
                 comparison_within_tolerance=metrics['comparison_within_tolerance'],
                 physics_criteria_met=metrics['physics_criteria_met'],
                 mode='FEM-assisted PINN',pinn_directory=str(record['pinn']),FEM_directory=str(record['fine']))
        rows.append(row)
        x,y=np.linspace(0,2,161),np.linspace(0,1,81);xx,yy=np.meshgrid(x,y)
        mask=(xx-1)**2+(yy-.5)**2>=.125**2
        xy=np.column_stack([xx[mask],yy[mask]])
        predicted=[]
        for p in np.array_split(xy,30):
            predicted.append(model(torch.tensor(p,dtype=torch.float64,requires_grad=True)).detach().numpy())
        predicted=np.vstack(predicted);fem=interpolation(xy.T).T
        arrays=[np.full(xx.shape,np.nan) for _ in range(5)]
        for k in range(3):arrays[k][mask]=predicted[:,k]
        arrays[3][mask]=fem[:,0];arrays[4][mask]=fem[:,1]
        u,v,p,uf,vf=arrays;error=np.hypot(u-uf,v-vf)
        field_data[record['case_id']]=(x,y,xx,yy,u,v,p)
        pd.DataFrame(dict(x=xy[:,0],y=xy[:,1],PINN_u=predicted[:,0],PINN_v=predicted[:,1],PINN_p=predicted[:,2],FEM_u=fem[:,0],FEM_v=fem[:,1])).to_csv(out/f"fields_{record['case_id']}.csv",index=False)
        fig,axes=plt.subplots(2,2,figsize=(11,6.4),layout='constrained')
        for ax,data,title,cmap in zip(axes.ravel(),[np.hypot(uf,vf),np.hypot(u,v),error,p],
                                      ['FEM speed','PINN speed','Velocity difference','PINN pressure'],['viridis','viridis','magma','coolwarm']):
            im=ax.pcolormesh(xx,yy,np.ma.masked_invalid(data),shading='auto',cmap=cmap,rasterized=True)
            ax.add_patch(Circle((1,.5),.125,color='#34495e',zorder=5))
            ax.set(xlabel='x / H',ylabel='y / H',title=title,aspect='equal');fig.colorbar(im,ax=ax,shrink=.85)
        axes[0,1].streamplot(x,y,np.ma.masked_invalid(u),np.ma.masked_invalid(v),color='white',density=.75,linewidth=.5,arrowsize=.6)
        save(fig,f"flow_{record['case_id']}",'Couette flow',
             f"{record['case_id']}: fields in the frame moving at ΔU/2. Cylinder center y/H=0.5 is prescribed. "
             f"PINN uses internally generated FEM velocity anchors; the relative velocity L2 error on independent points is {relative_velocity_error:.3%}. Article values are held out.")
        hist=pd.read_csv(Path(record['pinn'])/'training.csv')
        fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
        for key in ['momentum','continuity','torque','internal_FEM_anchors']:
            if key in hist:axes[0].semilogy(hist.step,hist[key].clip(lower=1e-16),label=key.replace('_',' '))
        axes[0].set(xlabel='Optimizer step',ylabel='Mean squared residual',title='Training terms');axes[0].legend(frameon=False,fontsize=8)
        axes[1].plot(hist.step,-hist.omega,label='Learned spin');axes[1].set(xlabel='Optimizer step',ylabel='|ω| H / ΔU',title='Rotation estimate')
        for ax in axes:ax.grid(alpha=.16)
        save(fig,f"training_{record['case_id']}",'Training history',f"{record['case_id']}: horizontal axis is optimizer iteration, not physical time. The paper target was not supplied to training or checkpoint selection.")
    frame=pd.DataFrame(rows);frame.to_csv(out/'comparison.csv',index=False)
    frame[['case_id','Rep','reference_spin','FEM_spin','PINN_spin','FEM_error_percent','PINN_error_percent']].to_latex(out/'comparison.tex',index=False,float_format=lambda a:f'{a:.6g}',escape=True)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    axes[0].plot(frame.Rep,frame.reference_spin,'o-',color='#223449',label='Article table')
    axes[0].plot(frame.Rep,frame.FEM_spin,'s--',color=PALETTE[2],label='Independent FEM')
    axes[0].plot(frame.Rep,frame.PINN_spin,'^--',color=PALETTE[1],label='FEM-assisted PINN')
    axes[0].set(xlabel='Rep',ylabel='|ω| H / ΔU',title='Equilibrium rotation',xticks=frame.Rep)
    k=np.arange(len(frame));axes[1].bar(k-.17,frame.FEM_error_percent,width=.34,label='FEM',color=PALETTE[2])
    axes[1].bar(k+.17,frame.PINN_error_percent,width=.34,label='PINN',color=PALETTE[1])
    axes[1].set(xticks=k,xticklabels=frame.case_id,ylabel='Relative error (%)',title='Reference difference')
    for ax in axes:ax.grid(alpha=.16);ax.legend(frameon=False)
    save(fig,'rotation_comparison','Equilibrium rotation',
         'Direct comparison with the supplied finest-grid article table at Rep=1 and 2. Lines connect two evaluated conditions; they are not a continuous Reynolds sweep. No parameter was tuned to either article spin value.')
    curve=digitize_reference(pack,out)
    if 'CO01' in models:
        primary=next(r for r in rows if r['case_id']=='CO01')
        fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
        ax.plot(curve.time_deltaU_over_H,curve.omega_magnitude,color='#2256b4',label='Article curve (digitized)')
        ax.hlines(primary['reference_spin'],20,30,color='#223449',lw=1,label='Article table')
        ax.hlines(primary['FEM_spin'],20,30,color=PALETTE[2],linestyle=':',lw=2,label='FEM steady level')
        ax.hlines(primary['PINN_spin'],20,30,color=PALETTE[1],linestyle='--',lw=2,label='PINN steady level')
        ax.set(xlabel='t ΔU / H',ylabel='|ω| H / ΔU',title='Rotation plateau',xlim=(0,30))
        ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8)
        save(fig,'rotation_plateau','Rotation plateau',
             'The blue curve is digitized from supplied Figure 2(b). FEM/PINN predict only the steady angular speed, shown over the final reference window t=20–30. They do not predict or validate the preceding transient relaxation. The numerical table is the primary target.')
    fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained');ax.set(xlim=(0,2),ylim=(-.15,1.2),aspect='equal')
    ax.plot([0,2],[0,0],color='#34495e',lw=3);ax.plot([0,2],[1,1],color='#34495e',lw=3)
    ax.add_patch(Circle((1,.5),.125,facecolor='#d7e7ef',edgecolor='#2563a6',lw=2))
    ax.annotate('',(.6,1.08),(.25,1.08),arrowprops=dict(arrowstyle='->',color=PALETTE[1],lw=2))
    ax.annotate('',(.25,-.08),(.6,-.08),arrowprops=dict(arrowstyle='->',color=PALETTE[1],lw=2))
    ax.text(.65,1.07,'Top: +ΔU/2',va='center');ax.text(.65,-.09,'Bottom: −ΔU/2',va='center')
    ax.text(1,.5,'ω',ha='center',va='center',fontsize=14);ax.text(1.2,.50,'D/H = 0.25\ny/H = 0.5\nTorque = 0',va='center')
    ax.text(.12,.53,'Periodic',ha='center',fontsize=9);ax.text(1.88,.53,'Periodic',ha='center',fontsize=9)
    ax.set(xlabel='x / H',ylabel='y / H',title='Benchmark geometry')
    save(fig,'geometry','Benchmark geometry','Original walls (0, ΔU) are transformed by a Galilean velocity ΔU/2. The cylinder translates with the symmetry-plane fluid velocity and rotates freely. Its symmetry-plane position is an input, not a validated prediction.')
    if 'CO01' in models:
        x,y,xx,yy,u,v,p=field_data['CO01'];model=models['CO01']
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        ax.pcolormesh(xx,yy,np.ma.masked_invalid(np.hypot(u,v)),shading='auto',cmap='viridis',rasterized=True)
        ax.add_patch(Circle((1,.5),.125,color='#34495e',zorder=4))
        line,=ax.plot([],[],color='white',lw=2.5,zorder=5);label=ax.text(.03,.94,'',transform=ax.transAxes,color='white')
        ax.set(xlim=(0,2),ylim=(0,1),xlabel='x / H',ylabel='y / H',aspect='equal',title='Free rotation')
        omega=float(model.omega.detach());times=np.linspace(0,12,61)
        def update(k):
            angle=omega*times[k]
            line.set_data([1-.105*np.cos(angle),1+.105*np.cos(angle)],[.5-.105*np.sin(angle),.5+.105*np.sin(angle)])
            label.set_text(f't ΔU/H = {times[k]:.1f}')
            return line,label
        animation=FuncAnimation(fig,update,frames=len(times),interval=80)
        animation.save(out/'free_rotation.gif',writer=PillowWriter(fps=12),dpi=130)
        files=['free_rotation.gif']
        if FFMpegWriter.isAvailable():
            animation.save(out/'free_rotation.mp4',writer=FFMpegWriter(fps=12,extra_args=['-pix_fmt','yuv420p']),dpi=120);files.append('free_rotation.mp4')
        plt.close(fig);items.append(dict(title='Free rotation',caption='Kinematic rendering of the PINN-predicted steady rotation. It does not show simulated startup, acceleration or lateral migration.',files=files))
    achieved=all(r['comparison_within_tolerance'] and r['physics_criteria_met'] for r in rows)
    summary=dict(scope='Conditional validation of equilibrium angular velocity for a resolved 2D circular cylinder at the symmetry plane.',
                 case_ids=[r['case_id'] for r in rows],literature_source=dict(authors='Garcia-Villalba et al.',year=2023,
                 doi='10.1016/j.compfluid.2023.105936',url='https://arxiv.org/abs/2305.16061',locator='Section 4.1; finest-grid equilibrium table and Figure 2(b)',
                 reference_sha256=hashlib.sha256(ref_file.read_bytes()).hexdigest(),reference_type='Published numerical simulation, manually transcribed table in user-provided pack'),
                 rows=rows,all_declared_criteria_met=achieved,
                 protocol=dict(geometry=[2,1],diameter=.25,center=[1,.5],density_ratio=1,
                     viscosity='1/(64 Rep)',boundary='Periodic x, moving no-slip walls; no-slip circular cylinder with unknown torque-free rotation',
                     frame='Translating at ΔU/2',reference_values_used_in_training=False,
                     FEM_used_as_PINN_velocity_anchors=True,reference_tolerance=.10,momentum_rms_tolerance=.01,
                     continuity_rms_tolerance=.01,normalized_torque_tolerance=.002),
                 limitations=['Center y/H=.5 is prescribed. Initial migration from y/H=.4 and its transient curve are not validated.',
                              'This is a resolved-surface extension. Original Gaussian parcel coupling/collision models are not certified by this comparison.',
                              'The PINN uses independently generated FEM field anchors. It is not a physics-only PINN result.',
                              'The reference is numerical literature data, not an experiment. Single-seed PINN uncertainty is not quantified.',
                              'Reference targets are from the supplied source pack; the external arXiv download was blocked by environment egress policy.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/'figure_manifest.json').write_text(json.dumps(items,indent=2))
    portable=[]
    for record in records:
        entry=dict(record)
        for key in ['pinn','fine','coarse']:
            if entry.get(key):
                p=Path(entry[key]).resolve()
                entry[key]=str(p.relative_to(out.resolve())) if p.is_relative_to(out.resolve()) else str(p)
        portable.append(entry)
    records=portable
    (out/'run_records.json').write_text(json.dumps(records,indent=2))
    lines=['# Couette equilibrium-spin comparison','',
           'Primary case: CO01. CO02 is a second operating condition from the same publication to test the Reynolds-number trend.',
           '', 'The validated observable is equilibrium spin at the prescribed symmetry-plane position. This report does not validate migration, startup dynamics, or the original Gaussian parcel closure.',
           '', '| Case | Paper | FEM | FEM error | PINN | PINN error | NS RMS | Divergence RMS |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case_id']} | {r['reference_spin']:.6f} | {r['FEM_spin']:.6f} | {r['FEM_error_percent']:.3f}% | {r['PINN_spin']:.6f} | {r['PINN_error_percent']:.3f}% | {r['PINN_momentum_rms']:.4g} | {r['PINN_continuity_rms']:.4g} |")
    lines+=['',f'All declared criteria met: {achieved}.', '',
            'The independently generated FEM solution uses conformal P2/P1 Taylor–Hood elements, exact periodic constraints and a free-rotation virtual-work equation. '
            'Two spatial meshes estimate sensitivity. PINN interior velocity anchors come from FEM; no article spin value or digitized source-curve point enters optimization.',
            '', 'The initial physics-only PINN pilots were insufficient. Their outputs remain in the repository for transparency; they are not used as successful validation results.',
            '', '## Limits','']+[f'- {s}' for s in summary['limitations']]
    lines+=['','## Figures','']
    for item in items:lines += [f"### {item['title']}",'',item['caption'],'',f"![{item['title']}]({item['files'][0]})",'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    def esc(a):return html.escape(str(a))
    cards=''.join(f'<article><h2>{esc(i["title"])}</h2><img src="{esc(i["files"][0])}"><p>{esc(i["caption"])}</p><p>'+ ' · '.join(f'<a href="{esc(f)}">{Path(f).suffix[1:].upper()}</a>' for f in i['files'])+'</p></article>' for i in items)
    table_html=frame[['case_id','reference_spin','FEM_spin','PINN_spin','FEM_error_percent','PINN_error_percent','physics_criteria_met']].to_html(index=False,float_format=lambda a:f'{a:.5g}')
    page='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Couette validation</title><style>
body{font:16px/1.6 system-ui;color:#203449;background:#edf2f6;margin:0}main{max-width:1100px;margin:auto;padding:30px}article{background:white;padding:25px;margin:25px 0;border-radius:12px}img{max-width:100%;height:auto}.note{background:#fff1d7;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #cad4df}a{color:#2563a6}</style><main>
<h1>Couette equilibrium rotation</h1><p dir="rtl">مقایسهٔ مستقیم سرعت دوران تعادلی استوانه با مقاله؛ مرکز استوانه در خط تقارن قرار داده شده و مسیر مهاجرت اولیه در این آزمون بررسی نمی‌شود.</p>
<div class="note">FEM-assisted PINN; article targets held out. This is a conditional equilibrium-spin benchmark. The original Gaussian parcel model is not validated by this comparison.</div>
<p><a href="summary.json">Full metrics and criteria</a> · <a href="comparison.csv">CSV table</a> · <a href="comparison.tex">LaTeX table</a> · <a href="REPORT.md">Scientific report</a> · <a href="reference_extraction.png">Reference extraction</a></p>'''+table_html+cards+'</main></html>'
    (out/'index.html').write_text(page)
    return summary
