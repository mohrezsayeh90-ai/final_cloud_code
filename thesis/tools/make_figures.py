"""Publication figures and numerical checks from the archived evidence only.

Illustrations are explicitly registered as schematics. No simulation fields,
trajectory samples, or numerical results are synthesized for the manuscript.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'build/matplotlib-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parents[1]/'build/xdg-cache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'evidence/basic/results/final_study'
G=ROOT/'evidence/gaussian/outputs/evaluated_2d_final'
F=ROOT/'figures'
NAVY='#173F5F'; TEAL='#087F8C'; GOLD='#D49A38'; INK='#233747'
CORAL='#B95745'; PALE='#EBF2F6'; GREY='#738390'
plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix', 'font.size':9,
 'axes.labelsize':9,'axes.titlesize':10,'axes.titleweight':'medium',
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5B2BC',
 'axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black',
 'text.color':'black','grid.color':'#DDE5EB','grid.alpha':.7,
 'lines.linewidth':1.6,'legend.frameon':False,'legend.fontsize':8,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':240,
 'figure.constrained_layout.use':True})
REGISTER=[]

def csv(p): return np.genfromtxt(p,delimiter=',',names=True,encoding='utf8')
def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def save(fig,name,kind,inputs):
    for label in fig.findobj(match=matplotlib.text.Text):
        label.set_color('black')
    for ext in ['pdf','png','svg']:
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',facecolor='white',metadata=metadata)
        if ext=='svg':
            target=F/f'{name}.{ext}'
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    resolved=[]
    for source in inputs:
        path,sep,field=source.partition(':')
        if path.startswith('basic/data/') or path=='basic/metrics.json':
            path='evidence/basic/results/final_study/'+path.removeprefix('basic/')
        elif path.startswith('gaussian/validation/') or path.startswith('gaussian/cases/'):
            path='evidence/gaussian/outputs/evaluated_2d_final/'+path.removeprefix('gaussian/')
        else:path='evidence/'+path
        assert (ROOT/path).is_file(),path
        resolved.append(path+('#'+field if sep else ''))
    REGISTER.append(dict(name=name,kind=kind,inputs=resolved))

def panel(ax,label,title):
    ax.set_title(f'{label}  {title}',loc='left',pad=9)

def walls(ax,x,y0,y1):
    ax.fill_between(x,y0,y1,color=PALE,zorder=0)
    ax.plot(x,y0,color=NAVY,lw=1.5);ax.plot(x,y1,color=NAVY,lw=1.5)

def route():
    fig,axs=plt.subplots(3,1,figsize=(7.4,7.6),height_ratios=[1,1,1.15])
    ax=axs[0]
    a=np.linspace(np.pi,0,200);R=.35;h=.15
    c=np.vstack([np.c_[R+R*np.cos(a),R*np.sin(a)],
                 np.c_[3*R+R*np.cos(a[1:]),-R*np.sin(a[1:])]])
    t=np.gradient(c,axis=0);t/=np.linalg.norm(t,axis=1)[:,None]
    n=np.c_[-t[:,1],t[:,0]]
    lower=c-h*n;upper=c+h*n
    poly=np.vstack([lower,upper[::-1]])
    ax.fill(poly[:,0],poly[:,1],color=PALE)
    for q in [lower,upper]:ax.plot(*q.T,color=NAVY,lw=1.5)
    ax.plot(*c.T,ls='--',color=TEAL,lw=1)
    for i in [30,95,160,260,340]:ax.scatter(*c[i],s=22,color=GOLD,zorder=3)
    ax.set_aspect('equal');ax.axis('off');panel(ax,'01','Dilute tracking in a curved channel')
    ax=axs[1];x=np.linspace(0,3,401);walls(ax,x,np.zeros_like(x),np.ones_like(x))
    xx,yy=np.meshgrid(x,np.linspace(0,1,100))
    z=sum(np.exp(-((xx-qx)**2+(yy-qy)**2)/(2*.14**2)) for qx,qy in [(.5,.4),(1.2,.68),(2.0,.43)])
    ax.contourf(xx,yy,z,levels=np.linspace(.04,1.25,12),cmap='Blues',alpha=.7)
    ax.scatter([.5,1.2,2.0],[.4,.68,.43],s=22,color=GOLD)
    ax.set_aspect('equal');ax.set(xlim=(0,3),ylim=(-.06,1.06));ax.axis('off')
    panel(ax,'02','Two-way Gaussian parcel coupling in a straight channel')
    ax=axs[2];x=np.linspace(0,8,801);delta=.4;ell=1.2
    f=np.where(abs(x-4)<ell,((1+np.cos(np.pi*(x-4)/ell))/2)**2,0)
    hh=(1-delta*f)/2;walls(ax,x,-hh,hh)
    ax.plot(x,np.zeros_like(x),ls='--',color=GREY,lw=.9)
    ax.annotate('',xy=(4,hh[400]),xytext=(4,-hh[400]),arrowprops=dict(arrowstyle='<->',color=TEAL))
    ax.text(4.18,0,r'$H_t=(1-\delta)H$',va='center',fontsize=9)
    ax.set_aspect('equal');ax.set(xlim=(0,8),ylim=(-.7,.7));ax.axis('off')
    panel(ax,'03','Smooth constriction: proposed extension')
    save(fig,'01_study_route','schematic',[])

def strand(ax,x0,x1,n0,n1,color):
    y0=np.linspace(.1,.9,n0);y1=np.linspace(.1,.9,n1)
    for i,a in enumerate(y0):
        for j,b in enumerate(y1):
            if (i+j)%2: continue
            verts=[(x0,a),(x0+.16,a),(x1-.16,b),(x1,b)]
            ax.add_patch(PathPatch(MPath(verts,[1,4,4,4]),fc='none',ec=color,lw=.42,alpha=.14))

def neural_row(ax,color,labels_in,labels_out,width,depth,title,footer):
    xs=np.linspace(.9,5.9,depth+2);ns=[len(labels_in)]+[6]*depth+[len(labels_out)]
    for a,b,na,nb in zip(xs[:-1],xs[1:],ns[:-1],ns[1:]):strand(ax,a,b,na,nb,color)
    for i,(x,n) in enumerate(zip(xs,ns)):
        for j,y in enumerate(np.linspace(.1,.9,n)):
            ax.add_patch(Circle((x,y),.038 if i in [0,len(xs)-1] else .027,
                               fc='white' if i in [0,len(xs)-1] else color,ec=color,lw=1,zorder=3))
            if i==0:ax.text(x-.09,y,labels_in[j],ha='right',va='center',fontsize=9)
            if i==len(xs)-1:ax.text(x+.09,y,labels_out[j],ha='left',va='center',fontsize=9)
        if 0<i<len(xs)-1:ax.text(x,1.03,str(width),ha='center',color=color,fontsize=9)
    ax.text(.05,1.38,title,weight='bold',fontsize=12,color=color)
    ax.text(3.5,-.16,footer,ha='center',fontsize=9)
    ax.set(xlim=(.0,6.7),ylim=(-.28,1.55));ax.axis('off')

def architecture():
    fig,axs=plt.subplots(2,1,figsize=(7.4,6.4))
    neural_row(axs[0],NAVY,[r'$x$',r'$y$',r'$R_{in}$',r'$R_{out}$',r'$Re$'],
       [r'$u$',r'$v$',r'$p$'],70,5,'FlowNet  |  continuous fluid field',
       'Five hidden layers · tanh activation · PDE and boundary losses')
    neural_row(axs[1],TEAL,[r'$u$',r'$v$',r'$p$',r'$d_p$',r'$x_p$',r'$y_p$',r'$u_p$',r'$v_p$'],
       [r'$F_x$',r'$F_y$'],90,5,'ForceNet  |  prescribed Stokes response',
       'Five hidden layers · exact zero-slip constraint · algebraic drag loss')
    save(fig,'02_baseline_architecture','schematic',['basic/particle_pinn/models.py'])

def basic_results(checks):
    p=csv(B/'data/poiseuille.csv');q=csv(B/'data/poiseuille_pressure.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))
    ax=axs[0];ax.plot(p['eta'],p['exact_u'],color=INK,label='Analytical')
    ax.plot(p['eta'][::5],p['fem_u'][::5],ls='',marker='o',mfc='white',mec=GOLD,ms=4,label='FEM')
    ax.plot(p['eta'][::5],p['pinn_u'][::5],ls='',marker='.',color=TEAL,ms=4,label='PINN')
    ax.set(xlabel=r'$y/h$',ylabel=r'$u/U_0$');panel(ax,'a','Plane Poiseuille velocity');ax.legend()
    ax=axs[1];order=np.argsort(q['x']);x=q['x'][order]
    for k,c,l in [('exact_p',INK,'Analytical'),('fem_p',GOLD,'FEM'),('pinn_p',TEAL,'PINN')]:
        y=q[k]-np.mean(q[k]);ax.plot(x,y[order],color=c,label=l,lw=1.5,ls='--' if k=='fem_p' else '-')
    ax.set(xlabel=r'$x/L_0$',ylabel=r'$p-\overline{p}$');panel(ax,'b','Gauge-aligned pressure')
    save(fig,'03_poiseuille','measured numerical result',['basic/data/poiseuille.csv','basic/data/poiseuille_pressure.csv'])
    a=np.load(B/'data/plotted_fields.npz');shape=tuple(a['shape']);xy=a['xy'];pred=a['predicted'];ref=a['reference']
    pc=pred[:,2]-pred[:,2].mean();rc=ref[:,2]-ref[:,2].mean()
    values=[np.linalg.norm(ref[:,:2],axis=1),np.linalg.norm(pred[:,:2],axis=1),np.linalg.norm(pred[:,:2]-ref[:,:2],axis=1),rc,pc,pc-rc]
    titles=['FEM speed','PINN speed','Velocity discrepancy','FEM pressure','PINN pressure','Pressure discrepancy']
    fig,axs=plt.subplots(2,3,figsize=(7.5,5.2))
    speedmax=max(values[0].max(),values[1].max());pressmax=max(abs(rc).max(),abs(pc).max())
    for i,(ax,z,t) in enumerate(zip(axs.flat,values,titles)):
        kw={}
        if i<2:kw=dict(vmin=0,vmax=speedmax,cmap='viridis')
        elif i in [3,4]:kw=dict(vmin=-pressmax,vmax=pressmax,cmap='RdBu_r')
        elif i==5:kw=dict(vmin=-abs(z).max(),vmax=abs(z).max(),cmap='RdBu_r')
        else:kw=dict(vmin=0,cmap='magma')
        im=ax.pcolormesh(xy[:,0].reshape(shape),xy[:,1].reshape(shape),z.reshape(shape),shading='gouraud',rasterized=True,**kw)
        for j in [0,-1]:ax.plot(xy[:,0].reshape(shape)[:,j],xy[:,1].reshape(shape)[:,j],color=INK,lw=.7)
        ax.set_aspect('equal');ax.set(xlabel=r'$x/L_0$',ylabel=r'$y/L_0$');panel(ax,chr(97+i),t)
        fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.86,pad=.08,ticks=plt.MaxNLocator(3))
    save(fig,'04_curved_fields','measured numerical result',['basic/data/plotted_fields.npz'])
    v=csv(B/'data/field_validation.csv');d=csv(B/'data/drag_validation.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.4,2.8))
    items=[(np.r_[v['fem_u'],v['fem_v']],np.r_[v['pinn_u'],v['pinn_v']],'Velocity components',NAVY),
           (v['fem_p']-v['fem_p'].mean(),v['pinn_p']-v['pinn_p'].mean(),'Centered pressure',TEAL),
           (np.r_[d['exact_x'],d['exact_y']],np.r_[d['predicted_x'],d['predicted_y']],'Normalized drag',GOLD)]
    for i,(ax,(x,y,title,c)) in enumerate(zip(axs,items)):
        ax.scatter(x,y,s=5,alpha=.25,color=c,rasterized=True,edgecolor='none')
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color=INK,ls='--',lw=.8)
        ax.set(xlabel='Independent reference',ylabel='Network prediction');panel(ax,chr(97+i),title)
    save(fig,'05_heldout_agreement','measured numerical result',['basic/data/field_validation.csv','basic/data/drag_validation.csv'])
    mesh=csv(B/'data/mesh_sensitivity.csv');seed=csv(B/'data/seed_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.1))
    ax=axs[0];ax.plot(mesh['triangles'][:-1],mesh['velocity_error_to_finest'][:-1]*100,'o-',color=TEAL)
    ax.set(xlabel='Number of triangles',ylabel='Velocity difference to finest (%)');panel(ax,'a','Reference-grid sensitivity')
    ax.annotate('Finest reference: 3,026 triangles',(.04,.88),xycoords='axes fraction',fontsize=8)
    ax=axs[1]
    for i,(k,c,l) in enumerate([('velocity_relative_l2',NAVY,'Velocity'),('pressure_relative_l2',GOLD,'Pressure')]):
        ax.scatter(np.full(2,i),seed[k]*100,s=35,facecolor='white',edgecolor=c,zorder=3)
        for val,num in zip(seed[k],seed['seed']):ax.annotate(str(int(num)),(i,val*100),xytext=(8,0),textcoords='offset points',va='center',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['Velocity','Pressure'],ylabel='Relative sample error (%)',xlim=(-.5,1.6));panel(ax,'b','Two independently trained seeds')
    save(fig,'06_reference_and_seeds','measured numerical result',['basic/data/mesh_sensitivity.csv','basic/data/seed_summary.csv'])
    m=json.loads((B/'metrics.json').read_text());flux=np.array(m['flow']['flux']);Q0=2/3*m['geometry']['width']
    fig,ax=plt.subplots(figsize=(7.4,2.8));ax.plot(np.linspace(0,1,len(flux)),flux/Q0,'o-',color=TEAL,ms=3,label='PINN sections')
    ax.axhline(1,color=INK,ls='--',lw=1,label='Prescribed flux');ax.set(xlabel=r'$s/L$',ylabel=r'$Q(s)/Q_0$',ylim=(.97,1.025));ax.legend(ncol=2)
    save(fig,'07_flux','measured numerical result',['basic/metrics.json:flow.flux'])
    # Recompute displayed aggregate errors, retaining CSV precision differences.
    for name,x,y,expected in [
       ('curved_velocity',np.c_[v['pinn_u'],v['pinn_v']],np.c_[v['fem_u'],v['fem_v']],m['flow']['velocity_relative_l2']),
       ('curved_pressure',v['pinn_p']-v['pinn_p'].mean(),v['fem_p']-v['fem_p'].mean(),m['flow']['pressure_relative_l2']),
       ('poiseuille_pressure',q['pinn_p']-q['pinn_p'].mean(),q['exact_p']-q['exact_p'].mean(),m['poiseuille']['pinn_pressure_relative_l2']),
       ]:
        measured=rel(x,y);assert np.isclose(measured,expected,rtol=2e-5,atol=2e-8),(name,measured,expected)
        checks.append(dict(metric=name,recomputed=measured,recorded=expected))
    # The velocity metric includes transverse velocity; the profile CSV alone
    # stores only u. Re-evaluate the archived checkpoint to retain that distinction.
    import importlib.util
    import sys
    import torch
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        out=importlib.util.module_from_spec(spec);sys.modules[name]=out
        spec.loader.exec_module(out);return out
    geo=module('thesis_geometry',ROOT/'evidence/basic/particle_pinn/geometry.py')
    mod=module('thesis_models',ROOT/'evidence/basic/particle_pinn/models.py')
    channel=geo.Channel('straight');model=mod.FlowNet(channel,1.)
    model.load_state_dict(torch.load(B/'models/poiseuille.pt',map_location='cpu',weights_only=True))
    xy=channel.map(np.ones(len(p)),p['eta']);uv=model.evaluate(xy)[:,:2]
    expected=np.c_[p['exact_u'],np.zeros(len(p))]
    measured=rel(uv,expected);recorded=m['poiseuille']['pinn_velocity_relative_l2']
    assert np.isclose(measured,recorded,rtol=2e-5)
    checks.append(dict(metric='poiseuille_full_velocity',recomputed=measured,recorded=recorded))
    channel=geo.Channel('serpentine');scale=2*channel.length/channel.half_width**2
    force=mod.ForceNet(channel,scale)
    force.load_state_dict(torch.load(B/'models/force.pt',map_location='cpu',weights_only=True))
    rng=np.random.default_rng(42+190)
    flow=rng.uniform(-1.3,1.3,(2048,3));vp=rng.uniform(-1.3,1.3,(2048,2));flow[:,2]*=scale
    inputs=np.column_stack((flow,rng.uniform(.04,.4,(2048,1))*channel.half_width,channel.sample(2048,rng),vp))
    with torch.no_grad():pred=force.normalized_drag(mod.tensor(inputs)).numpy()
    measured=rel(pred,flow[:,:2]-vp);recorded=m['force']['heldout_relative_l2']
    assert np.isclose(measured,recorded,rtol=2e-5)
    checks.append(dict(metric='drag_2048_heldout',recomputed=measured,recorded=recorded))
    checks.append(dict(metric='drag_1000_plot_samples',recomputed=rel(np.c_[d['predicted_x'],d['predicted_y']],np.c_[d['exact_x'],d['exact_y']]),recorded=None))

def gaussian_architecture():
    fig=plt.figure(figsize=(7.4,5.0));gs=fig.add_gridspec(2,1,height_ratios=[1.7,1])
    ax=fig.add_subplot(gs[0]);ax.set_aspect('equal');ax.set(xlim=(-3.8,3.8),ylim=(-1.9,1.9));ax.axis('off')
    for center,color,title,label in [((-2.1,.1),NAVY,'Fluid network',r'$(x,y,t)\mapsto(u,v,p)$'),((2.1,.1),TEAL,'Parcel network',r'$t\mapsto(\mathbf{q}_i,\mathbf{v}_i)$')]:
        cx,cy=center
        rings=[]
        for radius,n in [(.35,6),(.68,10),(1.02,14)]:
            angle=np.linspace(0,2*np.pi,n,endpoint=False)
            points=np.c_[cx+radius*np.cos(angle),cy+radius*np.sin(angle)];rings.append(points)
            ax.scatter(*points.T,s=12,c=color,zorder=3)
        for r0,r1 in zip(rings[:-1],rings[1:]):
            for i,p in enumerate(r0):
                for j in [i%len(r1),(i+3)%len(r1)]:ax.plot([p[0],r1[j,0]],[p[1],r1[j,1]],color=color,lw=.6,alpha=.17)
        ax.text(cx,-1.22,title,ha='center',weight='bold',color=color,fontsize=11)
        ax.text(cx,-1.58,label,ha='center',fontsize=10)
    ax.text(0,.45,r'$W_{ai}$',ha='center',fontsize=18,color=GOLD)
    ax.text(0,-.02,'shared Gaussian\nquadrature',ha='center',va='center',fontsize=9)
    for a,b,rad,c in [((-1.02,.4),(1.02,.4),-.55,NAVY),((1.02,-.25),(-1.02,-.25),-.55,TEAL)]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=c,lw=1.6,connectionstyle=f'arc3,rad={rad}'))
    ax.text(0,1.32,r'$\mathbf{U}_i=\sum_aW_{ai}\mathbf{u}_a\Delta A_a$',ha='center',fontsize=10)
    ax=fig.add_subplot(gs[1]);ax.axis('off')
    ax.text(.5,.78,'Residuals bind the two networks',ha='center',weight='bold',fontsize=11)
    ax.text(.5,.38,r'Fluid momentum  ·  incompressibility  ·  parcel kinematics  ·  parcel acceleration',ha='center',fontsize=9)
    ax.text(.5,.02,'Three hidden layers × 32 tanh units per network · float64',ha='center',fontsize=9,color=GREY)
    save(fig,'08_gaussian_architecture','schematic',['gaussian/parcel_pinn/model.py'])

def gaussian_results(checks):
    v1=csv(G/'validation/V01/comparison.csv');v2=csv(G/'validation/V02/comparison.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.1))
    for ax,ref,pred,label in [(axs[0],'v_over_vt_reference','v_over_vt_pinn',r'$v/v_\infty$'),(axs[1],'distance_over_vt_tau_reference','distance_over_vt_tau_pinn',r'$z/(v_\infty\tau_p)$')]:
        ax.plot(v1['time'],v1[ref],color=INK,label='Analytical')
        ax.plot(v1['time'][::20],v1[pred][::20],ls='',marker='o',color=TEAL,mfc='white',ms=4,label='PINN')
        ax.set(xlabel=r'$t/\tau_p$',ylabel=label);ax.legend()
    panel(axs[0],'a','Stokes velocity response');panel(axs[1],'b','Integrated displacement')
    save(fig,'09_stokes_response','measured numerical result',['gaussian/validation/V01/comparison.csv'])
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))
    ax=axs[0]
    for p,c,l in [('fluid_u',NAVY,'Fluid'),('particle_v',TEAL,'Particle')]:
        ax.plot(v2['time'],v2[p+'_reference'],color=c,label=l+' analytical')
        ax.plot(v2['time'][::20],v2[p+'_pinn'][::20],ls='',marker='o',mfc='white',mec=c,ms=4,label=l+' PINN')
    ax.set(xlabel='Dimensionless time',ylabel='Velocity');ax.legend(fontsize=7);panel(ax,'a','Two-way momentum exchange')
    ax=axs[1];ax.plot(v2['time'],v2['kinetic_energy_pinn'],color=TEAL,label='Kinetic energy')
    ax.axhline(v2['total_momentum_reference'][0],color=GOLD,ls='--',label='Total momentum')
    ax.plot(v2['time'][::20],v2['total_momentum_pinn'][::20],ls='',marker='.',color=GOLD,ms=3)
    ax.set(xlabel='Dimensionless time',ylabel='Nondimensional value');ax.legend();panel(ax,'b','Energy decay and momentum constraint')
    save(fig,'10_two_way_response','measured numerical result',['gaussian/validation/V02/comparison.csv'])
    v=csv(G/'validation/V10/comparison.csv');fig,axs=plt.subplots(1,3,figsize=(7.4,2.8))
    for ax,k,l in zip(axs,['u','v','p'],['Velocity u','Velocity v','Pressure p']):
        a=v[k+'_reference'];b=v[k+'_pinn'];ax.scatter(a,b,s=5,alpha=.3,color=TEAL,edgecolor='none',rasterized=True)
        low=min(a.min(),b.min());high=max(a.max(),b.max());ax.plot([low,high],[low,high],ls='--',color=INK,lw=.8)
        ax.set(xlabel='Analytical',ylabel='PINN');ax.set_title(l,loc='left')
    save(fig,'11_taylor_green','measured numerical result',['gaussian/validation/V10/comparison.csv'])
    v4=csv(G/'validation/V04.csv');fig,axs=plt.subplots(1,2,figsize=(7.4,3))
    ax=axs[0];ax.plot(v4['time'],v4['overlap_reference'],color=INK,label='Analytical overlap')
    ax.plot(v4['time'][::30],v4['overlap_pinn'][::30],ls='',marker='o',mfc='white',mec=TEAL,ms=4,label='Product quadrature')
    ax.set(xlabel='Reference trajectory parameter',ylabel='Gaussian overlap');ax.legend();panel(ax,'a','Independent 3D kernel integral')
    v3=csv(G/'validation/V03.csv');ax=axs[1];ax.scatter(np.arange(len(v3)),(v3['integral']-1)/np.finfo(float).eps,color=GOLD,s=30)
    ax.axhline(0,color=INK,lw=.8);ax.set(xticks=np.arange(len(v3)),xticklabels=['Seam L','Seam R','Wrapped','Lower wall','Upper wall'],ylabel=r'$(\sum_a W_{ai}\Delta A_a-1)/\epsilon_{mach}$')
    ax.tick_params(axis='x',labelrotation=25);panel(ax,'b','Discrete normalization near boundaries')
    save(fig,'12_kernel_verification','measured numerical result',['gaussian/validation/V04.csv','gaussian/validation/V03.csv'])
    ver=json.loads((G/'validation/verification.json').read_text())
    for case,data,columns in [('V01',v1,['v_over_vt','distance_over_vt_tau']),('V02',v2,['fluid_u','particle_v']),('V10',v,['u','v','p'])]:
        record=next(z for z in ver if z['case_id']==case)
        for k in columns:
            val=rel(data[k+'_pinn'],data[k+'_reference']);exp=record['metrics'][k]['relative_l2']
            assert np.isclose(val,exp,rtol=1e-8,atol=1e-11)
            checks.append(dict(metric=case+'_'+k,recomputed=val,recorded=exp))

def constriction():
    fig,axs=plt.subplots(2,1,figsize=(7.4,5.0),height_ratios=[1.2,1])
    ax=axs[0];x=np.linspace(0,8,801);delta=.4;ell=1.2
    f=np.where(abs(x-4)<ell,((1+np.cos(np.pi*(x-4)/ell))/2)**2,0);h=(1-delta*f)/2
    walls(ax,x,-h,h);ax.plot(x,x*0,ls='--',lw=.7,color=GREY)
    for xc,txt in [(1,'Inlet'),(4,'Throat'),(7,'Outlet')]:
        ax.axvline(xc,color=GREY,ls=':',lw=.8);ax.text(xc,.67,txt,ha='center',fontsize=9)
    ax.annotate('',xy=(4,h[400]),xytext=(4,-h[400]),arrowprops=dict(arrowstyle='<->',color=TEAL))
    ax.text(4.17,0,r'$H_t$',va='center',fontsize=10,color=TEAL)
    ax.annotate('',xy=(5.2,-.68),xytext=(2.8,-.68),arrowprops=dict(arrowstyle='<->',color=NAVY))
    ax.text(4,-.79,r'$2\ell$',ha='center',fontsize=10)
    ax.set(xlim=(0,8),ylim=(-1,1),aspect='equal');ax.axis('off');panel(ax,'a','Parameterized geometry — design illustration')
    ax=axs[1]
    for delta,c in [(0,NAVY),(.2,TEAL),(.4,GOLD)]:
        f=np.where(abs(x-4)<ell,((1+np.cos(np.pi*(x-4)/ell))/2)**2,0)
        ax.plot(x,1-delta*f,color=c,label=rf'$\delta={delta:.1f}$')
    ax.set(xlabel=r'$x/H$',ylabel=r'$2h(x)/H$',ylim=(.54,1.06));ax.legend(ncol=3)
    panel(ax,'b','Proposed height profiles; no flow solution is plotted')
    save(fig,'13_constriction_geometry','schematic',[])
    # Normalized kernel illustration in a curved boundary domain: explicitly design only.
    fig,ax=plt.subplots(figsize=(7.4,2.8));xx,yy=np.meshgrid(np.linspace(2.4,5.6,401),np.linspace(-.6,.6,151))
    f=np.where(abs(xx-4)<1.2,((1+np.cos(np.pi*(xx-4)/1.2))/2)**2,0);hh=(1-.4*f)/2
    z=np.exp(-((xx-3.9)**2+(yy-.2)**2)/(2*.16**2));z[abs(yy)>hh]=np.nan
    ax.contourf(xx,yy,z,levels=np.linspace(.02,1,14),cmap='Blues')
    x=np.linspace(2.4,5.6,401);f=np.where(abs(x-4)<1.2,((1+np.cos(np.pi*(x-4)/1.2))/2)**2,0);h=(1-.4*f)/2
    ax.plot(x,h,color=NAVY);ax.plot(x,-h,color=NAVY);ax.scatter([3.9],[.2],s=35,c=GOLD,zorder=3)
    ax.text(4.5,-.17,'Fluid-domain\nquadrature',ha='center',fontsize=9)
    ax.annotate('Truncated kernel support',xy=(3.95,.31),xytext=(4.45,.52),arrowprops=dict(arrowstyle='-',color=GREY),fontsize=8)
    ax.set(xlabel=r'$x/H$',ylabel=r'$y/H$',aspect='equal',xlim=(2.4,5.6),ylim=(-.58,.65))
    save(fig,'14_boundary_kernel','schematic',[])

def main():
    F.mkdir(exist_ok=True);checks=[]
    route();architecture();basic_results(checks);gaussian_architecture();gaussian_results(checks);constriction()
    from parcel_figures import make_parcel_figures
    make_parcel_figures(ROOT,save,panel)
    (ROOT/'generated/figure_manifest.json').write_text(json.dumps(REGISTER,indent=2)+'\n')
    (ROOT/'generated/numerical_validation.json').write_text(json.dumps(dict(
        passed=True,recomputed_metrics=checks,schematics_are_not_results=True),indent=2)+'\n')
    print(f'Created {len(REGISTER)} figures; independently recomputed {len(checks)} metrics.')

if __name__=='__main__':main()
