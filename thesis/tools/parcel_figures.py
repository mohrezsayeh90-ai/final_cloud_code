"""Postprocess one explicitly identified, incomplete-screen coupled parcel run.

Stored predictions remain unchanged. A dense cloud display evaluates the
specified Gaussian using stored centers and the original transfer normalizer.
It does not reconstruct or invent a higher-resolution fluid solution.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def make_parcel_figures(root, save, panel):
    directory=root/'evidence/gaussian/outputs/evaluated_2d_final/cases/high_inertia_seed7'
    cfg=json.loads((directory.parents[1]/'config.json').read_text())
    status=json.loads((directory/'case_status.json').read_text())
    data=np.load(directory/'fields.npz')
    particles=np.genfromtxt(directory/'particles.csv',delimiter=',',names=True)
    budgets=np.genfromtxt(directory/'diagnostics.csv',delimiter=',',names=True)
    t=data['times'];xy=data['xyz'];u=data['velocity'];alpha=data['volume_fraction']
    ids=np.unique(particles['particle']).astype(int)
    n=len(ids);nt=len(t)
    def states(column):
        return np.array([[particles[column][(particles['time']==time)&(particles['particle']==i)][0]
                          for i in ids] for time in t])
    q=np.stack([states('x_unwrapped'),states('y')],axis=-1)
    v=np.stack([states('vx'),states('vy')],axis=-1)
    st=status['physical']['stokes'];mass=status['physical']['represented_mass']
    rho=cfg['density_ratio'];sigma=cfg['width'];lx,ly=cfg['lengths']
    nx,ny=cfg['quadrature'];da=lx*ly/(nx*ny)
    assert (nt,n)==(41,6)
    assert np.array_equal(t,np.unique(particles['time']))
    assert np.array_equal(t,budgets['time'])
    assert np.all(np.diff(t)>0)
    for arr in [q,v,u,alpha,budgets.view(float).reshape(len(budgets),-1)]:assert np.isfinite(arr).all()
    assert np.all((q[:,:,1]>0)&(q[:,:,1]<ly))
    assert (alpha>=0).all() and (budgets['drag_dissipation']>=0).all()

    def raw_kernel(points,centers):
        dx=points[:,0,None]-centers[None,:,0]
        dx=dx-lx*np.floor(dx/lx+.5)
        dy=points[:,1,None]-centers[None,:,1]
        images=max(1,int(np.ceil(8*sigma/lx)))
        gx=sum(np.exp(-.5*((dx+k*lx)/sigma)**2) for k in range(-images,images+1))
        return gx*np.exp(-.5*(dy/sigma)**2)

    normalizers=[];sampled=[];diss=[];alpha_errors=[];norm_errors=[]
    for index in range(nt):
        raw=raw_kernel(xy,q[index]);z=raw.sum(axis=0)*da;w=raw/z
        ui=w.T@u[index]*da
        sampled.append(ui)
        calculated=mass/st*np.sum((ui-v[index])**2)
        diss.append(calculated)
        local=(w*mass/rho).sum(axis=1)
        alpha_errors.append(float(np.max(abs(local-alpha[index]))))
        norm_errors.append(float(np.max(abs(w.sum(axis=0)*da-1))))
        normalizers.append(z)
    sampled=np.array(sampled);diss=np.array(diss)
    ux_error=float(np.max(abs(sampled[:,:,0]-states('sampled_ux'))))
    assert np.allclose(sampled[:,:,0],states('sampled_ux'),rtol=1e-12,atol=1e-13)
    assert np.allclose(diss,budgets['drag_dissipation'],rtol=1e-12,atol=1e-13)
    assert max(alpha_errors)<1e-16
    represented_volume=alpha.sum(axis=1)*da
    target=n*mass/rho
    assert np.allclose(represented_volume,target,rtol=1e-12,atol=1e-17)
    assert not status['quality_criteria_met']
    audit=dict(passed=True,case='high_inertia_seed7',stokes=st,loading=status['physical']['loading'],
        time_samples=nt,parcels=n,all_arrays_finite=True,all_centers_inside=True,
        drag_dissipation_nonnegative=True,max_sampled_ux_reconstruction_error=ux_error,
        max_alpha_reconstruction_error=max(alpha_errors),
        max_drag_dissipation_reconstruction_error=float(np.max(abs(diss-budgets['drag_dissipation']))),
        max_normalization_error=max(norm_errors),represented_volume=target,
        max_represented_volume_error=float(np.max(abs(represented_volume-target))),
        final_vx_range=[float(v[-1,:,0].min()),float(v[-1,:,0].max())],
        downward_displacement_range=[float((q[0,:,1]-q[-1,:,1]).min()),float((q[0,:,1]-q[-1,:,1]).max())],
        complete_coupled_screen_met=False,momentum_rms=status['residuals']['momentum'])
    (root/'generated/parcel_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    colors=['#173F5F','#087F8C','#D49A38','#B95745','#7262A3','#5C8291']
    source='gaussian/cases/high_inertia_seed7/'
    chosen=[0,13,27,40]
    X=xy[:,0].reshape(nx,ny);Y=xy[:,1].reshape(nx,ny)
    def channel(ax):
        ax.axhline(0,color='#233747',lw=1);ax.axhline(ly,color='#233747',lw=1)
        ax.set(xlim=(0,lx),ylim=(0,ly),xlabel=r'$x/H$',ylabel=r'$y/H$')
        ax.set_aspect('equal')
    def markers(ax,index):
        for j in ids:ax.scatter(q[index,j,0]%lx,q[index,j,1],c=colors[j],s=26,edgecolors='white',lw=.8,zorder=5)

    fig,axs=plt.subplots(2,2,figsize=(7.4,4.5),layout='constrained')
    levels=np.linspace(0,float(np.linalg.norm(u,axis=-1).max()),19)
    for index,ax in zip(chosen,axs.flat):
        speed=np.linalg.norm(u[index],axis=1).reshape(nx,ny)
        im=ax.contourf(X,Y,speed,levels=levels,cmap='viridis')
        channel(ax);markers(ax,index);ax.set_title(rf'$tU_0/H={t[index]:.3f}$',loc='left')
    fig.colorbar(im,ax=list(axs.flat),shrink=.77,label=r'$|\mathbf{u}|/U_0$',pad=.025,ticks=np.linspace(levels[0],levels[-1],5),format='%.2f')
    save(fig,'15_parcel_transport','diagnostic coupled prediction',[source+'fields.npz',source+'particles.csv',source+'case_status.json'])

    fig=plt.figure(figsize=(7.4,5.7),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.05,1])
    ax=fig.add_subplot(gs[0,:]);channel(ax);ax.set_xlabel(r'Unwrapped $x/H$');panel(ax,'a','Computed parcel paths')
    for j in ids:
        ax.plot(q[:,j,0],q[:,j,1],c=colors[j],label=f'P{j+1}')
        ax.scatter(q[0,j,0],q[0,j,1],s=27,facecolors='white',edgecolors=colors[j],zorder=4)
        ax.scatter(q[-1,j,0],q[-1,j,1],s=25,c=colors[j],zorder=4)
    ax.legend(ncol=6,loc='upper center',fontsize=8)
    for k,ax,title in [(0,fig.add_subplot(gs[1,0]),'Streamwise acceleration'),(1,fig.add_subplot(gs[1,1]),'Wall-normal response')]:
        for j in ids:ax.plot(t,v[:,j,k],c=colors[j])
        ax.set(xlabel=r'$tU_0/H$',ylabel=[r'$v_{px}/U_0$',r'$v_{py}/U_0$'][k])
        ax.grid(alpha=.3);panel(ax,chr(98+k),title)
    save(fig,'16_parcel_paths_velocities','diagnostic coupled prediction',[source+'particles.csv',source+'case_status.json'])

    # Higher display density evaluates the specified kernel, not a new flow solution.
    xd,yd=np.meshgrid(np.linspace(0,lx,361),np.linspace(0,ly,121))
    points=np.c_[xd.ravel(),yd.ravel()];cloud=[]
    for i in chosen:
        display=(raw_kernel(points,q[i])/normalizers[i]*mass/rho).sum(axis=1)
        cloud.append(display.reshape(xd.shape))
    vmax=max(z.max() for z in cloud)*1e4
    fig,axs=plt.subplots(2,2,figsize=(7.4,4.5),layout='constrained')
    for ax,index,z in zip(axs.flat,chosen,cloud):
        im=ax.contourf(xd,yd,z*1e4,levels=np.linspace(0,vmax,19),cmap='YlGnBu')
        channel(ax);markers(ax,index);ax.set_title(rf'$tU_0/H={t[index]:.3f}$',loc='left')
    fig.colorbar(im,ax=list(axs.flat),shrink=.77,label=r'$10^4\alpha$',pad=.025,ticks=np.linspace(0,vmax,5),format='%.2f')
    save(fig,'17_gaussian_clouds','diagnostic coupled prediction',[source+'fields.npz',source+'particles.csv','gaussian/outputs/evaluated_2d_final/config.json'])

    fig,axs=plt.subplots(2,1,figsize=(7.4,4.3),layout='constrained')
    base=6*xy[:,1]*(1-xy[:,1]);du=u[-1,:,0]-base;vy=u[-1,:,1]
    for ax,z,title,cmap,symmetric in [(axs[0],du,'a  Streamwise change from imposed initial flow','Blues_r',False),
                                    (axs[1],vy,'b  Computed wall-normal carrier velocity','RdBu_r',True)]:
        maximum=float(np.max(abs(z)))
        levels=np.linspace(-maximum,maximum,19) if symmetric else np.linspace(float(z.min()),float(z.max()),19)
        im=ax.contourf(X,Y,z.reshape(nx,ny),levels=levels,cmap=cmap)
        channel(ax);markers(ax,nt-1);ax.set_title(title,loc='left')
        fig.colorbar(im,ax=ax,shrink=.78,pad=.025,ticks=np.linspace(levels[0],levels[-1],5),format='%.2g',label=r'$(u-u_0)/U_0$' if not symmetric else r'$u_y/U_0$')
    save(fig,'18_parcel_carrier_response','diagnostic coupled prediction',[source+'fields.npz',source+'particles.csv'])

    fig,axs=plt.subplots(2,2,figsize=(7.4,4.9),layout='constrained')
    specs=[('energy','Total kinetic energy',r'$K$',colors[0]),
           ('drag_dissipation','Nonnegative drag dissipation',r'$D_{drag}$',colors[1]),
           ('energy_balance_residual','Signed energy-balance discrepancy',r'$r_E$',colors[3]),
           ('momentum_balance_residual','Momentum-balance norm',r'$|\mathbf{r}_M|$',colors[2])]
    for i,(ax,(key,title,label,color)) in enumerate(zip(axs.flat,specs)):
        ax.plot(budgets['time'],budgets[key],c=color);ax.set(xlabel=r'$tU_0/H$',ylabel=label)
        if key=='energy_balance_residual':ax.axhline(0,c='#738390',ls='--',lw=.8)
        ax.grid(alpha=.3);panel(ax,chr(97+i),title)
    save(fig,'19_parcel_balances','diagnostic coupled prediction',[source+'diagnostics.csv',source+'case_status.json'])
    print('Added 5 parcel figures; reconstructed sampled velocity, cloud mass and drag dissipation from stored arrays.')
