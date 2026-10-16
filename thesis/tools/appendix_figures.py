"""Additional figures reconstructed from archived numerical records only."""
import importlib.util
import json
import os
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(_ROOT/'build/matplotlib-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(_ROOT/'build/xdg-cache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def make_appendix_figures(root,save,panel):
    b=root/'evidence/basic/results/final_study'
    g=root/'evidence/gaussian/outputs/evaluated_2d_final'
    c=root/'evidence/gaussian/outputs/couette_validation'
    def read(path):return np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8')
    metrics=json.loads((b/'metrics.json').read_text())
    verification={x['case_id']:x for x in json.loads((g/'validation/verification.json').read_text())}
    navy,teal,gold,coral='#173F5F','#087F8C','#D49A38','#B95745'
    checks={}

    a=read(b/'data/relaxation.csv');tau=metrics['relaxation']['tau']
    error=a['numerical']-a['analytical']
    maximum=float(np.max(abs(error)))
    assert np.isclose(maximum,metrics['relaxation']['max_absolute_error'],rtol=1e-9)
    assert np.allclose(a['analytical'],1-np.exp(-a['t']/tau),rtol=1e-13,atol=1e-14)
    checks['baseline_integrator']={'samples':len(a),'tau':tau,'max_absolute_error':maximum}
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
    axs[0].plot(a['t']/tau,a['analytical'],c=navy,label='Analytical')
    axs[0].plot(a['t'][::7]/tau,a['numerical'][::7],ls='',marker='o',ms=4,mfc='white',mec=gold,label='Radau integration')
    axs[0].set(xlabel=r'$t/\tau$',ylabel='Dimensionless velocity');axs[0].legend()
    axs[1].plot(a['t']/tau,error*1e10,c=teal)
    axs[1].axhline(0,c='black',lw=.7,ls=':')
    axs[1].set(xlabel=r'$t/\tau$',ylabel=r'$10^{10}(v_{\rm num}-v_{\rm exact})$')
    panel(axs[0],'a','Uniform-flow response');panel(axs[1],'b','Signed integration error')
    save(fig,'20_baseline_relaxation','additional verification result',[
        'basic/data/relaxation.csv','basic/metrics.json:relaxation'])

    tracks=read(b/'data/trajectories.csv')
    isolated=metrics['force_trajectory_isolation'];etas=np.array([x['seed_eta'] for x in isolated])
    width=metrics['geometry']['width'];errors=[];coverage=[]
    fig=plt.figure(figsize=(7.4,5.5));gs=fig.add_gridspec(2,2,height_ratios=[1.35,1])
    ax=fig.add_subplot(gs[0,:]);geo=load_module('appendix_geometry',root/'evidence/basic/particle_pinn/geometry.py')
    channel=geo.Channel('serpentine');s=np.linspace(0,channel.length,500)
    lower,upper=channel.map(s,-np.ones_like(s)),channel.map(s,np.ones_like(s))
    polygon=np.vstack([lower,upper[::-1]])
    ax.fill(polygon[:,0],polygon[:,1],color='#F3F5F6',zorder=0)
    for wall in [lower,upper]:ax.plot(wall[:,0],wall[:,1],c='black',lw=.7)
    colors=plt.get_cmap('tab10')(np.arange(7))
    for i,record in enumerate(isolated):
        group=[]
        for source,style in [('PINN + ForceNet','-'),('PINN + Stokes','--')]:
            subset=tracks[(tracks['source']==source)&(tracks['case']=='baseline')&np.isclose(tracks['eta'],record['seed_eta'],atol=1e-12)]
            assert len(np.unique(subset['track']))==1 and len(subset)>1
            assert np.all(np.diff(subset['t'])>0)
            group.append(subset)
            ax.plot(subset['x'],subset['y'],c=colors[i],ls=style,lw=1.25 if style=='-' else .85)
        p,q=group;end=min(p['t'][-1],q['t'][-1]);times=np.linspace(0,end,150)
        xy=[np.column_stack([np.interp(times,x['t'],x[k]) for k in ['x','y']]) for x in group]
        discrepancy=float(np.sqrt(np.mean(np.sum((xy[0]-xy[1])**2,axis=1))))
        common=float(end/max(p['t'][-1],q['t'][-1]))
        assert np.isclose(discrepancy,record['rms_position_error'],rtol=1e-8,atol=1e-13)
        assert np.isclose(common,record['common_fraction'],rtol=1e-12)
        assert p['outcome'][-1]==q['outcome'][-1] and record['outcome_agreement']
        errors.append(100*discrepancy/width);coverage.append(100*common)
    ax.set(xlabel=r'$x/L_0$',ylabel=r'$y/L_0$',aspect='equal')
    fig.legend(handles=[Line2D([],[],c='black',label='PINN + ForceNet'),Line2D([],[],c='black',ls='--',label='PINN + Stokes')],loc='outside upper right',ncol=2,fontsize=8)
    panel(ax,'a','All seven paths; the same carrier field')
    ax=fig.add_subplot(gs[1,0]);ax.bar(etas,errors,width=.13,color=colors,alpha=.85)
    ax.set(xlabel=r'Initial position $\eta$',ylabel='RMS position / width (%)',ylim=(0,.5),xticks=etas)
    ax.tick_params(axis='x',labelsize=7);ax.set_xticklabels([f'{eta:.2f}' for eta in etas])
    panel(ax,'b','Drag-isolation discrepancy')
    ax=fig.add_subplot(gs[1,1]);ax.scatter(etas,coverage,c=colors,s=35,zorder=3)
    ax.axhline(100,c='black',lw=.7,ls=':');ax.set(xlabel=r'Initial position $\eta$',ylabel='Common-time coverage (%)',ylim=(99.8,100.025),xticks=etas)
    ax.tick_params(axis='x',labelsize=7);ax.set_xticklabels([f'{eta:.2f}' for eta in etas])
    panel(ax,'c','Matched time intervals')
    checks['drag_isolation']={'paths':7,'endpoint_classes_matching':7,'max_rms_position_percent_width':max(errors),'minimum_common_time_percent':min(coverage),'all_recorded_metrics_recomputed':True}
    save(fig,'21_drag_isolation','additional verification result',[
        'basic/data/trajectories.csv','basic/metrics.json:force_trajectory_isolation','basic/particle_pinn/geometry.py'])

    fig,axs=plt.subplots(1,2,figsize=(7.4,3.0));monitors={}
    for seed,color in [(42,navy),(43,teal)]:
        a=read(b/f'data/flow_loss_seed_{seed}.csv')
        assert np.isfinite(a['monitor']).all() and (a['monitor']>0).all()
        axs[0].semilogy(a['epoch'],a['monitor'],c=color,label=f'Seed {seed}')
        monitors[str(seed)]={'logged_points':len(a),'initial':float(a['monitor'][0]),'final':float(a['monitor'][-1])}
    a=read(b/'data/force_loss.csv')
    for key,color,label in [('train',teal,'Training'),('validation',gold,'Validation')]:
        assert (a[key]>0).all() and np.isfinite(a[key]).all()
        axs[1].semilogy(a['epoch'],a[key],c=color,label=label)
    axs[0].set(xlabel='Recorded optimization counter',ylabel='Flow monitor')
    axs[1].set(xlabel='Recorded epoch',ylabel='ForceNet mean-square objective')
    for ax in axs:ax.legend();ax.grid(alpha=.3)
    panel(axs[0],'a','Two flow-training histories');panel(axs[1],'b','Learned drag: train and validation')
    checks['training_histories']={'flow':monitors,'force_logged_points':len(a),'smoothing':False}
    save(fig,'22_baseline_training','optimization history',[
        'basic/data/flow_loss_seed_42.csv','basic/data/flow_loss_seed_43.csv','basic/data/force_loss.csv'])

    fig,axs=plt.subplots(3,1,figsize=(7.4,6.2));error_checks={}
    for i,(test,series,title) in enumerate([
        ('V01',[('v_over_vt',navy,'Velocity'),('distance_over_vt_tau',gold,'Displacement')],'Reduced Stokes response'),
        ('V02',[('fluid_u',navy,'Fluid velocity'),('particle_v',teal,'Particle velocity')],'Uniform two-way relaxation'),
        ('V10',[('u',navy,r'$u$'),('v',teal,r'$v$'),('p',coral,r'$p$')],'Taylor–Green: all held-out space–time samples')]):
        a=read(g/f'validation/{test}/comparison.csv');test_checks={}
        for key,color,label in series:
            predicted,reference=a[key+'_pinn'],a[key+'_reference'];delta=predicted-reference
            relative=float(np.linalg.norm(delta)/np.linalg.norm(reference))
            assert np.isclose(relative,verification[test]['metrics'][key]['relative_l2'],rtol=1e-9)
            test_checks[key]={'relative_l2':relative,'max_absolute_error':float(np.max(abs(delta)))}
            if test=='V10':axs[i].scatter(a['time'],delta,s=7,c=color,alpha=.55,label=label,rasterized=True,edgecolors='none')
            else:axs[i].plot(a['time'],delta,c=color,label=label)
        axs[i].axhline(0,c='black',lw=.6,ls=':');axs[i].set(xlabel='Dimensionless time',ylabel='Signed discrepancy')
        axs[i].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axs[i].legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8)
        panel(axs[i],chr(97+i),test+': '+title)
        error_checks[test]={'samples':len(a),'observables':test_checks}
    checks['independent_verification_errors']=error_checks
    save(fig,'23_verification_errors','additional verification result',[
        'gaussian/validation/V01/comparison.csv','gaussian/validation/V02/comparison.csv',
        'gaussian/validation/V10/comparison.csv','gaussian/validation/verification.json'])

    a=read(g/'validation/V09.csv')
    assert np.array_equal(a['population'],[10000,10000,10000])
    assert np.isclose(a['max_difference'].max(),verification['V09']['max_force_density_difference'],rtol=1e-12,atol=0)
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.0));positions=np.arange(3)
    axs[0].bar(positions,a['population'],width=.55,color=[navy,teal,gold]);axs[0].set(xticks=positions,xticklabels=a['parcels'],xlabel='Coincident parcels',ylabel='Represented physical population',ylim=(0,12000))
    for j in positions:axs[0].text(j,10400,'10,000',ha='center',fontsize=9)
    axs[1].plot(positions,a['max_difference']*1e15,'o-',c=teal,ms=6)
    axs[1].set(xticks=positions,xticklabels=a['parcels'],xlabel='Coincident parcels',ylabel=r'Max. force-density difference $\times 10^{15}$',ylim=(-.1,3.2))
    panel(axs[0],'a','Population preserved');panel(axs[1],'b','Difference from 10-parcel reference')
    checks['parcel_splitting']={'counts':a['parcels'].tolist(),'population':a['population'].tolist(),'max_force_difference':float(a['max_difference'].max())}
    save(fig,'24_parcel_splitting','additional verification result',['gaussian/validation/V09.csv','gaussian/validation/verification.json'])

    physics=load_module('appendix_physics',root/'evidence/gaussian/parcel_pinn/physics.py')
    fig,axs=plt.subplots(1,3,figsize=(7.4,3.3));impacts={}
    for index,(ax,test,initial,e) in enumerate(zip(axs[:2],['V05','V06'],[([1.,0.],[-1.,0.]),([1.,.5],[-1.,-.5])],[.8,1.])):
        output=physics.binary_impact(initial[0],initial[1],1.,1.,[-1.,0.],e)
        assert np.allclose(output[0],verification[test]['velocity_1'],atol=1e-14)
        assert np.allclose(output[1],verification[test]['velocity_2'],atol=1e-14)
        assert np.allclose(np.sum(initial,axis=0),np.sum(output,axis=0),atol=1e-14)
        ein=sum(float(np.dot(v,v)) for v in initial)/2;eout=sum(float(np.dot(v,v)) for v in output)/2
        assert np.isclose(eout,verification[test]['final_energy'],rtol=1e-12)
        incoming=np.asarray(initial).ravel();outgoing=np.asarray(output).ravel()
        positions=np.arange(4)
        ax.bar(positions-.18,incoming,width=.34,color=gold,label='Incoming')
        ax.bar(positions+.18,outgoing,width=.34,color=teal,label='Outgoing')
        ax.set(xticks=positions,xticklabels=[r'$v_{1x}$',r'$v_{1y}$',r'$v_{2x}$',r'$v_{2y}$'],ylabel='Velocity component',ylim=(-1.25,1.5))
        ax.text(.5,.96,rf'$e={e:g}$; $E_{{out}}/E_{{in}}={eout/ein:.2f}$',transform=ax.transAxes,ha='center',va='top',fontsize=8)
        panel(ax,chr(97+index),test+': binary impact')
        impacts[test]={'input_velocities':initial,'output_velocities':[x.tolist() for x in output],'energy_ratio':eout/ein,'momentum_error':float(np.linalg.norm(np.sum(initial,axis=0)-np.sum(output,axis=0)))}
    initial=np.array([2.,-3.]);output=physics.wall_reflection(initial,[0.,1.],.6)
    assert np.allclose(output,verification['V07']['outgoing_velocity'],atol=1e-14)
    ax=axs[2]
    ax.bar(np.arange(2)-.18,initial,width=.34,color=gold)
    ax.bar(np.arange(2)+.18,output,width=.34,color=teal)
    ax.set(xticks=[0,1],xticklabels=[r'$v_x$',r'$v_y$'],ylabel='Velocity component',ylim=(-3.5,3.2))
    ax.text(.5,.96,r'$e=0.6$; $v_x$ preserved',transform=ax.transAxes,ha='center',va='top',fontsize=8)
    panel(ax,'c','V07: wall restitution')
    for ax in axs:ax.axhline(0,c='black',lw=.7,alpha=.7)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=9)
    impacts['V07']={'incoming':initial.tolist(),'outgoing':output.tolist()};checks['impact_algebra']=impacts
    save(fig,'25_impact_checks','standalone component verification',[
        'gaussian/validation/verification.json','gaussian/parcel_pinn/physics.py','gaussian/parcel_pinn/verification.py'])

    summary=json.loads((c/'summary.json').read_text());rows=summary['rows']
    reference=read(root/'evidence/gaussian/reference_inputs/Validation_Pack/data/couette_equilibrium.csv')
    assert [r['case_id'] for r in rows]==['CO01','CO02']
    comparisons=[]
    for row in rows:
        target=reference[reference['Re']==row['Rep']]['omega_magnitude_H_over_dU'][0]
        assert target==row['reference_spin'] and row['physics_criteria_met'] and row['mode']=='FEM-assisted PINN'
        for method in ['PINN','FEM']:
            error_percent=100*abs(row[method+'_spin']-target)/target
            assert np.isclose(error_percent,row[method+'_error_percent'],rtol=1e-10)
        comparisons.append({k:row[k] for k in ['case_id','Rep','reference_spin','FEM_spin','PINN_spin','PINN_error_percent','FEM_error_percent','PINN_velocity_relative_L2']})
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.2));re=np.array([x['Rep'] for x in rows])
    for key,color,marker,label,shift in [('reference_spin',navy,'o','Published numerical table',-.035),('FEM_spin',gold,'s','Independent FEM',0),('PINN_spin',teal,'^','FEM-assisted PINN',.035)]:
        axs[0].scatter(re+shift,[r[key] for r in rows],c=color,marker=marker,s=50,label=label,zorder=3)
    axs[0].set(xlabel=r'$\mathrm{Re}_p$',ylabel=r'$|\omega|H/\Delta U$',xticks=re,xlim=(.75,2.25),ylim=(0,.53));axs[0].legend(loc='lower left',fontsize=8)
    for method,color,shift in [('FEM',gold,-.12),('PINN',teal,.12)]:
        values=[r[method+'_error_percent'] for r in rows]
        axs[1].bar(re+shift,values,width=.22,color=color,label='FEM' if method=='FEM' else 'FEM-assisted PINN')
        for x,y in zip(re+shift,values):axs[1].text(x,y+.008,f'{y:.3f}%',ha='center',fontsize=8)
    axs[1].set(xlabel=r'$\mathrm{Re}_p$',ylabel='Difference from published table (%)',xticks=re,ylim=(0,.29));axs[1].legend(loc='upper right',fontsize=7)
    panel(axs[0],'a','Equilibrium angular speed');panel(axs[1],'b','Relative comparison errors')
    checks['couette_rotation']=comparisons
    save(fig,'26_couette_rotation','preliminary resolved-cylinder comparison',[
        'gaussian/outputs/couette_validation/summary.json','gaussian/reference_inputs/Validation_Pack/data/couette_equilibrium.csv'])

    arrays=[read(c/f'fields_{r["case_id"]}.csv') for r in rows]
    speedmax=max(float(np.max(np.hypot(a[k+'_u'],a[k+'_v']))) for a in arrays for k in ['FEM','PINN'])
    errmax=max(float(np.max(np.hypot(a['PINN_u']-a['FEM_u'],a['PINN_v']-a['FEM_v']))) for a in arrays)
    fig,axs=plt.subplots(2,3,figsize=(7.4,5.0));field_checks=[]
    for i,(a,row) in enumerate(zip(arrays,rows)):
        xs,ys=np.unique(a['x']),np.unique(a['y']);ix,iy=np.searchsorted(xs,a['x']),np.searchsorted(ys,a['y'])
        difference=np.hypot(a['PINN_u']-a['FEM_u'],a['PINN_v']-a['FEM_v'])
        relative=float(np.linalg.norm(np.c_[a['PINN_u']-a['FEM_u'],a['PINN_v']-a['FEM_v']])/np.linalg.norm(np.c_[a['FEM_u'],a['FEM_v']]))
        field_checks.append({'case':row['case_id'],'visualization_samples':len(a),'visualization_grid_velocity_relative_l2':relative,'independent_assessment_relative_l2':row['PINN_velocity_relative_L2'],'maximum_visualization_velocity_difference':float(np.max(difference))})
        for j,(values,title) in enumerate([(np.hypot(a['FEM_u'],a['FEM_v']),'FEM speed'),(np.hypot(a['PINN_u'],a['PINN_v']),'FEM-assisted PINN speed'),(difference,'Velocity discrepancy')]):
            z=np.full((len(ys),len(xs)),np.nan);z[iy,ix]=values
            assert np.isfinite(values).all()
            ax=axs[i,j];im=ax.pcolormesh(xs,ys,np.ma.masked_invalid(z),shading='nearest',rasterized=True,cmap='viridis' if j<2 else 'magma',vmin=0,vmax=speedmax if j<2 else errmax)
            ax.add_patch(Circle((1,.5),.125,fc='#EEEEEE',ec='black',lw=.7,zorder=3))
            ax.set(xlim=(0,2),ylim=(0,1),aspect='equal',xlabel=r'$x/H$',ylabel=r'$y/H$')
            panel(ax,chr(97+3*i+j),row['case_id']+': '+title)
            bar=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.9,pad=.12)
            if j==2:bar.formatter.set_powerlimits((0,0));bar.update_ticks()
    checks['couette_fields']=field_checks
    save(fig,'27_couette_fields','preliminary resolved-cylinder comparison',[
        'gaussian/outputs/couette_validation/fields_CO01.csv','gaussian/outputs/couette_validation/fields_CO02.csv','gaussian/outputs/couette_validation/summary.json'])

    report={'passed':True,'added_reference_keys':['garcia2023'],
            'figure_names':['20_baseline_relaxation','21_drag_isolation','22_baseline_training','23_verification_errors','24_parcel_splitting','25_impact_checks','26_couette_rotation','27_couette_fields'],
            'numerical_checks':checks,'new_simulations_run':False,'raw_numerical_values_modified':False,
            'selection_scope':'Supplementary verification, fixed-field drag isolation, and completed matched equilibrium Couette comparisons; excludes unsupported coupled transport trends.'}
    (root/'generated/appendix_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Prepared eight additional figures and recomputed their numerical checks.')

if __name__=='__main__':
    import make_figures as figures
    root=Path(__file__).resolve().parents[1]
    make_appendix_figures(root,figures.save,figures.panel)
    path=root/'generated/figure_manifest.json';manifest=json.loads(path.read_text())
    names={x['name'] for x in figures.REGISTER}
    manifest=[x for x in manifest if x['name'] not in names]+figures.REGISTER
    path.write_text(json.dumps(manifest,indent=2)+'\n')
