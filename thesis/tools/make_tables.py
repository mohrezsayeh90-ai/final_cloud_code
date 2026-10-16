"""Generate manuscript values and tables directly from recorded result files."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'evidence/basic/results/final_study'
G=ROOT/'evidence/gaussian/outputs/evaluated_2d_final'
T=ROOT/'generated'
TABLES=[]
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def sci(v):
    a,e=f'{float(v):.3e}'.split('e')
    return rf'${a}\times10^{{{int(e)}}}$'
def table(name,values):
    content='\n'.join(' & '.join(x)+r' \\' for x in values)+'\n'
    (T/(name+'.tex')).write_text(content)
    TABLES.append(r'\expandafter\def\csname Evidence'+name+r'\endcsname{'+content+'}\n')
def main():
    m=json.loads((B/'metrics.json').read_text());s=json.loads((G/'summary.json').read_text())
    v=json.loads((G/'validation/verification.json').read_text());d={x['case_id']:x for x in v}
    macros={'FlowVelocity':100*m['flow']['velocity_relative_l2'],
      'FlowPressure':100*m['flow']['pressure_relative_l2'],
      'DragError':100*m['force']['heldout_relative_l2'],
      'PoiseuilleVelocity':100*m['poiseuille']['pinn_velocity_relative_l2'],
      'PoiseuillePressure':100*m['poiseuille']['pinn_pressure_relative_l2'],
      'FluxDeviation':100*m['flow']['flux_relative_deviation'],
      'MeshDifference':100*m['reference']['penultimate_velocity_difference'],
      'InterpolationDifference':100*m['reference']['trajectory_interpolator_velocity_error'],
      'SeedVelocityMean':100*m['seed_statistics']['velocity_relative_l2']['mean'],
      'SeedVelocitySD':100*m['seed_statistics']['velocity_relative_l2']['std'],
      'VOneVelocity':100*d['V01']['metrics']['v_over_vt']['relative_l2'],
      'VOneDistance':100*d['V01']['metrics']['distance_over_vt_tau']['relative_l2'],
      'VTwoFluid':100*d['V02']['metrics']['fluid_u']['relative_l2'],
      'VTwoParticle':100*d['V02']['metrics']['particle_v']['relative_l2'],
      'TGU':100*d['V10']['metrics']['u']['relative_l2'],
      'TGV':100*d['V10']['metrics']['v']['relative_l2'],
      'TGP':100*d['V10']['metrics']['p']['relative_l2']}
    (T/'values.tex').write_text('\n'.join(rf'\newcommand{{\{k}}}{{{v:.3f}}}' for k,v in macros.items())+'\n')
    table('basic_errors',[
      ['Plane Poiseuille PINN','Velocity',rf'\PoiseuilleVelocity\%'],
      ['Plane Poiseuille PINN','Centered pressure',rf'\PoiseuillePressure\%'],
      ['Plane Poiseuille FEM','Velocity',sci(m['poiseuille']['fem_velocity_relative_l2'])],
      ['Curved-channel PINN','Velocity',rf'\FlowVelocity\%'],
      ['Curved-channel PINN','Centered pressure',rf'\FlowPressure\%'],
      ['ForceNet','Normalized drag',rf'\DragError\%'],
      ['Section flux','Maximum relative deviation',rf'\FluxDeviation\%'],
      ['Two finest FEM meshes','Velocity difference',rf'\MeshDifference\%']])
    table('mesh',[[r['triangles'],r['velocity_dofs'],r['pressure_dofs'],f"{100*float(r['velocity_error_to_finest']):.3f}" if i<2 else 'Reference',sci(r['nonlinear_residual'])] for i,r in enumerate(rows(B/'data/mesh_sensitivity.csv'))])
    table('seeds',[[r['seed'],f"{100*float(r['velocity_relative_l2']):.3f}",f"{100*float(r['pressure_relative_l2']):.3f}"] for r in rows(B/'data/seed_summary.csv')])
    table('gaussian_errors',[
      ['V01','Stokes velocity',rf'\VOneVelocity\%'],['V01','Integrated displacement',rf'\VOneDistance\%'],
      ['V02','Fluid velocity',rf'\VTwoFluid\%'],['V02','Particle velocity',rf'\VTwoParticle\%'],
      ['V10','Taylor--Green $u$',rf'\TGU\%'],['V10','Taylor--Green $v$',rf'\TGV\%'],['V10','Taylor--Green $p$',rf'\TGP\%']])
    table('coupled_scope',[[f"{c['physical']['stokes']:.2f}",f"{c['physical']['loading']:.2f}",f"{c['residuals']['momentum']:.4f}",f"{c['residuals']['particle']:.4f}",'Not met'] for c in s['cases']])
    table('trajectories',[[f"{c['seed_eta']:.3f}",f"{100*c['rms_position_error']/m['geometry']['width']:.2f}",f"{100*c['common_fraction']:.2f}",'Yes' if c['outcome_agreement'] else 'No'] for c in m['trajectory']['baseline_comparison']])
    cfg=json.loads((G/'config.json').read_text())
    selected=next(x for x in s['cases'] if x['case']=='high_inertia')
    phys=selected['physical']
    table('parcel_parameters',[
      [r'$\Rey_H,\ R,\ b$',rf"${cfg['reynolds']:g},\ {cfg['density_ratio']:g},\ {cfg['depth_ratio']:g}$"],
      [r'$\beta$, parcel count, seed',rf"${phys['loading']:.2f},\ {cfg['nparcels']},\ {cfg['seed']}$"],
      [r'$\sigma,\ T$',rf"${cfg['width']:g},\ {cfg['final_time']:g}$"],
      [r'$\vect{g}_{\rm eff}$',rf"$({cfg['gravity'][0]:g},\ {cfg['gravity'][1]:g})$"],
      [r'$\St,\ d/H$',rf"${phys['stokes']:.2f},\ {phys['diameter']:.6f}$"],
      [r'Multiplicity $N_i$',f"${phys['multiplicity']:.6f}$"],
      [r'Represented mass $m_i$',f"${phys['represented_mass']:.6f}$"],
      [r'Single-sphere mass in the planar scale',f"${phys['physical_mass']:.8f}$"],
      [r'Initial parcel velocity',r'$(0,0)$']])
    initial=[r for r in rows(G/'cases/high_inertia_seed7/particles.csv') if float(r['time'])==0]
    assert len(initial)==6
    table('parcel_initial_centers',[[f"P{int(r['particle'])+1}",f"{float(r['x_unwrapped']):.6f}",f"{float(r['y']):.6f}"] for r in initial])
    (T/'tables.tex').write_text(''.join(TABLES))
    print(f'Generated {len(TABLES)} tables and numerical macros.')
if __name__=='__main__':main()
