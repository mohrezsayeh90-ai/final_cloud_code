#!/usr/bin/env python3
"""Reproduce the finite-cylinder equilibrium-spin literature comparison."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from literature_validation.couette import Settings,train,assess
from literature_validation.fem_couette import solve
from literature_validation.report import generate_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/couette_validation'))
    parser.add_argument('--pack',type=Path,default=Path('reference_inputs/Validation_Pack'))
    parser.add_argument('--cases',type=int,nargs='+',choices=[1,2],default=[1,2])
    parser.add_argument('--adam',type=int,default=2200)
    parser.add_argument('--lbfgs',type=int,default=800)
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('choose a new output directory')
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    for rep in args.cases:
        folder=args.out/f'CO0{rep}';coarse=folder/'fem_coarse';fine=folder/'fem_fine';pinn=folder/'pinn'
        solve(float(rep),.07,96,coarse);solve(float(rep),.035,192,fine)
        settings=Settings(particle_re=float(rep),adam_steps=args.adam,lbfgs_steps=args.lbfgs,
                          fixed_points=2048,feature_set='multipole',teacher=str(fine/'fem_solution.npz'))
        model=train(settings,pinn);metrics=assess(model)
        # Final article comparison is performed by generate_report after training.
        metrics['physics_criteria_met']=(metrics['momentum_rms']<settings.residual_tolerance and
            metrics['continuity_rms']<settings.residual_tolerance and abs(metrics['torque_normalized'])<settings.torque_tolerance and
            metrics['cylinder_boundary_max_error']<1e-10 and metrics['wall_boundary_max_error']<1e-10)
        (pinn/'metrics.json').write_text(json.dumps(metrics,indent=2))
        records.append(dict(case_id=f'CO0{rep}',coarse=str(coarse),fine=str(fine),pinn=str(pinn)))
    result=generate_report(args.out,records,args.pack)
    print(json.dumps(result['rows'],indent=2))
    return 2 if args.strict and not result['all_declared_criteria_met'] else 0


if __name__=='__main__':raise SystemExit(main())
