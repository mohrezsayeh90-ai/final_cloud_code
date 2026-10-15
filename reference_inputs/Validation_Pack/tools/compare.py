#!/usr/bin/env python3
"""Compare a simulation CSV to a reference CSV on the reference sampling grid.

No implicit unit conversion, time shift, extrapolation, or universal pass criterion.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path,x,y):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty CSV: {path}')
    if x not in rows[0] or y not in rows[0]:raise ValueError(f'Missing {x!r} or {y!r} in {path}; columns: {list(rows[0])}')
    a=np.array([[float(r[x]),float(r[y])] for r in rows])
    if not np.all(np.isfinite(a)):raise ValueError(f'Non-finite data in {path}')
    a=a[np.argsort(a[:,0])]
    if len(a)<2 or np.any(np.diff(a[:,0])<=0):raise ValueError('Independent coordinate must have at least two distinct points and no duplicates.')
    return a[:,0],a[:,1]

def compare(rx,ry,sx,sy,scale=None,atol=None,rtol=None):
    mask=(rx>=sx[0])&(rx<=sx[-1])
    if mask.sum()<2:raise ValueError('Fewer than two reference samples in common support; no extrapolation is allowed.')
    x=rx[mask];ref=ry[mask];sim=np.interp(x,sx,sy);err=sim-ref
    rmse=float(np.sqrt(np.mean(err**2)));norm=float(np.linalg.norm(ref))
    result=dict(reference_samples=int(len(rx)),compared_samples=int(mask.sum()),sample_coverage_fraction=float(mask.mean()),
                independent_span_coverage_fraction=float((x[-1]-x[0])/(rx[-1]-rx[0])),comparison_start=float(x[0]),comparison_end=float(x[-1]),
                RMSE=rmse,MAE=float(np.mean(np.abs(err))),max_absolute_error=float(np.max(np.abs(err))),bias=float(np.mean(err)),
                relative_L2=float(np.linalg.norm(err)/norm) if norm>0 else None,
                normalized_RMSE=rmse/scale if scale is not None else None,normalization_scale=scale,
                all_points_within_tolerance=None,tolerance_scope=None)
    if atol is not None or rtol is not None:
        aa=atol or 0;rr=rtol or 0
        result.update(all_points_within_tolerance=bool(np.all(np.abs(err)<=aa+rr*np.abs(ref))),
                      tolerance_scope='compared samples only; not a whole-case pass verdict',absolute_tolerance=aa,relative_tolerance=rr)
    if not np.all(mask):result['coverage_warning']='Partial overlap. Agreement here does not validate the missing time or coordinate range.'
    return result,x,ref,sim

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--simulation',required=True)
    p.add_argument('--x',required=True);p.add_argument('--y',required=True)
    p.add_argument('--simulation-x');p.add_argument('--simulation-y')
    p.add_argument('--scale',type=float);p.add_argument('--abs-tol',type=float);p.add_argument('--rel-tol',type=float)
    p.add_argument('--out',default='comparison');p.add_argument('--title',default='Reference and simulation')
    args=p.parse_args()
    if args.scale is not None and args.scale<=0:p.error('--scale must be positive')
    for val in [args.abs_tol,args.rel_tol]:
        if val is not None and val<0:p.error('Tolerances must be nonnegative')
    try:
        rx,ry=read(args.reference,args.x,args.y)
        sx,sy=read(args.simulation,args.simulation_x or args.x,args.simulation_y or args.y)
        result,x,ref,sim=compare(rx,ry,sx,sy,args.scale,args.abs_tol,args.rel_tol)
    except (ValueError,KeyError) as e:p.error(str(e))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result.update(reference=str(args.reference),simulation=str(args.simulation),independent_variable=args.x,dependent_variable=args.y)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow([args.x,'reference','simulation','error']);w.writerows(zip(x,ref,sim,sim-ref))
    fig,ax=plt.subplots(2,1,figsize=(7,5.2),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(x,ref,label='reference',color='black');ax[0].plot(x,sim,'--',label='simulation',color='tab:blue')
    ax[0].set_ylabel(args.y);ax[0].set_title(args.title);ax[0].legend()
    ax[1].plot(x,sim-ref,color='tab:red');ax[1].set_xlabel(args.x);ax[1].set_ylabel('error')
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'overlay.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
