import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from .config import configuration
from .diagnostics import evaluate
from .figures import Gallery
from .model import CoupledPINN
from .training import seed_all, train_channel
from .verification import audit_catalog, verify


def locate_pack(path):
    if path and Path(path).is_file() and zipfile.is_zipfile(path):
        archive=Path(path)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        cache=Path(tempfile.gettempdir())/'parcel-pinn-inputs'/digest
        cache.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist())>128*1024**2:
                raise ValueError('validation archive exceeds the 128 MiB extraction limit')
            for item in z.infolist():
                target=(cache/item.filename).resolve()
                if not target.is_relative_to(cache.resolve()):
                    raise ValueError('unsafe archive path')
                if (item.external_attr>>16)&0o170000==0o120000:
                    raise ValueError('archive symlinks are not supported')
            z.extractall(cache)
        folders=list(cache.rglob('case_catalog.json'))
        if len(folders)!=1:
            raise ValueError('archive must contain exactly one case_catalog.json')
        path=folders[0].parent
    candidates=[Path(path)] if path else []
    candidates += [Path('/workspace/reference_inputs/Validation_Pack'), Path('Validation_Pack')]
    for candidate in candidates:
        if (candidate/'case_catalog.json').is_file() and (candidate/'data'/'analytic_stokes.csv').is_file():
            return candidate.resolve()
    raise FileNotFoundError('Unzip Validation_Pack.zip and supply --validation-pack /path/to/Validation_Pack')


def record_inputs(pack,out):
    rows=[]
    for p in sorted(pack.rglob('*')):
        if p.is_file():
            rows.append(dict(path=str(p.relative_to(pack)),bytes=p.stat().st_size,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (out/'reference_manifest.json').write_text(json.dumps(rows,indent=2))
    source=Path(__file__).resolve().parent
    files=(list(source.glob('*.py'))+[source.parent/'run.py',source.parent/'requirements.txt']
           if source.name=='parcel_pinn' else [Path(__file__).resolve()])
    files=[p for p in files if p.is_file()]
    relative_root=source.parent if source.name=='parcel_pinn' else source
    (out/'source_manifest.json').write_text(json.dumps([
        dict(path=str(p.relative_to(relative_root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],indent=2))


def report(out,cfg,gallery,verification,cases,audit,errors,elapsed):
    passed=sum(r['outcome']=='passed' for r in verification)
    failures=[r['case_id'] for r in verification if r['outcome']!='passed']
    required_plot_errors=[r for r in gallery.errors if r['figure']!='optional MP4']
    all_required=not failures and not errors and not required_plot_errors and len(cases)==len(cfg.cases)*len(cfg.seeds) and all(c['quality_criteria_met'] for c in cases)
    payload=dict(workflow_complete=not errors and not required_plot_errors and not any(r['outcome']=='error' for r in verification), research_quality_criteria_met=all_required,
                 dimension=cfg.dimension, preset=cfg.preset, seconds=elapsed,
                 verification_passed=passed, verification_total=len(verification),
                 verification=verification, cases=cases, literature_cases=audit,
                 physical_validation_cases_run=0, plot_errors=gallery.errors, errors=errors,
                 display_depth_only=cfg.dimension==2,
                 limitations=['No experimental or resolved-body validation claimed.',
                              'Regularized disturbed-velocity drag, no self-disturbance correction.',
                              'Low slip Reynolds number required by Stokes drag; reported explicitly.',
                              'No lubrication, lift, added mass, Basset history, particle rotation or finite-volume displacement.',
                              'Soft contact is an optional physical-sphere closure; default parcel cases are collision-free.',
                              'Residual thresholds are project criteria, not universal scientific accuracy.',
                              'Paper preset and spatial/time/width/parcel refinement need separate completed runs.'])
    # Strict JSON, with null for unsupported diagnostics rather than NaN/Infinity.
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    payload=clean(payload)
    (out/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    table_rows=[]
    for case in cases:
        data=pd.read_csv(out/case['directory']/'particles.csv')
        last=data[data.time==data.time.max()]
        table_rows.append(dict(case=case['case'],seed=case['seed'],Re=cfg.reynolds,
            St=case['physical']['stokes'],loading=case['physical']['loading'],
            parcels=cfg.nparcels,multiplicity=case['physical']['multiplicity'],
            diameter_over_H=case['physical']['diameter'],width_over_H=cfg.width,
            final_mean_vx=float(last.vx.mean()),final_mean_vy=float(last.vy.mean()),
            ns_residual_rms=case['residuals']['momentum'],
            continuity_rms=case['residuals']['continuity'],
            kinematics_rms=case['residuals']['position'],
            particle_ode_rms=case['residuals']['particle'],
            max_slip_Re=case['max_slip_re'],max_volume_fraction=case['max_volume_fraction'],
            screening_met=case['quality_criteria_met']))
    paper_table=pd.DataFrame(table_rows)
    paper_table.to_csv(out/'paper_metrics.csv',index=False)
    if len(paper_table):
        paper_table[['case','seed','St','loading','final_mean_vx','final_mean_vy','ns_residual_rms','particle_ode_rms','screening_met']].to_latex(
            out/'paper_metrics.tex',index=False,float_format=lambda value:f'{value:.4g}',escape=True)
    lines=['# Gaussian parcel PINN results','',
           f'Preset: {cfg.preset}. Carrier dimension: {cfg.dimension}D. Runtime: {elapsed:.1f} s.',
           f'Numerical verification: {passed}/{len(verification)} passed. Literature validation: 0/{len(audit)} run.',
           f'Research quality criteria met: {all_required}.', '',
           '## Interpretation','',
           'All plotted solver fields and trajectories are PINN predictions. Analytical targets are independent reduced-model verification data. '
           'The catalogue audit records every literature case in its supplied order; no mismatched experiment is scored. '
           'A completed execution or an attractive figure does not establish predictive accuracy.', '',
           'For 2D, parcels represent spherical populations moving in a plane. Forces and masses are converted to a per-depth convention using B/H. '
           'This is not the finite-size infinite-cylinder model. Extruded channel views solve no spanwise physics.', '',
           '## Case diagnostics','',
           '| Case | Seed | Held-out NS RMS | Parcel ODE RMS | Max slip Re | Max α | Quality criteria |',
           '|---|---:|---:|---:|---:|---:|---|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['seed']} | {c['residuals']['momentum']:.4g} | {c['residuals']['particle']:.4g} | {c['max_slip_re']:.4g} | {c['max_volume_fraction']:.4g} | {c['quality_criteria_met']} |")
    lines+=['','Acceptance is declared before training: all held-out residual RMS < '+str(cfg.residual_tolerance)+
            ', slip Re <= 1, local volume fraction < 0.01, all centers inside channel, trained checkpoint and no training error. '
            'Passing these screening criteria is necessary, not sufficient, for publication. Resolve residual, discretization and closure uncertainty.',
            '', '## Remaining checks','']+[f'- {s}' for s in payload['limitations']]
    if errors:lines+=['','## Execution errors','']+[f"- {e['stage']}: {e['error']}" for e in errors]
    lines+=['','## Figure captions','']
    for item in gallery.items:
        lines += [f"### {item['title']}",'',item['caption'],'',f"![{item['title']}]({item['files'][0]})",'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    def esc(s):return html.escape(str(s))
    cards=[]
    for item in gallery.items:
        first=item['files'][0]
        links=' · '.join(f'<a href="{esc(p)}">{esc(Path(p).suffix[1:].upper())}</a>' for p in item['files'])
        cards.append(f'<article><h2>{esc(item["title"])}</h2><img src="{esc(first)}" loading="lazy"><p>{esc(item["caption"])}</p><p>{links}</p></article>')
    status='Screening criteria met' if all_required else 'Further numerical or model assessment required'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian parcel PINN</title><style>body{{margin:0;background:#edf2f6;color:#203449;font:16px/1.6 system-ui}}main{{max-width:1150px;margin:auto;padding:35px}}h1{{font-size:36px}}article{{background:white;border-radius:14px;margin:22px 0;padding:25px}}img{{max-width:100%;height:auto}}a{{color:#2563a6}}.note{{background:#fff1d7;padding:20px;border-radius:12px}}p{{max-width:1000px}}</style><main>
<h1>Gaussian parcel PINN</h1><p>{cfg.dimension}D carrier flow · {esc(cfg.preset)} preset · independent reference audit</p>
<div class="note"><b>{esc(status)}</b><br>Numerical verification: {passed}/{len(verification)}. Literature validation: 0/{len(audit)}.
 Predictions from incomplete training remain diagnostic results. For 2D, the channel depth view is an extruded illustration.</div>
<p><a href="summary.json">Metrics and status</a> · <a href="paper_metrics.csv">Results table</a> · <a href="paper_metrics.tex">LaTeX table</a> · <a href="validation/literature_audit.csv">Literature audit</a> · <a href="config.json">Run configuration</a> · <a href="REPORT.md">Scientific report</a></p>
{''.join(cards)}</main></html>'''
    (out/'index.html').write_text(page)
    gallery.write_manifest()
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description='2D/3D Gaussian parcel PINN and honest validation audit')
    parser.add_argument('--preset',choices=['smoke','demo','paper'],default='demo')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--validation-pack',type=Path)
    parser.add_argument('--out',type=Path,default=Path('outputs/demo'))
    parser.add_argument('--strict',action='store_true',help='Exit 2 unless all declared quality checks pass')
    parser.add_argument('--fail-at',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    cfg=configuration(args.preset,args.config)
    pack=locate_pack(args.validation_pack)
    out=args.out.resolve()
    # Never overwrite results from a previous run, even accidentally.
    if out.exists() and any(out.iterdir()):
        parser.error(f'output folder is not empty: {out}; choose a new --out path')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();errors=[];cases=[]
    cfg.save(out/'config.json');record_inputs(pack,out)
    (out/'runtime.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
                                                  torch=torch.__version__,numpy=np.__version__,device='cpu',
                                                  threads=cfg.threads),indent=2))
    gallery=Gallery(out,cfg);gallery.safe('architecture')
    validation=out/'validation';validation.mkdir()
    audit=audit_catalog(pack,validation)
    print('Running V01--V10 numerical verification; literature cases audited separately.',flush=True)
    verification=verify(cfg,pack,validation,lambda *args: gallery.safe('verification_plot',*args))
    gallery.safe('verification_summary',verification)
    for seed in cfg.seeds:
        run_cfg=replace(cfg,seed=seed)
        for case in cfg.cases:
            name=f"{case['name']}_seed{seed}"
            directory=Path('cases')/name;folder=out/directory;folder.mkdir(parents=True)
            print(f'Training {name}: {cfg.dimension}D, epochs={cfg.epochs}',flush=True)
            try:
                seed_all(seed,cfg.threads)
                model=CoupledPINN(run_cfg,case)
                state,history=train_channel(model,folder,fail_at=args.fail_at)
                particles,budget,residuals=evaluate(model,folder)
                max_rep=float(budget.max_slip_re.max());max_alpha=float(budget.max_volume_fraction.max())
                quality=(all(r<cfg.residual_tolerance for r in residuals.values()) and max_rep<=1
                         and max_alpha<.01 and budget.wall_outside_count.max()==0 and state['best_step']>0
                         and not state['termination'].startswith('training_error'))
                result=dict(case=case['name'],seed=seed,directory=str(directory),residuals=residuals,
                            max_slip_re=max_rep,max_volume_fraction=max_alpha,quality_criteria_met=bool(quality),
                            training=state,physical=model.physical,
                            max_momentum_balance_residual=float(budget.momentum_balance_residual.max()),
                            max_energy_balance_residual=float(budget.energy_balance_residual.max()))
                cases.append(result)
                (folder/'case_status.json').write_text(json.dumps(result,indent=2))
                for method,params in [('training',(history,name,folder)),('fields',(model,name,folder)),
                                      ('trajectories',(particles,name,folder)),('budgets',(budget,name,folder)),
                                      ('animation',(particles,name,folder))]:
                    gallery.safe(method,*params)
                print(f"{name}: best audit={state['best_audit_loss']:.5g}, held-out NS={residuals['momentum']:.5g}, quality={quality}",flush=True)
            except Exception as exc:
                errors.append(dict(stage=name,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                (folder/'error.txt').write_text(traceback.format_exc())
                print(f'{name}: error recorded; continuing remaining cases.',flush=True)
    gallery.safe('sweep',cases)
    payload=report(out,cfg,gallery,verification,cases,audit,errors,time.monotonic()-started)
    print(f"Report: {out/'index.html'}\nVerification: {payload['verification_passed']}/{payload['verification_total']}; literature validation: 0/{len(audit)}",flush=True)
    return 2 if args.strict and not payload['research_quality_criteria_met'] else (1 if errors else 0)
