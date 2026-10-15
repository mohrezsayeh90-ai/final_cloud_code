"""Original transient Gaussian-parcel GPU workflow with only the channel geometry extended."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import pandas as pd
from parcel_pinn.config import configuration, Config
from parcel_pinn.training import seed_all
from parcel_pinn.pipeline import locate_pack, record_inputs, report
from parcel_pinn.verification import audit_catalog
from parcel_pinn.gpu_workflow import supervised, worker as original_worker
from .model import GeometrySettings, ConstrictedGaussianPINN, train
from .diagnostics import evaluate
from .figures import ConstrictedGallery


def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def worker(args):
    cfg = configuration('demo', args.out/'config.json')
    geometry = GeometrySettings.load(args.out/'geometry.json')
    if args.worker == 'verify':
        return original_worker('verify', args.out, 0, args.seconds, args.device, args.precision, args.validation_pack)
    case = cfg.cases[args.index]; seed_all(cfg.seed, cfg.threads)
    folder = args.out/'cases'/f"{case['name']}_seed{cfg.seed}"; folder.mkdir(parents=True, exist_ok=True)
    model = ConstrictedGaussianPINN(cfg, case, geometry)
    if args.worker == 'train':
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model.to(device=args.device, dtype=torch.float32 if args.precision == 'float32' else torch.float64)
        print(f"{case['name']}: two-way Gaussian parcels, {next(model.parameters()).device}, {next(model.parameters()).dtype}", flush=True)
        train(model, folder, args.seconds); return
    checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.cpu().double().eval()
    state = checkpoint['training']
    if state['termination'] == 'running': state['termination'] = 'worker_interrupted_best_checkpoint'
    particles, budget, residuals = evaluate(model, folder)
    max_rep, max_alpha = float(budget.max_slip_re.max()), float(budget.max_volume_fraction.max())
    quality = (all(r < cfg.residual_tolerance for r in residuals.values()) and max_rep <= 1 and max_alpha < .01
               and budget.wall_outside_count.max() == 0 and budget.obstacle_inside_count.max() == 0
               and state['best_step'] > 0 and state['termination'] in ['budget_exhausted', 'time_budget_exhausted'])
    result = dict(case=case['name'], seed=cfg.seed, directory=str(folder.relative_to(args.out)), residuals=residuals,
                  max_slip_re=max_rep, max_volume_fraction=max_alpha, quality_criteria_met=bool(quality),
                  training=state, physical=model.physical, fluid_volume_exact=float(model.fluid_volume_exact),
                  fluid_volume_quadrature=model.fluid_volume_quadrature,
                  effective_loading_fluid_volume=cfg.nparcels*model.physical['represented_mass']/model.fluid_volume_exact,
                  initial_centers_resampled=model.initial_centers_resampled,
                  max_obstacle_inside_count=int(budget.obstacle_inside_count.max()),
                  max_wall_outside_count=int(budget.wall_outside_count.max()),
                  max_kernel_error=float(budget.kernel_error.max()),
                  max_force_exchange_error=float(budget.force_exchange_error.max()),
                  max_momentum_balance_residual=float(budget.momentum_balance_residual.max()),
                  max_energy_balance_residual=float(budget.energy_balance_residual.max()))
    dump(folder/'case_status.json', result)
    gallery = ConstrictedGallery(args.out, cfg, geometry)
    for method, parameters in [
        ('training', (pd.read_csv(folder/'training.csv'), case['name'], folder)),
        ('fields', (model, case['name'], folder)), ('trajectories', (particles, case['name'], folder)),
        ('budgets', (budget, case['name'], folder)), ('animations', (model, particles, case['name'], folder)),
    ]:
        gallery.safe(method, *parameters)
        dump(folder/'gallery.json', dict(items=gallery.items, errors=gallery.errors))
    if any(e['figure'] != 'optional MP4' for e in gallery.errors): raise RuntimeError('Plot failures recorded in gallery.json')
    print(f"{case['name']}: momentum RMS={residuals['momentum']:.4g}, parcel RMS={residuals['particle']:.4g}, solid centers={result['max_obstacle_inside_count']}", flush=True)


def run(args):
    started = time.monotonic(); deadline = started+args.max_minutes*60
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('Enable CUDA GPU; no automatic CPU fallback.')
    out = args.out
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a fresh output folder.')
    cfg = configuration('smoke' if args.smoke else 'demo', args.config)
    if args.smoke: cfg.cases = Config().cases
    if cfg.dimension != 2 or len(cfg.seeds) != 1 or cfg.collision != 'none':
        raise ValueError('Use the original 2D collision-free parcel model with one seed per run.')
    cfg.seed = cfg.seeds[0]
    if not args.smoke:
        cfg.epochs, cfg.hidden, cfg.depth = 1000000, 48, 3
        cfg.time_batch, cfg.space_batch, cfg.quadrature = 4, 96, (24, 20)
        cfg.frames, cfg.dpi = 41, 160
    cfg.lbfgs_steps = 0; cfg.validate()
    geometry = GeometrySettings.load(args.geometry) if args.geometry else GeometrySettings()
    geometry.validate(cfg.lengths[0]); pack = locate_pack(args.validation_pack)
    out.mkdir(parents=True, exist_ok=True); (out/'validation').mkdir()
    cfg.save(out/'config.json'); geometry.save(out/'geometry.json'); record_inputs(pack, out)
    root = Path(__file__).resolve().parents[1]
    source = sorted((root/'parcel_pinn').glob('*.py'))+sorted((root/'gaussian_constriction').glob('*.py'))
    dump(out/'source_manifest.json', [dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source])
    dump(out/'runtime.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
         cuda_build=torch.version.cuda, training_device=args.device, training_precision=args.precision,
         export_device='cpu', export_precision='float64', gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None))
    dump(out/'run_settings.json', dict(max_minutes=args.max_minutes, train_minutes=args.train_minutes,
         device=args.device, precision=args.precision, smoke=args.smoke, particles_modeled=True,
         physical_time_modeled=True, two_way_gaussian_coupling=True,
         boundary='periodic x; no-slip flat and semicircular walls', body_force_x=12/cfg.reynolds))
    audit = audit_catalog(pack, out/'validation'); stages, errors, cases = [], [], []
    train_total = min(args.train_minutes*60, .65*args.max_minutes*60)
    train_deadline = min(deadline, time.monotonic()+train_total)
    print(f'Original transient two-way Gaussian model: {len(cfg.cases)} cases; only geometry extended.', flush=True)
    def launch(stage, i, limit, name):
        command = [sys.executable, '-u', str(root/'run_constriction.py'), '--worker', stage, '--index', str(i),
                   '--out', str(out), '--seconds', str(max(.01, limit-8)), '--device', args.device,
                   '--precision', args.precision, '--validation-pack', str(pack)]
        state = supervised(command, max(0, limit), out/f'{name}.log'); state['stage'] = name
        stages.append(state); dump(out/'stage_status.json', stages)
        if state['outcome'] != 'complete': errors.append(dict(stage=name, error=state['outcome']))
    for i, case in enumerate(cfg.cases):
        launch('train', i, max(0, train_deadline-time.monotonic())/(len(cfg.cases)-i), f"train_{case['name']}")
    launch('verify', 0, min(180, max(0, deadline-time.monotonic())*.18), 'verification')
    gallery = ConstrictedGallery(out, cfg, geometry); gallery.safe('geometry_plot'); gallery.safe('architecture')
    verification_path = out/'validation/verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.is_file() else [dict(case_id=f'V{k:02}', outcome='not_run', details='Worker did not complete within budget.') for k in range(1, 11)]
    dump(verification_path, verification)
    if (out/'verification_gallery.json').is_file():
        saved = json.loads((out/'verification_gallery.json').read_text()); gallery.items.extend(saved['items']); gallery.errors.extend(saved['errors'])
    gallery.safe('verification_summary', verification)
    for i, case in enumerate(cfg.cases):
        folder = out/'cases'/f"{case['name']}_seed{cfg.seed}"
        if not (folder/'checkpoint.pt').is_file():
            errors.append(dict(stage=case['name'], error='No finite checkpoint; no fabricated trajectory.')); continue
        launch('export', i, max(0, deadline-time.monotonic()-60)/(len(cfg.cases)-i), f"export_{case['name']}")
        if (folder/'case_status.json').is_file(): cases.append(json.loads((folder/'case_status.json').read_text()))
        if (folder/'gallery.json').is_file():
            saved = json.loads((folder/'gallery.json').read_text()); gallery.items.extend(saved['items']); gallery.errors.extend(saved['errors'])
    gallery.safe('sweep', cases)
    payload = report(out, cfg, gallery, verification, cases, audit, errors, time.monotonic()-started)
    payload.update(particles_modeled=True, physical_time_modeled=True, two_way_gaussian_coupling=True,
                   geometry=asdict(geometry), boundary='periodic x; no-slip flat and semicircular walls',
                   body_force_x=12/cfg.reynolds, original_parcel_model_preserved=True)
    payload['limitations'].extend(['Initial fluid field is adapted to the curved no-slip walls; r=0 recovers the original field exactly.',
        'Total represented particle mass and nominal loading retain the original rectangular reference volume; effective fluid-volume loading is also reported.',
        'Collision-free model retained. Center nonpenetration is penalized and audited, not enforced by an invented rebound or trajectory clipping.',
        'Kernel is renormalized over fluid-only quadrature. Filtered solid walls change the regularization locally.'])
    dump(out/'summary.json', payload)
    table_path = out/'paper_metrics.csv'
    if cases:
        table = pd.read_csv(table_path)
        for key in ['effective_loading_fluid_volume', 'max_obstacle_inside_count', 'max_kernel_error', 'max_force_exchange_error']:
            table[key] = [c[key] for c in cases]
        table['radius_over_H'] = geometry.radius
        table.to_csv(table_path, index=False)
    note = ('Transient two-way Gaussian parcels; periodic x and original body forcing retained. '
            'Only the geometry and required wall/domain/initial-field treatment change. '
            'No new contact/rebound law: center penetration is penalized and explicitly reported. '
            'Initial velocity is adapted to the constriction; r=0 reproduces the original model exactly.')
    with (out/'REPORT.md').open('a') as f: f.write('\n\n## Geometry extension\n\n'+note+'\n')
    page = (out/'index.html').read_text().replace('<h1>Gaussian parcel PINN</h1>', '<h1>Gaussian parcels · constricted channel</h1><p>'+note+'</p>')
    (out/'index.html').write_text(page)
    completion = dict(expected_cases=len(cfg.cases), actual_cases=len(cases), expected_animations=2*len(cfg.cases),
         actual_animations=len(list(out.glob('cases/*/*.gif'))), particles_modeled=True, physical_time_modeled=True,
         two_way_gaussian_coupling=True, seconds_before_archive=time.monotonic()-started, requested_seconds=args.max_minutes*60)
    dump(out/'completion.json', completion); truncated = False
    archive = out.parent/f'{out.name}_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if time.monotonic() >= deadline: truncated = True; break
            if p.is_file(): z.write(p, Path(out.name)/p.relative_to(out))
        for p in source+[root/'run_constriction.py', root/'requirements-gpu.txt', root/'docs/CONSTRICTION_GPU_FA.md']:
            if time.monotonic() >= deadline: truncated = True; break
            if p.is_file(): z.write(p, Path('source')/p.relative_to(root))
    completion.update(total_seconds=time.monotonic()-started, archive_complete=not truncated); dump(out/'completion.json', completion)
    with zipfile.ZipFile(archive, 'a', zipfile.ZIP_DEFLATED) as z: z.write(out/'completion.json', 'FINAL_COMPLETION.json')
    archive.with_suffix('.zip.sha256').write_text(f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(f"Report: {out/'index.html'}\nArchive: {archive}\nElapsed: {completion['total_seconds']/60:.2f} min", flush=True)
    return int(not payload['workflow_complete'] or truncated or len(cases) != len(cfg.cases))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--config', type=Path); p.add_argument('--geometry', type=Path)
    p.add_argument('--validation-pack', type=Path, default=Path('reference_inputs/Validation_Pack'))
    p.add_argument('--max-minutes', type=float, default=55); p.add_argument('--train-minutes', type=float, default=35)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda'); p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--worker', choices=['train', 'verify', 'export'], help=argparse.SUPPRESS)
    p.add_argument('--index', type=int, default=0, help=argparse.SUPPRESS); p.add_argument('--seconds', type=float, default=5, help=argparse.SUPPRESS)
    args = p.parse_args(argv); args.out = args.out.resolve(); args.validation_pack = args.validation_pack.resolve()
    if not 0 < args.max_minutes <= 55 or args.train_minutes <= 0: p.error('Use 0 < max-minutes <= 55 and train-minutes > 0.')
    if args.worker:
        try: worker(args)
        except Exception: traceback.print_exc(); return 1
        return 0
    return run(args)


if __name__ == '__main__': raise SystemExit(main())
