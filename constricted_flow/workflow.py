"""Time-budgeted workers for the fluid-only semicircular constriction notebook."""
import argparse
from dataclasses import replace
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
from parcel_pinn.training import seed_all
from parcel_pinn.gpu_workflow import supervised
from .model import Settings, FlowPINN, Geometry, train
from .report import FluidGallery, evaluate, dump, report


def worker(args):
    cfg = Settings.load(args.out/'config.json')
    case = cfg.cases()[args.index]
    seed_all(cfg.seed, cfg.threads)
    folder = args.out/'cases'/case['name']; folder.mkdir(parents=True, exist_ok=True)
    model = FlowPINN(cfg, case)
    if args.worker == 'train':
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dtype = torch.float32 if args.precision == 'float32' else torch.float64
        model.to(device=args.device, dtype=dtype)
        print(f"{case['name']}: {next(model.parameters()).device}, {dtype}", flush=True)
        train(model, folder, args.seconds)
        return
    checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.cpu().double().eval()
    state = checkpoint['training']
    if state['termination'] == 'running': state['termination'] = 'worker_interrupted_best_checkpoint'
    result, mesh, sections = evaluate(model, folder, state)
    gallery = FluidGallery(args.out, cfg)
    for method, parameters in [
        ('training', (pd.read_csv(folder/'training.csv'), case['name'], folder)),
        ('fields', (model, mesh, folder)), ('profiles', (model, sections, folder)),
        ('surfaces', (model, mesh, folder)), ('analytic', (model, folder)),
        ('section_scan', (model, mesh, folder)),
    ]:
        gallery.safe(method, *parameters)
        dump(folder/'gallery.json', dict(items=gallery.items, errors=gallery.errors))
    if any(x['figure'] != 'optional MP4' for x in gallery.errors):
        raise RuntimeError('Figure errors recorded in gallery.json')
    print(f"{case['name']}: momentum={result['heldout_momentum_rms']:.4g}, flux error={result['max_relative_flux_error']:.4g}", flush=True)


def run(args):
    started = time.monotonic(); deadline = started+args.max_minutes*60
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Enable a CUDA GPU runtime. There is no automatic CPU fallback.')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Output folder is not empty; use a fresh run name.')
    cfg = Settings.load(args.config) if args.config else Settings()
    if args.smoke:
        cfg = replace(cfg, hidden=16, depth=2, interior_points=64, boundary_points=12,
                      flux_points=16, epochs=3, frames=9, dpi=90)
    cfg.validate()
    out.mkdir(parents=True, exist_ok=True); cfg.save(out/'config.json')
    dump(out/'run_settings.json', dict(max_minutes=args.max_minutes, train_minutes=args.train_minutes,
         device=args.device, precision=args.precision, smoke=args.smoke, particles_modeled=False,
         physical_time_modeled=False, expected_cases=cfg.cases()))
    dump(out/'runtime.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
         cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(), training_device=args.device,
         training_precision=args.precision, export_device='cpu', export_precision='float64',
         gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None))
    root = Path(__file__).resolve().parents[1]
    source = sorted((root/'constricted_flow').glob('*.py'))+[
        root/'parcel_pinn'/name for name in ['model.py', 'physics.py', 'training.py', 'figures.py', 'gpu_workflow.py']]
    dump(out/'source_manifest.json', [dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source])
    stages, errors, cases = [], [], []
    definitions = cfg.cases()
    train_total = min(args.train_minutes*60, .65*args.max_minutes*60)
    train_deadline = min(deadline, time.monotonic()+train_total)
    print(f"Fluid-only steady flow: {len(definitions)} cases; training <= {train_total/60:.1f} min", flush=True)

    def launch(stage, i, limit):
        name = f'{stage}_{definitions[i]["name"]}'
        command = [sys.executable, '-u', str(root/'run_constriction.py'), '--worker', stage,
                   '--index', str(i), '--out', str(out), '--seconds', str(max(.01, limit-7)),
                   '--device', args.device, '--precision', args.precision]
        status = supervised(command, max(0, limit), out/f'{name}.log')
        status['stage'] = name; stages.append(status); dump(out/'stage_status.json', stages)
        if status['outcome'] != 'complete': errors.append(dict(stage=name, error=status['outcome']))

    for i, case in enumerate(definitions):
        limit = max(0, train_deadline-time.monotonic())/(len(definitions)-i)
        if case['radius'] == 0 and not args.smoke: limit = min(limit, 90.)
        launch('train', i, limit)
    gallery = FluidGallery(out, cfg)
    geo = Geometry(cfg.length, cfg.center_fraction, cfg.radius)
    gallery.safe('geometry', geo); gallery.safe('architecture'); gallery.safe('sampling', geo)
    for i, case in enumerate(definitions):
        folder = out/'cases'/case['name']
        if not (folder/'checkpoint.pt').is_file():
            errors.append(dict(stage=case['name'], error='No finite checkpoint; no fabricated replacement field.'))
            continue
        launch('export', i, max(0, deadline-time.monotonic()-60)/(len(definitions)-i))
        if (folder/'case_status.json').is_file(): cases.append(json.loads((folder/'case_status.json').read_text()))
        if (folder/'gallery.json').is_file():
            saved = json.loads((folder/'gallery.json').read_text())
            gallery.items.extend(saved['items']); gallery.errors.extend(saved['errors'])
    gallery.safe('comparison', cases)
    payload = report(out, cfg, gallery, cases, stages, errors, time.monotonic()-started)
    completion = dict(all_cases_exported=len(cases) == len(definitions), expected_cases=len(definitions),
                      actual_cases=len(cases), expected_animations=len(definitions),
                      actual_animations=len(list(out.glob('cases/*/*.gif'))),
                      requested_seconds=args.max_minutes*60, seconds_before_archive=time.monotonic()-started,
                      physical_time_modeled=False, particles_modeled=False, cuda_tested_in_this_run=args.device == 'cuda')
    dump(out/'completion.json', completion)
    archive = out.parent/f'{out.name}_results.zip'; truncated = False
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if time.monotonic() >= deadline: truncated = True; break
            if p.is_file(): z.write(p, Path(out.name)/p.relative_to(out))
        bundle_files = sorted((root/'constricted_flow').glob('*.py'))+sorted((root/'parcel_pinn').glob('*.py'))+[
            root/'run_constriction.py', root/'requirements-gpu.txt', root/'docs/CONSTRICTION_GPU_FA.md']
        for p in bundle_files:
            if time.monotonic() >= deadline: truncated = True; break
            if p.is_file(): z.write(p, Path('source')/p.relative_to(root))
    completion.update(total_seconds=time.monotonic()-started, archive_complete=not truncated)
    dump(out/'completion.json', completion)
    with zipfile.ZipFile(archive, 'a', zipfile.ZIP_DEFLATED) as z: z.write(out/'completion.json', 'FINAL_COMPLETION.json')
    archive.with_suffix('.zip.sha256').write_text(f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(f"Report: {out/'index.html'}\nArchive: {archive}\nElapsed: {completion['total_seconds']/60:.2f} min", flush=True)
    return int(not payload['workflow_complete'] or truncated)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--max-minutes', type=float, default=55)
    p.add_argument('--train-minutes', type=float, default=35)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', choices=['train', 'export'], help=argparse.SUPPRESS)
    p.add_argument('--seconds', type=float, default=5, help=argparse.SUPPRESS)
    p.add_argument('--index', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv); args.out = args.out.resolve()
    if not 0 < args.max_minutes <= 55 or args.train_minutes <= 0:
        p.error('Use 0 < max-minutes <= 55 and train-minutes > 0.')
    if args.worker:
        try: worker(args)
        except Exception: traceback.print_exc(); return 1
        return 0
    return run(args)


if __name__ == '__main__': raise SystemExit(main())
