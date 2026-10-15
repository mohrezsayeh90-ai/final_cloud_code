"""Bounded GPU training and CPU rendering of the Gaussian parcel channel model.

Workers isolate CUDA memory, CPU plotting and optional ffmpeg subprocesses. The
controller reserves export time and records incomplete stages rather than
silently presenting a timed-out run as successful.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import pandas as pd
import torch

from .config import configuration
from .diagnostics import evaluate
from .figures import Gallery
from .model import CoupledPINN
from .pipeline import locate_pack, record_inputs, report
from .training import seed_all, train_channel
from .verification import audit_catalog, verify


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def supervised(command, seconds, log):
    """A wall-clock timeout also covers hung CUDA calls and encoder children."""
    started = time.monotonic()
    if seconds <= 0:
        return dict(outcome='skipped_budget', seconds=0, returncode=None)
    log = Path(log)
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=os.name == 'posix')
        position, last_update = 0, 0.
        timed_out = False
        while proc.poll() is None:
            elapsed = time.monotonic()-started
            if elapsed >= seconds:
                timed_out = True
                try:
                    if os.name == 'posix':
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == 'posix':
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except ProcessLookupError:
                        pass
                    proc.wait()
                break
            if elapsed-last_update >= 15:
                with log.open(encoding='utf-8', errors='replace') as reader:
                    reader.seek(position)
                    new = reader.read()
                    position = reader.tell()
                print(new[-2000:] if new else f'{log.stem}: {elapsed:.0f}s', flush=True)
                last_update = elapsed
            time.sleep(.2)
        with log.open(encoding='utf-8', errors='replace') as reader:
            reader.seek(position)
            print(reader.read()[-3000:], flush=True)
    return dict(outcome='timeout' if timed_out else ('complete' if proc.returncode == 0 else 'error'),
                seconds=time.monotonic()-started, returncode=proc.returncode,
                log=log.name)


def worker(stage, out, index, seconds, device, precision, pack):
    cfg = configuration('demo', out/'config.json')
    seed_all(cfg.seed, cfg.threads)
    if stage == 'verify':
        # These are analytical/component checks, not literature validation.
        if cfg.preset == 'smoke':
            check_cfg = replace(cfg, hidden=24, depth=2, ode_epochs=700, tg_epochs=150,
                                ode_lbfgs_steps=0, tg_lbfgs_steps=0)
        else:
            check_cfg = replace(cfg, hidden=32, depth=3, ode_epochs=700, tg_epochs=1200,
                                ode_lbfgs_steps=120, tg_lbfgs_steps=200)
        check_cfg.save(out/'validation'/'verification_config.json')
        gallery = Gallery(out, cfg)
        verify(check_cfg, pack, out/'validation',
               lambda *args: gallery.safe('verification_plot', *args))
        save_json(out/'verification_gallery.json', dict(items=gallery.items, errors=gallery.errors))
        return
    case = cfg.cases[index]
    folder = out/'cases'/f"{case['name']}_seed{cfg.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model = CoupledPINN(cfg, case)
    if stage == 'train':
        dtype = torch.float32 if precision == 'float32' else torch.float64
        if device == 'cuda':
            # TF32/AMP are deliberately disabled for the PDE derivatives.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model = model.to(device=device, dtype=dtype)
        print(f"{case['name']}: actual device={next(model.parameters()).device}, dtype={dtype}", flush=True)
        train_channel(model, folder, max_seconds=seconds, audit_every=50, save_every_audit=True)
        return
    checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
    # Export in float64 on CPU: numpy, matplotlib and old rendering code stay compatible.
    model.load_state_dict(checkpoint['state_dict'])
    model.double().cpu().eval()
    state = checkpoint['training']
    if state['termination'] == 'running':
        state['termination'] = 'worker_interrupted_best_checkpoint'
    particles, budget, residuals = evaluate(model, folder)
    max_rep, max_alpha = float(budget.max_slip_re.max()), float(budget.max_volume_fraction.max())
    quality = (all(r < cfg.residual_tolerance for r in residuals.values()) and max_rep <= 1
               and max_alpha < .01 and budget.wall_outside_count.max() == 0 and state['best_step'] > 0
               and not state['termination'].startswith(('training_error', 'worker_interrupted')))
    result = dict(case=case['name'], seed=cfg.seed, directory=str(folder.relative_to(out)),
                  residuals=residuals, max_slip_re=max_rep, max_volume_fraction=max_alpha,
                  quality_criteria_met=bool(quality), training=state, physical=model.physical,
                  max_momentum_balance_residual=float(budget.momentum_balance_residual.max()),
                  max_energy_balance_residual=float(budget.energy_balance_residual.max()))
    save_json(folder/'case_status.json', result)
    gallery = Gallery(out, cfg)
    for method, args in [
        ('training', (pd.read_csv(folder/'training.csv'), case['name'], folder)),
        ('fields', (model, case['name'], folder)),
        ('trajectories', (particles, case['name'], folder)),
        ('budgets', (budget, case['name'], folder)),
        ('animation', (particles, case['name'], folder)),
        ('gaussian_transport', (model, case['name'], folder)),
    ]:
        gallery.safe(method, *args)
        save_json(folder/'gallery.json', dict(items=gallery.items, errors=gallery.errors))
    if any(e['figure'] != 'optional MP4' for e in gallery.errors):
        raise RuntimeError(f'Plotting failures are recorded in {folder / "gallery.json"}')


def run(args):
    started = time.monotonic()
    total = args.max_minutes*60
    deadline = started+total
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new, empty output folder; previous results are never overwritten.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Enable a GPU runtime; CPU is allowed only with --device cpu.')
    pack = locate_pack(args.validation_pack)
    out.mkdir(parents=True, exist_ok=True)
    (out/'validation').mkdir()
    cfg = configuration('smoke' if args.smoke else 'demo')
    if args.config:
        cfg = configuration(cfg.preset, args.config)
    if cfg.dimension != 2 or len(cfg.seeds) != 1:
        raise ValueError('This bounded notebook supports the requested 2D model and one seed per run.')
    cfg.seed = cfg.seeds[0]
    if not args.smoke:
        cfg.epochs, cfg.hidden, cfg.depth = 1000000, 48, 3
        cfg.time_batch, cfg.space_batch, cfg.quadrature = 4, 96, (24, 20)
        cfg.frames, cfg.dpi = 41, 160
    # A line-search step has unpredictable cost. Use deadline-controlled Adam here.
    cfg.lbfgs_steps = 0
    cfg.validate().save(out/'config.json')
    record_inputs(pack, out)
    settings = dict(device=args.device, precision=args.precision, max_minutes=args.max_minutes,
                    train_minutes=args.train_minutes, smoke=args.smoke,
                    note='Time budget limits work, not prediction error. No literature validation is claimed.')
    save_json(out/'run_settings.json', settings)
    runtime = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                   cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   training_device=args.device, training_precision=args.precision,
                   export_device='cpu', export_precision='float64', threads=cfg.threads,
                   gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None)
    save_json(out/'runtime.json', runtime)
    audit = audit_catalog(pack, out/'validation')
    stages, errors = [], []
    # At least 35% is reserved for CPU diagnostics, figures, encoding and archive.
    train_total = min(args.train_minutes*60, .65*total)
    if args.smoke:
        train_total = min(train_total, 20.)
    reserve = total-train_total
    print(f"{runtime['gpu'] or args.device}: {len(cfg.cases)} Gaussian-parcel cases, training <= {train_total/60:.1f} min", flush=True)

    def launch(stage, index, limit, name):
        cmd = [sys.executable, '-u', '-m', 'parcel_pinn.gpu_workflow', '--worker', stage,
               '--out', str(out), '--index', str(index), '--seconds', str(max(.01, limit-8)),
               '--device', args.device, '--precision', args.precision, '--validation-pack', str(pack)]
        result = supervised(cmd, max(0, limit), out/f'{name}.log')
        result.update(stage=name)
        stages.append(result)
        save_json(out/'stage_status.json', stages)
        if result['outcome'] != 'complete':
            errors.append(dict(stage=name, error=result['outcome']))

    # Train each case before spending the reserved budget on graphics.
    train_deadline = min(deadline-reserve, time.monotonic()+train_total)
    for index, case in enumerate(cfg.cases):
        remaining_cases = len(cfg.cases)-index
        available = max(0, train_deadline-time.monotonic())/remaining_cases
        launch('train', index, available, f"train_{case['name']}")

    launch('verify', 0, min(180., max(0, (deadline-time.monotonic())*.18)), 'verification')
    gallery = Gallery(out, cfg)
    gallery.safe('architecture')
    verification_path = out/'validation'/'verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.is_file() else [
        dict(case_id=f'V{k:02}', outcome='not_run', details='Verification worker did not finish within budget.')
        for k in range(1, 11)]
    if not verification_path.is_file():
        save_json(verification_path, verification)
    vg = out/'verification_gallery.json'
    if vg.is_file():
        saved = json.loads(vg.read_text())
        gallery.items.extend(saved['items']); gallery.errors.extend(saved['errors'])
    gallery.safe('verification_summary', verification)
    cases = []
    for index, case in enumerate(cfg.cases):
        folder = out/'cases'/f"{case['name']}_seed{cfg.seed}"
        if not (folder/'checkpoint.pt').is_file():
            errors.append(dict(stage=case['name'], error='No finite checkpoint available; no replacement trajectory generated.'))
            continue
        # Reserve one minute for summary and archiving, divide fairly across cases.
        limit = max(0, deadline-time.monotonic()-60)/(len(cfg.cases)-index)
        launch('export', index, limit, f"export_{case['name']}")
        if (folder/'case_status.json').is_file():
            cases.append(json.loads((folder/'case_status.json').read_text()))
        if (folder/'gallery.json').is_file():
            saved = json.loads((folder/'gallery.json').read_text())
            gallery.items.extend(saved['items']); gallery.errors.extend(saved['errors'])
    gallery.safe('sweep', cases)
    payload = report(out, cfg, gallery, verification, cases, audit, errors, time.monotonic()-started)
    completion = dict(seconds_before_archive=time.monotonic()-started,
                      requested_seconds=total, stages=stages,
                      all_requested_cases_exported=len(cases) == len(cfg.cases),
                      expected_animations=2*len(cfg.cases), actual_animations=len(list(out.glob('cases/*/*.gif'))),
                      physical_validation_cases_run=0)
    save_json(out/'completion.json', completion)
    archive = out.parent/f'{out.name}_results.zip'
    archive_deadline_hit = False
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(out.rglob('*')):
            if time.monotonic() >= deadline:
                archive_deadline_hit = True
                break
            if path.is_file():
                z.write(path, Path(out.name)/path.relative_to(out))
        # Include the exact source used, not just a changing GitHub URL.
        root = Path(__file__).resolve().parent.parent
        for source in sorted((root/'parcel_pinn').glob('*.py')):
            if time.monotonic() >= deadline:
                archive_deadline_hit = True
                break
            z.write(source, Path('source/parcel_pinn')/source.name)
        for name in ['run_gpu.py', 'GaussianParcelPINN_GPU.ipynb', 'docs/GPU_NOTEBOOK_FA.md', 'requirements-gpu.txt']:
            source = root/name
            if source.is_file() and time.monotonic() < deadline:
                z.write(source, Path('source')/name)
    completion.update(total_seconds=time.monotonic()-started, archive_complete=not archive_deadline_hit)
    save_json(out/'completion.json', completion)
    # The final timing manifest is also present inside the download.
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out/'completion.json', 'FINAL_COMPLETION.json')
    archive.with_suffix('.zip.sha256').write_text(
        f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(f"Report: {out/'index.html'}\nArchive: {archive}\nElapsed: {completion['total_seconds']/60:.1f} min", flush=True)
    required_errors = [e for e in gallery.errors if e['figure'] != 'optional MP4']
    return int(bool(errors or required_errors or archive_deadline_hit or not payload['workflow_complete']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--validation-pack', type=Path, default=Path('reference_inputs/Validation_Pack'))
    p.add_argument('--max-minutes', type=float, default=55.)
    p.add_argument('--train-minutes', type=float, default=35.)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--config', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', choices=['train', 'verify', 'export'], help=argparse.SUPPRESS)
    p.add_argument('--index', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--seconds', type=float, default=10., help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 0 < args.max_minutes <= 55 or args.train_minutes <= 0:
        p.error('Use 0 < max-minutes <= 55 and train-minutes > 0.')
    if args.worker:
        try:
            worker(args.worker, args.out.resolve(), args.index, args.seconds,
                   args.device, args.precision, args.validation_pack.resolve())
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
