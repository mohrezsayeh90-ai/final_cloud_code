#!/usr/bin/env python3
"""Regenerate all figures/reports from a completed run without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
import torch
from parcel_pinn.config import configuration
from parcel_pinn.figures import Gallery
from parcel_pinn.model import CoupledPINN
from parcel_pinn.pipeline import report
from parcel_pinn.training import seed_all


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args();out=args.run.resolve();started=time.monotonic()
    previous=json.loads((out/'summary.json').read_text())
    cfg=configuration(previous['preset'],out/'config.json')
    gallery=Gallery(out,cfg);gallery.safe('architecture')
    validation=out/'validation'
    verification=json.loads((validation/'verification.json').read_text())
    for kind,cols,mode,file in [
        ('V01',['v_over_vt','distance_over_vt_tau'],'response',validation/'V01'/'comparison.csv'),
        ('V02',['fluid_u','particle_v'],'response',validation/'V02'/'comparison.csv'),
        ('V04',['overlap'],'overlap',validation/'V04.csv'),
        ('V10',['u','v','p'],'taylor',validation/'V10'/'comparison.csv')]:
        if file.is_file():gallery.safe('verification_plot',mode,pd.read_csv(file),cols,kind,validation)
    gallery.safe('verification_summary',verification)
    for entry in previous['cases']:
        cfg.seed=entry['seed'];seed_all(cfg.seed,cfg.threads)
        case=next(c for c in cfg.cases if c['name']==entry['case'])
        model=CoupledPINN(cfg,case)
        folder=out/entry['directory']
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        particles=pd.read_csv(folder/'particles.csv');budgets=pd.read_csv(folder/'diagnostics.csv')
        gallery.safe('training',pd.read_csv(folder/'training.csv'),entry['case'],folder)
        gallery.safe('fields',model,entry['case'],folder)
        gallery.safe('trajectories',particles,entry['case'],folder)
        gallery.safe('budgets',budgets,entry['case'],folder)
        gallery.safe('animation',particles,entry['case'],folder)
    gallery.safe('sweep',previous['cases'])
    report(out,cfg,gallery,verification,previous['cases'],previous['literature_cases'],previous['errors'],previous['seconds'])
    renderer=Path(__file__).parent/'parcel_pinn'/'figures.py'
    (out/'render_manifest.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,
        figures_source_sha256=hashlib.sha256(renderer.read_bytes()).hexdigest(),
        note='Only figures and report regenerated. Training and stored numerical results unchanged.'),indent=2))
    return int(bool([e for e in gallery.errors if e['figure']!='optional MP4']))


if __name__=='__main__':raise SystemExit(main())
