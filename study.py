#!/usr/bin/env python3
"""Controlled studies: fixed loading, geometry and RNG. Each level retrains PINNs.

No notion of timestep exists in this continuous-time PINN. Budget and temporal
collocation studies replace claims of an integrator timestep refinement.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from parcel_pinn.config import configuration


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',choices=['quadrature','width','parcels','budget','time_sampling'],required=True)
    p.add_argument('--preset',choices=['smoke','demo','paper'],default='demo')
    p.add_argument('--validation-pack',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    c=configuration(a.preset);c.cases=[dict(name='baseline',stokes=.2,loading=.05)]
    levels={'quadrature':[(12,12),(24,24),(48,48)],'width':[.10,.14,.20],
            'parcels':[6,12,24],'budget':[400,1600,6400],
            'time_sampling':[2,4,8]}[a.study]
    rows=[]
    for i,level in enumerate(levels):
        values=asdict(c)
        key={'width':'width','parcels':'nparcels','budget':'epochs','quadrature':'quadrature','time_sampling':'time_batch'}[a.study]
        values[key]=level
        # Deterministic coincident splitting isolates parcel transfer invariance in V09.
        # The channel count sweep uses fresh matched-seed sampling, NOT coincident splitting.
        config=a.out/f'level_{i}.json';config.write_text(json.dumps(values,indent=2))
        target=a.out/f'level_{i}'
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),
                               '--preset',a.preset,'--config',str(config),'--validation-pack',a.validation_pack,
                               '--out',str(target)],check=False)
        rows.append(dict(level=level,returncode=result.returncode,results=str(target)))
    (a.out/'study_manifest.json').write_text(json.dumps(dict(study=a.study,levels=rows,
        interpretation='Fixed total represented mass. Channel parcel sweep changes the population sampling; use multiple seeds to quantify sampling noise. Kernel width is a model parameter, not a grid spacing.'),indent=2))
    return int(any(r['returncode'] for r in rows))


if __name__=='__main__':raise SystemExit(main())
