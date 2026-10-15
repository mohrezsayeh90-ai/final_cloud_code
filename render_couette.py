#!/usr/bin/env python3
"""Regenerate the Couette comparison from stored FEM/PINN solutions."""
import argparse,json
from pathlib import Path
from literature_validation.report import generate_report


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--pack',type=Path,default=Path('reference_inputs/Validation_Pack'))
    a=p.parse_args();out=a.run.resolve()
    records=json.loads((out/'run_records.json').read_text())
    for record in records:
        for key in ['pinn','fine','coarse']:
            if record.get(key) and not Path(record[key]).is_absolute():record[key]=str(out/record[key])
    generate_report(out,records,a.pack)


if __name__=='__main__':main()
