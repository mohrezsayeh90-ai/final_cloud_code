#!/usr/bin/env python3
"""Build the corrected transient Gaussian parcel notebook."""
from pathlib import Path
from build_gaussian_constriction_notebook import build

if __name__ == '__main__':
    build(Path(__file__).resolve().parent)
