# Fluid particle validation package

Read Validation_Atlas.pdf first. It contains 51 literature case variants from
10 primary papers, including 20 true 2D moving-cylinder variants, plus 10
independently specified verification checks. These are parameter variants,
not 51 independent papers. Three ER cases are literature verification tests.

The data-readiness grades are A (20), B (12), and C (19). A means numerical
targets are included; it does not mean the physics matches a drag-only parcel
model. B and C require the missing items recorded in case_catalog.json.

- data/: 31 official original MAT files and 32 reference CSV files.
- figures/: source excerpts and plots clearly labeled as reference data.
- case_catalog.json and case_catalog.csv: inputs, source locators, data status,
  missing details and compatibility.
- templates/: simulation output and run-log headers.
- tools/compare.py: error metrics and comparison figures.
- examples/: an intentionally biased analytical demonstration. This is NOT a
  simulation result or validation of a physical model.

Quick start from this folder:

    python -m pip install -r tools/requirements.txt
    python tools/compare.py --reference data/analytic_stokes.csv \
      --simulation examples/synthetic_demo_simulation.csv \
      --x time_over_tau --y v_over_vt --scale 1 \
      --out my_comparison

Replace the example with your solver's output in the same units and coordinates.
The tool does no hidden conversion or phase alignment. It reports common-support
coverage and does not extrapolate. Numerical tolerances are user-selected and
are not a universal model-validation verdict.

The official sphere data came from the KIT benchmark directory; an explicit
license was not located. Preserve attribution and consult source terms before
wider redistribution. The spheroid data came from 4TU, DOI 10.4121/13042793,
whose official record specifies CC0. The Mendeley experimental dataset was
verified and linked but could not be downloaded through the available transfer
path. No author or other person was contacted.

Source-table data are marked as manually transcribed. Source-figure curves have
not been silently digitized. Analytical curves are synthetic exact targets.
All source links and access states were checked on 15 October 2026.
