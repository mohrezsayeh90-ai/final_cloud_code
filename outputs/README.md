# Preserved run history

All generated run folders are retained at the user's explicit request. Some are smoke tests, injected-failure tests, styling checks or interrupted development runs; those are not scientific validation results. The original completed four-case 2D run is `evaluated_2d_final/`. Its `summary.json` states the actual residual criteria and the absence of a completed literature validation at that stage. Do not substitute earlier figures for that run or interpret a directory without a completed summary as a completed simulation.

The original delivered archives are in `../deliverables/`; uploaded source documents and ZIP are in `../inputs/`, with the extracted validation pack under `../reference_inputs/`.

The completed finite-cylinder equilibrium-spin literature comparison is in `couette_validation/`. Other folders beginning `co01_`, `co02_` or `couette_driver_smoke` include development pilots, mesh studies and command checks. The final report clearly distinguishes the FEM-assisted PINN from unsuccessful physics-only pilots.
