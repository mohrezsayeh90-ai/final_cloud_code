# Historical fluid-only interpretation

These files preserve the earlier, incorrect interpretation of the user's request as steady fluid-only flow.
They are not the default or requested Gaussian parcel workflow. The root BilateralConstriction_GPU.ipynb now runs the original transient two-way Gaussian parcel model with only geometry-related changes.

The complete previous source is also available in Git commit be351862a00dc894507d95e21a004699839ef7aa.
The archived notebook is self-contained and reproduces the old fluid-only formulation if intentionally run.
The archived builder is a source snapshot; use the historical commit for rebuilding that version.
The optional run_fluid_only_constriction.py entry point explicitly selects that old formulation.
The previous deliverables/BilateralConstriction_CPUchecks.zip contains only fluid-only checks; the corrected Gaussian checks are in deliverables/GaussianConstrictedPINN_CPUcheck_results.zip.
