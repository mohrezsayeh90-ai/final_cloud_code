# Executed checks

- Python 3.12, PyTorch 2.6.0+cpu, float64, CPU execution.
- Dependency consistency: `python -m pip check` passed.
- Unit suite: 10 tests executed and passed (including initial-pressure/gauge regression).
- V01--V10: 10 of 10 passed in `outputs/evaluated_2d_final` using `configs/evaluated.json`.
- Four 2D channel cases completed with 1200 Adam steps and 200 L-BFGS iterations each.
- Channel residual screens did not pass. These predictions are diagnostic, not certified paper results.
- All 51 literature cases were audited in catalogue order. None was scored as matching physical validation.
- Injected nonfinite training: best finite checkpoint restored; data, plots, GIF/MP4 and final report produced.
- Isolated single-file execution, outside the source tree, with original uploaded ZIP input: passed.
- Failure at the first optimizer step: initialized fallback explicitly recorded; strict mode returned exit code 2 after completing outputs.
- Checkpoint-only figure/report rendering: executed successfully without retraining.
- Optional 3D and physical-contact residual/gradient checks: finite outputs and backpropagation verified; no 3D or contact benchmark validation claimed.
- The paper preset, multi-seed study and refinement study drivers were not executed. They are available for follow-up research.

The default solver dimension is 2. Channel depth views are extruded illustrations. Analytical data were held out of PINN training. No synthetic demonstration from the validation pack was used as a model prediction or validation result.
