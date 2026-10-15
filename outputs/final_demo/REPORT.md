# Gaussian parcel PINN results

Preset: demo. Carrier dimension: 2D. Runtime: 289.1 s.
Numerical verification: 9/10 passed. Literature validation: 0/51 run.
Research quality criteria met: False.

## Interpretation

All plotted solver fields and trajectories are PINN predictions. Analytical targets are independent reduced-model verification data. The catalogue audit records every literature case in its supplied order; no mismatched experiment is scored. A completed execution or an attractive figure does not establish predictive accuracy.

For 2D, parcels represent spherical populations moving in a plane. Forces and masses are converted to a per-depth convention using B/H. This is not the finite-size infinite-cylinder model. Extruded channel views solve no spanwise physics.

## Case diagnostics

| Case | Seed | Held-out NS RMS | Parcel ODE RMS | Max slip Re | Max α | Quality criteria |
|---|---:|---:|---:|---:|---:|---|
| low_inertia | 7 | 0.6299 | 0.9914 | 0.2344 | 0.0004778 | False |
| baseline | 7 | 0.2499 | 0.0367 | 0.3707 | 0.0003746 | False |
| high_loading | 7 | 0.6049 | 0.04425 | 0.3707 | 0.001124 | False |
| high_inertia | 7 | 0.0777 | 0.00877 | 0.642 | 0.0003746 | False |

Acceptance is declared before training: all held-out residual RMS < 0.03, slip Re <= 1, local volume fraction < 0.01, all centers inside channel, trained checkpoint and no training error. Passing these screening criteria is necessary, not sufficient, for publication. Resolve residual, discretization and closure uncertainty.

## Remaining checks

- No experimental or resolved-body validation claimed.
- Regularized disturbed-velocity drag, no self-disturbance correction.
- Low slip Reynolds number required by Stokes drag; reported explicitly.
- No lubrication, lift, added mass, Basset history, particle rotation or finite-volume displacement.
- Soft contact is an optional physical-sphere closure; default parcel cases are collision-free.
- Residual thresholds are project criteria, not universal scientific accuracy.
- Paper preset and spatial/time/width/parcel refinement need separate completed runs.

## Figure captions

### Coupled PINN

Two separate fully connected tanh networks, both with 3 hidden layers of width 32. Trainable outputs are coupled through normalized Gaussian quadrature; reference arrays are used only after training.

![Coupled PINN](figures/architecture.png)

### Stokes response

V01: independent reduced-model verification. Analytical targets were excluded from PINN training. Errors and acceptance criteria are in verification.json; a plot is not a pass verdict.

![Stokes response](validation/V01.png)

### Momentum exchange

V02: independent reduced-model verification. Analytical targets were excluded from PINN training. Errors and acceptance criteria are in verification.json; a plot is not a pass verdict.

![Momentum exchange](validation/V02.png)

### Gaussian overlap

V04: independent reduced-model verification. Analytical targets were excluded from PINN training. Errors and acceptance criteria are in verification.json; a plot is not a pass verdict.

![Gaussian overlap](validation/V04.png)

### Taylor–Green flow

V10: independent reduced-model verification. Analytical targets were excluded from PINN training. Errors and acceptance criteria are in verification.json; a plot is not a pass verdict.

![Taylor–Green flow](validation/V10.png)

### Verification overview

Pass/fail classifications are implementation checks using declared project tolerances. They do not certify the 51 literature benchmarks.

![Verification overview](figures/verification_overview.png)

### Training history

low_inertia_seed7: no smoothing or loss truncation beyond the positive logarithmic floor. Checkpoint uses minimum fixed-audit loss.

![Training history](cases/low_inertia_seed7/training.png)

### Flow fields

low_inertia_seed7: direct PINN fields at t=1. Vorticity computed by plotting-grid differences. Uncertainty and residuals must be checked before quantitative interpretation.

![Flow fields](cases/low_inertia_seed7/fields.png)

### Channel profile

low_inertia_seed7: final profile at channel midpoint. The dashed initial unladen profile is not a two-way validation target.

![Channel profile](cases/low_inertia_seed7/profile.png)

### Speed surface

low_inertia_seed7: height encodes speed over the solved x–y plane; the vertical axis is a field value, not a third spatial coordinate.

![Speed surface](cases/low_inertia_seed7/speed_surface.png)

### Transport snapshots

low_inertia_seed7: four direct evaluations of PINN fluid and parcel states. Marker radii are illustrative. Colors use one common speed scale.

![Transport snapshots](cases/low_inertia_seed7/snapshots.png)

### Parcel paths

low_inertia_seed7: direct PINN parcel trajectories; open markers denote initial states and filled markers denote final states. No boundary clipping is applied.

![Parcel paths](cases/low_inertia_seed7/paths.png)

### Channel view

low_inertia_seed7: extruded illustration of a 2D solution; every particle has z=0. Display depth contains no solved physics.

![Channel view](cases/low_inertia_seed7/channel_view.png)

### Parcel velocities

low_inertia_seed7: PINN velocity outputs. Kinematic consistency with position derivatives is audited separately.

![Parcel velocities](cases/low_inertia_seed7/velocities.png)

### Conservation diagnostics

low_inertia_seed7: driven channel includes wall viscous and pressure tractions, body force and particle gravity. Total momentum and energy need not be constant. Quadrature errors enter global balances. Contact cases omit unsupported global balances.

![Conservation diagnostics](cases/low_inertia_seed7/budgets.png)

### Parcel transport

low_inertia_seed7: PINN positions, periodic wrapping in x. Marker sizes are illustrative, not physical radii.

![Parcel transport](cases/low_inertia_seed7/transport.gif)

### Training history

baseline_seed7: no smoothing or loss truncation beyond the positive logarithmic floor. Checkpoint uses minimum fixed-audit loss.

![Training history](cases/baseline_seed7/training.png)

### Flow fields

baseline_seed7: direct PINN fields at t=1. Vorticity computed by plotting-grid differences. Uncertainty and residuals must be checked before quantitative interpretation.

![Flow fields](cases/baseline_seed7/fields.png)

### Channel profile

baseline_seed7: final profile at channel midpoint. The dashed initial unladen profile is not a two-way validation target.

![Channel profile](cases/baseline_seed7/profile.png)

### Speed surface

baseline_seed7: height encodes speed over the solved x–y plane; the vertical axis is a field value, not a third spatial coordinate.

![Speed surface](cases/baseline_seed7/speed_surface.png)

### Transport snapshots

baseline_seed7: four direct evaluations of PINN fluid and parcel states. Marker radii are illustrative. Colors use one common speed scale.

![Transport snapshots](cases/baseline_seed7/snapshots.png)

### Parcel paths

baseline_seed7: direct PINN parcel trajectories; open markers denote initial states and filled markers denote final states. No boundary clipping is applied.

![Parcel paths](cases/baseline_seed7/paths.png)

### Channel view

baseline_seed7: extruded illustration of a 2D solution; every particle has z=0. Display depth contains no solved physics.

![Channel view](cases/baseline_seed7/channel_view.png)

### Parcel velocities

baseline_seed7: PINN velocity outputs. Kinematic consistency with position derivatives is audited separately.

![Parcel velocities](cases/baseline_seed7/velocities.png)

### Conservation diagnostics

baseline_seed7: driven channel includes wall viscous and pressure tractions, body force and particle gravity. Total momentum and energy need not be constant. Quadrature errors enter global balances. Contact cases omit unsupported global balances.

![Conservation diagnostics](cases/baseline_seed7/budgets.png)

### Parcel transport

baseline_seed7: PINN positions, periodic wrapping in x. Marker sizes are illustrative, not physical radii.

![Parcel transport](cases/baseline_seed7/transport.gif)

### Training history

high_loading_seed7: no smoothing or loss truncation beyond the positive logarithmic floor. Checkpoint uses minimum fixed-audit loss.

![Training history](cases/high_loading_seed7/training.png)

### Flow fields

high_loading_seed7: direct PINN fields at t=1. Vorticity computed by plotting-grid differences. Uncertainty and residuals must be checked before quantitative interpretation.

![Flow fields](cases/high_loading_seed7/fields.png)

### Channel profile

high_loading_seed7: final profile at channel midpoint. The dashed initial unladen profile is not a two-way validation target.

![Channel profile](cases/high_loading_seed7/profile.png)

### Speed surface

high_loading_seed7: height encodes speed over the solved x–y plane; the vertical axis is a field value, not a third spatial coordinate.

![Speed surface](cases/high_loading_seed7/speed_surface.png)

### Transport snapshots

high_loading_seed7: four direct evaluations of PINN fluid and parcel states. Marker radii are illustrative. Colors use one common speed scale.

![Transport snapshots](cases/high_loading_seed7/snapshots.png)

### Parcel paths

high_loading_seed7: direct PINN parcel trajectories; open markers denote initial states and filled markers denote final states. No boundary clipping is applied.

![Parcel paths](cases/high_loading_seed7/paths.png)

### Channel view

high_loading_seed7: extruded illustration of a 2D solution; every particle has z=0. Display depth contains no solved physics.

![Channel view](cases/high_loading_seed7/channel_view.png)

### Parcel velocities

high_loading_seed7: PINN velocity outputs. Kinematic consistency with position derivatives is audited separately.

![Parcel velocities](cases/high_loading_seed7/velocities.png)

### Conservation diagnostics

high_loading_seed7: driven channel includes wall viscous and pressure tractions, body force and particle gravity. Total momentum and energy need not be constant. Quadrature errors enter global balances. Contact cases omit unsupported global balances.

![Conservation diagnostics](cases/high_loading_seed7/budgets.png)

### Parcel transport

high_loading_seed7: PINN positions, periodic wrapping in x. Marker sizes are illustrative, not physical radii.

![Parcel transport](cases/high_loading_seed7/transport.gif)

### Training history

high_inertia_seed7: no smoothing or loss truncation beyond the positive logarithmic floor. Checkpoint uses minimum fixed-audit loss.

![Training history](cases/high_inertia_seed7/training.png)

### Flow fields

high_inertia_seed7: direct PINN fields at t=1. Vorticity computed by plotting-grid differences. Uncertainty and residuals must be checked before quantitative interpretation.

![Flow fields](cases/high_inertia_seed7/fields.png)

### Channel profile

high_inertia_seed7: final profile at channel midpoint. The dashed initial unladen profile is not a two-way validation target.

![Channel profile](cases/high_inertia_seed7/profile.png)

### Speed surface

high_inertia_seed7: height encodes speed over the solved x–y plane; the vertical axis is a field value, not a third spatial coordinate.

![Speed surface](cases/high_inertia_seed7/speed_surface.png)

### Transport snapshots

high_inertia_seed7: four direct evaluations of PINN fluid and parcel states. Marker radii are illustrative. Colors use one common speed scale.

![Transport snapshots](cases/high_inertia_seed7/snapshots.png)

### Parcel paths

high_inertia_seed7: direct PINN parcel trajectories; open markers denote initial states and filled markers denote final states. No boundary clipping is applied.

![Parcel paths](cases/high_inertia_seed7/paths.png)

### Channel view

high_inertia_seed7: extruded illustration of a 2D solution; every particle has z=0. Display depth contains no solved physics.

![Channel view](cases/high_inertia_seed7/channel_view.png)

### Parcel velocities

high_inertia_seed7: PINN velocity outputs. Kinematic consistency with position derivatives is audited separately.

![Parcel velocities](cases/high_inertia_seed7/velocities.png)

### Conservation diagnostics

high_inertia_seed7: driven channel includes wall viscous and pressure tractions, body force and particle gravity. Total momentum and energy need not be constant. Quadrature errors enter global balances. Contact cases omit unsupported global balances.

![Conservation diagnostics](cases/high_inertia_seed7/budgets.png)

### Parcel transport

high_inertia_seed7: PINN positions, periodic wrapping in x. Marker sizes are illustrative, not physical radii.

![Parcel transport](cases/high_inertia_seed7/transport.gif)

### Parameter effects

Parcel-mean velocities by case. Multiple-seed shading, when present, is ±1 sample standard deviation; it is not a confidence bound or a discretization error estimate.

![Parameter effects](figures/parameter_effects.png)
