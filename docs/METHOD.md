# Formulation and reporting conventions

## 2D carrier with planar sphere parcels

Use H, U, H/U and ρf U² to nondimensionalize length, velocity, time and pressure. The carrier lives in Ω=[0,Lx/H]×[0,1]. Let b=B/H be the specified reference spanwise depth, δ=d/H, R=ρp/ρf, Re=UH/ν. Physical particles are spheres; their representative centers move in the x–y plane. This is a planar approximation to a dilute sphere population, not a resolved circular cylinder with no-slip on its circumference.

Per-depth represented mass is

    m_i* = N_i R π δ_i³ / (6 b).
    St_i = R Re δ_i² / 18.
    β = Σ m_i* / |Ω*|.

In 3D remove the b factor and use the full dimensionless volume. The implementation reports the per-depth nondimensional physical mass as `physical_mass`; multiply by ρf H²B to recover a physical sphere's kg mass in the 2D convention. The dimensional 2D fluid source is force divided by reference depth and area. The 2D Gaussian has area⁻¹ units, not volume⁻¹. Effective local solid volume fraction is Σ m_i* W_i*/R.

## Equations solved by the PINN

    ∇·u = 0
    ∂t u + u·∇u = -∇p + Re⁻¹ Δu + S + (12/Re, 0)
    dq_i/dt = v_i
    dv_i/dt = (U_i - v_i)/St_i + g_eff + a_contact
    U_i = Σ_a W_ai u_a ΔA_a
    F_i = m_i* (U_i-v_i)/St_i
    S_a = -Σ_i W_ai F_i

In 3D replace areas with volumes and append the third component. The source deposits only the hydrodynamic drag reaction. Gravity, internal contacts and wall contact are not deposited a second time. `g_eff` is the user-selected effective acceleration; buoyancy must be included exactly once when mapping a physical case.

Gaussian weights are periodized in x (and z in 3D); retained image tails are at least eight widths away after choosing the nearest image. The bounded y direction is renormalized by the actual midpoint quadrature:

    G_ai = Π_d Σ_images exp[-(x_ad - q_id + k L_d)²/(2 h²)]
    Z_i = Σ_a G_ai ΔA_a
    W_ai = G_ai/Z_i.

At arbitrary collocation points the numerator uses their coordinates and the denominator uses the same fixed quadrature. Normalization is exact on the transfer quadrature, not an analytical guarantee for all possible integration rules. Both interpolation and spreading share W. Thus Σ_a S_a ΔA_a = -Σ_i F_i and Σ_a u_a·S_a ΔA_a = -Σ_i U_i·F_i. Interphase drag power is -Σ_i m_i*|U_i-v_i|²/St_i. These are semidiscrete identities; a PINN with residual error need not conserve global momentum exactly.

## Network and objective

The fluid network maps periodic sine/cosine features, wall-normal coordinate and time to velocity and pressure corrections. The parcel network maps time to 2DN values. Both use tanh and float64. For a channel of height one:

    u(x,t) = u_initial(y) + (t/T) y(1-y) NN_u(x,t)
    p(x,t) = NN_p(x,t)-NN_p(0,t)
    b_i(t) = 1 - exp(-t/St_i)
    a_i(t) = t - St_i b_i(t)
    q_i(t) = q_i0 + t v_i0 + a_i(t) NN_qi(t)
    v_i(t) = v_i0 + b_i(t) NN_vi(t).

The relaxation envelopes resolve the known response-time scale while leaving both neural functions trainable. They contain no reference velocity, trajectory or coupled solution. Set `relaxation_basis=false` to use the generic alternatives a=t²/T, b=t. The first parcel network block predicts position correction; the second predicts velocity correction. They are linked through dq/dt−v in the objective. This is a continuous-time coupled forward PINN, not a neural fit to reference trajectory data. Pressure gauge is p(0,t)=0. No independent initial-pressure condition is imposed: localized initial drag requires an elliptic pressure response. Periodicity follows from Fourier features. Initial velocity is (6y(1-y),0). Initial centers are seeded, and initial parcel velocity is zero.

Weighted mean squared residuals: fluid momentum 1, continuity 2, parcel kinematics 5, parcel acceleration 1, center-confinement penalty 100. Default walls have no explicit contact; a trained collision-free case leaving the channel violates quality criteria. Those paths are reported as predicted, without clipping. Increase training or select a matching contact model; clipping would hide failure.

Adam uses a decaying learning rate and gradient-norm clipping. Three periodic Fourier harmonics per periodic coordinate help represent localized sources. Training time samples are biased toward the initial response, and the audit includes endpoints. After Adam, optional L-BFGS refinement uses its own fixed collocation set and a strong-Wolfe line search. Checkpoint selection uses a distinct fixed residual audit set. A third, independent set measures reported residuals, including initial/early/final times. No reference values are used in optimization. The supplied V01/V02 analytical CSVs are read only after fitting reduced physics-only ODE PINNs. V10 is a separate 2D periodic Navier–Stokes PINN with hard initial conditions, tested against Taylor–Green decay over t∈[0,5].

## Optional physical contact

`soft_sphere` forces each multiplicity to one. For physical spheres in the planar representation, contact is modeled by a frictionless clipped normal spring/dashpot on overlap. In the per-depth formulation, use the per-depth mass and stiffness/damping consistently. Pair forces are equal and opposite and wall forces act on the wall. It is a deliberately limited closure: there is no lubrication, tangential spring, rotation, or calibrated wet restitution. Contact time must be resolved by temporal collocation and the network. Hard binary impulses and the Gaussian-cloud frozen-rate sampler are independently tested primitives, not the default PINN contact model. No complete time-evolving statistical Gaussian collision solver is claimed.

## Conservation diagnostics

Momentum budget includes imposed drive, particle effective gravity, pressure at the two channel walls and wall viscous tractions. Periodic face terms cancel. Energy budget includes external work, viscous dissipation and drag dissipation. Midpoint quadrature error and PINN residuals affect both budgets. Total energy can increase in a driven channel. Contact-mode budgets are intentionally omitted (null in the summary) until contact energy and wall forces are included consistently; drag dissipation remains available.

The disturbed fluid is sampled directly. No published undisturbed-velocity/self-interaction correction is implemented. No arbitrary Gaussian width can therefore establish accurate isolated-sphere mobility. Stokes drag is screened with the reported maximum Re_p=Re δ |U_i-v_i|. Re_p≤1 is only a loose screening bound, not an asymptotically small-Re guarantee. The local volume-fraction screening bound is 0.01.

## Evidence levels

1. Unit checks: algebra, derivatives, transfer and recovery behavior.
2. V01–V10: reduced-model analytical and conservation verification. V08 is frozen-rate ensemble verification, not validation of collision-driven dynamics.
3. Channel results: coupled PINN predictions with held-out residuals, balances, model-validity screens and raw arrays.
4. Literature/experimental validation: not claimed by this baseline. The complete catalogue audit records missing data and absent physical mechanisms for all 51 cases, in source order. No manual figure digitization, unit retuning or hidden phase alignment is performed.

To support a paper, execute seeds, quadrature refinement, temporal collocation refinement and training-budget studies. Change h separately because it changes the regularized model. At fixed population, coincident splitting tests deterministic parcel invariance (V09); dispersed parcel refinement needs ensemble statistics. Use independent experimental or numerical references only after matching physical model, geometry, dimension, boundary conditions, initial states and units. A `paper` preset does not automatically complete this work.
