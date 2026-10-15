# Executed validation checks

- 16 unit tests completed and passed, including hard cylinder/wall no-slip, exact periodicity, centrosymmetry, sampling outside the body, multipole features, and an independent analytic rotating-cylinder test for both the NS operator and torque integral.
- Independent conformal P2/P1 FEM solutions completed at Rep=1 and Rep=2 on both 889-vertex and 3041-vertex meshes.
- Newton residuals reached below 1e-10 without an article target or fitted angular velocity.
- Two simulation-assisted PINNs completed 2200 Adam steps and 800 L-BFGS iterations each; same architecture and training settings for both Reynolds conditions.
- Article spin values and source-curve points were excluded from training and checkpoint selection.
- Held-out momentum, continuity, torque and no-slip criteria passed for both final PINNs.
- PINN field comparison used 1200 independent FEM evaluation points, not the 1024 training anchors.
- The end-to-end benchmark command was separately tested with deliberately insufficient training; reports and animations were produced and strict mode correctly returned exit code 2.

Scope: equilibrium angular velocity at a prescribed symmetry-plane center. No experimental, migration-trajectory, startup, Gaussian point-parcel or collision validation is claimed. The successful PINNs use internally generated FEM field anchors and are not physics-only networks.
