# Reference data conventions

The CSV arrays are direct conversions, not fitted or digitized curves.
See mat_inventory.json and the original MATLAB readers and data sheets.

- sphere190_timeseries.csv: time_evol is t Ug/d; uv and uh are velocities / Ug;
  omh is angular velocity times d/Ug. Preserve the source projection conventions.
- spheroid_C15_timeseries.csv: t is t Ug/d; upV and upH are velocities / Ug;
  omHzp is angular velocity times d/Ug. This spheroid Ug differs from sphere Ug.
- *_pdf_*.csv: bin is (variable minus its mean) divided by its standard deviation.
  pdf is the standardized density. Do not interpret these bins as absolute velocity/Ug.
- *_acf_*.csv: lag follows the source nondimensional time convention; autocorrelation
  is the normalized correlation supplied by the authors.
- sphere144_axis_*.csv: x_over_d is the source particle-trajectory-axis coordinate;
  u_relative_over_Ug is the corresponding projected relative fluid velocity.
- couette_equilibrium.csv: manually transcribed finest-grid numerical tables.
- analytic_*.csv: independently generated exact reduced-model verification targets.

Original sphere data: Uhlmann and Dusek 2014, official KIT benchmark directory.
Original spheroid data: Moriche et al., DOI 10.4121/13042793, CC0.
The nominal dimension 2.67d in the spheroid data sheet is rounded; original MAT
metadata give domain diameter 5.333333333333333d. Prefer the original metadata
when reproducing the exact archived dataset.
