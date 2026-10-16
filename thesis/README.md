# Particle transport thesis

English thesis manuscript by **Mohammad Reza Sayah**, supervisor **Mohammad Ali Bijarchi**, Sharif University of Technology.

The study order is: the baseline particle problem; Gaussian parcels in a straight channel; a proposed smooth constriction. The document follows the uploaded template's report format, margins, typography and chapter structure. The original template is preserved under `template/` as a reference file, not as manuscript content.

## Files

- `Particle_Transport_Thesis.pdf`: compiled manuscript.
- `thesis.tex`, `metadata.tex`, `references.tex`, `chapters/`: editable LaTeX.
- `figures/`: nineteen redesigned figures in PDF, SVG and PNG, including five plots of the computed Gaussian parcel response.
- `evidence/`: unmodified copies of selected numerical records and relevant method definitions.
- `generated/`: numerical macros, tables, figure provenance and numerical audit.
- `generated/citation_plan.json`: the 36 contextual citation placements and metadata provenance for the 25 added references (40 references in total).
- `build_thesis.py`: build and validation command.
- `DELIVERY_NOTES_FA.md`: scope, metadata assumptions and delivery notes in Persian.

## Compile without Python dependencies

Upload the source bundle to Overleaf and select `thesis.tex` as the main document with pdfLaTeX. Alternatively, use TeX Live and:

```bash
latexmk -pdf -pdflatex='pdflatex -no-shell-escape %O %S' -interaction=nonstopmode -halt-on-error -outdir=build thesis.tex
```

Typical packages: `texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-fonts-recommended`, `lmodern`, and `latexmk`. Use a complete TeX Live installation or Overleaf if package names differ on your operating system. Figures and numerical tables are already included; training is not required.

## Rebuild and audit the manuscript

```bash
python build_thesis.py
```

The build needs `latexmk`, pdfLaTeX, `pdftotext` and `pdfinfo`. It verifies evidence hashes, figures, citations, references, missing glyphs and overfull boxes. It does not modify or re-run the original simulations.

To regenerate every figure and repeat the numerical audit:

```bash
python build_thesis.py --regenerate
```

Regeneration additionally requires Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and PyTorch 2.6.0+cpu (or a compatible CPU environment). The archived baseline model definitions and two checkpoints support reconstruction of the original Poiseuille and held-out drag metrics. No external network requests or credentials are needed. The script creates writable caches under `build/`.

`tools/prepare_evidence.py` is an archival helper used in the original workspace. It requires both complete project checkouts at their `/workspace` paths and is not needed for portable compilation or figure regeneration. To preserve the archived record, do not run it merely to rebuild the PDF.

## Evidence boundaries

Accepted numerical comparisons use the baseline field/drag results and the Gaussian model's reduced verification problems. Five additional figures show the actual finite coupled predictions of `high_inertia_seed7` (`St=0.6`, loading `0.05`): transport snapshots, paths and velocities, represented clouds, carrier response, and energy/drag/balance diagnostics. This case has the lowest audited momentum residual among the four evaluated cases. The coupled-results table presents measured residuals and their ratios to the original reference criterion, replacing binary verdict labels with quantitative comparisons. The plots illustrate the computed response without claiming independent coupled validation. Unsupported deposition maps and extruded views are excluded. No constricted-channel simulation result, experiment, GPU speedup, or completed physical validation is invented.

The data-availability section and repository URLs have been removed from the thesis text. This revision makes no promise of public data access. Repository access settings are separate from the manuscript wording and remain unchanged.

The citation revision follows the uploaded review's placement plan. Added references support physical and methodological background, not the thesis-specific error values. It clarifies that all four coupled approximations were computed, distinguishes residual screening from optimizer convergence, and documents actual parcel settings, quadrature, sampling and balance definitions. Measured errors, numerical arrays, acceptance thresholds and pass/fail outcomes are preserved. Live Crossref/DOI lookups were blocked by the network proxy; the added bibliographic metadata comes from the supplied review and is not described as newly verified against publishers.

The archived Git revisions identify the inspected checkouts. Historical result manifests preserve their own source hashes. Some current driver/configuration files differ from those historical manifests; the central baseline model/geometry and Gaussian physics/model/verification files match. This package supports evidence reproduction and typesetting, not a claim of exact historical retraining from every current driver.

## Create a portable source archive

```bash
python build_thesis.py --bundle
```

The ZIP includes all typesetting inputs and evidence. Build caches, temporary reviews, absolute-path auxiliary files and the ZIP itself are excluded. Original simulation code and numerical results remain unchanged.
