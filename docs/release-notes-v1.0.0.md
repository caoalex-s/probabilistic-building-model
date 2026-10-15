# Version 1.0.0 release notes

Version 1.0.0 provides the stationary model associated with the Structural
Safety paper [*Probabilistic modelling of buildings as networks*](https://doi.org/10.1016/j.strusafe.2026.102783),
Article 102783.
It packages the simulator, analysis methods, portable commands, documentation
and scientific evidence for use outside the authors' working environment.

## Included

- Monte Carlo collapse-size distributions with reproducible random streams,
  percolation thresholds, risk equivalence and notional-damage robustness.
- Installable command-line tools, authoritative SQL resources and Python APIs.
- A getting-started guide and small interactive notebook, with explained model
  equations, collapse-size distributions and a baseline/damage comparison.
- Portable notebooks, a pinned Python 3.12 paper environment and campaign scripts.
- Curated Figures 3–5, scientific counts, provenance and artifact hashes in
  [the reproduction package](../reproduction/README.md).
- A saved-count validator that checks the evidence without running simulations,
  plus a small fresh smoke profile and explicit larger campaign profiles.

## Validation and limits

The latest verified code candidate passed 192 tests in each of five CI
environments, with six optional publication-profile tests skipped per environment.
A newly installed wheel and extracted source distribution passed
runtime, SQL-resource, CLI, fresh-smoke and saved-count checks. The
[validation workflow](../.github/workflows/validation.yml) also tests Linux on
Python 3.11–3.13 and Windows/macOS on Python 3.12; inspect the run for the exact
release commit before publishing.

All four notebooks also completed before and after the source cleanup. The
beginner notebook ran unchanged; paper notebooks used 2,000 samples per run or
scan point with their original grids, model inputs and seed. Scientific database
contents and risk diagnostics matched exactly, and all 57 PDF renders and two
PNG figures matched pixel for pixel. These operational checks do not replace
the manuscript-scale evidence described below.

All 20 fresh damage-scenario distributions at one million realizations per
scenario reproduce the retained publication counts exactly. Reduced Figure 3
campaigns and an independent-seed check satisfy the documented agreement gates.
The full ten-million-per-point Figure 3 campaign has not been rerun. In Case 2,
the reduced estimate rounds to 0.58 while the manuscript estimate rounds to 0.59;
the reduced figure is labelled with its actual sample counts. The engineering
agreement tolerance is not a statistical confidence interval. See
[manuscript validation](manuscript-validation.md) for numerical results and limits.

Publication PDF exports retain the manuscript plotting style. New standard plots
use portable Matplotlib mathtext; generating publication-style plots requires
external LaTeX. Larger fresh campaigns remain computationally intensive; inspect
the plan with `--dry-run` before executing them.

## Citation and release status

Software authors, license, version and the associated article citation are in
[CITATION.cff](../CITATION.cff). The paper DOI is
[10.1016/j.strusafe.2026.102783](https://doi.org/10.1016/j.strusafe.2026.102783),
and its article number is 102783. Version 1.0.0 is dated 16 October 2026.
A software DOI can be added after the release is archived.
The maintainer has selected the standard MIT license, permitting commercial
reuse. The README and citation file request citation of the paper and software;
this request adds no license condition. ETH Zurich is identified as copyright
holder, with Alex Sixie Cao and Asim Kumalic credited as software creators.
The maintainer has confirmed approval for the MIT release.
