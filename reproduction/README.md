# Reproduction evidence and commands

This directory contains curated evidence from the completed stationary-model
campaigns at source revision `71c774ec89f68ca732626e262a782560bf561952`.
The files are available in a source checkout and source distribution. Install
the package and the Python 3.12 paper requirements as described in the main
README before running the commands below from the source root.

## Check the saved evidence

```bash
python scripts/validate_reproduction.py --out output/saved-count-validation.json
```

This verifies artifact hashes and recomputes all three Figure 3 thresholds,
11,895 retained-prefix PMF comparisons, 1,235 independent-seed PMF comparisons,
Figure 4–5 robustness values, probability limits, and numerical/analytical
risk equivalence. It runs no Monte Carlo simulations. Optional
`--plot-dir output/saved-count-plots` regenerates three portable Figure 3 PNGs
using the saved fresh counts.

The validation uses a `0.015` engineering tolerance for threshold agreement
and smoothing/grid stability. It is not a confidence interval. Common random
numbers couple scan points; no independent-point bootstrap interval is used.
The retained-prefix comparison conditions on historical 10-million counts and
the recorded random-stream contract. The separate seed check uses exact
conditional two-sample tests. Both use Bonferroni correction with a family
error allowance of `0.01`.

## Run a small fresh example

```bash
python scripts/reproduce.py smoke --out output/smoke-example --workers 2
```

This uses an eight-node lattice, 2,000 samples per point and a three-point
scan, plus a baseline: 8,000 realizations. It checks integer PMFs, the
zero-propagation binomial law, shared initial-failure counts and the
full-propagation limit. It exercises the installed simulator and SQL resources;
it does not reproduce a manuscript threshold.

Every simulation command requires a new output directory and refuses to
overwrite one. It retains input/result databases, logs and `campaign.json`
with portable commands, full counts, environment information and hashes.
Choose a new directory when repeating a command. `--dry-run` prints the plan
without creating files or running simulations. Worker count defaults to one
and does not change the random-stream contract.

## Regenerate scientific campaigns

| Profile | Realizations including baselines | Purpose |
| --- | ---: | --- |
| `figure3-reduced` | 330 million | Original grids, 1/2/2 million samples per point |
| `figure3-seed-check` | 22 million | Seed 456, 19 transition-window points |
| `figure3-full` | 1.86 billion | Original grids, 10 million samples per point |
| `damage` | 20 million | All twenty Figure 4–5 baseline/damage scenarios |
| `risk` | 55 million | Focused numerical crossing, broad anchors and probability endpoints |

For example:

```bash
python scripts/reproduce.py figure3-reduced --out output/figure3-reduced --workers 8
python scripts/reproduce.py figure3-seed-check --out output/figure3-seed-check --workers 8
python scripts/reproduce.py damage --out output/damage-reproduction --workers 8
python scripts/reproduce.py risk --out output/risk-reproduction --workers 8
```

These commands run simulations, export counts and record threshold, robustness
or risk-equivalence analysis in `campaign.json` as appropriate. The independent
seed windows provide probability counts without estimating new thresholds.
Damage scenarios share one result database per propagation setting, preserving
their baseline/damaged run IDs for the post-processing CLI. Use the manuscript
notebooks and post-processing CLI for figure generation from the new databases.
The saved-count validator above checks the curated completed campaigns, rather
than arbitrary new campaign directories. The risk profile regenerates its
three anchors; the recorded campaign reused three damage baselines, adding
52 million new realizations to those 3 million already completed.

The reduced Figure 3 campaign passed all recorded checks and used 81.1% fewer
realizations including its independent-seed check. Its thresholds are
`0.669039`, `0.579861`, and `0.088178`. Case 2 rounds to `0.58` rather than the
manuscript's `0.59`. Full publication-sample Figure 3 regeneration remains a
separate task. Bitwise agreement across unrecorded dependency environments
is not guaranteed.

## Curated artifacts

- [Reduced Figure 3](figures/Figure-3-reduced.pdf): three pages with explicit
  sample-count labels and publication plotting fonts/colours; standalone layout.
- [Figure 4](figures/Figure-4.pdf): six panels, original order.
- [Figure 5](figures/Figure-5.pdf): two panels; original inputs `0.094` and `0.0927`.
- [Validation diagnostics](figures/Validation-diagnostics.pdf): three pages
  showing probability limits and focused risk-equivalence checks.
- [Validation report](data/validation.json): recomputed numerical results.
- [Provenance](data/provenance.json): source revision, package versions,
  environment, timings and realization totals.
- [Artifact hashes](SHA256SUMS.json): SHA-256 values for curated data and figures.

`data/figure3.json`, `data/damage.json`, and `data/risk.json` contain integer
counts extracted from the completed fresh campaigns. Historical comparison
counts are in `tests/scientific/data/manuscript_reference.json`, including all
65 bins at every Figure 3 point. Reported estimates are derived from the counts,
not substituted for them. The archived PDFs are unchanged copies of the
visually inspected exports; captions and assembly can differ from the manuscript.

This package contains portable scientific evidence. Raw result databases and
private working files are excluded. Scientific data, figures and scripts use
the repository's MIT license; see [LICENSE](../LICENSE).
