# Reproducing Figures 3–5

The manuscript figures are reproduced with the notebooks retained in
`notebooks/`. They implement the stationary model only.

New users should begin with the
[introductory notebook](../notebooks/getting_started.ipynb), which uses a small
eight-compartment model. Its example inputs and sample counts are illustrative;
the three paper notebooks below retain their manuscript-scale settings.

The [curated reproduction package](../reproduction/README.md) contains the
completed fresh counts, reviewed PDF exports, environment/provenance records
and commands for saved-count validation or fresh simulation profiles. These
public commands do not depend on ignored local working files.

## Environment

From the repository root, create and activate a Python environment and install
the paper requirements:

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install .
python -m pip install -r requirements-paper.txt
jupyter lab
```

Launch Jupyter from the repository root and open the files in `notebooks/`.
The notebooks accept either the repository root or the notebook directory as
their working directory.

The pinned environment targets Python 3.12. Use `PLOT_MODE = "standard"` for
portable Matplotlib output. Publication mode
requires a working `pdflatex` installation.

Completed result databases can be re-plotted without rerunning the simulation
sections. For PDF/PGF publication exports, select Matplotlib's `pgf` backend
before importing `pyplot`, then use `plot_mode="publication"` in the plotting
function. The verified publication exports preserve the manuscript panel fonts,
colours, and ordering; combined captions are re-typeset separately. See
[manuscript validation](manuscript-validation.md) for the comparison evidence.

## Figure 3: percolation behaviour

Run `notebooks/percolation.ipynb` with:

- lattice: `4x4x4`;
- `PF_NODE = 1e-4`;
- `SAMPLES = 10_000_000` per scan point;
- `THRESHOLD = 2`;
- `SEED = 123`;
- scan spacing over `[0,1]`: `0.025` for Cases 1 and 2, `0.010` for Case 3;
- case 1: `pe_x = pe_y = pe_z` varied together;
- case 2: `pe_z = 0.8`, with `pe_x = pe_y` varied together;
- case 3: `pe_x = pe_y = 0.8`, with `pe_z` varied.

The manuscript reports thresholds of approximately `0.67`, `0.59`, and `0.09`
for Cases 1, 2, and 3, respectively. A valid general-purpose no-threshold
diagnostic must not be mistaken for the expected result of manuscript Case 2.

The notebook writes the percolation database and figures below
`output/figs_percolation/`.

### Lower-cost Figure 3 verification

For code verification, keep the original lattice, axis configurations, seed,
and scan grids, and reduce the samples per point separately for each case:

| Case | Samples per point | Scan | Points |
| --- | --- | --- | --- |
| 1 | `1_000_000` | `xyz:0,1,0.025` | 41 |
| 2 | `2_000_000` | `xy:0,1,0.025` | 41 |
| 3 | `2_000_000` | `z:0,1,0.01` | 101 |

Run each case separately with a fresh output database. The CLI accepts these
scan specifications through `--percolation-scan` and the sample count through
`--samples`; the fixed axes take their probabilities from the input database.
Keep `--threshold 2 --seed 123`. Set all input edges to `0.01` for Case 1
and `0.8` for Cases 2 and 3 before applying the scan. The Case 1 baseline is
an extra diagnostic; its scanned axes override this initial edge probability.

These runs require 330 million realizations including the three CLI baselines,
versus 1.86 billion for the corresponding 10-million-sample campaign.
Assess input fingerprints, complete integer PMFs and summary consistency,
exact probability limits, and threshold stability under smoothing and grid
perturbations. The recorded verification uses a predeclared absolute threshold
tolerance of `0.015` relative to the retained scan estimates. This is an
engineering acceptance tolerance, not a confidence interval.

A separate seed `456` check uses 1 million samples per point in the transition
windows `xyz:0.6,0.725,0.025`, `xy:0.525,0.65,0.025`, and
`z:0.06,0.12,0.01`, with the same inputs. It adds 22 million realizations,
including baselines. Compare all 65 bins at these 19 points with exact
conditional two-sample tests and a family-wise error allowance of `0.01`.
This checks probability agreement across seeds; it does not estimate three
new thresholds from the short windows.

Common random numbers couple points within a scan. Do not bootstrap aggregate
point counts as independent observations to obtain a threshold interval.
See [manuscript validation](manuscript-validation.md) for the measured results
and statistical comparison method. Reduced verification exports must state
their sample counts; reproducing the manuscript at its stated sample counts
still uses the 10-million-sample settings above.

## Risk-equivalent propagation probability

After the coupled `xyz` scan has completed, run
`notebooks/verify_risk_equivalence_pe_star.ipynb`.

For the paper configuration, the expected values are approximately:

- numerical `p_e* = 0.093`;
- analytical `p_e* = 0.094` (unrounded: `0.09386758`).

The analytical approximation and the finite-lattice numerical estimate are
close for this example; the approximation has no guaranteed direction of bias
for other topologies. Numerical values also depend on scan resolution and
Monte Carlo sampling.

## Figure 4: notional-damage robustness

Run the relevant sections of `notebooks/damage_scenarios.ipynb` with:

- lattice: `4x4x4`;
- `PF_NODE = 1e-4`;
- `SAMPLES = 1_000_000` per scenario;
- `THRESHOLD = 2`;
- `SEED = 123`;
- `pe = 0.67`, `0.40`, and `0.20`;
- undamaged baseline and corner `(0,0,0)`, perimeter `(1,0,0)`, and internal
  `(1,1,0)` damage;
- adverse states `N_f >= 2` and `N_f >= 16`.

Each adverse state uses its own undamaged baseline. The expected robustness
ordering is corner, perimeter, then internal, from highest to lowest.

## Figure 5: risk-equivalent cases

In `notebooks/damage_scenarios.ipynb`, run the cases:

- analytical risk-equivalent probability `pe = 0.094`;
- numerical risk-equivalent probability `pe = 0.0927`.

Use the same lattice, node probability, sample count, seed, and damage
locations as for Figure 4, with adverse state `N_f >= 2`.

## Before publication runs

1. Use fresh input and output databases or archive the existing files.
2. Run `pytest -q`. Opt-in higher-sample tests do not replace figure generation.
3. Retain the output databases, figures, run metadata, Git commit, and package
   versions.
4. Record the activated Python environment.

Each damage-notebook section stops if its output database already exists.
Archive that database or choose a fresh `OUT_DB` path before rerunning; the
notebook does not delete completed campaigns. Its input lattice uses
`input/input_damage_scenarios.db`.

The publication campaigns are computationally expensive. Reduced tests verify
the model invariants and expected qualitative behaviour but do not replace the
stated publication sample counts.

Retained publication counts and their source revisions are described in
[manuscript validation](manuscript-validation.md). They support analysis
regression checks, while fresh figure campaigns establish reproduction with
the release environment.
