# Probabilistic Building Model -- PBMpy

Reference implementation accompanying the *Structural Safety* paper
**[Probabilistic modelling of buildings as networks](https://doi.org/10.1016/j.strusafe.2026.102783)**.
Version **1.0.0** implements the following: a building is represented as
a network of compartments, and simulations estimate how an initial failure can
spread through that network.

Start with a small example below or open the
[introductory notebook](notebooks/getting_started.ipynb). The notebook explains
each step, plots collapse-size distributions, and compares an undamaged baseline
with a specified damage scenario.

## Getting started

### Install from this repository

Use Python **3.11 or later** for the package. Use **Python 3.12** for the pinned
notebook and paper environment. Clone the repository and enter its directory:

```bash
git clone --branch main --single-branch https://github.com/caoalex-s/probabilistic-building-model.git
cd probabilistic-building-model
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

### Run a small simulation

This example builds a $2\times2\times2$ lattice of eight compartments and runs
2,000 realizations. The schema defaults are spontaneous failure probability
$p_f=0.01$ per compartment and propagation probability $p_e=0.1$ per allowed
transmission attempt. These are illustrative inputs, not the paper's inputs.

```bash
python -c "from pathlib import Path; Path('output/quickstart').mkdir(parents=True, exist_ok=True)"
pbm-build --nx 2 --ny 2 --nz 2 --db output/quickstart/input.db --truncate yes
pbm-simulate --db output/quickstart/input.db --outdb output/quickstart/results.db --samples 2000 --threshold 2 --seed 123
pbm-postprocess --outdb output/quickstart/results.db --figs-root output/quickstart/figs --make-plots yes --plot-mode standard
```

The simulator reports probabilities for any failure, complete collapse, and at
least two failed compartments. Standard plots are written to
`output/quickstart/figs/`. A collapse-size distribution includes zero:
many realizations can have no initial failure and therefore no collapse.
The small sample count is suitable for learning; rare-event estimates need
larger samples.

The commands write an input database and a results database under
`output/quickstart/`. Repeating them rebuilds this example's input
and appends a new simulation run to its results database. Post-processing selects
the latest run unless a run ID is supplied.

### Learn interactively

For notebooks, install the pinned requirements in a Python 3.12 environment:

```bash
python -m pip install -r requirements-paper.txt
jupyter lab
```

Launch Jupyter from the repository root and open
[`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb).
Select the same Python environment as the notebook kernel. Run all cells to:

1. Build an eight-compartment model and inspect its nodes and edges.
2. Simulate an undamaged baseline and the forced loss of one corner compartment.
3. Plot their collapse-size distributions.
4. Define an adverse event and calculate its entropy-based robustness index.

The notebook runs **20,000 realizations per scenario**, uses one worker, and
creates a fresh directory under `output/getting-started/` on each
execution. It uses standard Matplotlib plots and requires no external LaTeX.

## How the model works

### Compartments, initial failures, and propagation

The building is a graph $G=(\mathcal N,\mathcal E)$ with
$N=|\mathcal N|$ compartments. A node represents a redundant compartment; an
edge represents a possible route for failure propagation.

For each compartment outside a prescribed damage set, its initial failure
indicator is sampled independently:

$$
F_i(0)\sim\mathrm{Bernoulli}(p_i).
$$

Here $p_i$ is the node's input probability, stored as `pf_node`.

If a reliability index $\beta_i$ is used to prepare the model, convert it to
$p_i=\Phi(-\beta_i)$, where $\Phi$ is the standard normal cumulative distribution
function. The simulator takes probabilities as inputs.

A newly failed compartment attempts transmission along each allowed outgoing
arc once, with the assigned propagation probability. Failure is irreversible.
A notional-damage scenario forces its specified compartments to fail at the
start of propagation. Propagation steps order the cascade; they are not
physical time. Model probabilities stay fixed throughout a realization.

### Collapse-size distribution

Let $N_f$ be the number of failed compartments at the end of a realization.
With $M$ Monte Carlo realizations, the estimated probability mass function is

$$
\widehat P(N_f=n)
=\frac{1}{M}\sum_{r=1}^{M}\mathbf 1(N_f^{(r)}=n),
\qquad n=0,\ldots,N.
$$

Here $\mathbf 1$ is one when its condition holds and zero otherwise.
Complete collapse is the event $N_f=N$. An adverse event can instead be defined
by a chosen severity limit $S_{\mathrm{lim}}$:

$$
P(N_f\ge S_{\mathrm{lim}})
=\sum_{n=S_{\mathrm{lim}}}^{N}P(N_f=n).
$$

The CLI's `--threshold` selects this severity limit. The meaning of
a result therefore depends on both the model and the chosen adverse event.

### Percolation and risk equivalence

A percolation scan varies propagation probability and evaluates
$p_{\mathrm{full}}(p_e)=P(N_f=N)$. This implementation estimates a threshold at
an identifiable interior convex-to-concave inflection:

$$
\frac{\mathrm d^2p_{\mathrm{full}}}{\mathrm dp_e^2}=0.
$$

Insufficient resolution or ambiguous roots return diagnostics. A tiny
introductory simulation is not sufficient to establish a percolation threshold.

Under the paper's additive-consequence assumption, the numerical
risk-equivalent propagation probability $p_e^*$ balances isolated failure
against the additional consequences of larger collapses:

$$
P(N_f=1)=\sum_{n=2}^{N}(n-1)P(N_f=n).
$$

See [analysis methods](docs/analysis-methods.md) for the analytical approximation,
root selection, assumptions, and uncertainty diagnostics.

### Entropy-based robustness

Choose one adverse event, such as $N_f\ge S_{\mathrm{lim}}$. Let $p_0$ be its
probability in the undamaged baseline and $p_j$ its probability after one
specified notional-damage scenario. Compare otherwise identical models.

For the scale used in the paper, $c=1$:

$$
I_{\mathrm{rob}}
=\exp\left[-D_{\mathrm{KL}}\left(
\mathrm{Bern}(p_0)\Vert\mathrm{Bern}(p_j)
\right)\right],
$$

$$
D_{\mathrm{KL}}
=p_0\ln\frac{p_0}{p_j}
+(1-p_0)\ln\frac{1-p_0}{1-p_j}.
$$

The logarithms are natural, and the divergence direction is baseline to
damaged. Equal probabilities give $I_{\mathrm{rob}}=1$; greater sensitivity to
damage gives a lower index. Each damage scenario and severity limit has its own
comparison. The formulation assumes a coherent perturbation, $p_j\ge p_0$;
the package checks coherence and handles probability boundaries explicitly.

The index and its thresholds are developed in Cao and Beck (2026),
[*On the quantification of robustness and its thresholds*](https://doi.org/10.1016/j.strusafe.2026.102688).
Read the [analysis guide](docs/analysis-methods.md) for the assumptions and
threshold interpretation. The introductory notebook demonstrates the index;
it does not classify the example using the paper's reference thresholds.

## Repository and notebook guide

| Location | What it contains |
| --- | --- |
| [`notebooks/`](notebooks/) | Interactive introduction and paper-analysis workflows |
| [`pbm/`](pbm/) | Model construction, loading, simulation, and analysis APIs |
| [`pbm/resources/sql/`](pbm/resources/sql/) | Authoritative database schemas, also included in the installed package |
| [`cli/`](cli/) | Simulation and post-processing command-line interfaces |
| [`scripts/`](scripts/) | Fresh reproduction campaigns and saved-count validation |
| [`reproduction/`](reproduction/README.md) | Reviewed figures, scientific counts, provenance, and hashes |
| [`docs/`](docs/) | Detailed methods, usage, validation, and release instructions |
| [`tests/`](tests/) | Unit, integration, and scientific regression checks |
| `output/` | Generated local databases, figures, and reports; ignored by Git |

| Notebook | Purpose and prerequisites |
| --- | --- |
| [Getting started](notebooks/getting_started.ipynb) | Start here: small fresh simulations, PMF plot, and baseline/damage robustness |
| [Percolation](notebooks/percolation.ipynb) | Figure 3 scans; manuscript settings use ten million realizations per point |
| [Risk equivalence](notebooks/verify_risk_equivalence_pe_star.ipynb) | Numerical and analytical comparison; requires a compatible fully coupled scan from the percolation workflow |
| [Damage scenarios](notebooks/damage_scenarios.ipynb) | Figures 4–5; compares baseline, corner, perimeter, and internal damage at manuscript-scale sample counts |

The last three notebooks reproduce the manuscript workflow and are
computationally intensive. Their execution order, inputs, and reduced-cost
alternatives are in [Figures 3–5 reproduction](docs/figures-3-5-reproduction.md).

## Reproducing and checking the paper

After installation, validate the curated counts without running simulations:

```bash
python scripts/validate_reproduction.py --out output/saved-count-validation.json
```

For a small fresh code check:

```bash
python scripts/reproduce.py smoke --out output/smoke-example --workers 2
```

The runner requires a new output directory. Inspect larger campaign plans with
`--dry-run` before executing them. The
[reproduction package](reproduction/README.md) lists profiles and costs.

Fresh Figures 4–5 damage distributions were reproduced at the manuscript sample
count. Figure 3 was verified with reduced samples and an independent seed;
the full ten-million-per-point campaign has not been rerun. The reduced Case 2
threshold rounds to 0.58, versus the manuscript's 0.59. See
[manuscript validation](docs/manuscript-validation.md) for evidence and limits.

## Further documentation and development

- [CLI examples, Python API, and database tables](docs/usage.md)
- [Stationary model contract](docs/model-contract.md)
- [Analysis methods](docs/analysis-methods.md)
- [Version 1.0.0 release notes](docs/release-notes-v1.0.0.md)
- [Release and archive guide](docs/release-guide.md)

Install the test dependencies and run the standard suite:

```bash
python -m pip install ".[test]"
python -m pytest -q
```

The [validation workflow](.github/workflows/validation.yml) checks Linux on
Python 3.11–3.13 and Windows/macOS on Python 3.12. Higher-sample tests are opt-in;
their scope is explained in the validation guide.

## Acknowledgments

We acknowledge **ETH Zurich**, **Empa**, and **Nanyang Technological University
(NTU), Singapore**, for their roles in the research and software development
behind this project. The development history and individual contributions are
recorded in [AUTHORS.md](AUTHORS.md).

## Citation

Please cite both the associated paper and the software version used when
publishing work that uses this software. For commercial products without an
accompanying publication, please include the references in the product
documentation or credits. This is a scholarly attribution request, not an
additional condition of the MIT license.

**Associated paper**

> Cao, A. S., Kumalic, A., & Thöns, S. (2026). Probabilistic modelling of
> buildings as networks. *Structural Safety*, Article 102783.
> [doi:10.1016/j.strusafe.2026.102783](https://doi.org/10.1016/j.strusafe.2026.102783)

**Software**

> Cao, A. S., & Kumalic, A. (2026). *Probabilistic Building Model*
> (version 1.0.0) [Computer software].
> [GitHub repository](https://github.com/caoalex-s/probabilistic-building-model/tree/v1.0.0).

Until a release DOI is available, include the version and full commit identifier
used. Obtain it from `git rev-parse HEAD` in your checkout.

Machine-readable references are in [`CITATION.cff`](CITATION.cff).
Contributor roles are in [`AUTHORS.md`](AUTHORS.md), and changes
are in [`CHANGELOG.md`](CHANGELOG.md).

## License

Copyright (c) 2025-2026 ETH Zurich. Software creators: Alex Sixie Cao and
Asim Kumalic; see [AUTHORS.md](AUTHORS.md) for individual contributions.

The software is licensed under the [MIT license](LICENSE). Commercial and
noncommercial use, modification and redistribution are permitted subject to
its terms, including retention of the copyright and license notices in copies
or substantial portions of the software. The citation request above does not
amend the license.
