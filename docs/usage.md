# CLI, Python API, and database reference

Start with the [main README](../README.md) or the
[introductory notebook](../notebooks/getting_started.ipynb). This page contains
additional command examples and database details. Shell examples below use
POSIX line continuation; on PowerShell enter each command on one line.

## Database model

The authoritative SQL files are packaged under `pbm/resources/sql/` and remain
available after wheel installation.

### Input database

- `nodes(id, x, y, z, pf_node)`
- `edges(id, org, dst, dir, pe, pe_fwd, pe_rev, directed, blocked)`
- `initial_damage(node_id)`

An empty `initial_damage` table denotes the undamaged case. Multiple rows form
one simultaneous notional-damage scenario.

### Output database

The principal tables are:

- `runs`: sample count, threshold, seed, timestamps, package version, RNG
  contract, model fingerprint, notional-damage set, and provenance;
- `pmf`: integer counts and probabilities for `N_f=0,...,|N|`;
- `collapse_probs`: `P(N_f>=1)`, `P(N_f=|N|)`, and
  `P(N_f>=threshold)`;
- `percolation_multi`: scan-point summaries;
- `percolation_pmf`: authoritative integer PMF at each scan point;
- `node_influence`, `edge_usage`, and `edge_attempts` when requested.

## Python API

Build a lattice and run one realization:

```python
from pathlib import Path
import numpy as np

from pbm.io.loader import load_all
from pbm.network.builder import build_and_insert
from pbm.sim import simulate_once

input_db = Path("example/input.db")
input_db.parent.mkdir(parents=True, exist_ok=True)
build_and_insert(4, 4, 4, input_db, truncate=True)
model = load_all(input_db)

rng = np.random.default_rng(123)
outcome = simulate_once(
    model,
    rng,
    damage_coordinates=[(0, 0, 0)],
)
print(int(outcome.failed_mask.sum()))
```

Public analysis APIs are exported from `pbm.post`, including percolation
threshold estimation, numerical and analytical risk equivalence, and
Bernoulli Kullback–Leibler robustness. The detailed contracts are documented
in `docs/`.

## Command-line interface

### Build a regular lattice

Create the parent directories for these command examples first:

```bash
python -c "from pathlib import Path; Path('input').mkdir(exist_ok=True); Path('output').mkdir(exist_ok=True)"
```

```bash
pbm-build --nx 4 --ny 4 --nz 4 --db input/input.db --truncate yes
```

### Run a stationary simulation

```bash
pbm-simulate \
  --db input/input.db \
  --outdb output/results.db \
  --samples 100000 \
  --threshold 2 \
  --seed 123 \
  --mc-workers 4 \
  --mc-chunk-size 25000
```

Use `--damage-node-ids 1,2` or `--damage-coords "0,0,0;1,0,0"` for one
simultaneous runtime-authored damage set. Runtime damage cannot be combined
with a non-empty database `initial_damage` table.

A coupled percolation scan is specified as:

```bash
pbm-simulate \
  --db input/input.db \
  --outdb output/results_percolation.db \
  --samples 100000 \
  --threshold 2 \
  --seed 123 \
  --percolation-scan "xyz:0,1,0.05"
```

### Post-process

```bash
pbm-postprocess \
  --outdb output/results_percolation.db \
  --figs-root output/figs \
  --run-id 1 \
  --make-plots yes \
  --plot-mode standard
```

`standard` uses portable Matplotlib mathtext. `publication` uses external
LaTeX through Matplotlib's PGF backend and requires a working `pdflatex`
installation.

Numerical and analytical risk equivalence:

```bash
pbm-postprocess \
  --outdb output/results_percolation.db \
  --run-id 1 \
  --make-plots no \
  --risk-equivalence \
  --pe-crit 0.67
```

Entropy robustness requires explicitly paired undamaged and damaged runs and
an adverse macro-state:

```bash
pbm-postprocess \
  --outdb output/results_damage.db \
  --entropy-robustness \
  --baseline-run-id 1 \
  --damaged-run-ids 2,3,4 \
  --s-lim 2,16 \
  --coherence-mode strict
```
