# Stationary model contract

This document defines the scientific scope and executable meaning of version
1.0 of the probabilistic building model. It follows the manuscript
*Probabilistic modelling of buildings as networks*.

## Scope

The building is represented by a stationary graph
$G=(\mathcal N,\mathcal E)$. Nodes are redundant compartments and edges are
possible paths for failure propagation. All model parameters remain fixed
within one Monte Carlo realization.

Version 1.0 supports stationary simulations, percolation scans, risk
equivalence, simultaneous notional-damage scenarios, and entropy-based
robustness.

## Nodes

For every node $i$ not included in the notional-damage set,

$$
F_i(0)\sim\mathrm{Bernoulli}(p_i).
$$

The input field `pf_node` stores $p_i$ for node $i$.

Initial node failures are mutually independent and independent of edge
transmission. `pf_node` is the executable node-failure probability and must be
finite and lie in `[0,1]`.

A reliability index may be used during model preparation, but it must be
converted before simulation through

$$
p_i=\Phi(-\beta_i).
$$

## Edges

Each physical adjacency is stored once with canonical endpoints `org < dst`.
The `directed` field is authoritative.

- `directed=0`: `pe` is required and applies in both directions;
  `pe_fwd` and `pe_rev` must be null.
- `directed=1`: `pe` must be null and both directional probabilities are
  required. Use an explicit zero for a non-transmitting direction.
- `blocked=1`: transmission is suppressed in both directions.

All probabilities must be finite and lie in `[0,1]`. Self-loops, duplicate
adjacencies, duplicate directed arcs, contradictory rows, and missing endpoint
nodes are invalid.

## Propagation

Failure propagation is coherent and irreversible. Newly failed nodes form a
breadth-first frontier. Each newly failed node attempts transmission once
along each outgoing arc to a surviving neighbour. Newly triggered nodes are
committed synchronously, and failed nodes do not recover.

Propagation pseudo-time orders the failure waves within a cascade. It is not
physical time or service life.

The final collapse extent is

$$
N_f=\sum_i F_i,
$$

from which the model estimates $P(N_f=n)$, $P(N_f\ge 1)$,
$P(N_f=|\mathcal N|)$, and $P(N_f\ge S_{\mathrm{lim}})$.

## Notional damage

Notional damage is the simultaneous deterministic loss of a set of nodes
$D\subseteq\mathcal N$. It is applied once at propagation pseudo-time
$t=0$:

$$
F_i(0)=
\begin{cases}
1, & i\in D,\\
\mathrm{Bernoulli}(p_i), & i\notin D.
\end{cases}
$$

An empty `initial_damage` table denotes the undamaged case. Multiple rows form
one simultaneous damage scenario. Unknown node IDs or coordinates are errors.
Runtime damage and database damage cannot be combined silently.

## Random-number generation

Realization `i` uses a NumPy `PCG64` stream derived only from
`(base_seed, realization_index)`. Results are therefore independent of worker
count, chunk size, task scheduling, output-database history, and `run_id`.
Comparable scan points and damage scenarios use common random numbers when the
same base seed is supplied.

## Analysis contract

- The percolation response is $P(N_f=|\mathcal N|)$. A threshold is an
  identifiable interior inflection point; no identifiable threshold is a
  valid result.
- Numerical risk equivalence uses

  $$
  P(N_f=1)=\sum_{n=2}^{|\mathcal N|}(n-1)P(N_f=n).
  $$

- Entropy-based robustness compares otherwise identical undamaged and damaged
  runs for one explicitly specified adverse state $N_f\ge S_{\mathrm{lim}}$.
  Each value of $S_{\mathrm{lim}}$ requires its own undamaged baseline.
- `p_any` means only $P(N_f\ge1)$; it is not a progressive-collapse
  probability and is not used as the perturbed probability in robustness
  calculations.

## Excluded from version 1.0

Version 1.0 does not support time-varying node or edge probabilities,
degradation, shocks, spatial copulas, recovery, repair, sequential damage,
physical-time propagation, partial compartment states, or within-run topology
changes.
