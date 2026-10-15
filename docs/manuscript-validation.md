# Manuscript validation

The scientific reference is the supplied R1 manuscript, *Probabilistic modelling
of buildings as networks*. Its publication examples use a `4x4x4` lattice and
spontaneous node-failure probability `1e-4`.

Portable fresh counts, reviewed figures, environment/provenance records and
the executable saved-count check are available in the
[curated reproduction package](../reproduction/README.md).

## Paper-to-code mapping

| Manuscript quantity | Implementation | Verification |
| --- | --- | --- |
| Independent initial failures and irreversible propagation | `pbm/sim/engine.py`, `pbm/network/propagation.py` | Initial-failure, propagation, and scientific limit tests |
| Figure 3 inflection threshold, Equation 15 | `pbm/post/percolation_threshold.py` | Generic shape tests and retained publication scan counts |
| Numerical risk equivalence, Equations 16–19 | `pbm/post/risk_equivalence.py` | PMF residual and retained fully coupled scan |
| Finite-lattice approximation, Equations 23–27 | `pbm/post/risk_equivalence.py` | Degree-count tests and direct evaluation of the manuscript equation |
| Notional-damage robustness, Equation 28 | `pbm/post/robustness.py` | KL boundary/coherence tests and retained scenario PMFs |
| Figures 3–5 inputs and workflow | `notebooks/`, `docs/figures-3-5-reproduction.md` | Explicit settings, retained counts, and fresh campaigns before final release |

## Retained publication results

`tests/scientific/data/manuscript_reference.json` contains scientific counts
exported from the author's local result databases. It omits machine names,
local paths, and submission correspondence. Each run retains its source Git
revision, completion timestamp, sample count, and scan/damage configuration.
These are historical simulator results rather than synthetic response curves.

The retained scans have 10,000,000 realizations per point, seed `123`, and
the manuscript's axis configurations. Reanalysis with the inspected code gives:

| Case | Axis configuration | Spacing | Threshold | Manuscript rounded value |
| --- | --- | --- | --- | --- |
| 1 | `pe_x = pe_y = pe_z` varied | `0.025` | `0.671507` | `0.67` |
| 2 | `pe_x = pe_y` varied; `pe_z = 0.8` | `0.025` | `0.586145` | `0.59` |
| 3 | `pe_z` varied; `pe_x = pe_y = 0.8` | `0.010` | `0.091729` | `0.09` |

The fully coupled PMF gives numerical risk equivalence `0.0931840` on its
`0.025` grid. The analytical result is `0.09386758`; directly evaluating
Equations 23–27 gives the same result to approximately `2e-13`.
The numerical result is an interpolated Monte Carlo estimate, not an exact
root; its interval and grid bracket are retained in analysis diagnostics.
The Figure 5 notebook uses the publication settings `0.0927` and `0.094`.

Retained damage PMFs use 1,000,000 realizations per scenario and seed `123`.
They give the manuscript's robustness ordering: corner, perimeter, internal,
from highest to lowest, for the Figure 4 and Figure 5 configurations.

## Fresh Figure 4–5 reproduction

On 2026-10-09, the complete damage notebook ran from source commit
`71c774ec89f68ca732626e262a782560bf561952`, using Python
3.12.14 on Windows and the pinned `requirements-paper.txt` environment.
The executing kernel's package versions were verified against the applicable
pins. Eight workers completed the campaign in approximately 24.5 minutes.
Only output/input paths and worker count were changed for this execution.

All twenty scenarios used the manuscript's 1,000,000 realizations each,
seed `123`, `4x4x4` lattice, and node-failure probability `1e-4`.
All twenty 65-bin probability distributions matched the retained counts
exactly, and model fingerprints matched their historical counterparts.
The robustness values also agreed with an independent evaluation of
Equation 28 to within `1e-12`.

| Figure | `pe` | `S_lim` | Corner robustness | Perimeter robustness | Internal robustness |
| --- | --- | --- | --- | --- | --- |
| 4 | `0.67` | `2` | `0.037682151` | `0.012509914` | `0.004239903` |
| 4 | `0.67` | `16` | `0.048598939` | `0.018671404` | `0.005723758` |
| 4 | `0.40` | `2` | `0.223496547` | `0.134811673` | `0.081104260` |
| 4 | `0.40` | `16` | `0.476079017` | `0.369434920` | `0.260219208` |
| 4 | `0.20` | `2` | `0.521616914` | `0.418141076` | `0.334367787` |
| 4 | `0.20` | `16` | `0.985647974` | `0.977756936` | `0.964861117` |
| 5 | `0.0927` | `2` | `0.752821486` | `0.683855973` | `0.620787299` |
| 5 | `0.094` | `2` | `0.749703771` | `0.680083279` | `0.616486264` |

Every adverse state reproduced corner > perimeter > internal robustness.
Both Figure 5 cases reproduced the medium robustness category. The eight
robustness panels were rendered and inspected against the manuscript;
portable Matplotlib styling was used for the initial exports. Execution notebooks, result databases,
environment metadata, validation diagnostics, and artifact hashes were retained
locally. Exact count agreement is specific to the recorded seed and
simulator/random-stream contract; it is not a cross-environment guarantee.

The saved results were subsequently exported with the repository's publication
mode and a working LaTeX installation, without rerunning simulations. Figure 4
was assembled in the original six-panel order and Figure 5 in its original
two-panel order. All eight panels match the manuscript's font-family sets,
filled-colour sets, and scaled maximum font size. Independent value checks and
before/after database hashes confirm that the results are unchanged. The final
two figure pages and three accompanying diagnostic pages were visually
inspected. Captions were re-typeset, so bounding boxes and spacing can differ
slightly; byte-for-byte PDF identity is not claimed. Figure 5 retains the
manuscript inputs `0.094` and `0.0927`; the fresh numerical estimate below is
shown separately in the diagnostics.

## Fresh plausibility and numerical risk-equivalence checks

A separate campaign at the same source revision and pinned environment added
52,000,000 realizations on 2026-10-09, taking 10.3 minutes
with eight workers. It reused three completed undamaged Figure 4 baselines.

At `pf_node=1e-4`, the propagation settings `[0, 0.20, 0.40, 0.67, 1]` each
have 1,000,000 realizations and satisfy all four Section 2.8 plausibility checks:

- At zero propagation, all 65 bins agree with the exact binomial law using
  simultaneous 99% marginal acceptance regions. The single-failure probability
  is `0.006320` versus exact `0.006359805`.
- Any-failure probability is `0.006343` at every setting,
  versus exact `0.006379882`. Common random numbers give
  exactly the same `6343` any-failure realizations.
- All observed tail probabilities are nondecreasing and single-failure
  probabilities are nonincreasing; simultaneous exact interval checks pass.
- At full propagation, every nonempty initial state reaches all 64 nodes,
  with no intermediate final collapse sizes.

The fresh numerical risk-equivalence estimate is **`0.093521677`**.
Five points `[0.08, 0.085, 0.09, 0.095, 0.10]` use 10,000,000 realizations each,
seed `123`; fresh anchors at `0`, `0.20`, `0.40`, and `0.67` use 1,000,000 each.
No historical counts enter the estimate. The manuscript setting `0.67` supplies
the upper bound; its percolation threshold was not re-estimated.

The crossing bracket is `[0.09, 0.095]`. The nominal 95% Monte Carlo
interval is `[0.092252993, 0.094820405]`, conditional on local linear interpolation.
The interval derives from simultaneous normal residual bands and handles
common-random-number covariance conservatively; it excludes interpolation bias.
Endpoint signs resolved by these bands: **True**. Direct evaluation of
Equation 19, interpolation, and residual standard errors agrees with the implementation.
The analytical approximation remains `0.093867580`; the historical coarse-grid
estimate `0.093184008` above is retained as a separate comparison.

The fresh `pe=0.10` result matches all 65 retained fully coupled Figure 3
counts exactly at the same 10,000,000-realization sample count. This verifies
one Figure 3 scan point at the full publication sample count.

## Fresh reduced Figure 3 verification

On 2026-10-09, all three original Figure 3 grids were regenerated at the same
source revision and pinned environment described above, with eight workers.
The inputs, threshold `2`, seed `123`, and grid spacing were unchanged.
Sample counts were selected before the corresponding fresh simulations using
stress screening of retained counts. The screening surrogate preserves the
shared initial-failure count but does not recover propagation dependence
between points; its empirical acceptance fractions are not confidence levels.

| Case | Samples per point | Points | Fresh threshold | Retained threshold | Absolute difference |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,000,000 | 41 | `0.669039` | `0.671507` | `0.002468` |
| 2 | 2,000,000 | 41 | `0.579861` | `0.586145` | `0.006284` |
| 3 | 2,000,000 | 101 | `0.088178` | `0.091729` | `0.003551` |

All three thresholds are identified and pass the predeclared absolute
agreement tolerance of `0.015`. Smoothing factors `0.25x`, `1x`, and `4x`
and alternating-grid fits also pass this stability tolerance. The Case 2
fresh estimate rounds to `0.58`, whereas the manuscript reports `0.59`;
the reduced run does not reproduce every rounded threshold exactly.

All 11,895 PMF bins pass simultaneous exact marginal finite-population
acceptance regions with a family error allowance of `0.01`. These comparisons
condition on the retained 10-million counts and the deterministic random-stream
prefix contract; the lower-sample and historical runs are not independent.
The audit verified unchanged simulator/random-stream source at the historical
revisions, fresh input fingerprints, all integer counts and summaries, identical
same-input baselines, and shared any-failure counts. Exact binomial
any-failure checks pass for all cases; zero- and full-propagation limits pass
for Case 1. Historical dependency environments remain unknown, so this
contract is evidence for the recorded implementation rather than a portable
bitwise guarantee.

The primary campaign used 330 million realizations including baselines and
took approximately 78.5 minutes on the recorded machine. A corresponding
full-sample campaign would use 1.86 billion realizations. This establishes
verification at reduced cost; full 10-million-per-point reproduction remains
a separate artifact-generation task. Aggregate scan points are coupled by
common random numbers, so no independent-point bootstrap threshold interval
is reported. The `0.015` acceptance tolerance is not a confidence interval.

A further independent-seed check used seed `456` and 1 million samples per
point at 19 transition-window points (six for Case 1, six for Case 2, seven
for Case 3). All 1,235 PMF-bin comparisons pass exact conditional two-sided
equal-tail tests against the fresh primary runs, with Bonferroni correction
and a family error allowance of `0.01`. No independence between scan points
is assumed. Input fingerprints, shared initial-failure counts and same-input
baselines also pass. These windows check probability agreement across seeds;
they do not provide new full-grid threshold estimates.

The independent campaign added 22 million realizations including baselines
and took 10.1 minutes. Together the two campaigns used 352 million
realizations, **81.1% fewer** than the corresponding full-sample campaign.

The three reduced-result pages were exported from saved counts with the
repository's publication mode and the PGF/LaTeX backend. They retain the
publication plotting fonts and colours, carry explicit sample-count titles,
and place legends outside the data axes. All three rendered pages and the
retained-count comparison plot were visually inspected. Their standalone
page layout differs from the manuscript's combined Figure 3 assembly.
Final hashes confirm that plotting left the simulation databases unchanged.
Commands, environment records, full counts, statistical diagnostics and
artifact hashes are retained locally alongside the figures.

## Limits of this evidence

The reference data verify analysis of historical runs. Fresh execution now
establishes Figure 4–5 reproduction and reduced-sample Figure 3 verification
for the environment and commit above. Figure 3 has not been regenerated
at 10 million samples per point. Source revisions differ
between runs, although the propagation, simulation, random-stream, notional-damage,
and simulation CLI source files match the inspected code at all three recorded
simulator revisions. Historical dependency environments have not been recovered.
A final release must retain the figure-generation environment,
source commit, and run evidence. Monte Carlo comparisons across different
random streams require uncertainty and interpolation tolerances rather than
exact count agreement. The local validation includes pointwise exact 95%
binomial intervals; robustness uncertainty was not separately propagated.

Synthetic response tests are explicitly labelled as estimator checks. Opt-in
tests marked `publication` use higher sample counts but include synthetic
curves and conditional simulations, so they do not replace the notebook
campaigns described in the reproduction guide.
