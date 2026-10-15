# Analysis methods

This document summarizes the three post-processing methods used by the
stationary paper. Detailed implementation behaviour is covered by the tests
and public function docstrings.

## 1. Percolation threshold

The percolation response is the complete-collapse probability

$$
p_{\mathrm{full}}(p_e)=P(N_f=|\mathcal N|).
$$

The threshold is an identifiable interior root of

$$
\frac{\mathrm d^2p_{\mathrm{full}}}{\mathrm dp_e^2}=0.
$$

The estimator validates the ordered scan grid, probability range, retained
binomial counts, and monotonicity within Monte Carlo uncertainty. It fits a
weighted smoothing spline and searches for a sign-changing interior root of
the fitted second derivative.

A result is reported only when the convex-to-concave root is interior, has
positive slope, and is stable with respect to smoothing and scan spacing.
If several derivative peaks exist, one must be clearly dominant; comparable
peaks return a multiple-root diagnostic. Linear,
convex, concave, boundary-only, or insufficiently resolved curves return
`threshold=None` with diagnostics. A maximum-slope location is never used as a
substitute.

The result contains the threshold or `None`, root bracket, slope, boundary
status, uncertainty interval, fit diagnostics, and the reason for a
no-threshold result.

## 2. Risk equivalence

For common additive consequences, the numerical risk-equivalence condition is

$$
P(N_f=1)=\sum_{n=2}^{N}(n-1)P(N_f=n).
$$

The numerical method evaluates both sides from the retained PMF at every
fully coupled `xyz` scan point. A solution is returned only for one
sign-changing bracket within the admissible interval
$[0,p_{e,\mathrm{crit}}]$. The root is linearly interpolated within that
bracket. No crossing or multiple crossings produce an explicit diagnostic.

For a regular $n_x\times n_y\times n_z$ lattice, the analytical approximation
uses the degree distribution and mean excess degree

$$
\kappa=\frac{E[\Delta(\Delta-1)]}{E[\Delta]}.
$$

It solves

$$
\frac{E[\Delta]p_e}{1-\kappa p_e}-E[(1-p_e)^\Delta]=0
$$

within

$$
0\le p_e\le\min(p_{e,\mathrm{crit}},\kappa^{-1}).
$$

The approximation assumes small spontaneous node-failure probability,
homogeneous propagation probability, a uniformly selected initial
compartment, and locally tree-like early propagation. The initiating node has
mean degree $E[\Delta]$; subsequent generations use the mean excess degree
$\kappa$. The approximation has no guaranteed direction of bias for arbitrary
finite topologies; the structured result labels its conservatism as
`case-dependent`.

For the paper's `4x4x4` lattice, the reference values are approximately:

- numerical $p_e^*\approx0.093$;
- analytical $p_e^*\approx0.094$ (unrounded: `0.09386758`).

For this lattice, the degree counts are `{3: 8, 4: 24, 5: 24, 6: 8}`,
$E[\Delta]=4.5$, and $\kappa=11/3$. These are the finite-lattice
quantities in Equations 23–27 of the manuscript.

## 3. Entropy-based robustness

For an undamaged baseline probability $p_0$ and the corresponding damaged
probability $p_j$, the robustness index is the Bernoulli specialization of
Cao and Beck (2026),
[*On the quantification of robustness and its thresholds*](https://doi.org/10.1016/j.strusafe.2026.102688).
With the paper's scale $c=1$, it is

$$
I_{\mathrm{rob}}=\exp[-D_{\mathrm{KL}}],
$$

with

$$
D_{\mathrm{KL}}
=(1-p_0)\ln\frac{1-p_0}{1-p_j}
+p_0\ln\frac{p_0}{p_j}.
$$

Both probabilities must refer to the same adverse macro-state
$N_f\ge S_{\mathrm{lim}}$ and otherwise identical stationary models. Each
notional-damage scenario is paired with one undamaged run, and each
$S_{\mathrm{lim}}$ has its own baseline probability.

The formulation assumes a coherent perturbation, $p_j\ge p_0$. A violation
beyond the stated Monte Carlo tolerance raises an error in strict mode or
returns an explicit diagnostic in diagnostic mode. Probabilities are not
silently clipped, and incoherent cases are not replaced by a robustness value
of one.

Derivative-based robustness thresholds are found in the
$\pi=-\Phi^{-1}(p_j)$ space. Missing roots return diagnostics rather than
fallback positions. For the published reference reliability
$\beta=3.7$, the corresponding index thresholds are approximately
$I^-_{\mathrm{rob}}=0.50$ and $I^+_{\mathrm{rob}}=0.84$.

## Tests

Run the complete unit, integration, CLI, and reduced scientific regression
suite with:

```bash
pytest -q
```

Higher-sample tests are optional because of their computational cost:

```bash
pytest -q -m publication --run-publication
```

These tests include synthetic estimator shapes and conditional simulations;
they do not regenerate the full publication figures. The scientific suite
also checks retained publication counts in
`tests/scientific/data/manuscript_reference.json`. See
[manuscript validation](manuscript-validation.md) for provenance and limits.
