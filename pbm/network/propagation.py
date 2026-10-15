# model/propagation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

try:
    from scipy import sparse
except Exception:  # SciPy is in your requirements, but keep a soft fallback
    sparse = None


@dataclass
class PropagationMatrices:
    """
    Sparse propagation kernel for the network contagion / percolation model.

    Attributes
    ----------
    P : scipy.sparse.csr_matrix
        CSR matrix of conditional propagation probabilities with semantics:
        P[i, j] = Pr(j fails at next step due to i failing now).
    n : int
        Number of nodes.
    edge_ids : np.ndarray
        Mapping from each CSR nonzero (arc) to its row index in `edges_tbl`.
    rowptr, colind, data : np.ndarray
        CSR arrays mirroring P.indptr, P.indices, P.data. Kept for fast MC loops.
    """
    P: "sparse.csr_matrix"  # CSR, shape (n_nodes, n_nodes)
    n: int
    edge_ids: np.ndarray        # len(nnz) → edges_tbl row index per arc
    rowptr: np.ndarray          # CSR row pointer (len n+1)
    colind: np.ndarray          # CSR column indices (len nnz)
    data: np.ndarray            # CSR data (len nnz)


def build_transition_matrix(
    n_nodes: int,
    edges_tbl: np.ndarray,
) -> PropagationMatrices:
    """
    Build sparse propagation matrix P with conditional transmission probabilities.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edges_tbl : np.ndarray
        Structured array with fields:
        org, dst : int
            Node indices (0-based) in the simulation ordering.
        directed : int
            1 if edge uses directional probabilities; 0 for undirected.
        pe : float
            Undirected propagation probability (used when edge is undirected).
        pe_fwd, pe_rev : float
            Directional propagation probabilities for org->dst and dst->org.
        blocked : int
            If 1, both effective propagation probabilities are zero.

    Returns
    -------
    PropagationMatrices
        CSR propagation matrix plus CSR arrays used by Monte Carlo simulation.

    Notes
    -----
    `directed` is authoritative. Undirected rows require only `pe`; directed
    rows require both `pe_fwd` and `pe_rev`. NaN is the structured-array
    representation of SQL NULL. Contradictory, incomplete, non-canonical,
    out-of-range, or duplicate input is rejected before CSR construction.
    """
    if sparse is None:
        raise RuntimeError("scipy.sparse required")

    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 0:
        raise ValueError(f"n_nodes must be a non-negative integer; got {n_nodes!r}.")
    n_nodes = int(n_nodes)

    edges_tbl = np.asarray(edges_tbl)
    names = set(edges_tbl.dtype.names or ())
    required = {"org", "dst", "directed", "pe", "pe_fwd", "pe_rev", "blocked"}
    missing = required - names
    if missing:
        raise ValueError(
            f"edges_tbl is missing required edge-contract fields: {sorted(missing)}."
        )

    def row_label(index: int) -> str:
        if "id" in names:
            return f"edge id={edges_tbl['id'][index]!r}"
        return f"edge row {index}"

    def integer_value(value, *, field: str, label: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label}: {field} must be an integer, not boolean.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: {field} must be an integer; got {value!r}.") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label}: {field} must be an integer; got {value!r}.")
        return int(numeric)

    def binary_flag(value, *, field: str, label: str) -> int:
        flag = integer_value(value, field=field, label=label)
        if flag not in (0, 1):
            raise ValueError(f"{label}: {field} must be 0 or 1; got {value!r}.")
        return flag

    def nullable_probability(value, *, field: str, label: str) -> Optional[float]:
        if value is None:
            return None
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label}: {field} must be NULL or a finite probability in [0,1]; "
                f"got {value!r}."
            ) from exc
        if math.isnan(probability):
            return None
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{label}: {field} must be a finite probability in [0,1]; "
                f"got {value!r}."
            )
        return probability

    arc_rows: list[int] = []
    arc_cols: list[int] = []
    arc_data: list[float] = []
    arc_edge_indices: list[int] = []
    seen_adjacencies: dict[tuple[int, int], int] = {}
    seen_arcs: dict[tuple[int, int], int] = {}

    for index in range(len(edges_tbl)):
        label = row_label(index)
        org = integer_value(edges_tbl["org"][index], field="org", label=label)
        dst = integer_value(edges_tbl["dst"][index], field="dst", label=label)
        directed = binary_flag(
            edges_tbl["directed"][index], field="directed", label=label
        )
        blocked = binary_flag(
            edges_tbl["blocked"][index], field="blocked", label=label
        )

        if not 0 <= org < n_nodes or not 0 <= dst < n_nodes:
            raise ValueError(
                f"{label}: endpoints ({org}, {dst}) must both be in "
                f"[0,{n_nodes - 1}]."
            )
        if org == dst:
            raise ValueError(f"{label}: self-loop ({org}, {dst}) is not allowed.")
        if org > dst:
            raise ValueError(
                f"{label}: endpoints must use canonical ordering org < dst; "
                f"got ({org}, {dst})."
            )

        adjacency = (org, dst)
        if adjacency in seen_adjacencies:
            first = seen_adjacencies[adjacency]
            raise ValueError(
                f"{label}: duplicate adjacency ({org}, {dst}); it already appears "
                f"in edge row {first}."
            )
        seen_adjacencies[adjacency] = index

        pe = nullable_probability(edges_tbl["pe"][index], field="pe", label=label)
        pe_fwd = nullable_probability(
            edges_tbl["pe_fwd"][index], field="pe_fwd", label=label
        )
        pe_rev = nullable_probability(
            edges_tbl["pe_rev"][index], field="pe_rev", label=label
        )

        if directed == 0:
            if pe is None:
                raise ValueError(f"{label}: directed=0 requires non-NULL pe.")
            if pe_fwd is not None or pe_rev is not None:
                raise ValueError(
                    f"{label}: directed=0 requires pe_fwd and pe_rev to be NULL."
                )
            p_fwd = pe
            p_rev = pe
        else:
            if pe is not None:
                raise ValueError(f"{label}: directed=1 requires pe to be NULL.")
            missing_direction = [
                field
                for field, value in (("pe_fwd", pe_fwd), ("pe_rev", pe_rev))
                if value is None
            ]
            if missing_direction:
                raise ValueError(
                    f"{label}: directed=1 requires non-NULL pe_fwd and pe_rev; "
                    f"missing {', '.join(missing_direction)}."
                )
            p_fwd = pe_fwd
            p_rev = pe_rev

        if blocked:
            p_fwd = 0.0
            p_rev = 0.0

        for origin, destination, probability in (
            (org, dst, p_fwd),
            (dst, org, p_rev),
        ):
            arc = (origin, destination)
            if arc in seen_arcs:
                first = seen_arcs[arc]
                raise ValueError(
                    f"{label}: duplicate directed arc {arc}; it already comes "
                    f"from edge row {first}."
                )
            seen_arcs[arc] = index
            if (
                probability is None
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    f"{label}: effective probability for arc {arc} must be "
                    f"finite and in [0,1]; got {probability!r}."
                )
            if probability > 0.0:
                arc_rows.append(origin)
                arc_cols.append(destination)
                arc_data.append(probability)
                arc_edge_indices.append(index)

    rows = np.asarray(arc_rows, dtype=np.int64)
    cols = np.asarray(arc_cols, dtype=np.int64)
    data = np.asarray(arc_data, dtype=float)

    P = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    P.eliminate_zeros()
    P.sort_indices()

    arc_to_edge_index = {
        (origin, destination): edge_index
        for origin, destination, edge_index in zip(
            arc_rows, arc_cols, arc_edge_indices
        )
    }
    edge_ids = np.empty(P.nnz, dtype=np.int64)
    rp, ci, da = P.indptr, P.indices, P.data
    k = 0
    for i in range(P.shape[0]):
        for j_idx in range(rp[i], rp[i + 1]):
            j = int(ci[j_idx])
            edge_ids[k] = arc_to_edge_index[(i, j)]
            k += 1

    return PropagationMatrices(P=P, n=n_nodes, edge_ids=edge_ids, rowptr=rp, colind=ci, data=da)


def step_prob(F_prev: np.ndarray, mats: PropagationMatrices) -> np.ndarray:
    """
    Deterministic synchronous one-step update in probability space.

    Given current failure probabilities F_prev[i] in [0,1], compute the
    probability that each node j is triggered at the next step by any
    currently failed source i:

        F_next[j] = 1 - Π_i (1 - P[i,j] * F_prev[i])
        
    This computes the *one-step induced* probabilities (not cumulative over time).
    """
    P = mats.P  # CSR (i->j)
    F_prev = np.asarray(F_prev, dtype=float)
    # Only rows with nonzero failure probability contribute.
    active = np.nonzero(F_prev > 0)[0]
    if active.size == 0:
        return np.zeros_like(F_prev, dtype=float)

    # Gather active rows and compute product over (1 - p_ij * f_i)
    # We aggregate Π(1 - x_ij) per column; for typical sparse degrees this is stable enough.
    # If you ever hit underflow for very long columns, switch to log-sum-exp.
    sub = P[active, :]  # still CSR
    # broadcast F over each active row
    fi = F_prev[active].reshape(-1, 1)  # (k,1)
    x = sub.multiply(fi)               # elementwise scaling of each row by f_i
    # 1 - summed? no → product of (1 - x_ij) across rows i for each column j
    # Convert to CSC for efficient column aggregation
    Xc = x.tocsc()
    one_minus = 1.0 - Xc.data
    one_minus = np.clip(one_minus, 0.0, 1.0)
    # For each column, we need Π over entries in that column. We’ll do it by manual CSC sweep.
    prod = np.ones(P.shape[1], dtype=float)
    cp = Xc.indptr
    for j in range(P.shape[1]):
        if cp[j] < cp[j + 1]:
            col_slice = slice(cp[j], cp[j + 1])
            prod[j] = one_minus[col_slice].prod()
    F_next = 1.0 - prod
    return np.clip(F_next, 0.0, 1.0)


@dataclass
class MCResult:
    F_final: np.ndarray           # bool, final failed set
    steps: int
    attempts_per_edge: Optional[np.ndarray]  # len(nnz) counts (if log_attempts), else None
    successes_per_edge: Optional[np.ndarray] # len(nnz) counts (if log_attempts), else None


def simulate_once_mc(
    F0: np.ndarray,
    mats: PropagationMatrices,
    *,
    rng: np.random.Generator,
    max_steps: int = 10_000,
    log_attempts: bool = False,
) -> MCResult:
    """
    Run one Monte Carlo realization of the propagation process.

    Algorithm
    ---------
    - Start with initial failed set F0 (boolean mask).
    - Maintain a frontier of nodes that failed in the previous step.
    - For each frontier node i, attempt each outgoing arc (i->j) once.
      Each attempt succeeds with probability P[i,j] and triggers failure of j.
    - Iterate synchronously until no new failures occur or max_steps is reached.
    - Each frontier node attempts each outgoing arc once per *time step*.

    Logging
    -------
    If `log_attempts=True`, counts are returned per CSR nonzero entry (per arc),
    aligned with mats.data/mats.indices ordering.
    """
    if not isinstance(max_steps, (int, np.integer)) or int(max_steps) < 1:
        raise ValueError(f"max_steps must be a positive integer; got {max_steps!r}.")
    max_steps = int(max_steps)

    P = mats.P
    rp, ci, da = mats.rowptr, mats.colind, mats.data
    n = mats.n

    F = np.array(F0, dtype=bool, copy=True)
    frontier = F.copy()
    attempts = np.zeros(P.nnz, dtype=np.int64) if log_attempts else None
    successes = np.zeros(P.nnz, dtype=np.int64) if log_attempts else None

    steps = 0
    while frontier.any():
        if steps >= max_steps:
            raise RuntimeError(
                f"Propagation did not converge within max_steps={max_steps}."
            )

        new = np.zeros(n, dtype=bool)

        active_idx = np.nonzero(frontier)[0]

        for i in active_idx:
            start, end = rp[i], rp[i + 1]
            if start == end:
                continue

            all_positions = np.arange(start, end)
            all_destinations = ci[start:end]
            eligible = ~F[all_destinations]
            if not np.any(eligible):
                continue

            positions = all_positions[eligible]
            destinations = all_destinations[eligible]
            probs = da[positions]
            if probs.size == 0:
                continue
            rnd = rng.random(probs.shape)
            trials = rnd < probs
            if log_attempts:
                attempts[positions] += 1
                successes[positions] += trials.astype(np.int64)
            if np.any(trials):
                new[destinations[trials]] = True

        if not new.any():
            break

        F[new] = True
        frontier = new
        steps += 1

    return MCResult(F_final=F, steps=steps, attempts_per_edge=attempts, successes_per_edge=successes)
