from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import sqlite3

from pbm import __version__
from pbm.utils.logging_setup import setup_logging
from pbm.io import writer
from pbm.post.core import run_postprocessing
from pbm.post.plot_style import PLOT_MODES
from pbm.post.robustness_plots import plot_entropy_robustness_index


def _as_bool(s: str) -> bool:
    """Parse a yes/no style CLI flag into bool.

    Accepts: 1, true, yes, y, on  → True
             0, false, no, n, off → False
    Case-insensitive.
    """
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stationary probabilistic building model post-processing")
    ap.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    ap.add_argument(
        "--outdb",
        default="output/results.db",
        help="Path to results DB (default: output/results.db)",
    )
    ap.add_argument(
        "--figs-root",
        default="output/figs",
        help="Root directory for plot outputs (default: output/figs)",
    )
    ap.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run id in 'runs' table (default: latest run)",
    )
    ap.add_argument(
        "--make-plots",
        default="yes",
        help="yes/no — whether to generate image files (default: yes)",
    )
    ap.add_argument(
        "--plot-mode",
        choices=PLOT_MODES,
        default="standard",
        help=(
            "Plot mode: 'standard' uses portable Matplotlib mathtext; "
            "'publication' exports PDF/PGF with external LaTeX via the PGF backend "
            "(default: standard)"
        ),
    )
    ap.add_argument(
        "--risk-equivalence",
        action="store_true",
        help=(
            "Compute numerical scan-PMF and analytical regular-lattice "
            "risk-equivalence estimates"
        ),
    )
    ap.add_argument(
        "--pe-crit",
        "--risk-equivalence-pe-crit",
        dest="risk_equivalence_pe_crit",
        type=float,
        default=None,
        help=(
            "Risk-equivalence upper bound. Defaults to the identified "
            "percolation threshold from the same run."
        ),
    )
    ap.add_argument(
        "--lattice-shape",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help=(
            "Regular lattice shape for the analytical approximation. "
            "Defaults to the shape recorded by the simulation."
        ),
    )
    ap.add_argument(
        "--risk-equivalence-confidence-level",
        type=float,
        default=0.95,
        help="Simultaneous Monte Carlo diagnostic confidence level (default: 0.95)",
    )
    ap.add_argument(
        "--risk-equivalence-json",
        default=None,
        help=(
            "Machine-readable output path. Defaults to "
            "<figs-root>/run_<id>/risk_equivalence.json."
        ),
    )
    ap.add_argument(
        "--entropy-robustness",
        action="store_true",
        help=(
            "Analyze and plot Bernoulli KL robustness for explicitly paired "
            "undamaged/damaged runs and adverse macrostates"
        ),
    )
    ap.add_argument(
        "--baseline-run-id",
        type=int,
        default=None,
        help=(
            "Undamaged baseline run_id (required with --entropy-robustness)"
        ),
    )
    ap.add_argument(
        "--damaged-run-ids",
        default="",
        help=(
            "Comma-separated run_ids with specified notional damage "
            "(required with --entropy-robustness)"
        ),
    )
    ap.add_argument(
        "--s-lim",
        default="",
        help=(
            "Explicit adverse macrostate threshold(s) in N_f >= S_lim, "
            "comma-separated (required with --entropy-robustness)"
        ),
    )
    ap.add_argument(
        "--coherence-mode",
        choices=("strict", "diagnostic"),
        default="strict",
        help=(
            "Strict mode rejects pj < p0 beyond tolerance; diagnostic mode "
            "returns an invalid result with the violation recorded"
        ),
    )
    ap.add_argument(
        "--mc-tolerance",
        type=float,
        default=0.0,
        help=(
            "Absolute probability tolerance for Monte Carlo coherence "
            "reversals (default: 0)"
        ),
    )
    ap.add_argument(
        "--robustness-json",
        default=None,
        help=(
            "Machine-readable robustness output path. Defaults to "
            "<figs-root>/robustness_baseline_<id>.json."
        ),
    )

    args = ap.parse_args()

    # Configure logging (file path can be adjusted later if desired)
    setup_logging("output/logs/app.log")

    out_db = Path(args.outdb)
    figs_root = Path(args.figs_root)

    # Connect to DB
    conn: sqlite3.Connection = writer.connect(out_db)

    # Determine run_id (default: latest run)
    run_id = args.run_id
    if run_id is None:
        cur = conn.execute("SELECT MAX(id) FROM runs;")
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise SystemExit("No runs found in output DB; run scripts/cli_simulation.py first.")
        run_id = int(row[0])

    make_plots = _as_bool(args.make_plots)
    if args.risk_equivalence_json and not args.risk_equivalence:
        raise SystemExit(
            "--risk-equivalence-json requires --risk-equivalence"
        )
    if args.robustness_json and not args.entropy_robustness:
        raise SystemExit("--robustness-json requires --entropy-robustness")
    if args.entropy_robustness and args.risk_equivalence:
        raise SystemExit(
            "--entropy-robustness and --risk-equivalence are separate "
            "post-processing workflows."
        )

    # --- campaign-level entropy robustness plot (multi-run) -------------------
    if args.entropy_robustness:
        if args.baseline_run_id is None:
            raise SystemExit(
                "--baseline-run-id is required with --entropy-robustness"
            )
        if not str(args.damaged_run_ids).strip():
            raise SystemExit(
                "--damaged-run-ids is required with --entropy-robustness"
            )
        if not str(args.s_lim).strip():
            raise SystemExit("--s-lim is required with --entropy-robustness")

        try:
            damaged_ids = [
                int(token.strip())
                for token in str(args.damaged_run_ids).split(",")
                if token.strip()
            ]
            if any(run_id < 1 for run_id in damaged_ids):
                raise ValueError("Damaged run IDs must be positive integers.")
            campaign = plot_entropy_robustness_index(
                conn,
                baseline_run_id=int(args.baseline_run_id),
                damaged_run_ids=damaged_ids,
                s_lim=str(args.s_lim),
                figs_dir=figs_root,
                formats=("pdf",),
                coherence_mode=args.coherence_mode,
                mc_tolerance=args.mc_tolerance,
                plot_mode=args.plot_mode,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        json_path = (
            Path(args.robustness_json)
            if args.robustness_json
            else figs_root
            / f"robustness_baseline_{int(args.baseline_run_id)}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                campaign.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        for output in campaign.output_paths:
            print(f"Wrote entropy robustness plot: {output}")
        print(f"Wrote entropy robustness diagnostics: {json_path}")
        return

    summary = run_postprocessing(
        conn,
        run_id=run_id,
        make_plots=make_plots,
        compute_risk_equivalence=args.risk_equivalence,
        risk_equivalence_pe_crit=args.risk_equivalence_pe_crit,
        lattice_shape=(
            tuple(args.lattice_shape)
            if args.lattice_shape is not None
            else None
        ),
        risk_equivalence_confidence_level=(
            args.risk_equivalence_confidence_level
        ),
        figs_root=figs_root,
        plot_mode=args.plot_mode,
    )

    if args.risk_equivalence:
        risk_result = summary["risk_equivalence_result"]
        json_path = (
            Path(args.risk_equivalence_json)
            if args.risk_equivalence_json
            else figs_root / f"run_{run_id}" / "risk_equivalence.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": int(run_id),
            "risk_equivalence": risk_result.to_dict(),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["risk_equivalence_json"] = str(json_path)

    # Simple console output for quick inspection
    print(f"Post-processing summary for run {run_id}:")
    for k, v in sorted(summary.items()):
        if k in {
            "percolation_threshold_result",
            "percolation_scan_metadata",
            "risk_equivalence_result",
        }:
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
