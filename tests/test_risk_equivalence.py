from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from pbm.io.schema import ensure_schema
from pbm.post.core import run_postprocessing
from pbm.post.risk_equivalence import (
    analyze_risk_equivalence,
    estimate_analytical_risk_equivalence,
    estimate_numerical_risk_equivalence,
    expected_no_propagation,
    regular_lattice_degree_counts,
    regular_lattice_degree_distribution,
)
from pbm.post.run_selection import (
    CompatibleRunNotFoundError,
    select_latest_compatible_xyz_run,
)


# Reduced-cost single-initial-compartment Monte Carlo fixture for the paper's
# 4x4x4 lattice (200,000 conditional realizations per point, fixed seeds).
# Adding 31,050,000 no-failure realizations gives the unconditional
# P(N_f>=1)=0.0064 implied to first order by 64 compartments with pf=1e-4.
# Conditioning/reweighting is the small-pf limit used in Equations 20-23.
_PAPER_CASE_NONZERO_COUNTS = {
    0.08: (
        0, 137716, 39482, 13529, 5278, 2147, 1011, 445, 204, 106, 44,
        20, 11, 3, 0, 2, 2,
    ),
    0.09: (
        0, 131128, 41276, 15105, 6661, 2942, 1470, 696, 338, 185, 102,
        43, 24, 18, 5, 4, 3,
    ),
    0.10: (
        0, 124963, 41901, 17019, 8053, 3878, 1968, 986, 578, 307, 163,
        77, 52, 26, 15, 6, 2, 4, 0, 1, 0, 1,
    ),
    0.11: (
        0, 118781, 42892, 18114, 9160, 4741, 2706, 1527, 856, 500, 303,
        178, 99, 61, 36, 18, 12, 8, 2, 2, 0, 1, 1, 0, 1, 0, 0, 1,
    ),
}


def paper_case_scan() -> tuple[np.ndarray, np.ndarray]:
    pe = np.asarray([0.0, 0.08, 0.09, 0.10, 0.11], dtype=float)
    counts = np.zeros((pe.size, 65), dtype=np.int64)
    counts[0, 1] = 200_000
    for index, probability in enumerate(pe[1:], start=1):
        values = _PAPER_CASE_NONZERO_COUNTS[float(probability)]
        counts[index, :len(values)] = values
    counts[:, 0] += 31_050_000
    assert np.all(np.sum(counts, axis=1) == 31_250_000)
    return pe, counts


def make_paper_case_results_db(path: Path) -> Path:
    pe, counts = paper_case_scan()
    samples = int(np.sum(counts[0]))
    conn = ensure_schema(path, "output")
    notes = {
        "regular_lattice_shape": [4, 4, 4],
        "notional_damage_node_ids": [],
        "notional_damage_source": "none",
        "pf_node_provenance": {
            "schema_version": 1,
            "source": "input_database.nodes.pf_node",
            "input_database": "synthetic-paper-case.db",
            "authoritative_field": "pf_node",
            "node_count": 64,
            "uniform": True,
            "uniform_value": 1e-4,
            "minimum": 1e-4,
            "maximum": 1e-4,
            "unique_value_count": 1,
            "values_sha256": "0" * 64,
            "sha256_encoding": (
                "concatenated big-endian int64 node id and float64 pf_node, "
                "ordered by node id"
            ),
        },
        "percolation_scan_metadata": {
            "scan_spec": "xyz:0,0.11,variable",
            "mode": "1d_xyz",
            "varied_axes": ["x", "y", "z"],
            "fixed_axes": [],
            "varied_axis_groups": [["x", "y", "z"]],
            "fixed_axis_values": {},
        },
    }
    with conn:
        conn.execute(
            """
            INSERT INTO runs(
              id, scenario_id, samples, threshold, seed, log_attempts,
              started_at, finished_at, notes
            ) VALUES (1, 1, ?, 2, 123, 0, 'now', 'now', ?);
            """,
            (samples, json.dumps(notes)),
        )
        for probability, histogram in zip(pe, counts):
            trials = int(np.sum(histogram))
            count_any = trials - int(histogram[0])
            count_one = int(histogram[1])
            count_full = int(histogram[-1])
            mean_nf = float(
                np.dot(np.arange(histogram.size), histogram) / trials
            )
            conn.execute(
                """
                INSERT INTO percolation_multi(
                  run_id, pe_x, pe_y, pe_z, p_any, p_one, p_full,
                  mean_nf, n_samples, count_any, count_one, count_full
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    1,
                    float(probability),
                    float(probability),
                    float(probability),
                    count_any / trials,
                    count_one / trials,
                    count_full / trials,
                    mean_nf,
                    trials,
                    count_any,
                    count_one,
                    count_full,
                ),
            )
            conn.executemany(
                """
                INSERT INTO percolation_pmf(
                  run_id, pe_x, pe_y, pe_z, n_failed, count, p, n_samples
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        1,
                        float(probability),
                        float(probability),
                        float(probability),
                        int(n_failed),
                        int(count),
                        float(count / trials),
                        trials,
                    )
                    for n_failed, count in enumerate(histogram)
                ],
            )
    conn.close()
    return path


def _copy_paper_case_run(
    conn: sqlite3.Connection,
    *,
    source_run_id: int,
    target_run_id: int,
) -> None:
    source = conn.execute(
        """
        SELECT scenario_id, samples, threshold, seed, log_attempts, notes
        FROM runs
        WHERE id = ?;
        """,
        (source_run_id,),
    ).fetchone()
    assert source is not None
    conn.execute(
        """
        INSERT INTO runs(
          id, scenario_id, samples, threshold, seed, log_attempts,
          started_at, finished_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, 'later', 'later', ?);
        """,
        (target_run_id, *source),
    )
    conn.execute(
        """
        INSERT INTO percolation_multi(
          run_id, pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf,
          n_samples, count_any, count_one, count_full
        )
        SELECT ?, pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf,
               n_samples, count_any, count_one, count_full
        FROM percolation_multi
        WHERE run_id = ?;
        """,
        (target_run_id, source_run_id),
    )
    conn.execute(
        """
        INSERT INTO percolation_pmf(
          run_id, pe_x, pe_y, pe_z, n_failed, count, p, n_samples
        )
        SELECT ?, pe_x, pe_y, pe_z, n_failed, count, p, n_samples
        FROM percolation_pmf
        WHERE run_id = ?;
        """,
        (target_run_id, source_run_id),
    )


def test_selects_latest_compatible_xyz_run_and_validates_provenance(
    tmp_path: Path,
):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    with sqlite3.connect(db_path) as conn:
        with conn:
            _copy_paper_case_run(
                conn,
                source_run_id=1,
                target_run_id=2,
            )
            # A newer unrelated run must not override the latest compatible
            # fully coupled run.
            source = conn.execute(
                """
                SELECT scenario_id, samples, threshold, seed, log_attempts, notes
                FROM runs
                WHERE id = 1;
                """
            ).fetchone()
            conn.execute(
                """
                INSERT INTO runs(
                  id, scenario_id, samples, threshold, seed, log_attempts,
                  started_at, finished_at, notes
                ) VALUES (3, ?, ?, ?, ?, ?, 'newest', 'newest', ?);
                """,
                source,
            )

        selected = select_latest_compatible_xyz_run(
            conn,
            expected_samples=31_250_000,
            expected_pf_node=1e-4,
            lattice_shape=(4, 4, 4),
            pe_crit=0.11,
        )

    assert selected.run_id == 2
    assert selected.samples == 31_250_000
    assert selected.scan_samples == 31_250_000
    assert selected.sample_count_validated is True
    assert selected.pf_node == pytest.approx(1e-4)
    assert selected.pf_provenance_validated is True
    assert selected.pf_empirical_check_resolved is True
    assert abs(selected.p_any_at_zero_z_score) < 6.0
    assert selected.scan_bounds == (0.0, 0.11)


def test_run_selection_rejects_missing_pf_provenance(tmp_path: Path):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    with sqlite3.connect(db_path) as conn:
        notes = json.loads(
            conn.execute("SELECT notes FROM runs WHERE id = 1;").fetchone()[0]
        )
        notes.pop("pf_node_provenance")
        with conn:
            conn.execute(
                "UPDATE runs SET notes = ? WHERE id = 1;",
                (json.dumps(notes),),
            )
        with pytest.raises(
            CompatibleRunNotFoundError,
            match="missing pf_node_provenance",
        ):
            select_latest_compatible_xyz_run(
                conn,
                expected_samples=31_250_000,
                expected_pf_node=1e-4,
                lattice_shape=(4, 4, 4),
                pe_crit=0.11,
            )


def test_run_selection_rejects_sample_count_mismatch(tmp_path: Path):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(
            CompatibleRunNotFoundError,
            match=r"runs\.samples=31250000, expected 10000000",
        ):
            select_latest_compatible_xyz_run(
                conn,
                expected_samples=10_000_000,
                expected_pf_node=1e-4,
                lattice_shape=(4, 4, 4),
                pe_crit=0.11,
            )


def test_run_selection_checks_pf_provenance_against_zero_scan_pmf(
    tmp_path: Path,
):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    with sqlite3.connect(db_path) as conn:
        notes = json.loads(
            conn.execute("SELECT notes FROM runs WHERE id = 1;").fetchone()[0]
        )
        provenance = notes["pf_node_provenance"]
        provenance["uniform_value"] = 0.01
        provenance["minimum"] = 0.01
        provenance["maximum"] = 0.01
        with conn:
            conn.execute(
                "UPDATE runs SET notes = ? WHERE id = 1;",
                (json.dumps(notes),),
            )
        with pytest.raises(
            CompatibleRunNotFoundError,
            match="p_e=0 PMF is inconsistent",
        ):
            select_latest_compatible_xyz_run(
                conn,
                expected_samples=31_250_000,
                expected_pf_node=0.01,
                lattice_shape=(4, 4, 4),
                pe_crit=0.11,
            )


def test_verification_notebook_does_not_hard_code_run_id():
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            repo_root
            / "notebooks"
            / "verify_risk_equivalence_pe_star.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "RUN_ID = 1" not in source
    assert "select_latest_compatible_xyz_run" in source
    assert "expected_samples=EXPECTED_SAMPLES" in source
    assert "expected_pf_node=EXPECTED_PF_NODE" in source


def test_regular_4x4x4_degree_formulas():
    counts = regular_lattice_degree_counts(4, 4, 4)
    assert counts == {3: 8, 4: 24, 5: 24, 6: 8}

    distribution = regular_lattice_degree_distribution(4, 4, 4)
    expected_mean = 6.0 - 2.0 * (1 / 4 + 1 / 4 + 1 / 4)
    expected_second_factorial = (
        30.0
        - 20.0 * (1 / 4 + 1 / 4 + 1 / 4)
        + 8.0 * (1 / 16 + 1 / 16 + 1 / 16)
    )
    assert distribution.node_count == 64
    assert distribution.mean_degree == pytest.approx(expected_mean)
    assert distribution.second_factorial_moment == pytest.approx(
        expected_second_factorial
    )
    assert distribution.mean_excess_degree == pytest.approx(11 / 3)
    assert distribution.inverse_mean_excess_degree == pytest.approx(3 / 11)


def test_degree_counts_handle_degenerate_lattice_dimensions():
    distribution = regular_lattice_degree_distribution(1, 2, 3)
    assert dict(distribution.degree_counts) == {2: 4, 3: 2}
    assert sum(dict(distribution.degree_counts).values()) == 6


def test_expected_no_propagation_uses_finite_degree_mixture():
    counts = regular_lattice_degree_counts(4, 4, 4)
    pe = 0.1
    expected = (
        8 * 0.9**3 + 24 * 0.9**4 + 24 * 0.9**5 + 8 * 0.9**6
    ) / 64
    assert expected_no_propagation(pe, counts) == pytest.approx(expected)


def test_analytical_4x4x4_paper_reference():
    result = estimate_analytical_risk_equivalence(
        4,
        4,
        4,
        pe_crit=0.65,
    )

    assert result.identified
    assert result.estimate == pytest.approx(
        0.09386757997141897,
        abs=1e-9,
    )
    assert result.pe_max == pytest.approx(3 / 11)
    assert abs(result.residual_at_estimate) < 1e-9
    assert result.conservatism == "case-dependent"


def test_analytical_bound_can_exclude_the_crossing():
    result = estimate_analytical_risk_equivalence(
        4, 4, 4, pe_crit=0.05
    )
    assert not result.identified
    assert result.estimate is None
    assert result.reason == "no_crossing_within_bounds"
    assert result.bounds == (0.0, 0.05)


def test_numerical_4x4x4_paper_reference_from_scan_pmfs():
    pe, counts = paper_case_scan()
    result = estimate_numerical_risk_equivalence(
        pe, counts, pe_crit=0.11
    )
    assert result.identified
    assert result.estimate == pytest.approx(0.093, abs=0.006)
    assert result.root_bracket == (0.09, 0.10)
    assert result.residual_at_estimate == 0.0
    assert result.resolution == pytest.approx(0.01)
    assert result.monte_carlo_interval is not None
    assert result.monte_carlo_interval[0] < result.estimate
    assert result.monte_carlo_interval[1] > result.estimate
    assert result.monte_carlo_crossing_resolved is True


def test_numerical_no_crossing_has_diagnostics_without_fallback():
    pe = np.asarray([0.0, 0.1, 0.2])
    counts = np.asarray(
        [
            [0, 750, 250],
            [0, 750, 250],
            [0, 750, 250],
        ],
        dtype=np.int64,
    )
    result = estimate_numerical_risk_equivalence(
        pe, counts, pe_crit=0.2
    )
    assert not result.identified
    assert result.estimate is None
    assert result.reason == "no_crossing_within_bounds"
    assert result.closest_grid_point is not None
    assert result.closest_grid_residual == pytest.approx(0.5)


def test_zero_failure_realizations_do_not_create_spurious_exact_roots():
    pe = np.asarray([0.0, 0.1, 0.2])
    counts = np.asarray(
        [[1000, 0, 0], [1000, 0, 0], [1000, 0, 0]],
        dtype=np.int64,
    )
    result = estimate_numerical_risk_equivalence(
        pe, counts, pe_crit=0.2
    )
    assert result.estimate is None
    assert result.reason == "no_crossing_within_bounds"
    assert result.curve.effective_failure_samples == (0, 0, 0)
    assert any("not treated as a root" in warning for warning in result.warnings)


def test_numerical_multiple_crossings_are_not_selected_arbitrarily():
    pe = np.asarray([0.0, 0.1, 0.2, 0.3])
    counts = np.asarray(
        [
            [0, 750, 250],
            [0, 250, 750],
            [0, 750, 250],
            [0, 250, 750],
        ],
        dtype=np.int64,
    )
    result = estimate_numerical_risk_equivalence(
        pe, counts, pe_crit=0.3
    )
    assert not result.identified
    assert result.estimate is None
    assert result.reason == "multiple_crossings"
    assert len(result.crossing_brackets) == 3


def test_numerical_scan_must_cover_the_requested_bound():
    pe = np.asarray([0.05, 0.10])
    counts = np.asarray([[0, 750, 250], [0, 250, 750]])
    result = estimate_numerical_risk_equivalence(
        pe, counts, pe_crit=0.20
    )
    assert result.estimate is None
    assert result.reason == "scan_does_not_cover_bounds"


def test_combined_result_is_machine_readable_and_reports_difference():
    pe, counts = paper_case_scan()
    result = analyze_risk_equivalence(
        pe,
        counts,
        pe_crit=0.11,
        lattice_shape=(4, 4, 4),
    )
    payload = result.to_dict()
    assert payload["numerical"]["estimate"] == pytest.approx(
        result.numerical.estimate
    )
    assert payload["analytical"]["estimate"] == pytest.approx(
        result.analytical.estimate
    )
    assert payload["analytical"]["bounds"] == (0.0, 0.11)
    assert payload["relative_difference"] == pytest.approx(
        (
            result.analytical.estimate - result.numerical.estimate
        )
        / result.numerical.estimate
    )
    assert payload["absolute_relative_difference"] > 0.0
    assert payload["analytical"]["conservatism"] == "case-dependent"


def test_postprocessing_flag_executes_package_analysis(tmp_path: Path):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    with sqlite3.connect(db_path) as conn:
        summary = run_postprocessing(
            conn,
            run_id=1,
            make_plots=False,
            compute_risk_equivalence=True,
            risk_equivalence_pe_crit=0.11,
        )
    assert summary["risk_equivalence_numerical"] == pytest.approx(
        0.093, abs=0.006
    )
    assert summary["risk_equivalence_analytical"] == pytest.approx(
        0.0938675800, abs=1e-9
    )
    assert summary["risk_equivalence_result"].relative_difference is not None


def test_postprocessing_cli_writes_machine_readable_output(tmp_path: Path):
    db_path = make_paper_case_results_db(tmp_path / "results.db")
    output_path = tmp_path / "risk-equivalence.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_post_processing.py"),
            "--outdb",
            str(db_path),
            "--run-id",
            "1",
            "--make-plots",
            "no",
            "--risk-equivalence",
            "--pe-crit",
            "0.11",
            "--risk-equivalence-json",
            str(output_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    analysis = payload["risk_equivalence"]
    assert analysis["numerical"]["estimate"] == pytest.approx(
        0.093, abs=0.006
    )
    assert analysis["analytical"]["estimate"] == pytest.approx(
        0.0938675800, abs=1e-9
    )
    assert analysis["numerical"]["bounds"] == [0.0, 0.11]
    assert analysis["analytical"]["conservatism"] == "case-dependent"
    assert analysis["relative_difference"] is not None
