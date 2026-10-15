PRAGMA foreign_keys = ON;

-- =============================================================================
-- Output/results database schema (authoritative)
-- =============================================================================
--
-- This schema is aligned with the current simulation CLI + post-processing.
-- It intentionally contains no duplicate/conflicting DDL.
--
-- Foreign keys to input entities (nodes/edges)
-- -------------------------------------------
-- When writing a standalone results.db, we optionally mirror lightweight
-- (id-only) tables so output tables can reference node_id/edge_id.
-- If you keep input.db and results.db as separate files, these FKs are not
-- enforced unless both DBs are ATTACHed; they still document intent.

CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY
);

-- =============================================================================
-- Runs
-- =============================================================================
-- notes: JSON string (configuration, resolved notional damage, stationary
-- model fingerprint, random-stream contract, git SHA, host information, etc.)

CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY,
  scenario_id   INTEGER NOT NULL,
  samples       INTEGER NOT NULL,
  threshold     INTEGER NOT NULL,
  seed          INTEGER NOT NULL,
  log_attempts  INTEGER NOT NULL DEFAULT 0,
  started_at    TEXT    NOT NULL,
  finished_at   TEXT,
  notes         TEXT
);

-- =============================================================================
-- PMF over number of failed nodes
-- =============================================================================

CREATE TABLE IF NOT EXISTS pmf (
  run_id    INTEGER NOT NULL,
  n_failed  INTEGER NOT NULL,
  count     INTEGER NOT NULL,
  p         REAL    NOT NULL,
  PRIMARY KEY (run_id, n_failed),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- =============================================================================
-- Collapse probabilities
-- =============================================================================

CREATE TABLE IF NOT EXISTS collapse_probs (
  run_id      INTEGER PRIMARY KEY,
  p_any       REAL NOT NULL,
  p_full      REAL NOT NULL,
  p_ge_thresh REAL NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- =============================================================================
-- Edge usage / attempts
-- =============================================================================

CREATE TABLE IF NOT EXISTS edge_usage (
  run_id    INTEGER NOT NULL,
  edge_id   INTEGER NOT NULL,
  successes INTEGER NOT NULL,
  PRIMARY KEY (run_id, edge_id),
  FOREIGN KEY (run_id)  REFERENCES runs(id)  ON DELETE CASCADE,
  FOREIGN KEY (edge_id) REFERENCES edges(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edge_attempts (
  run_id   INTEGER NOT NULL,
  edge_id  INTEGER NOT NULL,
  attempts INTEGER NOT NULL,
  PRIMARY KEY (run_id, edge_id),
  FOREIGN KEY (run_id)  REFERENCES runs(id)  ON DELETE CASCADE,
  FOREIGN KEY (edge_id) REFERENCES edges(id) ON DELETE CASCADE
);

-- =============================================================================
-- Percolation scans (legacy)
-- =============================================================================
-- Axis-specific percolation curves.
-- Kept temporarily for backwards compatibility; current code writes
-- percolation_multi only.

CREATE TABLE IF NOT EXISTS percolation (
  run_id   INTEGER NOT NULL,
  dir      TEXT    NOT NULL CHECK (dir IN ('x','y','z')),
  pe_value REAL    NOT NULL,
  p_any    REAL    NOT NULL,
  PRIMARY KEY (run_id, dir, pe_value),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- =============================================================================
-- Percolation multi-axis scans (current)
-- =============================================================================
-- p_any  = P(N_f >= 1)
-- p_one  = P(N_f = 1)
-- p_full = P(N_f = |N|)
-- Integer counts and n_samples are retained for binomial uncertainty analysis.

CREATE TABLE IF NOT EXISTS percolation_multi (
  run_id    INTEGER NOT NULL,
  pe_x      REAL NOT NULL,
  pe_y      REAL NOT NULL,
  pe_z      REAL NOT NULL,
  p_any     REAL NOT NULL CHECK (p_any >= 0.0 AND p_any <= 1.0),
  p_one     REAL NOT NULL CHECK (p_one >= 0.0 AND p_one <= 1.0),
  p_full    REAL NOT NULL CHECK (p_full >= 0.0 AND p_full <= 1.0),
  mean_nf   REAL NOT NULL,
  n_samples INTEGER NOT NULL CHECK (n_samples > 0),
  count_any INTEGER NOT NULL CHECK (count_any >= 0 AND count_any <= n_samples),
  count_one INTEGER NOT NULL CHECK (count_one >= 0 AND count_one <= n_samples),
  count_full INTEGER NOT NULL CHECK (count_full >= 0 AND count_full <= n_samples),
  CHECK (ABS(p_any - CAST(count_any AS REAL) / n_samples) <= 1.0e-12),
  CHECK (ABS(p_one - CAST(count_one AS REAL) / n_samples) <= 1.0e-12),
  CHECK (ABS(p_full - CAST(count_full AS REAL) / n_samples) <= 1.0e-12),
  PRIMARY KEY (run_id, pe_x, pe_y, pe_z),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Full integer PMF retained at every scan point.  The aggregate probabilities
-- in percolation_multi are convenient summaries; this table is authoritative
-- for risk-equivalence post-processing and Monte Carlo uncertainty.

CREATE TABLE IF NOT EXISTS percolation_pmf (
  run_id     INTEGER NOT NULL,
  pe_x       REAL NOT NULL,
  pe_y       REAL NOT NULL,
  pe_z       REAL NOT NULL,
  n_failed   INTEGER NOT NULL CHECK (n_failed >= 0),
  count      INTEGER NOT NULL CHECK (count >= 0),
  p          REAL NOT NULL CHECK (p >= 0.0 AND p <= 1.0),
  n_samples  INTEGER NOT NULL CHECK (n_samples > 0),
  CHECK (count <= n_samples),
  CHECK (ABS(p - CAST(count AS REAL) / n_samples) <= 1.0e-12),
  PRIMARY KEY (run_id, pe_x, pe_y, pe_z, n_failed),
  FOREIGN KEY (run_id, pe_x, pe_y, pe_z)
    REFERENCES percolation_multi(run_id, pe_x, pe_y, pe_z)
    ON DELETE CASCADE
);

-- =============================================================================
-- Node influence metric
-- =============================================================================

CREATE TABLE IF NOT EXISTS node_influence (
  run_id    INTEGER NOT NULL,
  node_id   INTEGER NOT NULL,
  influence REAL NOT NULL,
  PRIMARY KEY (run_id, node_id),
  FOREIGN KEY (run_id)  REFERENCES runs(id)  ON DELETE CASCADE,
  FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);
