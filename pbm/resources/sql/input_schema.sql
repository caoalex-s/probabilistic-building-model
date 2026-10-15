PRAGMA foreign_keys = ON;

-- ==== Nodes (compartments) ===================================================
-- 0-based lattice coordinates, canonical primary key id
-- pf_node is the sole executable node-failure probability.
CREATE TABLE IF NOT EXISTS nodes (
  id         INTEGER PRIMARY KEY,
  x          INTEGER NOT NULL,
  y          INTEGER NOT NULL,
  z          INTEGER NOT NULL,
  pf_node    REAL NOT NULL DEFAULT 0.01
             CHECK (
               typeof(pf_node) IN ('integer','real')
               AND pf_node BETWEEN 0.0 AND 1.0
             ),
  UNIQUE (x,y,z)
);

-- Helpful spatial index
CREATE INDEX IF NOT EXISTS idx_nodes_xyz ON nodes(x,y,z);

-- ==== Edges (one canonical adjacency row: org < dst) =========================
-- dir: stored axis of this edge ('x','y','z') to support axis scans
-- directed=0: pe is required; pe_fwd and pe_rev must be NULL
-- directed=1: pe must be NULL; pe_fwd and pe_rev are both required
-- blocked=1: both effective propagation probabilities are zero
CREATE TABLE IF NOT EXISTS edges (
  id          INTEGER PRIMARY KEY,
  org         INTEGER NOT NULL,
  dst         INTEGER NOT NULL,
  dir         TEXT    NOT NULL CHECK (dir IN ('x','y','z')),
  pe          REAL    DEFAULT 0.1,
  pe_fwd      REAL,
  pe_rev      REAL,
  directed    INTEGER NOT NULL DEFAULT 0 CHECK (directed IN (0,1)),
  blocked     INTEGER NOT NULL DEFAULT 0 CHECK (blocked IN (0,1)),
  CHECK (org < dst),
  CHECK (pe IS NULL OR pe BETWEEN 0.0 AND 1.0),
  CHECK (pe_fwd IS NULL OR pe_fwd BETWEEN 0.0 AND 1.0),
  CHECK (pe_rev IS NULL OR pe_rev BETWEEN 0.0 AND 1.0),
  CHECK (
    (directed = 0 AND pe IS NOT NULL AND pe_fwd IS NULL AND pe_rev IS NULL)
    OR
    (directed = 1 AND pe IS NULL AND pe_fwd IS NOT NULL AND pe_rev IS NOT NULL)
  ),
  UNIQUE (org, dst),
  FOREIGN KEY (org) REFERENCES nodes(id) ON DELETE CASCADE,
  FOREIGN KEY (dst) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_dir ON edges(dir);

-- ==== Simultaneous pre-run notional damage ===================================
-- An empty table is the undamaged case. Each row identifies one node that is
-- deterministically failed once at pseudo-time t=0. Multiple rows are one
-- simultaneous damage set, not staged or delayed events.
CREATE TABLE IF NOT EXISTS initial_damage (
  node_id INTEGER PRIMARY KEY,
  FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);
