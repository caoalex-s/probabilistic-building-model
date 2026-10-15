from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from pbm.io.loader import load_all
from pbm.network.builder import build_and_insert

from ._helpers import PaperLattice


@pytest.fixture(scope="session")
def paper_lattice(tmp_path_factory: pytest.TempPathFactory) -> PaperLattice:
    """Build the manuscript's non-periodic 4x4x4 lattice once per session."""

    directory = tmp_path_factory.mktemp("paper-lattice")
    database = Path(directory) / "input.db"
    build_and_insert(
        nx=4,
        ny=4,
        nz=4,
        db_path=database,
        truncate=True,
    )
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE nodes SET pf_node=0.03;")
        connection.execute(
            """
            UPDATE edges
            SET pe=0.35,
                pe_fwd=NULL,
                pe_rev=NULL,
                directed=0,
                blocked=0;
            """
        )
        connection.commit()
    return PaperLattice(db_path=database, model=load_all(database))
