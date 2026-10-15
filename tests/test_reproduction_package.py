"""Reject corrupt evidence and read the correct run from paired databases."""
import hashlib
import json
import sqlite3
from contextlib import closing

import pytest
from pbm.io.schema import read_schema_sql
from scripts.reproduce import read_result
from scripts.validate_reproduction import pmf,verify_hashes


@pytest.mark.parametrize('counts,samples',[
    ([100]+[0]*63,100),
    ([101]+[0]*64,100),
    ([101,-1]+[0]*63,100),
    ([100.]+[0.]*64,100),
])
def test_invalid_scientific_counts_are_rejected(counts,samples):
    with pytest.raises(ValueError):
        pmf(counts,samples)


def test_changed_artifact_is_rejected(tmp_path):
    artifact=tmp_path/'counts.json'
    artifact.write_text('original',encoding='utf8')
    expected=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps({'counts.json':expected}),encoding='utf8')
    artifact.write_text('changed',encoding='utf8')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_hashes(tmp_path)


def test_inventory_cannot_reference_a_parent_file(tmp_path):
    directory=tmp_path/'evidence'
    directory.mkdir()
    (directory/'SHA256SUMS.json').write_text(json.dumps({'../outside.json':'unused'}),encoding='utf8')
    with pytest.raises(ValueError,match='leaves evidence'):
        verify_hashes(directory)


def test_paired_results_export_the_latest_run(tmp_path):
    database=tmp_path/'results.db'
    with closing(sqlite3.connect(database)) as con:
        con.executescript(read_schema_sql('output'))
        for rid,n,counts in [(1,2,[2,0]),(2,3,[0,3])]:
            con.execute('INSERT INTO runs(id,scenario_id,samples,threshold,seed,started_at,finished_at,notes) VALUES(?,?,?,2,123,?,?,?)',
                        (rid,rid-1,n,'start','finish','{}'))
            con.executemany('INSERT INTO pmf(run_id,n_failed,count,p) VALUES(?,?,?,?)',
                            [(rid,i,k,k/n) for i,k in enumerate(counts)])
        con.commit()
    result=read_result(database)
    assert result['run_id']==2 and result['samples']==3
    assert result['baseline_counts']==[0,3]
