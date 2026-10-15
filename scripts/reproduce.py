"""Run stationary manuscript campaigns with fresh databases and portable records.

Install the repository package first. See reproduction/README.md for costs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
from scipy.stats import binom

from pbm.network.builder import build_and_insert


def plan(profile: str) -> list[dict]:
    if profile == 'smoke':
        return [dict(label='smoke',shape=[2,2,2],pf=.01,pe=.1,samples=2000,
                     seed=123,scan='xyz:0,1,0.5',damage=[])]
    common = dict(shape=[4,4,4],pf=1e-4,seed=123,damage=[])
    if profile.startswith('figure3'):
        independent = profile == 'figure3-seed-check'
        scans = ['xyz:0.6,0.725,0.025','xy:0.525,0.65,0.025','z:0.06,0.12,0.01'] if independent else ['xyz:0,1,0.025','xy:0,1,0.025','z:0,1,0.01']
        samples = [1_000_000]*3 if independent else [10_000_000]*3 if profile=='figure3-full' else [1_000_000,2_000_000,2_000_000]
        return [dict(common,label=f'case{i+1}',case=i+1,pe=.01 if i==0 else .8,
                     samples=n,scan=scan,seed=456 if independent else 123)
                for i,(scan,n) in enumerate(zip(scans,samples))]
    if profile == 'damage':
        return [dict(common,label=f'pe{pe:g}-scenario{i}',pe=pe,samples=1_000_000,
                     damage=damage,scenario=i)
                for pe in (.67,.4,.2,.094,.0927)
                for i,damage in enumerate(([],[[0,0,0]],[[1,0,0]],[[1,1,0]]))]
    return [dict(common,label=f'pe{pe:g}',pe=pe,samples=10_000_000 if .08<=pe<=.1 else 1_000_000)
            for pe in (0.,.08,.085,.09,.095,.1,.2,.4,.67,1.)]


def read_result(path: Path) -> dict:
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as con:
        run_id,n,seed,threshold,finished,raw = con.execute(
            'SELECT id,samples,seed,threshold,finished_at,notes FROM runs ORDER BY id DESC LIMIT 1').fetchone()
        if not finished:
            raise ValueError('Simulation did not finish')
        notes=json.loads(raw)
        baseline=con.execute('SELECT n_failed,count FROM pmf WHERE run_id=? ORDER BY n_failed',(run_id,)).fetchall()
        counts=[row[1] for row in baseline]
        if [row[0] for row in baseline]!=list(range(len(counts))) or sum(counts)!=n:
            raise ValueError('Invalid baseline PMF')
        points=[]
        for coordinate in con.execute('SELECT pe_x,pe_y,pe_z FROM percolation_multi WHERE run_id=? ORDER BY pe_x,pe_y,pe_z',(run_id,)).fetchall():
            rows=con.execute('SELECT n_failed,count FROM percolation_pmf WHERE run_id=? AND pe_x=? AND pe_y=? AND pe_z=? ORDER BY n_failed',(run_id,*coordinate)).fetchall()
            if [row[0] for row in rows]!=list(range(len(counts))) or sum(row[1] for row in rows)!=n:
                raise ValueError('Invalid scan PMF')
            points.append(dict(coordinates=list(coordinate),counts=[r[1] for r in rows]))
    return dict(run_id=run_id,samples=n,seed=seed,threshold=threshold,finished_at=finished,
                source_commit=notes.get('git_sha'),model_provenance=notes.get('stationary_model_provenance'),
                damage_node_ids=notes.get('notional_damage_node_ids'),baseline_counts=counts,points=points,
                database_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def analyze_campaign(profile: str, runs: list[dict]) -> dict:
    """Analyze new counts without comparing them to a selected historical seed."""
    if profile=='damage':
        from pbm.post.robustness import bernoulli_kl_robustness
        rows=[]
        for pe in sorted({r['pe'] for r in runs}):
            cases=sorted((r for r in runs if r['pe']==pe),key=lambda r:r['scenario'])
            for s_lim in ([2,16] if pe>.1 else [2]):
                probabilities=[sum(r['result']['baseline_counts'][s_lim:])/r['samples'] for r in cases]
                rows.append(dict(pe=pe,s_lim=s_lim,robustness=[bernoulli_kl_robustness(probabilities[0],p).value for p in probabilities[1:]]))
        return dict(robustness=rows)
    if profile=='risk':
        from pbm.post.risk_equivalence import analyze_risk_equivalence
        cases=[r for r in runs if r['pe']<=.67]
        return analyze_risk_equivalence([r['pe'] for r in cases],[r['result']['baseline_counts'] for r in cases],pe_crit=.67,lattice_shape=(4,4,4)).to_dict()
    if profile in ('figure3-reduced','figure3-full'):
        from pbm.post.percolation_threshold import AxisScanMetadata,estimate_percolation_threshold
        rows=[]
        for run in runs:
            number=run['case']
            groups=('x','y','z') if number==1 else ('x','y') if number==2 else ('z',)
            fixed=() if number==1 else ('z',) if number==2 else ('x','y')
            metadata=AxisScanMetadata(scan_spec=run['scan'],mode=['1d_xyz','1d_xy','1d_z'][number-1],
                varied_axes=groups,fixed_axes=fixed,varied_axis_groups=(groups,),fixed_axis_values=tuple((a,.8) for a in fixed))
            h=np.array([p['counts'] for p in run['result']['points']])
            grid=[p['coordinates'][2 if number==3 else 0] for p in run['result']['points']]
            fit=estimate_percolation_threshold(grid,h[:,-1]/run['samples'],trials=run['samples'],successes=h[:,-1],scan_metadata=metadata,sensitivity_tolerance=.015)
            rows.append(dict(case=number,threshold=fit.to_dict()))
        return dict(figure3=rows,threshold_tolerance=.015,limits='Stability diagnostic, not a threshold confidence interval.')
    return dict(scope='Probability checks only; no full-grid threshold estimate.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',choices=['smoke','figure3-reduced','figure3-full','figure3-seed-check','damage','risk'])
    parser.add_argument('--out',type=Path,help='New campaign directory; existing directories are never overwritten')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true',help='Print the plan without creating files or running simulations')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    runs=plan(args.profile)
    if args.dry_run:
        print(json.dumps(runs,indent=2))
        return
    if args.out is None:
        parser.error('--out is required unless --dry-run is used')
    directory=args.out.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    record=dict(schema_version=1,profile=args.profile,status='running',
                started_at=datetime.now(timezone.utc).isoformat(),workers=args.workers,
                python=platform.python_version(),platform=platform.system(),
                packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},runs=[])
    def save():
        (directory/'campaign.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf8')
    environment=dict(os.environ,MPLBACKEND='Agg',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    save()
    start=time.perf_counter()
    try:
        for item in runs:
            label=item['label']
            print(f"Running {label}: {item['samples']:,} samples per setting",flush=True)
            input_name=f'input-{label}.db'
            output_name=f"results-pe{item['pe']:g}.db" if args.profile=='damage' else f'results-{label}.db'
            build_and_insert(*item['shape'],directory/input_name,truncate=False)
            with sqlite3.connect(directory/input_name) as con:
                con.execute('UPDATE nodes SET pf_node=?',(item['pf'],))
                con.execute('UPDATE edges SET pe=?,pe_fwd=NULL,pe_rev=NULL,directed=0,blocked=0',(item['pe'],))
            command=['-m','cli.cli_simulation','--db',input_name,'--outdb',output_name,
                     '--samples',str(item['samples']),'--threshold','2','--seed',str(item['seed']),
                     '--mc-workers',str(args.workers)]
            if item.get('scan'):
                command+=['--percolation-scan',item['scan'],'--percolation-workers',str(args.workers)]
            if item['damage']:
                command+=['--damage-coords',';'.join(','.join(map(str,p)) for p in item['damage'])]
            if args.profile=='damage':
                command+=['--scenario-id',str(item['scenario'])]
            with (directory/f'{label}.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,*command],cwd=directory,env=environment,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            record['runs'].append(dict(item,command=['python',*command],output_database=output_name,result=read_result(directory/output_name)))
            save()
        if args.profile=='smoke':
            r=record['runs'][0]['result']
            zero,_,full=r['points']
            expected=binom.pmf(np.arange(9),8,.01)
            observed=np.asarray(zero['counts'])
            if not (np.all(observed>=binom.ppf(.01/(2*9),2000,expected)) and
                    np.all(observed<=binom.ppf(1-.01/(2*9),2000,expected))):
                raise ValueError('Zero-propagation binomial limit failed')
            if any(full['counts'][1:-1]) or full['counts'][-1]!=sum(zero['counts'][1:]):
                raise ValueError('Full-propagation limit failed')
            if len({sum(p['counts'][1:]) for p in r['points']})!=1:
                raise ValueError('Initial-failure count is not scan invariant')
        record['analysis']=analyze_campaign(args.profile,record['runs'])
        for run in record['runs']:
            run['result']['database_sha256']=hashlib.sha256((directory/run['output_database']).read_bytes()).hexdigest()
        record.update(status='completed',elapsed_seconds=time.perf_counter()-start,
                      finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status='failed',error=type(error).__name__)
        raise
    finally:
        save()
    print(f'Completed: {directory}',flush=True)


if __name__=='__main__':
    main()
