"""Recompute scientific checks from curated counts; no Monte Carlo runs.

Run from a source checkout or unpacked source distribution after installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom
from pbm.post.percolation_threshold import axis_scan_metadata_from_mapping, estimate_percolation_threshold
from pbm.post.risk_equivalence import analyze_risk_equivalence
from pbm.post.robustness import bernoulli_kl_robustness

ROOT=Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pmf(value: list, samples: int | list) -> np.ndarray:
    raw=np.asarray(value)
    require(raw.dtype.kind in 'iu' and raw.ndim in (1,2), 'PMF must contain integer counts')
    require(raw.shape[-1]==65 and bool((raw>=0).all()), 'PMF must have 65 nonnegative bins')
    counts=raw.astype(np.int64)
    require(bool(np.all(counts.sum(axis=-1)==samples)), 'PMF counts do not sum to samples')
    return counts


def verify_hashes(directory: Path) -> None:
    inventory=json.loads((directory/'SHA256SUMS.json').read_text(encoding='utf8'))
    require(bool(inventory),'Empty artifact inventory')
    for name,expected in inventory.items():
        path=(directory/name).resolve()
        require(path.is_relative_to(directory.resolve()),'Artifact path leaves evidence directory')
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,
                f'Artifact hash mismatch: {name}')


def validate(directory: Path, reference_path: Path, plot_dir: Path | None=None) -> dict:
    verify_hashes(directory)
    def read(name): return json.loads((directory/'data'/name).read_text(encoding='utf8'))
    reference=json.loads(reference_path.read_text(encoding='utf8'))
    figure3,damage,risk,provenance=(read(n) for n in ('figure3.json','damage.json','risk.json','provenance.json'))
    require(len({d['source_commit'] for d in (figure3,damage,risk,provenance)})==1,'Campaign source revisions disagree')
    require(figure3['seed']==damage['seed']==risk['seed']==123,'Unexpected primary seed')
    require(figure3['independent_seed']['seed']==456,'Unexpected independent seed')
    require(all(d['lattice_shape']==[4,4,4] and d['pf_node']==1e-4 for d in (figure3,damage,risk)), 'Unexpected manuscript inputs')
    require([c['case'] for c in figure3['cases']]==[1,2,3], 'Missing Figure 3 case')
    require([c['case'] for c in figure3['independent_seed']['cases']]==[1,2,3], 'Missing independent-seed case')
    require(figure3['family_alpha']==.01 and figure3['threshold_tolerance']==.015,'Changed acceptance criteria')
    summary=dict(source_commit=figure3['source_commit'],artifact_hashes_verified=True,figure3=[],damage=[],risk={})
    exact_any=-math.expm1(64*math.log1p(-1e-4))
    family_bins=183*65
    independent_bins=19*65
    if plot_dir:
        import matplotlib
        matplotlib.use('Agg')
        from pbm.post.percolation_plots import make_1d_plot
        plot_dir.mkdir(parents=True,exist_ok=True)
    for case,old,ind in zip(figure3['cases'],reference['figure3'],figure3['independent_seed']['cases']):
        n,number=case['samples'],case['case']
        require(n==[1_000_000,2_000_000,2_000_000][number-1],'Unexpected reduced sample count')
        counts=pmf(case['pmf_counts'],n)
        retained=pmf(old['pmf_counts'],old['samples'])
        grid=np.asarray(case['grid'])
        require(counts.shape==retained.shape and np.array_equal(grid,old['grid']), 'Figure 3 grid mismatch')
        require(case['metadata']==old['metadata'],'Axis configuration mismatch')
        # CDF/SF evaluation is equivalent to inclusive equal-tail quantile
        # regions, and avoids expensive repeated inverse-CDF searches.
        tail=.01/(2*family_bins)
        require(bool(np.all(hypergeom.cdf(counts,old['samples'],retained,n)>=tail)
                     and np.all(hypergeom.sf(counts-1,old['samples'],retained,n)>tail)),
                f'Case {number}: retained-prefix PMF comparison failed')
        any_counts=counts[:,1:].sum(axis=1)
        require(len(set(any_counts.tolist()))==1,'Any-failure count varies within scan')
        require(binom.ppf(.005/3,n,exact_any)<=any_counts[0]<=binom.ppf(1-.005/3,n,exact_any),'Any-failure probability check failed')
        if number==1:
            expected=binom.pmf(np.arange(65),64,1e-4)
            require(bool(np.all(counts[0]>=binom.ppf(.01/(2*65),n,expected)) and np.all(counts[0]<=binom.ppf(1-.01/(2*65),n,expected))), 'Zero-propagation limit failed')
            require(counts[-1,1:-1].sum()==0 and counts[-1,-1]==any_counts[0],'Full-propagation limit failed')
        metadata=axis_scan_metadata_from_mapping(case['metadata'])
        fit=estimate_percolation_threshold(grid,counts[:,-1]/n,trials=n,successes=counts[:,-1],scan_metadata=metadata,sensitivity_tolerance=.015)
        historical=estimate_percolation_threshold(grid,retained[:,-1]/old['samples'],trials=old['samples'],successes=retained[:,-1],scan_metadata=metadata,sensitivity_tolerance=.015)
        require(fit.identified and historical.identified,'Threshold not identified/stable')
        require(abs(fit.threshold-historical.threshold)<=.015,'Threshold agreement tolerance exceeded')
        require(abs(fit.threshold-case['recorded_threshold'])<1e-12,'Recorded threshold does not match counts')
        fresh=pmf(ind['pmf_counts'],figure3['independent_seed']['samples'])
        require(len(ind['grid'])==[6,6,7][number-1], 'Independent window size mismatch')
        previous=[]
        for pe in ind['grid']:
            index=np.flatnonzero(np.isclose(grid,pe,atol=1e-12,rtol=0))
            require(index.size==1,'Independent point missing from primary grid')
            previous.append(counts[index[0]])
        previous=np.asarray(previous)
        total=fresh+previous
        pvalue=np.minimum(1.,2*np.minimum(hypergeom.cdf(fresh,n+1_000_000,total,1_000_000),hypergeom.sf(fresh-1,n+1_000_000,total,1_000_000)))
        require(bool(np.all(pvalue>=.01/independent_bins)), 'Independent-seed comparison failed')
        summary['figure3'].append(dict(case=number,threshold=fit.threshold,retained_threshold=historical.threshold,
            pmf_bins_verified=int(counts.size),independent_bins_verified=int(fresh.size),stability_passed=True))
        if plot_dir:
            make_1d_plot(grid,any_counts/n,counts[:,1]/n,counts[:,-1]/n,plot_dir/f'case{number}.png',threshold_result=fit,plot_mode='standard')
    require(figure3['cases'][1]['baseline_pmf_counts']==figure3['cases'][2]['baseline_pmf_counts'],'Same-input baselines disagree')
    for case in damage['cases']:
        old=next(c for c in reference['damage'] if c['pe']==case['pe'])
        require([r['scenario'] for r in case['runs']]==[0,1,2,3],'Missing damage scenario')
        histograms=[]
        for run,historical in zip(case['runs'],old['runs']):
            counts=pmf(run['pmf_counts'],run['samples'])
            require(np.array_equal(counts,historical['pmf_counts']) and run['damage_node_ids']==historical['damage_node_ids'],'Fresh damage counts/configuration differ from retained run')
            histograms.append(counts)
        h=np.asarray(histograms)
        for s_lim in ([2,16] if case['pe']>.1 else [2]):
            probabilities=h[:,s_lim:].sum(axis=1)/1_000_000
            values=[bernoulli_kl_robustness(probabilities[0],p).value for p in probabilities[1:]]
            require(values[0]>values[1]>values[2],'Robustness ordering failed')
            summary['damage'].append(dict(pe=case['pe'],s_lim=s_lim,robustness=values))
    by_pe={r['pe']:r for r in risk['runs']}
    require(len(by_pe)==len(risk['runs']) and set(by_pe)=={0.,.08,.085,.09,.095,.1,.2,.4,.67,1.},'Missing risk/plausibility point')
    grid=np.array([0.,.08,.085,.09,.095,.1,.2,.4,.67])
    h=pmf([by_pe[p]['counts'] for p in grid],[by_pe[p]['samples'] for p in grid])
    result=analyze_risk_equivalence(grid,h,pe_crit=.67,lattice_shape=(4,4,4))
    require(result.numerical.identified,'Risk-equivalence crossing unidentified')
    totals=h.sum(axis=1)
    residual=h[:,1]/totals-(h@np.arange(65)/totals-h[:,1:].sum(axis=1)/totals)
    i,j=[int(np.flatnonzero(np.isclose(grid,p,atol=1e-12,rtol=0))[0]) for p in result.numerical.root_bracket]
    direct=grid[i]-residual[i]*(grid[j]-grid[i])/(residual[j]-residual[i])
    require(abs(direct-result.numerical.estimate)<1e-12,'Independent Equation 19 interpolation failed')
    require(abs(direct-.09352167732764706)<1e-12 and abs(result.analytical.estimate-.093867580)<5e-10,'Recorded risk estimates differ')
    limits=pmf([by_pe[p]['counts'] for p in [0.,.2,.4,.67,1.]],1_000_000)
    require(len(set(limits[:,1:].sum(axis=1).tolist()))==1,'Plausibility any-failure invariance failed')
    require(limits[-1,1:-1].sum()==0 and limits[-1,-1]==limits[0,1:].sum(),'Plausibility full-propagation limit failed')
    expected=binom.pmf(np.arange(65),64,1e-4)
    require(bool(np.all(limits[0]>=binom.ppf(.01/(2*65),1_000_000,expected)) and np.all(limits[0]<=binom.ppf(1-.01/(2*65),1_000_000,expected))), 'Plausibility zero-propagation limit failed')
    tails=np.cumsum(limits[:,::-1],axis=1)[:,::-1]
    require(bool(np.all(np.diff(tails[:,1:],axis=0)>=0) and np.all(np.diff(limits[:,1])<=0)), 'Observed plausibility monotonicity failed')
    summary['risk']=dict(numerical=direct,analytical=result.analytical.estimate,bracket=result.numerical.root_bracket,
                        nominal_monte_carlo_interval=result.numerical.monte_carlo_interval,plausibility_passed=True)
    summary['all_checks_passed']=True
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT/'reproduction')
    parser.add_argument('--reference',type=Path,default=ROOT/'tests/scientific/data/manuscript_reference.json')
    parser.add_argument('--out',type=Path,help='Optional JSON report destination')
    parser.add_argument('--plot-dir',type=Path,help='Optional portable Figure 3 PNG exports; no LaTeX needed')
    args=parser.parse_args()
    summary=validate(args.evidence,args.reference,args.plot_dir)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
