"""Family-paired uncertainty on saved original VALID historical-CE results."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from hidate.io import write_json
from run_core_mechanisms import read,rows


METHODS=('Full_LM18','Original_LM10','RRF')
METRICS=('target_mrr','target_recall_at_four')


def paired_panel(samples,*,draws=2000):
    """Equal cell means; resample whole families, retaining all matched rows."""
    families=sorted({s['family_id'] for s in samples});cells=sorted({s['cell'] for s in samples})
    fi={x:i for i,x in enumerate(families)};ci={x:i for i,x in enumerate(cells)}
    count=np.zeros((len(families),len(cells)));total=np.zeros((len(families),len(cells),3,2))
    for row in samples:
        i,j=fi[row['family_id']],ci[row['cell']];count[i,j]+=1
        for m,name in enumerate(METHODS):
            for k,metric in enumerate(METRICS):total[i,j,m,k]+=row['conditions'][name][metric]
    point=(total.sum(axis=0)/count.sum(axis=0)[:,None,None]).mean(axis=0)
    rng=np.random.default_rng(42)
    selected=rng.integers(0,len(families),size=(draws,len(families)))
    weights=np.asarray([np.bincount(r,minlength=len(families)) for r in selected])
    denominators=weights@count;valid=(denominators>0).all(axis=1)
    numerators=np.einsum('bg,gcmk->bcmk',weights[valid],total)
    replicates=(numerators/denominators[valid,:,None,None]).mean(axis=1)
    estimates={name:{metric:{'mean':float(point[m,k]),
        'ci95':np.quantile(replicates[:,m,k],[.025,.975]).tolist()}
        for k,metric in enumerate(METRICS)} for m,name in enumerate(METHODS)}
    contrasts=[]
    for ref in (1,2):
        for k,metric in enumerate(METRICS):
            contrasts.append({'estimate':'Full_LM18','reference':METHODS[ref],'metric':metric,
                'delta':float(point[0,k]-point[ref,k]),
                'ci95':np.quantile(replicates[:,0,k]-replicates[:,ref,k],[.025,.975]).tolist()})
    return {'n_queries':len(samples),'n_families':len(families),'families':families,
        'cell_counts':{c:int(count[:,ci[c]].sum()) for c in cells},
        'n_valid_bootstrap_draws':int(valid.sum()),'estimates':estimates,'comparisons':contrasts}


def analyze(args):
    source=read(args.input/'evaluation.json')
    if not source['complete'] or source['split']!='valid' or source['queries']!=2490:
        raise ValueError('Use complete original VALID history-stage results')
    queries={q['query_id']:q for p in args.queries for q in rows(p) if q['split']=='valid'}
    panels=defaultdict(list);seen=set()
    for row in rows(args.input/'predictions.jsonl'):
        q=queries[row['query_id']]
        if row['query_id'] in seen or row['split']!='valid':raise ValueError('Duplicate/non-VALID predictions')
        if any(row[k]!=q[k] for k in ('family_id','query_variant','stratum','target_status')):
            raise ValueError('Saved VALID metadata differs')
        seen.add(row['query_id'])
        if q['target_status']!='available':continue
        values={name:{k:float(row['conditions'][name]['metrics'][k]) for k in METRICS} for name in METHODS}
        sample={'family_id':q['family_id'],'cell':q['query_variant']+'|'+q['stratum'],'conditions':values}
        panels['four_cell_macro'].append(sample)
        one={**sample,'cell':'query_weighted'}
        panels[q['query_variant']+'/available'].append(one)
        if q.get('source_quote_absent_from_target') is True:
            panels[q['query_variant']+'/quote_absent'].append(one)
    if len(queries)!=2490 or seen!=set(queries):raise ValueError('All original VALID query IDs required')
    result={'complete':True,'split':'valid','role':'development uncertainty; the same VALID selected the recipe',
        'seed':42,'bootstrap_seed':42,'bootstrap_draws':2000,'bootstrap_unit':'original source family',
        'queries':len(seen),'available_queries':len(panels['four_cell_macro']),
        'methods':list(METHODS),'contrast_direction':'Full_LM18 historical CE minus reference historical CE',
        'primary_statistic':'unweighted mean of the four available variant/date cell means',
        'panels':{name:paired_panel(values) for name,values in panels.items()},
        'input':str(args.input/'predictions.jsonl'),'source_job':source['job_id'],
        'training_runs':0,'model_forward_calls':0,'new_TEST_rows':0,
        'job_id':os.environ.get('PBS_JOBID'),'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    primary=result['panels']['four_cell_macro']
    for name in METHODS:
        if not np.isclose(primary['estimates'][name]['target_mrr']['mean'],source['heads'][name]['macro_mrr'],atol=1e-12,rtol=0):
            raise ValueError('Original four-cell macro point estimate changed')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'paired-analysis.json',result)
    with (args.output/'contrasts.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['panel','estimate','reference','metric','delta','ci95_low','ci95_high','n_queries','n_families','n_valid_draws']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for panel,stats in result['panels'].items():
            for c in stats['comparisons']:
                writer.writerow({'panel':panel,**{k:v for k,v in c.items() if k!='ci95'},
                    'ci95_low':c['ci95'][0],'ci95_high':c['ci95'][1],
                    'n_queries':stats['n_queries'],'n_families':stats['n_families'],
                    'n_valid_draws':stats['n_valid_bootstrap_draws']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path('runs/hidate/revision-aware/diagnostics')
    p.add_argument('--input',type=Path,default=root/'history-stage')
    p.add_argument('--output',type=Path,default=root/'history-valid-analysis')
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    args=p.parse_args()
    d=analyze(args);print(json.dumps(d['panels']['four_cell_macro']['comparisons']),flush=True)


if __name__=='__main__':main()
