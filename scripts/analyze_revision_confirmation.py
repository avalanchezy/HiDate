"""Family-paired estimates for the already frozen additional-family test."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

from hidate.io import write_json
from run_revision_scale import read, rows


METRICS=('target_mrr','target_recall_at_four','top_parent_correct')


def paired(records, methods, selected, *, macro_cells):
    families=sorted({r['family_id'] for r in records})
    cells=sorted({r['query_variant']+'|'+r['stratum'] for r in records}) if macro_cells else ['all']
    fi={f:i for i,f in enumerate(families)};ci={c:i for i,c in enumerate(cells)}
    sums=np.zeros((len(families),len(cells),len(methods),len(METRICS)))
    counts=np.zeros((len(families),len(cells)))
    for r in records:
        i=fi[r['family_id']];j=ci[r['query_variant']+'|'+r['stratum']] if macro_cells else 0
        counts[i,j]+=1
        for k,name in enumerate(methods):
            sums[i,j,k]+=[r['conditions'][name]['structural'][m] for m in METRICS]
    observed=(sums.sum(axis=0)/counts.sum(axis=0)[:,None,None]).mean(axis=0)
    rng=np.random.default_rng(42);draws=[]
    for ids in rng.integers(0,len(families),size=(2000,len(families))):
        den=counts[ids].sum(axis=0)
        if np.all(den>0):draws.append((sums[ids].sum(axis=0)/den[:,None,None]).mean(axis=0))
    draws=np.asarray(draws);si=methods.index(selected)
    contrasts=[];estimates={}
    for k,name in enumerate(methods):
        estimates[name]={m:{'mean':float(observed[k,mi]),
            'ci95':np.quantile(draws[:,k,mi],[.025,.975]).tolist()} for mi,m in enumerate(METRICS)}
        if name==selected:continue
        for mi,m in enumerate(METRICS):
            delta=draws[:,si,mi]-draws[:,k,mi]
            contrasts.append({'estimate':selected,'reference':name,'metric':m,
                'delta':float(observed[si,mi]-observed[k,mi]),
                'ci95_low':float(np.quantile(delta,.025)),'ci95_high':float(np.quantile(delta,.975))})
    return {'n':len(records),'families':len(families),'cells':len(cells),'valid_draws':len(draws),
        'estimates':estimates,'contrasts':contrasts}


def analyze(directory):
    report=read(directory/'evaluation.json')
    if not report['complete']:raise ValueError('Complete frozen confirmation predictions first')
    records=[r for r in rows(directory/'predictions.jsonl') if r['target_status']=='available']
    methods=sorted(report['panels']);selected=report['selected_method']
    panels={'available_four_cell_macro':paired(records,methods,selected,macro_cells=True)}
    for variant in ('exact_quote','generated_paraphrase'):
        subset=[r for r in records if r['query_variant']==variant]
        panels[variant]=paired(subset,methods,selected,macro_cells=False)
        panels[variant+'|quote_absent']=paired([r for r in subset if r['quote_absent']],methods,selected,macro_cells=False)
    # Exact aggregate replay checks the denominators used in the main objective.
    for name,metrics in panels['available_four_cell_macro']['estimates'].items():
        for m,value in metrics.items():
            if not np.isclose(value['mean'],report['available_four_cell_macro'][name][m],atol=1e-12,rtol=0):
                raise ValueError('Saved macro metric differs from paired input aggregation')
    role={'regression_scale':'original TEST at 1000-law scale; regression and transfer',
          'regression_original':'original TEST at 120-law scale; regression'}.get(
              report.get('dataset'),'frozen additional-family confirmation')
    result={'complete':True,'role':role,'selected_method':selected,
        'seed':42,'draws':2000,'unit':'paired whole family; both variants and dates stay together',
        'panels':panels,'selection_updates':0,'new_training_runs':0,'new_model_predictions':0,
        'job_id':os.environ.get('PBS_JOBID')}
    write_json(directory/'paired-analysis.json',result)
    contrasts=[{'panel':p,**c} for p,v in panels.items() for c in v['contrasts']]
    with (directory/'contrasts.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]));writer.writeheader();writer.writerows(contrasts)
    transitions=[]
    for r in records:
        ours=r['conditions'][selected]['structural']['target_mrr']
        for name in ('common_Historical_CE','RRF_native','Original_LM10_historical_CE','Original_HiDate_LR_CE'):
            baseline=r['conditions'][name]['structural']['target_mrr']
            if ours!=baseline:
                transitions.append({'query_id':r['query_id'],'family_id':r['family_id'],
                    'variant':r['query_variant'],'stratum':r['stratum'],'quote_absent':r['quote_absent'],
                    'method':selected,'reference':name,'method_mrr':ours,'reference_mrr':baseline,'delta':ours-baseline})
    if transitions:
        with (directory/'rank-transitions.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(transitions[0]));writer.writeheader()
            writer.writerows(sorted(transitions,key=lambda r:(-r['delta'],r['query_id'],r['reference'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/hidate/revision-aware/confirmation/final'))
    args=p.parse_args()
    result=analyze(args.input)
    print(json.dumps({'complete':result['complete'],'method':result['selected_method'],
        'n':result['panels']['available_four_cell_macro']['n']}),flush=True)
