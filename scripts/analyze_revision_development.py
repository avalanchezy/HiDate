"""Family-paired four-cell VALID analysis of saved revision-development outputs."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

from hidate.io import write_json
from run_revision_development import rows, read


def analyze(directory):
    report = read(directory / 'evaluation.json')
    records = [r for r in rows(directory / 'predictions.jsonl') if r['target_status'] == 'available']
    families = sorted({r['family_id'] for r in records})
    cells = sorted({r['query_variant'] + '|' + r['stratum'] for r in records})
    conditions = sorted(report['conditions'])
    fi, ci = {f:i for i,f in enumerate(families)}, {c:i for i,c in enumerate(cells)}
    totals = np.zeros((len(families), len(cells), len(conditions), 2))
    counts = np.zeros((len(families), len(cells)))
    for r in records:
        i = fi[r['family_id']]; j = ci[r['query_variant'] + '|' + r['stratum']]
        counts[i,j] += 1
        for k, condition in enumerate(conditions):
            totals[i,j,k] += [r['conditions'][condition]['mrr4'], r['conditions'][condition]['recall4']]
    observed = (totals.sum(axis=0) / counts.sum(axis=0)[:,None,None]).mean(axis=0)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(families), size=(2000, len(families)))
    draws = []
    for sample in samples:
        den = counts[sample].sum(axis=0)
        if np.all(den > 0):
            draws.append((totals[sample].sum(axis=0) / den[:,None,None]).mean(axis=0))
    draws = np.asarray(draws)
    comparisons = []
    references = ['original_lambdamart_native','history_rrf_native','lambdamart_base',
                  'lambdamart_dual_relevance','lambdamart_no_alignment','lambdamart_no_query_change','gate_revision']
    for scope in ('own_pool','common_rrf10'):
        estimate = 'lambdamart_revision|' + scope; ei = conditions.index(estimate)
        for reference in references:
            key = reference + '|' + scope
            if key not in conditions:
                continue
            ri = conditions.index(key)
            for mi, metric in enumerate(('mrr4','recall4')):
                diff = draws[:,ei,mi] - draws[:,ri,mi]
                comparisons.append({'scope':scope,'estimate':'lambdamart_revision','reference':reference,
                    'metric':metric,'delta':float(observed[ei,mi]-observed[ri,mi]),
                    'ci95_low':float(np.quantile(diff,.025)),'ci95_high':float(np.quantile(diff,.975))})
    result = {'complete':True,'split':'original VALID','available_queries':len(records),
        'families':len(families),'cell_count':len(cells),'bootstrap_draws':2000,
        'valid_draws':len(draws),'bootstrap_seed':42,'job_id':os.environ.get('PBS_JOBID'),
        'unit':'paired whole family, preserving both query variants and original four-cell macro objective',
        'role':'development diagnostics; not a confirmatory test estimate',
        'comparisons':comparisons,'new_training_runs':0,'new_model_predictions':0}
    write_json(directory / 'paired-analysis.json', result)
    with (directory / 'contrasts.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(comparisons[0]))
        writer.writeheader();writer.writerows(comparisons)
    # Save readable rank transitions for later case selection; no source/gold text feeds a model.
    transitions = []
    for r in records:
        for scope in ('own_pool','common_rrf10'):
            a = r['conditions']['lambdamart_revision|' + scope]['mrr4']
            b = r['conditions']['original_lambdamart_native|' + scope]['mrr4']
            if a != b:
                transitions.append({'query_id':r['query_id'],'family_id':r['family_id'],
                    'variant':r['query_variant'],'stratum':r['stratum'],'scope':scope,
                    'original_mrr':b,'revision_mrr':a,'delta':a-b})
    with (directory / 'rank-transitions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(transitions[0]))
        writer.writeheader();writer.writerows(sorted(transitions,key=lambda r:(-r['delta'],r['query_id'],r['scope'])))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('runs/hidate/revision-aware/native-development/valid-components'))
    args=parser.parse_args()
    print(json.dumps(analyze(args.input)),flush=True)
