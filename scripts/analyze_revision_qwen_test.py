"""Family-paired original-TEST comparisons of frozen Qwen methods."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from hidate.io import write_json
from analyze_revision_confirmation import paired
from run_revision_scale import read,rows


def merge_controls(main_records, control_records, selected):
    controls={r['query_id']:r for r in control_records}
    if set(controls)!={r['query_id'] for r in main_records}:
        raise ValueError('Main and CE controls must contain identical original TEST queries')
    merged=[]
    for row in main_records:
        control=controls[row['query_id']]
        for field in ('family_id','query_variant','stratum','target_asof','target_status','quote_absent'):
            if row[field]!=control[field]:
                raise ValueError(f'Paired grouping differs for {row["query_id"]}: {field}')
        merged.append({**row,'conditions':{selected:row['conditions'][selected],**control['conditions']}})
    return merged


def analyze(directory, *, controls=False):
    report=read(directory/'evaluation.json')
    if not report['complete'] or report['queries']!=3172:
        raise ValueError('Complete original3172 Qwen TEST predictions required')
    selected=report['selected_condition']
    output=directory
    methods=sorted(name for name in report['methods'] if name.endswith('__historical_ce'))
    records=list(rows(directory/'predictions.jsonl'))
    expected=report['available_four_cell_macro']
    if controls:
        output=directory/'ce-controls'
        extra=read(output/'evaluation.json')
        if not extra['complete'] or extra['queries']!=report['queries']:
            raise ValueError('Complete matched original TEST CE controls required')
        methods=sorted([selected,*extra['methods']])
        records=merge_controls(records,list(rows(output/'predictions.jsonl')),selected)
        expected={selected:expected[selected],**extra['available_four_cell_macro']}
    records=[r for r in records if r['target_status']=='available']
    panels={'available_four_cell_macro':paired(records,methods,selected,macro_cells=True)}
    for variant in ('exact_quote','generated_paraphrase'):
        subset=[r for r in records if r['query_variant']==variant]
        panels[variant]=paired(subset,methods,selected,macro_cells=False)
        panels[variant+'|quote_absent']=paired([r for r in subset if r['quote_absent']],methods,selected,macro_cells=False)
    for name,metrics in panels['available_four_cell_macro']['estimates'].items():
        for metric,value in metrics.items():
            if not np.isclose(value['mean'],expected[name][metric],atol=1e-12,rtol=0):
                raise ValueError('Paired original TEST objective differs from the completed evaluator')
    result={'complete':True,'role':'original TEST regression and Qwen backbone transfer; no new-family confirmation',
        'selected_method':selected,'methods':methods,
        'comparison':'own-Qwen TRAIN CE13/20 controls' if controls else 'same historical CE address heads',
        'seed':42,'draws':2000,
        'unit':'paired whole family; original variants and dates remain together; four-cell macro retained',
        'panels':panels,'selection_updates':0,'new_training_runs':0,'new_model_predictions':0,
        'job_id':os.environ.get('PBS_JOBID')}
    write_json(output/'paired-analysis.json',result)
    contrasts=[{'panel':name,**c} for name,panel in panels.items() for c in panel['contrasts']]
    with (output/'paired-contrasts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]));writer.writeheader();writer.writerows(contrasts)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/hidate/revision-aware/transfer/qwen-test'))
    p.add_argument('--controls',action='store_true',help='Compare main17 frozen method against completed four CE controls')
    args=p.parse_args()
    analyze(args.input,controls=args.controls)
