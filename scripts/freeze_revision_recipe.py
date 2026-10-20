"""Select one declared HiDate recipe using completed original VALID outcomes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from hidate.io import write_json


ROOT=Path('runs/hidate/revision-aware')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def choose(candidates):
    # Treat floating-point replay differences as ties, not model improvements.
    best=max(c['mrr4'] for c in candidates)
    tied=[c for c in candidates if best-c['mrr4']<=1e-12]
    return min(tied,key=lambda c:(c['CE_views_per_candidate'],c['feature_count'],c['method']))


def freeze(root):
    target=root/'frozen-recipe.json'
    if target.exists():return read(target)
    native=read(root/'native-development/valid-components/evaluation.json')
    historical=read(root/'diagnostics/history-stage/evaluation.json')
    ce=read(root/'ce-training/evaluation.json')
    if not all(d['complete'] and d['split']=='valid' and d['queries']==2490 for d in (native,historical,ce)):
        raise ValueError('All declared original VALID development comparisons must finish')
    candidates=[]
    mapping={'history_rrf_native':('RRF_native',0),
        'original_lambdamart_native':('Original_LM10_native',10),
        'lambdamart_dual_relevance':('Dual_relevance_LM11_native',11),
        'lambdamart_revision':('Full_LM18_native',18),
        'lambdamart_no_alignment':('Lean_noalignment_LM16_native',16),
        'original_lr_native':('Original_LR_native',10)}
    for key,(name,count) in mapping.items():
        result=native['conditions'][key+'|own_pool']
        candidates.append({'method':name,'mrr4':result['mrr4'],'recall4':result['recall4'],
            'CE_views_per_candidate':0,'feature_count':count,'candidate_pool':'own_address_head_ten'})
    for key,result in historical['heads'].items():
        count={'RRF':0,'Original_LM10':10,'Dual_relevance_LM11':11,'Full_LM18':18,
               'Lean_noalignment_LM16':16,'Original_LR':10,'RRF_identity_view':0}[key]
        name=key if key=='RRF_identity_view' else key+'_historical_CE'
        candidates.append({'method':name,'mrr4':result['macro_mrr'],
            'recall4':result['macro_recall_at4'],'CE_views_per_candidate':1,
            'feature_count':count,'candidate_pool':'own_address_head_ten'})
    for key,result in ce['heads'].items():
        name='ce_'+key if key.startswith(('lambdamart_','gate_')) else 'common_'+key
        count=13 if key.endswith('base13') else 20 if key.endswith('revision20') else 0
        views=0 if key=='RRF_native' else 1 if key in ('Historical_CE','Current_CE') else 2
        candidates.append({'method':name,'mrr4':result['macro_mrr'],'recall4':result['macro_recall_at4'],
            'CE_views_per_candidate':views,'feature_count':count,'candidate_pool':'common_historical_RRF_ten'})
    winner=choose(candidates)
    result={'complete':True,'selected_method':winner['method'],'selected_validation_result':winner,
        'selection_split':'original_valid','selection_metric':'available four-cell macro target-parent MRR@4',
        'tie_rule':'numerical ties within 1e-12: fewer CE views, then fewer features, then method name',
        'confirmation_results_used':False,'original_test_results_used_for_this_choice':False,
        'seed':42,'training_split':'original TRAIN; one fixed fit per declared variant',
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ.get('PBS_JOBID'),
        'candidate_results':sorted(candidates,key=lambda c:-c['mrr4']),
        'development_sources':['native-development/valid-components/evaluation.json',
            'diagnostics/history-stage/evaluation.json','ce-training/evaluation.json'],
        'evidence_budget':4,'candidate_budget':10,'discovery_chunk_budget':50,
        'new_family_manifest':'data/processed/hidate-holdout/family-manifest.json',
        'confirmation_rule':'evaluate every predeclared strong control; selected method stays fixed'}
    write_json(target,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    result=freeze(args.root)
    print(json.dumps({'selected':result['selected_method'],'valid':result['selected_validation_result']},indent=2))
