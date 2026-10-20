"""Compare frozen address heads with the same historical-text CE final stage.

Original VALID only. Each head keeps its own ten bound parents; historical CE
orders those parents and delivers their unchanged requested-date current text.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.lambdamart import lightgbm_module
from hidate.ranking import address_probabilities
from hidate.reranking import PairScoreCache
from evaluate_final import structural_result
from run_core_mechanisms import check_pair_context, prefill, read, rows, save_row
from run_revision_development import deliverable_indices, ordered_indices
from run_revision_dualview import fill_missing


HEADS = ('RRF', 'Original_LM10', 'Dual_relevance_LM11', 'Full_LM18',
         'Lean_noalignment_LM16', 'Original_LR')
METRICS = ('target_recall_at_four', 'target_mrr', 'top_parent_correct',
           'top_revision_correct', 'top_address_correct')


def load_heads(args):
    """Load existing TRAIN fits without fitting or selecting any parameter."""
    models = args.development / 'models'
    fit, controls, oldfit = (read(models / 'fit.json'),
        read(models / 'component-controls.json'), read(args.original_tree.parent / 'fit.json'))
    if not all(d['complete'] and d['seed'] == 42 for d in (fit, controls, oldfit)):
        raise ValueError('Use complete seed42 frozen TRAIN fits')
    if fit['validation_used_for_fitting'] or controls['validation_used_for_fitting'] or oldfit['validation_used']:
        raise ValueError('All address heads must have been fitted without VALID')
    original_features = read(args.protocol)['rank_features']
    if oldfit['feature_names'] != original_features:
        raise ValueError('Original LambdaMART columns differ from the original protocol')
    lgb = lightgbm_module()
    definitions = {
        'Original_LM10': (args.original_tree, original_features),
        'Dual_relevance_LM11': (models/'lambdamart_dual_relevance.txt',
            controls['fits']['lambdamart_dual_relevance']['feature_names']),
        'Full_LM18': (models/'lambdamart_revision.txt',
            fit['fits']['lambdamart_revision']['feature_names']),
        'Lean_noalignment_LM16': (models/'lambdamart_no_alignment.txt',
            controls['fits']['lambdamart_no_alignment']['feature_names'])}
    if [len(v[1]) for v in definitions.values()] != [10, 11, 18, 16]:
        raise ValueError('Requested frozen feature definitions differ')
    trees = {k: lgb.Booster(model_file=str(path)) for k, (path, names) in definitions.items()}
    lr = read(args.original_lr)
    def score(addresses):
        result = {'RRF': np.asarray([a['features']['max_history_rrf'] for a in addresses]),
                  'Original_LR': address_probabilities(lr, [a['features'] for a in addresses])}
        for name, tree in trees.items():
            X = np.asarray([[a['features'][k] for k in definitions[name][1]] for a in addresses])
            result[name] = tree.predict(X, raw_score=True, num_threads=4)
        return result
    return score, {'tree_models': {k: {'path': str(p), 'feature_names': names}
        for k, (p, names) in definitions.items()}, 'original_lr': str(args.original_lr),
        'fit_receipts': [str(models/'fit.json'), str(models/'component-controls.json'),
                         str(args.original_tree.parent/'fit.json')], 'new_fits': 0}


def load_compact(cache, path, queries, chunks):
    count = 0
    for row in rows(path):
        result = row['result']
        for i, ref in enumerate(row['pair_refs']):
            key = (queries[ref['query_id']]['query_text'], chunks[ref['chunk_id']]['text'])
            cache.cache[key] = {'reranker_score': result['scores'][i],
                'reranker_logit': result['logits'][i], 'pair_token_count': result['token_counts'][i]}
            count += 1
    return {'path': str(path), 'pairs': count}


def summarize(values):
    return {'n': len(values), **{k: float(np.mean([v[k] for v in values]))
        for k in ('pool_parent_at10', 'pool_address_at10', 'target_recall_at_four',
                  'target_mrr', 'top_parent_correct', 'native_recall4', 'native_mrr4')}}


def evaluate(args, *, factory=None, head_bundle=None, expected_valid=2490):
    started = time.perf_counter(); out = args.output
    if (out/'evaluation.json').exists() and read(out/'evaluation.json')['complete']:
        return read(out/'evaluation.json')
    prep = read(args.development/'features/preparation.json')
    if not prep['complete'] or prep['seed'] != 42 or prep['queries']['valid'] != expected_valid:
        raise ValueError('Use complete original VALID feature preparation')
    queries = {q['query_id']: q for path in args.queries for q in rows(path) if q['split'] == 'valid'}
    if len(queries) != expected_valid: raise ValueError('Original VALID coverage differs')
    score_heads, provenance = head_bundle or load_heads(args)
    chunks = {c['chunk_id']: c for c in rows(args.index/'chunks.jsonl')}
    parents = {cid: c['parent_passage_id'] for cid, c in chunks.items()}
    prepared, references, seen = [], [], set()
    for row in rows(args.development/'features/valid.jsonl'):
        qid = row['query']['query_id']; q = queries[qid]; addresses = row['addresses']
        if qid in seen or row['query']['split'] != 'valid': raise ValueError('Duplicate/non-VALID feature row')
        if any(row['query'][k] != q[k] for k in ('family_id','query_variant','stratum','target_status','target_passage_id')):
            raise ValueError('Feature metadata differs from original VALID')
        seen.add(qid); scores = score_heads(addresses)
        if set(scores) != set(HEADS): raise ValueError('All six frozen heads are required')
        gold_address = (q['target_key'][0], q['target_key'][2], q['target_key'][3])
        candidates = {}
        for name in HEADS:
            if len(scores[name]) != len(addresses) or not np.isfinite(scores[name]).all():
                raise ValueError('Each frozen head must score the complete same address pool')
            indices = deliverable_indices(addresses, ordered_indices(addresses, scores[name]), parents)
            cc = []
            for rank, i in enumerate(indices, 1):
                a = addresses[i]; h = a['historical_representative_chunk_id']
                if h is None: raise ValueError('Historical representative missing from a retrieved address')
                c = {'address': a['address'], 'historical_chunk_id': h,
                    'current_chunk_id': a['current_best_chunk_id'], 'selection_rank': rank,
                    'head_score': float(scores[name][i])}
                cc.append(c); references.append((qid, h))
            candidates[name] = cc
        bindable = [a for a in addresses if a['status']=='available' and a['current_best_chunk_id'] is not None]
        prepared.append({'query_id': qid, 'candidates': candidates,
            'pool_address_count': len(addresses), 'bindable_address_count': len(bindable),
            'gold_address_in_discovery_pool': any(tuple(a['address'])==gold_address for a in addresses),
            'gold_parent_in_bindable_pool': q['target_status']=='available' and any(
                parents[a['current_best_chunk_id']]==q['target_passage_id'] for a in bindable)})
        if len(seen)%500==0: print(json.dumps({'phase':'preselection','valid_queries':len(seen)}),flush=True)
    if seen != set(queries): raise ValueError('Original VALID feature rows incomplete')
    pin=read(args.stack)['models']['reranker']; dev=read(args.dev_metadata)
    source_scoring=read(args.historical_cache/'scoring.json')
    if not dev['complete'] or not source_scoring['complete']: raise ValueError('Source CE caches are incomplete')
    check_pair_context(dev,pin); check_pair_context(source_scoring,pin)
    out.mkdir(parents=True,exist_ok=True); scoring_path=out/'scoring.json'
    if scoring_path.exists(): check_pair_context(read(scoring_path),pin)
    scoring={'complete':False,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'split':'valid',
        'model_pin':{k:pin[k] for k in ('model_id','revision')},'max_length':1024,'sdpa_backend':'math',
        'input_ID_arrays_saved':False,'training_runs':0,'new_TEST_rows':0}
    write_json(scoring_path,scoring)
    cache=PairScoreCache(None); scoring['original_prefill']=prefill(cache,[args.dev_pairs])
    scoring['historical_prefill']=load_compact(cache,args.historical_cache/'new-ce-pair-scores.jsonl',queries,chunks)
    scoring['work']=fill_missing(cache,references,queries,chunks,pin,out,factory=factory,batch_size=args.batch_size)
    scoring['complete']=True;write_json(scoring_path,scoring)
    cells=defaultdict(lambda:defaultdict(list));panels=defaultdict(list)
    identity_view_counts=defaultdict(int)
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in prepared:
            q=queries[row['query_id']]; results={}
            gold_address=(q['target_key'][0],q['target_key'][2],q['target_key'][3])
            for name,cc in list(row['candidates'].items())+[('RRF_identity_view',row['candidates']['RRF'])]:
                for c in cc:
                    c['historical_ce']=cache.cache[(q['query_text'],chunks[c['historical_chunk_id']]['text'])]
                views=[]; final_scores=[]
                for c in cc:
                    same_parent=parents[c['historical_chunk_id']]==parents[c['current_chunk_id']]
                    view='current' if name=='RRF_identity_view' and same_parent else 'historical'
                    views.append(view)
                    if view=='current':
                        c['current_ce']=cache.cache[(q['query_text'],chunks[c['current_chunk_id']]['text'])]
                    final_scores.append(c[view+'_ce']['reranker_score'])
                    if name=='RRF_identity_view':identity_view_counts[view]+=1
                order=sorted(range(len(cc)),key=lambda i:(-final_scores[i],
                    cc[i]['selection_rank'],cc[i]['current_chunk_id']))[:4]
                final=[chunks[cc[i]['current_chunk_id']] for i in order]
                native=[chunks[c['current_chunk_id']] for c in cc[:4]]
                m=structural_result(q,final,'natural'); n=structural_result(q,native,'natural')
                values={k:m[k] for k in METRICS}
                values.update(pool_parent_at10=q['target_status']=='available' and any(
                    parents[c['current_chunk_id']]==q['target_passage_id'] for c in cc),
                    pool_address_at10=any(tuple(c['address'])==gold_address for c in cc),
                    native_recall4=n['target_recall_at_four'],native_mrr4=n['target_mrr'])
                results[name]={'metrics':values,'preselected10':cc,'final4_common_indices':order,
                    'selected_CE_views':views}
                if q['target_status']=='available':
                    cells[name][q['query_variant']+'|'+q['stratum']].append(values)
                    panels[name+'|'+q['query_variant']+'|available'].append(values)
                    if q.get('source_quote_absent_from_target') is True:
                        panels[name+'|'+q['query_variant']+'|quote_absent'].append(values)
            meta={k:q[k] for k in ('query_id','family_id','query_variant','stratum','target_status','target_asof')}
            save_row(stream,{**meta,'split':'valid','quote_absent':q.get('source_quote_absent_from_target'),
                'discovery_reachability':{k:v for k,v in row.items() if k not in ('query_id','candidates')},'conditions':results})
    heads={name:{'cells':{k:summarize(v) for k,v in group.items()},
        **{'macro_'+k:float(np.mean([summarize(v)[source] for v in group.values()]))
            for k,source in [('mrr','target_mrr'),('recall_at4','target_recall_at_four'),
                ('pool_parent_at10','pool_parent_at10'),('native_mrr','native_mrr4'),('native_recall_at4','native_recall4')]}}
        for name,group in cells.items()}
    if expected_valid==2490:
        prior=read(args.historical_cache/'evaluation.json')
        if not math.isclose(heads['RRF']['macro_mrr'],prior['heads']['Historical_CE_before_binding']['macro_mrr'],abs_tol=1e-12):
            raise ValueError('RRF historical-CE control differs from completed P1')
        if any(len(head['cells'])!=4 for head in heads.values()):raise ValueError('Four original available VALID cells required')
        original=read(args.rank_head_receipt)
        for new,old in [('Original_LM10','LambdaMART_native_own10'),('Original_LR','LR_native_own10'),('RRF','RRF_native_own10')]:
            if not math.isclose(heads[new]['macro_native_mrr'],original['heads'][old]['macro_mrr'],abs_tol=1e-12):
                raise ValueError('Frozen own-ten native control differs: '+new)
    result={'complete':True,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'split':'valid','queries':len(queries),
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started,
        'heads':heads,'panels':{k:summarize(v) for k,v in panels.items()},'scoring':scoring,'frozen_heads':provenance,
        'candidate_policy':'same full original historical address pool; each frozen head owns its ten distinct available bound parents',
        'historical_representative':'highest actual historical RRF hit, chunk-ID tie break; frozen feature cache',
        'final_stage':'same historical-text CE only within each own ten; deliver unchanged bound-current chunks; four distinct parents',
        'identity_view_rule':{'scope':'additional RRF common-ten VALID development control',
            'definition':'current CE iff retrieved historical representative and bound canonical have identical parent_passage_id; otherwise historical CE; always deliver bound current',
            'logical_view_counts':dict(identity_view_counts),'new_current_CE_pairs':0,
            'uses_source_date_query_stratum_or_gold':False,'threshold_search':False},
        'feature_source':str(args.development/'features/valid.jsonl'),'training_runs':0,'new_TEST_rows':0,
        'selection':'all six fixed heads reported; no model/policy selected by this worker'}
    write_json(out/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path('runs/hidate/revision-aware')
    p.add_argument('--development',type=Path,default=root/'native-development')
    p.add_argument('--output',type=Path,default=root/'diagnostics/history-stage')
    p.add_argument('--historical-cache',type=Path,default=root/'diagnostics/revision-dualview')
    p.add_argument('--rank-head-receipt',type=Path,default=root/'diagnostics/rank-head-valid/evaluation.json')
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    p.add_argument('--index',type=Path,default=Path('data/index/hidate'))
    p.add_argument('--protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--original-lr',type=Path,default=Path('runs/hidate/final/ranker-used.json'))
    p.add_argument('--original-tree',type=Path,default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--dev-metadata',type=Path,default=Path('runs/hidate/dev/scoring.json'))
    p.add_argument('--dev-pairs',type=Path,default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    d=evaluate(args);print(json.dumps({'complete':d['complete'],'queries':d['queries'],'heads':d['heads']}),flush=True)


if __name__=='__main__':main()
