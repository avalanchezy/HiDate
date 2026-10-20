"""Supervised same-history-pool LambdaMART baseline; no raw-score/probability fusion."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from hidate.io import write_json, write_jsonl
from hidate.lambdamart import (FIXED_PARAMETERS, grouped_training_data, fit_model,
    load_model, rank_addresses, canonical_candidates)
from hidate.reranking import PairScoreCache
from hidate.selection import selected_evidence
from evaluate_final import compact_candidate, structural_result
from run_core_mechanisms import (read, rows, save_row, reachability, metric_panels,
                                     LazyReranker, check_pair_context, prefill)


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs(args):
    queries = {}
    for path in args.queries:
        for q in rows(path):
            if q['query_id'] in queries:
                raise ValueError('Duplicate original query ID')
            queries[q['query_id']] = q
    protocol, reference = read(args.protocol), read(args.reference_ranker)
    if (protocol['seed'] != 42 or len(protocol['rank_features']) != 10
            or not reference['complete'] or reference['random_state'] != 42
            or protocol['rank_features'] != reference['feature_names']):
        raise ValueError('Use seed42 and the same ten features as the completed history LR')
    if [protocol[k] for k in ('discovery_top_k','reranker_top_k','reader_max_passages')] != [50,10,4]:
        raise ValueError('Keep the original50/10/4 budgets')
    return queries, protocol, reference


def split_discovery(paths, queries, split):
    expected = {key for key,q in queries.items() if q['split'] == split}
    seen = set()
    for path in paths:
        if not read(Path(path).parent/'discovery.json')['complete']:
            raise ValueError('Source history discovery must be complete')
        for row in rows(path):
            q = queries[row['query_id']]
            if q['split'] != split:
                continue
            if q['query_id'] in seen:
                raise ValueError('Duplicate original discovery query')
            if row.get('candidate_pool') == 'asof':
                raise ValueError('LambdaMART baseline requires the original history address pool')
            if any(row[k] != q[k] for k in ('split','query_variant')):
                raise ValueError('History discovery metadata differs from original queries')
            seen.add(q['query_id']); yield row
    if seen != expected:
        raise ValueError(f'Incomplete original {split} IDs: {len(seen)}/{len(expected)}')


def fit_phase(args, *, expected_train=14030):
    queries, protocol, reference = inputs(args)
    if sum(q['split']=='train' for q in queries.values()) != expected_train:
        raise ValueError('LambdaMART requires the original complete TRAIN')
    output = Path(args.output_dir)/'ranker'
    fit_path, model_path = output/'fit.json', output/'model.txt'
    if fit_path.exists():
        saved = read(fit_path)
        if (not saved['complete'] or saved['parameters'] != FIXED_PARAMETERS
                or saved['training_statistics']['query_count'] != expected_train
                or not model_path.exists()):
            raise ValueError('Existing one-fit LambdaMART receipt differs')
        return saved
    if model_path.exists():
        raise ValueError('A model exists without a complete fitting receipt; inspect before replacing it')
    started = time.perf_counter()
    X,y,sizes,groups,stats = grouped_training_data(
        split_discovery(args.train_discovery,queries,'train'),queries,protocol['rank_features'])
    original = reference['training_statistics']
    for field in ('query_count','candidate_count','positive_addresses','positive_unavailable'):
        if stats[field] != original[field]:
            raise ValueError('TRAIN rows/labels differ from original LR: '+field)
    if set(stats['queries_by_variant']) != {'exact_quote','generated_paraphrase'}:
        raise ValueError('Both original TRAIN variants are required')
    fitted = fit_model(X,y,sizes,protocol['rank_features'],model_path)
    write_jsonl(output/'train-groups.jsonl',groups)
    fitted.update(complete=True,seed=42,training_runs=1,training_statistics=stats,
        job_id=os.environ.get('PBS_JOBID'),finished_at_utc=now(),wall_seconds=time.perf_counter()-started,
        train_discovery_paths=[str(p) for p in args.train_discovery],
        reference_ranker_path=str(args.reference_ranker),baseline_role='supervised learning-to-rank baseline, not a core module ablation')
    write_json(fit_path,fitted)
    return fitted


def reference_lr(args, queries, expected_test):
    expected = {key for key,q in queries.items() if q['split']=='test'}
    if len(expected) != expected_test:
        raise ValueError('Use all original natural TEST queries')
    result = {}
    for row in rows(args.reference_retrieval):
        if row['evaluation_scope'] != 'natural': continue
        key = row['query_id']
        if key not in expected or key in result:
            raise ValueError('Original LR natural TEST IDs differ')
        if any(row[k] != queries[key][k] for k in ('query_variant','target_asof','source_role','target_key')):
            raise ValueError('Original LR TEST metadata differs from the frozen queries')
        result[key] = row['conditions']['history_learned']
    if set(result) != expected:
        raise ValueError('Original LR reference is incomplete')
    return result


def same_history_pool(discovery, reference, features):
    left = {tuple(a['address']):a for a in discovery['addresses']}
    right = {tuple(a['address']):a for a in reference['address_diagnostics']}
    if set(left) != set(right):
        raise ValueError('LambdaMART and saved LR candidate address pools differ')
    for key,a in left.items():
        b = right[key]
        if a['current_best_chunk_id'] != b['current_best_chunk_id'] or a['status'] != b['status']:
            raise ValueError('Canonical target chunks/binding differ from saved LR')
        if any(a['features'][name] != b['features'][name] for name in features):
            raise ValueError('History feature values differ from saved LR')


def output_condition(query, discovery, candidates, evidence, ranked, score_kind):
    return {'score_kind':score_kind,'fusion':'none','alpha':None,
        'ranked_addresses':ranked,'native_top10_addresses':ranked[:10],'native_top4_addresses':ranked[:4],
        'preselected_candidates':[compact_candidate(c) for c in candidates],
        'evidence':[compact_candidate(c) for c in evidence],
        'structural':structural_result(query,evidence,'natural'),
        'stage_reachability':reachability(query,discovery,{'reranked_candidates':candidates},evidence),
        'support_calibration':'not applicable; raw ranking margins are not probabilities'}


def add_metrics(cells, query, conditions):
    subsets=['all_queries']
    if query['target_status']=='available':
        subsets.append('available')
        flag=query.get('source_quote_absent_from_target')
        if flag is not None: subsets.append('quote_absent' if flag else 'quote_present')
    else: subsets.append('unavailable')
    for label,condition in conditions.items():
        for subset in subsets:
            cells['|'.join((label,query['query_variant'],subset))].append(condition['structural'])


def native_phase(args, *, expected_test=3172):
    started=time.perf_counter()
    queries,protocol,_=inputs(args)
    fitted=read(Path(args.output_dir)/'ranker/fit.json')
    if not fitted['complete'] or fitted['parameters'] != FIXED_PARAMETERS:
        raise ValueError('The one frozen LambdaMART fit must be complete')
    model=load_model(Path(args.output_dir)/'ranker/model.txt')
    chunks={c['chunk_id']:c for c in rows(args.chunks)}
    references=reference_lr(args,queries,expected_test)
    output=Path(args.output_dir)/'native';output.mkdir(parents=True,exist_ok=True)
    summary={'complete':False,'seed':42,'started_at_utc':now(),'job_id':os.environ.get('PBS_JOBID'),
        'conditions':['LambdaMART_native','LR_native'],'queries':0,'fusion':'none',
        'new_neural_model_calls':0,'tree_predict_calls':0,'pair_budget':10,'final_parent_budget':4,
        'fit_path':str(Path(args.output_dir)/'ranker/fit.json'),'reference_retrieval':str(args.reference_retrieval)}
    write_json(output/'evaluation.json',summary);cells=defaultdict(list)
    with (output/'retrieval.jsonl').open('w',encoding='utf-8') as stream:
        for discovery in split_discovery(args.test_discovery,queries,'test'):
            q=queries[discovery['query_id']];lr=references[q['query_id']]
            same_history_pool(discovery,lr,protocol['rank_features'])
            ranked=rank_addresses(model,discovery['addresses'],protocol['rank_features'])
            candidates=canonical_candidates(ranked,chunks)
            rank_fields=('address','status','current_best_chunk_id','features','lambdamart_score','native_address_rank')
            native_rank=[{k:a[k] for k in rank_fields} for a in ranked]
            lr_candidates=sorted(lr['reranked_candidates'],key=lambda c:c['selection_rank'])
            lr_evidence=selected_evidence(lr,alpha=1.)
            conditions={'LambdaMART_native':output_condition(q,discovery,candidates,candidates[:4],native_rank,'raw LambdaMART margin'),
                'LR_native':output_condition(q,discovery,lr_candidates,lr_evidence,
                    [{'address':a['address'],'learned_probability':a['learned_probability']}
                     for a in sorted(lr['address_diagnostics'],
                         key=lambda a:(-a['learned_probability'],tuple(a['address'])))],
                    'saved LR probability used only as an address rank')}
            record={k:q[k] for k in ('query_id','query_variant','family_id','stratum','target_status','target_asof','source_role')}
            record.update(variant=q['query_variant'],quote_absent=q.get('source_quote_absent_from_target'),
                split='test',evaluation_scope='natural',conditions=conditions)
            save_row(stream,record);add_metrics(cells,q,conditions)
            summary['queries']+=1;summary['tree_predict_calls']+=int(bool(discovery['addresses']))
            if summary['queries']%100==0:
                stream.flush();write_json(output/'evaluation.json',summary)
                print(json.dumps({'phase':'native','queries':summary['queries']}),flush=True)
    summary.update(complete=True,finished_at_utc=now(),wall_seconds=time.perf_counter()-started,panels=metric_panels(cells))
    write_json(output/'evaluation.json',summary)
    return summary


def ce_phase(args, *, expected_test=3172, reranker_factory=None):
    queries,_,_=inputs(args)
    stack,dev,full=read(args.stack),read(args.dev_metadata),read(args.reference_evaluation)
    pin=stack['models']['reranker'];check_pair_context(dev,pin);check_pair_context(full,pin)
    if not dev['complete'] or not full['complete']:
        raise ValueError('Original pair-cache work must be complete')
    native=Path(args.output_dir)/'native'
    if not read(native/'evaluation.json')['complete']:
        raise ValueError('Native preselection must be complete before CE-only evaluation')
    output=Path(args.output_dir)/'ce';output.mkdir(parents=True,exist_ok=True)
    lazy=LazyReranker(pin,'cuda',reranker_factory);cache=PairScoreCache(lazy)
    sources=[args.dev_pair_receipts,args.final_pair_receipts]
    receipt_path=output/'model-pair-receipts.jsonl'
    if receipt_path.exists():
        check_pair_context(read(output/'evaluation.json'),pin);sources.append(receipt_path)
    loaded=prefill(cache,sources)
    chunks={c['chunk_id']:c for c in rows(args.chunks)}
    summary={'complete':False,'seed':42,'job_id':os.environ.get('PBS_JOBID'),'started_at_utc':now(),
        'queries':0,'fusion':'none; CE-only on each saved head top10','model_pin':{k:pin[k] for k in ('model_id','revision')},
        'max_length':1024,'sdpa_backend':'math','cache_prefill':loaded,'conditions':['LambdaMART_CE_only','LR_CE_only']}
    write_json(output/'evaluation.json',summary);cells=defaultdict(list);seen=set()
    with receipt_path.open('a',encoding='utf-8') as receipts, (output/'retrieval.jsonl').open('w',encoding='utf-8') as stream:
        def callback(row):save_row(receipts,row);receipts.flush()
        cache.receipt_callback=callback
        for row in rows(native/'retrieval.jsonl'):
            q=queries[row['query_id']]
            if q['split']!='test' or q['query_id'] in seen:raise ValueError('Native TEST IDs differ')
            seen.add(q['query_id'])
            candidates=row['conditions']['LambdaMART_native']['preselected_candidates']
            scores=cache.score_pairs([(q['query_text'],chunks[c['chunk_id']]['text']) for c in candidates])
            lm=[{**c,**s} for c,s in zip(candidates,scores)]
            lr=row['conditions']['LR_native']['preselected_candidates']
            conditions={}
            for label,selected in (('LambdaMART_CE_only',lm),('LR_CE_only',lr)):
                ordered=sorted(selected,key=lambda c:(-c['reranker_score'],c['selection_rank'],c['chunk_id']))
                evidence=[{**c,'final_score':c['reranker_score']} for c in ordered[:4]]
                conditions[label]={'score_kind':'original Ruri CE score only','fusion':'none','alpha':None,
                    'reranked_candidates':ordered,'evidence':evidence,
                    'structural':structural_result(q,evidence,'natural'),
                    'stage_reachability':{**row['conditions']['LambdaMART_native' if label.startswith('Lambda') else 'LR_native']['stage_reachability'],
                        'gold_address_in_final4':any((c['law_id'],c['scope'],c['article_num'])==(q['law_id'],q['scope'],q['article_num']) for c in evidence),
                        'gold_parent_in_final4':any(c['parent_passage_id']==q['target_passage_id'] for c in evidence) if q['target_status']=='available' else None}}
            save_row(stream,{**{k:v for k,v in row.items() if k!='conditions'},'conditions':conditions})
            add_metrics(cells,q,conditions);summary['queries']+=1
            if summary['queries']%100==0:
                stream.flush();summary['physical_pair_count']=cache.physical_pair_count;write_json(output/'evaluation.json',summary)
    expected={key for key,q in queries.items() if q['split']=='test'}
    if len(expected)!=expected_test or seen!=expected:raise ValueError('CE-only needs all original TEST IDs')
    summary.update(complete=True,finished_at_utc=now(),panels=metric_panels(cells),
        physical_pair_count=cache.physical_pair_count,logical_pair_requests=cache.logical_pair_requests,
        neural_model_calls=cache.model_call_count,encoded_input_tokens=cache.encoded_input_tokens,
        model_seconds=cache.model_wall_seconds_with_cuda_sync,model_loaded=lazy.model is not None,
        LR_scores_reused_from_original=True)
    write_json(output/'evaluation.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('fit','native','ce'),required=True)
    parser.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    parser.add_argument('--train-discovery',nargs='+',type=Path,default=[Path('runs/hidate/discovery/exact-train-valid/candidates.jsonl'),Path('runs/hidate/discovery/paraphrase-train-valid/candidates.jsonl')])
    parser.add_argument('--test-discovery',nargs='+',type=Path,default=[Path('runs/hidate/final-discovery/test-natural-exact/candidates.jsonl'),Path('runs/hidate/final-discovery/test-natural-paraphrase/candidates.jsonl')])
    parser.add_argument('--protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    parser.add_argument('--reference-ranker',type=Path,default=Path('runs/hidate/final/ranker-used.json'))
    parser.add_argument('--reference-retrieval',type=Path,default=Path('runs/hidate/final/retrieval.jsonl'))
    parser.add_argument('--reference-evaluation',type=Path,default=Path('runs/hidate/final/evaluation.json'))
    parser.add_argument('--chunks',type=Path,default=Path('data/index/hidate/chunks.jsonl'))
    parser.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    parser.add_argument('--dev-metadata',type=Path,default=Path('runs/hidate/dev/scoring.json'))
    parser.add_argument('--dev-pair-receipts',type=Path,default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    parser.add_argument('--final-pair-receipts',type=Path,default=Path('runs/hidate/final/model-pair-receipts.jsonl'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/hidate/mechanism-analysis/lambdamart'))
    args=parser.parse_args()
    result={'fit':fit_phase,'native':native_phase,'ce':ce_phase}[args.phase](args)
    print(json.dumps({'phase':args.phase,'complete':result['complete']}),flush=True)


if __name__=='__main__':main()
