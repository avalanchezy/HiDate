"""Frozen ranking comparisons on confirmation or original scale regression.

Discovery is prepared separately from public text/date inputs. This worker
requires a completed development decision before joining the held-out labels.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.lambdamart import lightgbm_module
from hidate.reranking import PairScoreCache
from hidate.ranking import address_probabilities
from hidate.revision import gate_scores
from evaluate_final import structural_result
from run_core_mechanisms import check_pair_context, prefill
from run_revision_development import deliverable_indices, ordered_indices
from run_revision_scale import read, rows, save_row, load_trees, compact_chunks, METRICS
from run_revision_ce_training import add_ce_features, fixed_scores
from run_revision_dualview import fill_missing
from run_revision_history_stage import load_compact


ROOT=Path('runs/hidate/revision-aware')
DATA=Path('data/processed/hidate-holdout/confirmatory')


def expected_queries(args):return 14912 if args.dataset=='confirmation' else 3172


def frozen(args):
    decision=read(args.freeze)
    if not decision['complete'] or decision['selection_split']!='original_valid':
        raise ValueError('Freeze the development decision before confirmation evaluation')
    if decision['confirmation_results_used']:
        raise ValueError('Confirmation cannot select the recipe')
    return decision


def select_pools(addresses, scores, parents):
    scores={'RRF':[a['features']['max_history_rrf'] for a in addresses],**scores}
    return {name:deliverable_indices(addresses,ordered_indices(addresses,values),parents)
            for name,values in scores.items()}


def prepare(args):
    decision=frozen(args); out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists() and read(out/'preparation.json')['complete']:
        return read(out/'preparation.json')
    source=read(args.discovery/'discovery.json')
    expected=expected_queries(args)
    if args.dataset=='confirmation':
        valid=source['complete'] and source['completed_queries']==expected and not source['sealed_gold_read']
    else:
        scale='120' if args.dataset=='regression_original' else '1000'
        valid=source['complete'] and source['completed_query_scale_rows']==expected and source['scales']==[scale] and source['original_test_only']
    if not valid:raise ValueError('Complete the matching frozen query/corpus discovery first')
    started=time.perf_counter()
    chunks={c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    parents={cid:c['parent_passage_id'] for cid,c in chunks.items()}
    trees,definitions=load_trees(args)
    lr=read(args.original_lr)
    count=0;timing=Counter();views=Counter()
    with (out/'prepared.jsonl').open('w',encoding='utf-8') as stream:
        for d in rows(args.discovery/'candidates.jsonl'):
            addresses=d['addresses'];scores={'Original_LR':address_probabilities(lr,[a['features'] for a in d['addresses']])}
            for name,tree in trees.items():
                begin=time.perf_counter()
                X=np.asarray([[a['features'][k] for k in definitions[name]['feature_names']] for a in addresses])
                scores[name]=tree.predict(X,raw_score=True,num_threads=4) if len(X) else np.array([])
                timing[name+'_predict_seconds']+=time.perf_counter()-begin
            pools=select_pools(addresses,scores,parents)
            needed=sorted(set(i for ids in pools.values() for i in ids))
            mapping={old:new for new,old in enumerate(needed)}
            candidates=[{'address':addresses[i]['address'],
                'historical_chunk_id':addresses[i]['historical_representative_chunk_id'],
                'current_chunk_id':addresses[i]['current_best_chunk_id'],
                'features':addresses[i]['features'], 'original_lr_score':float(scores['Original_LR'][i])} for i in needed]
            pools={name:[mapping[i] for i in ids] for name,ids in pools.items()}
            direct=[];seen=set()
            for hit in d['asof']['hybrid']:
                cid=hit['chunk_id'];parent=parents[cid]
                if parent in seen:continue
                seen.add(parent);direct.append({'current_chunk_id':cid,'native_score':hit['score'],
                    'selection_rank':len(direct)+1})
                if len(direct)==10:break
            record={k:d[k] for k in ('query_id','query_text','query_variant','target_asof')}
            record.update(candidates=candidates,pools=pools,asof=direct,discovery_address_count=len(addresses),
                discovered_addresses=[a['address'] for a in addresses],
                bindable_parent_ids=[parents[a['current_best_chunk_id']] for a in addresses
                    if a['status']=='available' and a['current_best_chunk_id'] is not None],
                discovery_timing=d.get('timing',{}))
            save_row(stream,record);count+=1
            views['history']+=len(candidates);views['current_rrf']+=len(pools['RRF']);views['asof']+=len(direct)
    result={'complete':count==expected,'queries':count,'dataset':args.dataset,'freeze':decision,'models':definitions,
        'logical_requested_views':dict(views),'sealed_gold_read':False,'training_runs':0,
        'timing':dict(timing),'wall_seconds':time.perf_counter()-started,'job_id':os.environ.get('PBS_JOBID')}
    write_json(out/'preparation.json',result)
    return result


def all_queries(args):
    return {q['query_id']:q for path in args.original_queries for q in rows(path)} | {
        q['query_id']:q for q in rows(args.output/'prepared.jsonl')}


def score(args):
    frozen(args);out=args.output
    if (out/'scoring.json').exists() and read(out/'scoring.json')['complete']:
        return read(out/'scoring.json')
    if not read(out/'preparation.json')['complete']:raise ValueError('Complete frozen candidate preparation first')
    queries=all_queries(args);chunks={c['chunk_id']:c for c in rows(args.index/'chunks.jsonl')}
    pin=read(args.stack)['models']['reranker'];cache=PairScoreCache(None)
    check_pair_context(read(args.dev_pairs.parent/'scoring.json'),pin)
    sources=[prefill(cache,[args.dev_pairs])]
    for path in args.additional_pair_caches:
        check_pair_context(read(path.parent/'evaluation.json'),pin)
        sources.append(prefill(cache,[path]))
    for directory in args.compact_caches:
        check_pair_context(read(directory/'scoring.json'),pin)
        sources.append(load_compact(cache,directory/'new-ce-pair-scores.jsonl',queries,chunks))
    references=[]
    for r in rows(out/'prepared.jsonl'):
        qid=r['query_id'];cc=r['candidates']
        references.extend((qid,c['historical_chunk_id']) for c in cc)
        references.extend((qid,cc[i]['current_chunk_id']) for i in set(r['pools']['RRF']+r['pools']['Original_LR']))
        references.extend((qid,c['current_chunk_id']) for c in r['asof'])
    result={'complete':False,'model_pin':{k:pin[k] for k in ('model_id','revision')},
        'max_length':1024,'sdpa_backend':'math','cache_sources':sources,'seed':42,
        'input_ID_arrays_saved':False,'job_id':os.environ.get('PBS_JOBID'),'training_runs':0}
    write_json(out/'scoring.json',result)
    result['work']=fill_missing(cache,references,queries,chunks,pin,out,batch_size=args.batch_size)
    with (out/'scored.jsonl').open('w',encoding='utf-8') as stream:
        for r in rows(out/'prepared.jsonl'):
            text=r['query_text']
            for i,c in enumerate(r['candidates']):
                c['historical_ce']=cache.cache[(text,chunks[c['historical_chunk_id']]['text'])]
                if i in set(r['pools']['RRF']+r['pools']['Original_LR']):
                    c['current_ce']=cache.cache[(text,chunks[c['current_chunk_id']]['text'])]
            for c in r['asof']:c['current_ce']=cache.cache[(text,chunks[c['current_chunk_id']]['text'])]
            save_row(stream,r)
    result['complete']=True;write_json(out/'scoring.json',result)
    return result


def result_rows(query,candidates,scores,chunks,views=()):
    order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),
        candidates[i]['selection_rank'],candidates[i]['current_chunk_id']))
    chosen=[candidates[i]['current_chunk_id'] for i in order[:4]]
    evidence=[chunks[cid] for cid in chosen]
    return {'structural':{k:v for k,v in structural_result(query,evidence,'natural').items() if k in METRICS},
        'evidence_chunk_ids':chosen,'candidate_parent_at10':int(query['target_status']=='available' and any(
            chunks[c['current_chunk_id']]['parent_passage_id']==query['target_passage_id'] for c in candidates)),
        'logical_CE_views':len(candidates)*len(views),
        'logical_CE_tokens':sum(c[v]['pair_token_count'] for c in candidates for v in views)}


def target_queries(args):
    if args.dataset in ('regression_scale','regression_original'):
        return {q['query_id']:q for path in args.original_queries for q in rows(path) if q['split']=='test'}
    gold={q['query_id']:q for q in rows(DATA/'sealed/queries-exact.jsonl')}
    result={}
    for path in args.public_queries:
        for q in rows(path):
            source=gold[q.get('base_query_id',q['query_id'])]
            result[q['query_id']]={**source,**{k:q[k] for k in ('query_id','query_text','query_variant','target_asof')}}
    return result


def evaluate(args):
    decision=frozen(args);out=args.output
    if (out/'evaluation.json').exists() and read(out/'evaluation.json')['complete']:
        return read(out/'evaluation.json')
    if not read(out/'scoring.json')['complete']:raise ValueError('Complete frozen CE scoring before evaluation')
    queries=target_queries(args);chunks={c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    fits=read(args.ce_models/'fit.json')['models'];lgb=lightgbm_module()
    alpha=read(args.original_policy)['alpha']
    trees={name:lgb.Booster(model_file=str(args.ce_models/(name+'.txt'))) for name in fits if name.startswith('lambdamart')}
    panels=defaultdict(Counter);cost=defaultdict(Counter);count=0
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for r in rows(out/'scored.jsonl'):
            q=queries[r['query_id']];conditions={}
            for head,pool in r['pools'].items():
                cc=[{**r['candidates'][i],'selection_rank':rank} for rank,i in enumerate(pool,1)]
                conditions[head+'_native']=result_rows(q,cc,-np.arange(len(cc)),chunks)
                conditions[head+'_historical_CE']=result_rows(q,cc,[c['historical_ce']['reranker_score'] for c in cc],
                    chunks,('historical_ce',))
            cc=[{**r['candidates'][i],'selection_rank':rank} for rank,i in enumerate(r['pools']['RRF'],1)]
            view_names=['current_ce' if chunks[c['historical_chunk_id']]['parent_passage_id']==chunks[c['current_chunk_id']]['parent_passage_id']
                        else 'historical_ce' for c in cc]
            identity=result_rows(q,cc,[c[v]['reranker_score'] for c,v in zip(cc,view_names)],chunks)
            identity.update(logical_CE_views=len(cc),logical_CE_tokens=sum(c[v]['pair_token_count'] for c,v in zip(cc,view_names)))
            conditions['RRF_identity_view']=identity
            features,h,c=add_ce_features(cc);fixed=fixed_scores(cc)
            for name,values in fixed.items():
                views=() if name=='RRF_native' else (('historical_ce',) if name=='Historical_CE' else
                    ('current_ce',) if name=='Current_CE' else ('historical_ce','current_ce'))
                conditions['common_'+name]=result_rows(q,cc,values,chunks,views)
            for name,model in fits.items():
                X=np.asarray([[f[k] for k in model['feature_names']] for f in features])
                start=time.perf_counter()
                values=(trees[name].predict(X,raw_score=True,num_threads=4) if name in trees else gate_scores(model,X,h,c)[0]) if cc else np.array([])
                cost['ce_'+name]['prediction_seconds']+=time.perf_counter()-start
                conditions['ce_'+name]=result_rows(q,cc,values,chunks,('historical_ce','current_ce'))
            cc=[{**r['candidates'][i],'selection_rank':rank} for rank,i in enumerate(r['pools']['Original_LR'],1)]
            conditions['Original_HiDate_LR_CE']=result_rows(q,cc,[alpha*c['original_lr_score']+(1-alpha)*c['current_ce']['reranker_score']
                for c in cc],chunks,('current_ce',))
            cc=r['asof']
            conditions['AsOf_hybrid_native']=result_rows(q,cc,[c['native_score'] for c in cc],chunks)
            conditions['AsOf_hybrid_CE']=result_rows(q,cc,[c['current_ce']['reranker_score'] for c in cc],chunks,('current_ce',))
            if decision['selected_method'] not in conditions:raise ValueError('Frozen method is absent from complete comparison')
            groups=['all']
            if q['target_status']=='available':
                groups+=['available','variant='+q['query_variant'],'cell='+q['query_variant']+'|'+q['stratum']]
                if q.get('source_quote_absent_from_target') is True:groups+=['quote_absent','quote_absent|'+q['query_variant']]
            for name,condition in conditions.items():
                for group in groups:
                    a=panels[(name,group)];a['n']+=1
                    for k,v in condition['structural'].items():a[k]+=v
                    a['candidate_parent_at10']+=condition['candidate_parent_at10']
                cost[name]['queries']+=1;cost[name]['logical_CE_views']+=condition['logical_CE_views']
                cost[name]['logical_CE_tokens']+=condition['logical_CE_tokens']
            gold_address=(q['target_key'][0],q['target_key'][2],q['target_key'][3])
            save_row(stream,{**{k:q[k] for k in ('query_id','query_variant','family_id','stratum','target_status')},
                'quote_absent':q.get('source_quote_absent_from_target'),
                'discovery_address_hit':gold_address in map(tuple,r['discovered_addresses']),
                'bindable_parent_hit':q['target_status']=='available' and q['target_passage_id'] in r['bindable_parent_ids'],
                'conditions':conditions})
            count+=1
    aggregates=defaultdict(dict)
    for (name,group),acc in panels.items():
        aggregates[name][group]={'n':acc['n'],**{k:acc[k]/acc['n'] for k in METRICS+('candidate_parent_at10',)}}
    macro={name:{k:float(np.mean([v[k] for key,v in group.items() if key.startswith('cell=')]))
        for k in METRICS} for name,group in aggregates.items()}
    result={'complete':count==len(queries)==expected_queries(args),'queries':count,'dataset':args.dataset,'selected_method':decision['selected_method'],
        'family_count':len({q['family_id'] for q in queries.values()}),'freeze':decision,
        'panels':dict(aggregates),'available_four_cell_macro':macro,'cost':dict(cost),
        'cost_note':'logical per-method views/tokens; shared physical scoring is separately recorded',
        'selection_updates':0,'training_runs':0,'job_id':os.environ.get('PBS_JOBID')}
    write_json(out/'evaluation.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('confirmation','regression_scale','regression_original'),default='confirmation')
    p.add_argument('--phase',choices=('prepare','score','evaluate','all'),default='all')
    p.add_argument('--freeze',type=Path,default=ROOT/'frozen-recipe.json')
    p.add_argument('--discovery',type=Path,default=ROOT/'confirmation/discovery')
    p.add_argument('--output',type=Path,default=ROOT/'confirmation/final')
    p.add_argument('--index',type=Path,default=Path('data/index/hidate/extended-benchmark/scale'))
    p.add_argument('--development',type=Path,default=ROOT/'native-development')
    p.add_argument('--original-tree',type=Path,default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    p.add_argument('--original-lr',type=Path,default=Path('runs/hidate/final/ranker-used.json'))
    p.add_argument('--original-policy',type=Path,default=Path('runs/hidate/policy/frozen.json'))
    p.add_argument('--rank-protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--ce-models',type=Path,default=ROOT/'ce-training/models')
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--original-queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    p.add_argument('--public-queries',nargs='+',type=Path,default=[DATA/'unlabeled/queries-exact-unlabeled.jsonl',DATA/'unlabeled/queries-paraphrase-unlabeled.jsonl'])
    p.add_argument('--dev-pairs',type=Path,default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    p.add_argument('--additional-pair-caches',nargs='*',type=Path,default=[Path('runs/hidate/final/model-pair-receipts.jsonl')])
    p.add_argument('--compact-caches',nargs='+',type=Path,default=[ROOT/'diagnostics/revision-dualview',ROOT/'diagnostics/history-stage',ROOT/'ce-training'])
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    for name,worker in [('prepare',prepare),('score',score),('evaluate',evaluate)]:
        if args.phase in ('all',name):
            result=worker(args);print(json.dumps({'phase':name,'complete':result['complete']}),flush=True)


if __name__=='__main__':main()
