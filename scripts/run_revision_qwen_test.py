"""Qwen3 original-TEST native heads and historical/current CE transfer.

Prepare and score do not compute TEST metrics. Evaluation additionally requires
a VALID-frozen recipe. No address or CE model is fitted by this worker.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.revision import gate_scores
from discover_final import query_segmenter, query_tokens
from evaluate_final import structural_result
from run_revision_ce_training import add_ce_features
from run_revision_development import deliverable_indices, ordered_indices
from run_revision_scale import (TREE_HEADS, METRICS, add_panels, compact_chunks,
    compact_discovery, load_trees, original_queries, read, rows, save_row)


INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
HEADS = ('RRF_native',) + TREE_HEADS
PAIR_FIELDS = ('scores','logits','token_counts','total_tokens','input_shape',
    'sdpa_backend','seconds','score_format','logit_format','yes_token_id','no_token_id','no_yes_logits')


def receipt(**values):
    return {'seed':42, 'job_id':os.environ.get('PBS_JOBID'), 'split':'test',
        'at_utc':datetime.now(timezone.utc).isoformat(), 'training_runs':0,
        'encoder_calls':0, 'reader_calls':0, 'new_confirmatory_queries':0, **values}


def check_context(meta, pin):
    saved = meta.get('model_pin') or meta.get('model_pins',{}).get('reranker')
    if pin['model_id'] != 'Qwen/Qwen3-Reranker-0.6B' or not saved or any(
            saved[k] != pin[k] for k in ('model_id','revision')):
        raise ValueError('Pair cache must use the frozen Japanese Qwen3 reranker pin')
    if meta.get('max_length',meta.get('reranker_max_length')) != 1024:
        raise ValueError('Qwen pair cache must retain max_length=1024')
    if meta.get('sdpa_backend',meta.get('reranker_sdpa_backend')) != 'default':
        raise ValueError('Qwen pair cache uses default attention, not Ruri math attention')
    instructions = [pin.get('conventions',{}).get('instruction',INSTRUCTION),
        saved.get('conventions',{}).get('instruction',INSTRUCTION), meta.get('instruction',INSTRUCTION)]
    if any(value != INSTRUCTION for value in instructions):
        raise ValueError('Japanese and French Qwen instruction contexts cannot share scores')


def check_pair_result(result, count):
    if (result['sdpa_backend'] != 'default' or result['score_format'] != 'conditional_yes_no'
            or result['logit_format'] != 'yes_minus_no'
            or (result['yes_token_id'],result['no_token_id']) != (9693,2152)):
        raise ValueError('Qwen conditional yes/no pair-score context differs')
    if (any(len(result[k]) != count for k in ('scores','logits','token_counts'))
            or sum(result['token_counts']) != result['total_tokens']
            or not all(math.isfinite(x) for x in result['scores']+result['logits'])):
        raise ValueError('Qwen pair-score arrays are not aligned and finite')


def pair_value(result, i):
    return {'reranker_score':result['scores'][i], 'reranker_logit':result['logits'][i],
            'pair_token_count':result['token_counts'][i]}


def preselect(cached, chunks, scores):
    addresses = cached['addresses']; parents = {cid:c['parent_passage_id'] for cid,c in chunks.items()}
    heads = {}
    for name, values in scores.items():
        if len(values) != len(addresses) or not np.isfinite(values).all():
            raise ValueError('Every frozen head must score the same full address pool')
        indices = deliverable_indices(addresses,ordered_indices(addresses,values),parents)
        heads[name] = []
        for rank,i in enumerate(indices,1):
            a = addresses[i]; hid = a['historical_representative_chunk_id']; cid = a['current_best_chunk_id']
            heads[name].append({'address_index':i, 'address':a['address'], 'selection_rank':rank,
                'native_score':float(values[i]), 'features':a['features'],
                'historical_chunk_id':hid, 'historical_parent_passage_id':parents[hid],
                'current_chunk_id':cid, 'current_parent_passage_id':parents[cid]})
    asof, seen = [], set()
    for hit in cached['asof']['hybrid']:
        cid=hit['chunk_id']; parent=parents[cid]
        if parent in seen:continue
        seen.add(parent); asof.append({'current_chunk_id':cid, 'current_parent_passage_id':parent,
            'selection_rank':len(asof)+1, 'native_score':float(hit['score'])})
        if len(asof)==10:break
    return {'heads':heads, 'asof':asof}


def prepare(args):
    started=time.perf_counter(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists() and read(out/'preparation.json')['complete']:
        return read(out/'preparation.json')
    queries,_=original_queries(args,read(args.protocol)); stack=read(args.stack)
    chunks={c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    tokens={r['chunk_id']:r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    trees,definitions=load_trees(args); segment=query_segmenter(); seen=set(); timing=Counter()
    manifest=receipt(complete=False, expected_queries=len(queries), completed_queries=0,
        models=definitions, index_path=str(args.index), discovery_paths=[str(p) for p in args.discovery],
        neural_model_calls=0, metrics_computed=False, full_pool_normalization=True,
        candidate_policy='each fixed head owns ten distinct available bound parents; all views deliver the same bound current chunk',
        controls=[name+'__'+stage for name in HEADS for stage in ('native','historical_ce','current_ce')]
            +['AsOf_hybrid_native','AsOf_hybrid_current_CE'])
    write_json(out/'preparation.json',manifest)
    with (out/'prepared.jsonl').open('w',encoding='utf-8') as stream:
        for path in args.discovery:
            source=read(path.parent/'discovery.json')
            if (not source['complete'] or source['seed']!=42 or source['top_k']!=50 or source['rrf_k']!=60
                    or source['evaluation_scope']!='natural'
                    or any(source['model_pin'][k]!=stack['models']['embedding'][k] for k in ('model_id','revision'))):
                raise ValueError('Use completed original natural Qwen discovery, 50/60, seed42')
            for d in rows(path):
                qid=d['query_id']; q=queries[qid]
                if qid in seen or any(d[k]!=q[k] for k in ('split','query_variant','source_role','target_asof')):
                    raise ValueError('Qwen discovery must match every original TEST request once')
                public={k:q[k] for k in ('query_id','query_text','query_variant','split','source_role','target_asof')}
                stamp=time.perf_counter()
                cached=compact_discovery(public,d,chunks,tokens,query_tokens(q['query_text'],segment),'120')
                timing['feature_seconds']+=time.perf_counter()-stamp
                scores={'RRF_native':[a['features']['max_history_rrf'] for a in cached['addresses']]}
                stamp=time.perf_counter()
                for name,tree in trees.items():
                    X=np.asarray([[a['features'][k] for k in definitions[name]['feature_names']] for a in cached['addresses']])
                    scores[name]=tree.predict(X,raw_score=True,num_threads=4) if len(X) else []
                timing['predict_seconds']+=time.perf_counter()-stamp
                selected=preselect(cached,chunks,scores)
                save_row(stream,{**cached, **selected,
                    'full_address_scores':{name:[float(v) for v in vals] for name,vals in scores.items()}})
                seen.add(qid)
                if len(seen)%200==0:
                    stream.flush();manifest.update(completed_queries=len(seen),timing=dict(timing))
                    write_json(out/'preparation.json',manifest)
                    print(json.dumps({'phase':'prepare','queries':len(seen)}),flush=True)
    if seen!=set(queries):raise ValueError('Original TEST preparation is incomplete')
    manifest.update(complete=True,completed_queries=len(seen),timing=dict(timing),wall_seconds=time.perf_counter()-started)
    write_json(out/'preparation.json',manifest);return manifest


def references_for(row):
    refs=[(row['query_id'],c[view+'_chunk_id']) for cc in row['heads'].values() for c in cc
          for view in ('historical','current')]
    return refs+[(row['query_id'],c['current_chunk_id']) for c in row['asof']]


def prefill_sources(args, cache, requested, pin):
    provenance=[]
    for metadata,pairs in ((args.dev_metadata,args.dev_pairs),(args.final_metadata,args.final_pairs)):
        meta=read(metadata)
        if not meta['complete']:raise ValueError('Original Qwen pair caches must be complete')
        check_context(meta,pin); count=accepted=0
        for row in rows(pairs):
            result=row['result']; check_pair_result(result,len(row['pairs']))
            for i,pair in enumerate(row['pairs']):
                key=(pair['query_text'],pair['document_text']);count+=1
                if key in requested:cache[key]=pair_value(result,i);accepted+=1
        provenance.append({'metadata':str(metadata),'pairs':str(pairs),'scanned_pair_records':count,
            'requested_pair_records_reused':accepted,'instruction':INSTRUCTION})
    return provenance


def fill_missing(cache,references,queries,chunks,pin,output,*,factory=None,batch_size=24):
    source=output/'new-ce-pair-scores.jsonl'; total=Counter(calls=0,pairs=0,input_tokens=0,padded_tokens=0,model_seconds=0.)
    if source.exists():
        for row in rows(source):
            result=row['result'];check_pair_result(result,len(row['pair_refs']))
            for i,ref in enumerate(row['pair_refs']):
                key=(queries[ref['query_id']]['query_text'],chunks[ref['chunk_id']]['text'])
                cache[key]=pair_value(result,i)
            total.update(calls=1,pairs=len(row['pair_refs']),input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1],model_seconds=result['seconds'])
    unique={}
    for qid,cid in references:
        key=(queries[qid]['query_text'],chunks[cid]['text'])
        if key not in cache and key not in unique:unique[key]={'query_id':qid,'chunk_id':cid}
    pending=list(unique.items()); model=None
    this_job=Counter(calls=0,pairs=0,input_tokens=0,padded_tokens=0,model_seconds=0.)
    with source.open('a',encoding='utf-8') as stream:
        for offset in range(0,len(pending),batch_size):
            selected=pending[offset:offset+batch_size]
            if model is None:
                if factory is None:
                    from hidate.models import create_reranker
                    model=create_reranker(pin,device='cuda',max_length=1024)
                else:model=factory(pin)
            raw=model.score([key for key,_ in selected],include_inputs=False)
            result={k:raw[k] for k in PAIR_FIELDS}; check_pair_result(result,len(selected))
            save_row(stream,{'pair_refs':[ref for _,ref in selected],'result':result});stream.flush()
            for i,(key,_) in enumerate(selected):cache[key]=pair_value(result,i)
            work=dict(calls=1,pairs=len(selected),input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1],model_seconds=result['seconds'])
            total.update(work);this_job.update(work)
            if this_job['calls']%20==0:
                print(json.dumps({'phase':'Qwen_CE','new_pairs':this_job['pairs'],'pending_pairs':len(pending)}),flush=True)
    return {'logical_view_pairs':len(references),'unique_requested_pairs':len({
        (queries[qid]['query_text'],chunks[cid]['text']) for qid,cid in references}),
        'missing_at_start':len(pending),'this_job':dict(this_job),'all_new_pair_work':dict(total),
        'model_loaded_this_job':model is not None}


def score(args):
    out=args.output; preparation=read(out/'preparation.json')
    if not preparation['complete']:raise ValueError('Complete CPU preparation is required')
    path=out/'scoring.json';pin=read(args.stack)['models']['reranker']
    if path.exists():
        check_context(read(path),pin)
        if read(path)['complete']:return read(path)
    queries,_=original_queries(args,read(args.protocol))
    chunks={c['chunk_id']:c for c in rows(args.index/'chunks.jsonl')}
    refs=[ref for row in rows(out/'prepared.jsonl') for ref in references_for(row)]
    requested={(queries[qid]['query_text'],chunks[cid]['text']) for qid,cid in refs}; cache={}
    result=receipt(complete=False,model_pin={k:pin[k] for k in ('model_id','revision')},
        max_length=1024,sdpa_backend='default',instruction=INSTRUCTION,metrics_computed=False,
        input_ID_arrays_saved=False,cache_key='exact raw query and complete chunk text',
        expected_queries=preparation['expected_queries'])
    check_context(result,pin);write_json(path,result)
    result['sources']=prefill_sources(args,cache,requested,pin)
    result['cached_unique_requested_pairs']=len(cache)
    write_json(path,result)
    result['work']=fill_missing(cache,refs,queries,chunks,pin,out,batch_size=args.batch_size)
    seen=set()
    with (out/'scored.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows(out/'prepared.jsonl'):
            qid=row['query_id'];q=queries[qid]
            for cc in row['heads'].values():
                for c in cc:
                    for view in ('historical','current'):
                        c[view+'_ce']=cache[(q['query_text'],chunks[c[view+'_chunk_id']]['text'])]
            for c in row['asof']:c['current_ce']=cache[(q['query_text'],chunks[c['current_chunk_id']]['text'])]
            save_row(stream,{'query_id':qid,'heads':row['heads'],'asof':row['asof']});seen.add(qid)
    if seen!=set(queries):raise ValueError('Scored original TEST candidate coverage differs')
    result.update(complete=True,completed_queries=len(seen));write_json(path,result);return result


def rank_candidates(candidates,stage,*,model=None,tree=None):
    if not candidates:return [],[],None
    gates=None
    if stage=='native':scores=np.array([c['native_score'] for c in candidates])
    elif stage in ('historical_ce','current_ce'):
        scores=np.array([c[stage]['reranker_score'] for c in candidates])
    elif stage=='identity_view':
        scores=np.array([c['current_ce' if c['historical_parent_passage_id']==c['current_parent_passage_id']
                              else 'historical_ce']['reranker_score'] for c in candidates])
    elif stage in ('gate','lambdamart'):
        features,h,current=add_ce_features(candidates)
        X=np.asarray([[f[k] for k in model['feature_names']] for f in features])
        if stage=='gate':scores,gates=gate_scores(model,X,h,current)
        else:scores=tree.predict(X,raw_score=True,num_threads=4)
    else:raise ValueError('Unknown frozen final stage: '+stage)
    order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),
        candidates[i]['selection_rank'],candidates[i]['current_chunk_id']))[:4]
    return order,scores,gates


def load_recipe(path, expected_pin, ce_development=None):
    recipe=read(path)
    if not recipe['complete'] or recipe['selection_split']!='original_valid':
        raise ValueError('Evaluation requires a completed original-VALID freeze')
    if recipe['selected_method']!='Full_LM18_historical_CE':
        raise ValueError('Update the Qwen transfer mapping if the actual frozen recipe differs')
    definitions=[{'name':name+'__'+stage,'head':name,'stage':stage}
        for name in HEADS for stage in ('native','historical_ce','current_ce')]
    definitions += [{'name':'AsOf_hybrid_native','head':'AsOf','stage':'native'},
                    {'name':'AsOf_hybrid_current_CE','head':'AsOf','stage':'current_ce'}]
    definitions += recipe.get('conditions',[])
    if ce_development is not None:
        fits=read(ce_development/'models/fit.json')
        names=('lambdamart_base13','lambdamart_revision20','gate_base13','gate_revision20')
        if not fits['complete'] or fits['seed']!=42 or set(fits['models'])!=set(names):
            raise ValueError('Four completed own-Qwen TRAIN CE fits are required')
        for name in names:
            definitions.append({'name':'CE_'+name,'head':'RRF_native',
                'stage':'gate' if name.startswith('gate_') else 'lambdamart',
                'model_json':str(ce_development/'models'/(name+'.json')),
                'model_txt':str(ce_development/'models'/(name+'.txt'))})
    if len({d['name'] for d in definitions})!=len(definitions):raise ValueError('Condition names must be unique')
    models={};trees={}
    for d in definitions:
        if d['head'] not in HEADS+('AsOf',):raise ValueError('Frozen head was not prepared')
        if d['stage'] in ('gate','lambdamart'):
            if d['head']!='RRF_native':raise ValueError('CE TRAIN heads transfer on their fixed common RRF ten')
            model_path=Path(d['model_json']);model=read(model_path)
            source_scoring=read(model_path.parent.parent/'scoring.json')
            if not source_scoring['complete']:raise ValueError('CE TRAIN parent scoring must be complete')
            check_context(source_scoring,expected_pin)
            if (not model['complete'] or model['seed']!=42 or model['validation_used_for_fitting']
                    or len(model['feature_names']) not in (13,20)
                    or model['feature_names'][-2:]!=['historical_CE_query_rank','current_CE_query_rank']):
                raise ValueError('Use a complete frozen TRAIN CE13/20 model with saved column order')
            models[d['name']]=model
            if d['stage']=='lambdamart':
                from hidate.lambdamart import lightgbm_module
                tree=lightgbm_module().Booster(model_file=d['model_txt'])
                if tree.feature_name()!=model['feature_names']:raise ValueError('CE tree columns differ from receipt')
                trees[d['name']]=tree
    return recipe,definitions,models,trees


def evaluate(args, *, controls_only=False):
    if args.frozen_recipe is None:raise ValueError('Supply the completed original-VALID frozen recipe')
    scoring=read(args.output/'scoring.json')
    if not scoring['complete']:raise ValueError('Complete actual Qwen CE scores required')
    pin=read(args.stack)['models']['reranker'];check_context(scoring,pin)
    recipe,definitions,models,trees=load_recipe(args.frozen_recipe,pin,args.ce_development)
    output=args.output
    if controls_only:
        if args.ce_development is None:raise ValueError('Supply completed own-Qwen CE training for additional controls')
        definitions=[d for d in definitions if d['name'].startswith('CE_')]
        output=args.output/'ce-controls';output.mkdir(exist_ok=True)
    queries,_=original_queries(args,read(args.protocol))
    chunks={c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    panels=defaultdict(Counter);seen=set()
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows(args.output/'scored.jsonl'):
            q=queries[row['query_id']];conditions={}
            for d in definitions:
                cc=row['asof'] if d['head']=='AsOf' else row['heads'][d['head']]
                order,scores,gates=rank_candidates(cc,d['stage'],model=models.get(d['name']),tree=trees.get(d['name']))
                evidence=[chunks[cc[i]['current_chunk_id']] for i in order]
                conditions[d['name']]={'head':d['head'],'stage':d['stage'],'selected_candidate_indices':order,
                    'final_stage_scores':[float(v) for v in scores],
                    'evidence_chunk_ids':[c['chunk_id'] for c in evidence],
                    'evidence_parent_ids':[c['parent_passage_id'] for c in evidence],
                    'structural':structural_result(q,evidence,'natural')}
                if gates is not None:conditions[d['name']]['historical_gate_weights']=[float(g) for g in gates]
            prediction={k:q[k] for k in ('query_id','family_id','query_variant','stratum','target_asof','target_status')}
            prediction.update(scale='Qwen_original',quote_absent=q.get('source_quote_absent_from_target'),conditions=conditions)
            save_row(stream,prediction);add_panels(panels,prediction);seen.add(q['query_id'])
    if seen!=set(queries):raise ValueError('Original TEST evaluation coverage differs')
    grouped=defaultdict(dict)
    for (_,method,group),acc in panels.items():
        grouped[method][group]={'n':acc['n'],**{m:acc[m]/acc['n'] for m in METRICS}}
    macro={name:{m:float(np.mean([v[m] for k,v in groups.items() if k.startswith('cell=')]))
                 for m in METRICS} for name,groups in grouped.items()}
    result=receipt(complete=True,queries=len(seen),methods=[d['name'] for d in definitions],
        panels=dict(grouped),frozen_recipe_path=str(args.frozen_recipe),frozen_recipe=recipe,
        available_four_cell_macro=macro,
        selected_method=recipe['selected_method'],
        selected_condition='Full_LM18__historical_ce' if not controls_only else None,
        evaluation_role='additional frozen TRAIN CE controls' if controls_only else 'frozen main method and native/CE controls',
        model_definitions=definitions,neural_model_calls=0,reranker_calls=0,
        method_selection=False,preparation=str(args.output/'preparation.json'),scoring=str(args.output/'scoring.json'))
    write_json(output/'evaluation.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path('runs/hidate/revision-aware')
    prior=Path('runs/hidate/extended-benchmark/qwen3')
    p.add_argument('--phase',choices=('prepare','score','evaluate','evaluate-controls'),required=True)
    p.add_argument('--protocol',type=Path,default=Path('configs/scale_protocol.json'))
    p.add_argument('--rank-protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--original-prepared',type=Path,default=Path('data/processed/hidate'))
    p.add_argument('--index',type=Path,default=Path('data/index/hidate-qwen3'))
    p.add_argument('--stack',type=Path,default=Path('models/qwen3/model-stack.json'))
    p.add_argument('--development',type=Path,default=root/'qwen-native-development')
    p.add_argument('--original-tree',type=Path,default=root/'qwen-original-lambdamart/ranker/model.txt')
    p.add_argument('--discovery',type=Path,nargs='+',default=[prior/'final-discovery'/('test-natural-'+k)/'candidates.jsonl' for k in ('exact','paraphrase')])
    p.add_argument('--dev-metadata',type=Path,default=prior/'dev/scoring.json')
    p.add_argument('--dev-pairs',type=Path,default=prior/'dev/model-pair-receipts.jsonl')
    p.add_argument('--final-metadata',type=Path,default=prior/'final/evaluation.json')
    p.add_argument('--final-pairs',type=Path,default=prior/'final/model-pair-receipts.jsonl')
    p.add_argument('--output',type=Path,default=root/'transfer/qwen-test')
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--frozen-recipe',type=Path)
    p.add_argument('--ce-development',type=Path,help='Completed own-Qwen CE13/20 TRAIN output; used only for extra controls')
    args=p.parse_args()
    if args.phase=='evaluate-controls':evaluate(args,controls_only=True)
    else:{'prepare':prepare,'score':score,'evaluate':evaluate}[args.phase](args)


if __name__=='__main__':main()
