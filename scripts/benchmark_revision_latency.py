"""Independent per-query method latency on 100 fixed original VALID requests.

Weights and static date banks are resident; query encoding, dense/BM25 ranking,
binding/features, heads and CE execute anew for every method/request. No TEST
metrics, reader inference, or cross-method neural-result cache is used.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace

import numpy as np

from hidate.io import write_json
from hidate.discovery import DiscoveryEngine
from hidate.revision import enrich_addresses
from discover_final import query_segmenter, query_tokens, vector_bank
from run_core_mechanisms import check_pair_context
from run_revision_development import deliverable_indices, ordered_indices
from run_revision_qwen_test import rank_candidates
from run_revision_scale import read, rows, save_row


VARIANTS=('exact_quote','generated_paraphrase')
PUBLIC_FIELDS=('query_id','query_text','query_variant','split','source_role','target_asof')
BASE_METHODS=(
    {'name':'RRF_native','branch':'history','head':'rrf','stage':'native'},
    {'name':'AsOf_hybrid_native','branch':'asof','head':'rrf','stage':'native'},
    {'name':'AsOf_hybrid_current_CE','branch':'asof','head':'rrf','stage':'current_ce'},
    {'name':'Historical_CE','branch':'history','head':'rrf','stage':'historical_ce'},
    {'name':'Original_LM10_historical_CE','branch':'history','head':'original_lm','stage':'historical_ce'})


def sample_queries(queries, per_variant=50):
    rng=random.Random(42);selected=[]
    valid=[q for q in queries if q['split']=='valid' and q['source_role']=='full']
    for variant in VARIANTS:
        pool=sorted((q for q in valid if q['query_variant']==variant),key=lambda q:q['query_id'])
        selected.extend(rng.sample(pool,per_variant))
    ids={q['query_id'] for q in selected}
    warmup=next(q for q in sorted(valid,key=lambda q:q['query_id']) if q['query_id'] not in ids)
    return selected,warmup


def prepare(args):
    source=[q for path in args.queries for q in rows(path)]
    selected,warmup=sample_queries(source)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'requests.jsonl').open('w',encoding='utf-8') as stream:
        for q in selected:save_row(stream,{k:q[k] for k in PUBLIC_FIELDS})
    write_json(args.output/'warmup.json',{k:warmup[k] for k in PUBLIC_FIELDS})
    result={'complete':True,'seed':42,'split':'valid','queries':len(selected),
        'queries_by_variant':dict(Counter(q['query_variant'] for q in selected)),
        'query_ids':[q['query_id'] for q in selected],'warmup_query_id':warmup['query_id'],
        'source_paths':[str(p) for p in args.queries],
        'sampling':'sort each original VALID/full variant by ID; Random(42), sequential sample of 50 per variant',
        'performance_used_for_sampling':False,'gold_fields_saved':False,'model_calls':0,
        'target_dates':dict(Counter(q['target_asof'] for q in selected))}
    write_json(args.output/'preparation.json',result);return result


def score_request(query,candidates,views,reranker,batch_size):
    """A fresh local dictionary deduplicates only this method/request's pairs."""
    unique={}; logical=0
    for candidate in candidates:
        for view in views:
            key=(query['query_text'],candidate[view+'_text']);logical+=1
            unique.setdefault(key,None)
    keys=list(unique);actual=Counter(calls=0,input_tokens=0,padded_tokens=0,model_seconds=0.)
    for offset in range(0,len(keys),batch_size):
        batch=keys[offset:offset+batch_size]
        result=reranker.score(batch,include_inputs=False)
        if (result['sdpa_backend']!='math' or len(result['scores'])!=len(batch)
                or sum(result['token_counts'])!=result['total_tokens']
                or not np.isfinite(result['scores']+result['logits']).all()):
            raise ValueError('Independent Ruri CE receipt differs from the frozen math backend')
        for i,key in enumerate(batch):
            unique[key]={'reranker_score':result['scores'][i],'reranker_logit':result['logits'][i],
                         'pair_token_count':result['token_counts'][i]}
        actual.update(calls=1,input_tokens=result['total_tokens'],
            padded_tokens=result['input_shape'][0]*result['input_shape'][1],model_seconds=result['seconds'])
    for candidate in candidates:
        for view in views:candidate[view+'_ce']=unique[(query['query_text'],candidate[view+'_text'])]
    return {**dict(actual),'logical_pairs':logical,'unique_pairs':len(keys)}


def measure_query(query,method,runtime):
    start=time.perf_counter();stage=Counter()
    stamp=time.perf_counter()
    encoded=runtime.encoder.encode([query['query_text']],kind='query',include_inputs=False)
    vector=np.asarray(encoded['embeddings'][0],dtype=np.float32);stage['query_encoding_seconds']=time.perf_counter()-stamp
    stamp=time.perf_counter();terms=query_tokens(query['query_text'],runtime.segmenter)
    stage['query_tokenization_seconds']=time.perf_counter()-stamp
    pool=(query['source_role'],query['target_asof']);mode=method['branch'];bank=runtime.banks[pool+(mode,)]
    stamp=time.perf_counter()
    indices,vectors=runtime.vector_views[pool+(mode,)]
    scores=np.full(runtime.vector_count,np.nan,dtype=np.result_type(vectors.dtype,vector.dtype))
    scores[indices]=vectors@vector
    hits=runtime.engine._rank_bank(bank,terms,scores)
    stage['discovery_seconds']=time.perf_counter()-stamp
    candidates=[];head_cpu=0.;head_wall=0.
    if mode=='asof':
        stamp=time.perf_counter();cpu=time.process_time();seen=set()
        for hit in hits['hybrid']:
            cid=hit['chunk_id'];parent=runtime.parents[cid]
            if parent in seen:continue
            seen.add(parent);candidates.append({'current_chunk_id':cid,'current_parent_passage_id':parent,
                'native_score':hit['score'],'selection_rank':len(candidates)+1})
            if len(candidates)==10:break
        head_cpu+=time.process_time()-cpu;head_wall+=time.perf_counter()-stamp
        pool_count=len(hits['hybrid'])
    else:
        stamp=time.perf_counter()
        addresses=runtime.engine._addresses(hits,runtime.banks[pool+('asof',)],terms,scores,query['target_asof'])
        stage['binding_and_base_features_seconds']=time.perf_counter()-stamp
        stamp=time.perf_counter()
        if method['stage'] in ('gate','lambdamart') or method['head']=='full_lm':
            addresses=enrich_addresses({'addresses':addresses},terms,runtime.tokens)
        else:
            addresses=[{**a,'historical_representative_chunk_id':min(a['history_chunk_hits'],
                key=lambda h:(-h['score'],h['chunk_id']))['chunk_id']} for a in addresses]
        stage['revision_features_seconds']=time.perf_counter()-stamp
        stamp=time.perf_counter();cpu=time.process_time()
        if method['head'] in ('original_lm','full_lm'):
            tree,names=((runtime.original_tree,runtime.original_features) if method['head']=='original_lm'
                        else (runtime.full_tree,runtime.full_features))
            X=np.asarray([[a['features'][k] for k in names] for a in addresses])
            values=tree.predict(X,raw_score=True,num_threads=4) if len(X) else []
        else:values=[a['features']['max_history_rrf'] for a in addresses]
        chosen=deliverable_indices(addresses,ordered_indices(addresses,values),runtime.parents)
        for i in chosen:
            a=addresses[i];h=a['historical_representative_chunk_id'];c=a['current_best_chunk_id']
            candidates.append({'address':a['address'],'features':a['features'],'selection_rank':len(candidates)+1,
                'historical_chunk_id':h,'historical_parent_passage_id':runtime.parents[h],
                'current_chunk_id':c,'current_parent_passage_id':runtime.parents[c],
                'native_score':float(values[i])})
        head_cpu+=time.process_time()-cpu;head_wall+=time.perf_counter()-stamp;pool_count=len(addresses)
    views={'native':(), 'historical_ce':('historical',), 'current_ce':('current',),
           'gate':('historical','current'), 'lambdamart':('historical','current')}[method['stage']]
    for c in candidates:
        for view in views:c[view+'_text']=runtime.chunks[c[view+'_chunk_id']]['text']
    stamp=time.perf_counter()
    ce=score_request(query,candidates,views,runtime.reranker,runtime.batch_size)
    stage['CE_seconds']=time.perf_counter()-stamp
    stamp=time.perf_counter();cpu=time.process_time()
    order,_,_=rank_candidates(candidates,method['stage'],model=method.get('model'),tree=method.get('tree'))
    head_cpu+=time.process_time()-cpu;head_wall+=time.perf_counter()-stamp
    elapsed=time.perf_counter()-start
    return {'query_id':query['query_id'],'query_variant':query['query_variant'],'method':method['name'],
        'latency_seconds':elapsed,'stages':{**dict(stage),'head_wall_seconds':head_wall,'head_cpu_seconds':head_cpu},
        'encoder':{'calls':1,'input_tokens':encoded['total_tokens'],
            'padded_tokens':encoded['input_shape'][0]*encoded['input_shape'][1],
            'model_seconds':encoded['seconds'],'sdpa_backend':encoded['sdpa_backend']},
        'CE':ce,'discovery_bank':mode,'dense_unique_vector_rows':len(indices),'pool_count':pool_count,
        'preselected_parents':len(candidates),'evidence_chunk_ids':[candidates[i]['current_chunk_id'] for i in order]}


def load_methods(args,pin):
    from hidate.lambdamart import lightgbm_module
    recipe=read(args.frozen_recipe)
    if not recipe['complete'] or recipe['selection_split']!='original_valid':
        raise ValueError('Benchmark methods require the original-VALID frozen recipe')
    if recipe['selected_method']!='Full_LM18_historical_CE':
        raise ValueError('Update the benchmark definition if the actual frozen method differs')
    methods=[dict(m) for m in BASE_METHODS];lgb=lightgbm_module()
    methods.append({'name':recipe['selected_method'],'branch':'history','head':'full_lm','stage':'historical_ce'})
    cost_name=args.ce_cost_model
    cost_model=args.ce_development/'models'/cost_name
    cost_items=[{'name':'Development_'+cost_name+'_dual_CE','head':'RRF_native',
        'stage':'gate' if cost_name.startswith('gate_') else 'lambdamart',
        'model_json':str(cost_model.with_suffix('.json')),'model_txt':str(cost_model.with_suffix('.txt'))}]
    for item in cost_items:
        if item['stage'] not in ('gate','lambdamart'):continue
        if item['head']!='RRF_native':raise ValueError('CE learned heads retain their TRAIN common RRF ten')
        model_path=Path(item['model_json']);model=read(model_path)
        scoring=read(model_path.parent.parent/'scoring.json')
        if not scoring['complete']:raise ValueError('CE models require completed parent TRAIN scoring')
        check_pair_context(scoring,pin)
        if (not model['complete'] or model['seed']!=42 or model['validation_used_for_fitting']
                or len(model['feature_names']) not in (13,20)):
            raise ValueError('Use saved seed42 TRAIN CE13/20 heads')
        method={'name':item['name'],'branch':'history','head':'rrf','stage':item['stage'],
            'model':model,'model_json':item['model_json']}
        if item['stage']=='lambdamart':
            tree=lgb.Booster(model_file=item['model_txt'])
            if tree.feature_name()!=model['feature_names']:raise ValueError('Saved CE columns differ')
            method.update(tree=tree,model_txt=item['model_txt'])
        methods.append(method)
    return methods,recipe


def load_runtime(args,requests):
    from hidate.models import create_encoder,create_reranker
    from hidate.lambdamart import lightgbm_module
    stack=read(args.stack);protocol=read(args.protocol)
    docs=vector_bank(args.index/'embeddings','chunk-ids.json')
    if (not docs['manifest']['complete'] or any(docs['manifest']['model_pin'][k]!=stack['models']['embedding'][k]
            for k in ('model_id','revision'))):raise ValueError('Use the complete original Ruri corpus vectors')
    chunks=list(rows(args.index/'chunks.jsonl'));chunkmap={c['chunk_id']:c for c in chunks}
    if [c['chunk_id'] for c in chunks]!=docs['ids']:raise ValueError('Document IDs differ from the frozen vector order')
    if len({c['law_id'] for c in chunks if c['source_kind']=='statute' and c['source_role']=='full'})!=120:
        raise ValueError('Benchmark retains the original 120-law corpus')
    tokens={r['chunk_id']:r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    targets={tuple(r['address_key']):r for r in rows(args.prepared/'target-addresses.jsonl')}
    bindings={(r['law_id'],r['requested_asof']):r for r in rows(args.source/'asof-bindings.jsonl')}
    engine=DiscoveryEngine(chunks,tokens,docs['vectors'],docs['aliases'],docs['ids'],targets,bindings,top_k=50,rrf_k=60)
    banks={};views={}
    for role,date in sorted({(q['source_role'],q['target_asof']) for q in requests}):
        for mode in ('asof','history'):
            key=(role,date,mode);bank=engine._bank(role,date,mode);banks[key]=bank
            # Immutable eligible-vector view is an offline part of each date bank.
            indices=np.unique(bank['vector_rows']);views[key]=(indices,docs['vectors'][indices])
    old=read(args.original_tree.parent/'fit.json')
    if not old['complete'] or old['seed']!=42 or old['validation_used'] or old['feature_names']!=protocol['rank_features']:
        raise ValueError('Original LM10 must retain its fixed TRAIN fit')
    tree=lightgbm_module().Booster(model_file=str(args.original_tree))
    if tree.feature_name()!=old['feature_names']:raise ValueError('Original LM10 columns differ')
    full_fit=read(args.full_tree.parent/'fit.json')
    if not full_fit['complete'] or full_fit['seed']!=42 or full_fit['validation_used_for_fitting']:
        raise ValueError('Full LM18 must retain its fixed TRAIN fit')
    full_names=full_fit['fits']['lambdamart_revision']['feature_names']
    full_tree=lightgbm_module().Booster(model_file=str(args.full_tree))
    if len(full_names)!=18 or full_tree.feature_name()!=full_names:raise ValueError('Full LM18 columns differ')
    return SimpleNamespace(encoder=create_encoder(stack['models']['embedding'],device='cuda',max_length=1024),
        reranker=create_reranker(stack['models']['reranker'],device='cuda',max_length=1024),
        engine=engine,banks=banks,vector_views=views,vector_count=len(docs['vectors']),
        chunks=chunkmap,tokens=tokens,parents={cid:c['parent_passage_id'] for cid,c in chunkmap.items()},
        segmenter=query_segmenter(),original_tree=tree,original_features=old['feature_names'],
        full_tree=full_tree,full_features=full_names,batch_size=10),stack


def summarize(records):
    def distribution(values):
        return {'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),
                'p95':float(np.percentile(values,95))}
    result={}
    for name in dict.fromkeys(r['method'] for r in records):
        rr=[r for r in records if r['method']==name]
        stages=sorted({k for r in rr for k in r['stages']})
        result[name]={'queries':len(rr),'latency_seconds':distribution([r['latency_seconds'] for r in rr]),
            'stage_seconds':{k:distribution([r['stages'].get(k,0.) for r in rr]) for k in stages},
            'actual_encoder_calls':sum(r['encoder']['calls'] for r in rr),
            'actual_CE':{k:sum(r['CE'][k] for r in rr) for k in
                ('calls','unique_pairs','logical_pairs','input_tokens','padded_tokens','model_seconds')},
            'CE_pairs_per_request':distribution([r['CE']['unique_pairs'] for r in rr]),
            'CE_input_tokens_per_request':distribution([r['CE']['input_tokens'] for r in rr])}
    return result


def run(args):
    if args.frozen_recipe is None:raise ValueError('Supply the frozen original-VALID recipe before benchmarking')
    import torch
    preparation=read(args.output/'preparation.json')
    if not preparation['complete'] or preparation['queries']!=100:raise ValueError('Prepare the fixed 100 VALID requests')
    requests=list(rows(args.output/'requests.jsonl'));warmup=read(args.output/'warmup.json')
    gpu=torch.cuda.get_device_properties(0)
    if 'A6000' not in gpu.name:raise ValueError('This benchmark was specified for an A6000 allocation')
    setup=time.perf_counter();runtime,stack=load_runtime(args,requests+[warmup])
    methods,recipe=load_methods(args,stack['models']['reranker']);setup_seconds=time.perf_counter()-setup
    cpu_name=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name')),platform.processor())
    hardware={'gpu':gpu.name,'gpu_memory_bytes':gpu.total_memory,'cpu':cpu_name,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_version':torch.__version__,
        'torch_num_threads':torch.get_num_threads(),'hostname':platform.node(),
        'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')}
    with (args.output/'warmup-measurements.jsonl').open('w',encoding='utf-8') as stream:
        for method in methods:save_row(stream,measure_query(warmup,method,runtime))
    records=[];path=args.output/'benchmark.json'
    method_receipts=[{k:v for k,v in m.items() if k not in ('tree','model')} for m in methods]
    result={'complete':False,'seed':42,'job_id':os.environ.get('PBS_JOBID'),'hardware':hardware,
        'expected_requests_per_method':100,'methods':method_receipts,'frozen_recipe':recipe,
        'setup_seconds':setup_seconds,'model_pins':{k:stack['models'][k] for k in ('embedding','reranker')},
        'warmup_queries':1,'warmup_repeated_per_method':True,'warmup_in_statistics':False,
        'timing_mode':'independent end-to-end single-query measurements; no stage composition',
        'excluded_setup':'model loading; resident lexical/date banks and eligible document-vector views',
        'cache_policy':'no query/model-result reuse across methods or queries; dedup identical raw CE pairs only within one method/request',
        'CE_batch_size':10,'head_threads':4,'reader_calls':0,'training_runs':0,'TEST_queries':0,
        'method_order':'round-robin rotation per query; each method sees the same fixed 100 requests',
        'percentile_method':'numpy linear interpolation','metrics_computed':False,
        'binding_timing':'shared existing date binding plus original bound-current ten-feature construction',
        'new_head_feature_timing':'full-address revision features before common-RRF ten selection'}
    write_json(path,result)
    with (args.output/'measurements.jsonl').open('w',encoding='utf-8') as stream:
        for number,q in enumerate(requests):
            order=methods[number%len(methods):]+methods[:number%len(methods)]
            for method in order:
                record=measure_query(q,method,runtime);records.append(record);save_row(stream,record);stream.flush()
            if (number+1)%10==0:
                result['completed_queries']=number+1;write_json(path,result)
                print(json.dumps({'phase':'latency','queries':number+1,'method_requests':len(records)}),flush=True)
    result.update(complete=True,completed_queries=len(requests),method_requests=len(records),
        results=summarize(records),finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(path,result);return result


def export(args):
    """Flat scientific CSVs from completed, independently recomputed measurements."""
    report=read(args.output/'benchmark.json');records=list(rows(args.output/'measurements.jsonl'))
    warmup=list(rows(args.output/'warmup-measurements.jsonl'))
    if not report['complete'] or len(records)!=700 or len(warmup)!=7:
        raise ValueError('Export requires the completed 100-query seven-method benchmark')
    recomputed=summarize(records)
    if recomputed!=report['results']:raise ValueError('Stored summary differs from the measured requests')
    expected=set(read(args.output/'preparation.json')['query_ids'])
    for name in recomputed:
        rr=[r for r in records if r['method']==name]
        if len(rr)!=100 or {r['query_id'] for r in rr}!=expected:
            raise ValueError('Each method must cover the same 100 requests exactly once')
    summary=[]
    for name,r in recomputed.items():
        summary.append({'method':name,'queries':r['queries'],
            **{k+'_ms':1000*v for k,v in r['latency_seconds'].items()},
            'encoder_calls':r['actual_encoder_calls'],'CE_calls':r['actual_CE']['calls'],
            'CE_logical_pairs':r['actual_CE']['logical_pairs'],'CE_actual_unique_pairs':r['actual_CE']['unique_pairs'],
            'CE_input_tokens':r['actual_CE']['input_tokens'],'CE_padded_tokens':r['actual_CE']['padded_tokens'],
            'CE_model_seconds':r['actual_CE']['model_seconds'],
            'head_p50_wall_ms':1000*r['stage_seconds']['head_wall_seconds']['p50'],
            'head_p50_CPU_ms':1000*r['stage_seconds']['head_cpu_seconds']['p50'],
            'source_file':'benchmark.json'})
    details=[]
    for r in records:
        details.append({**{k:r[k] for k in ('query_id','query_variant','method','latency_seconds')},
            **{'stage_'+k:v for k,v in r['stages'].items()},
            **{'encoder_'+k:v for k,v in r['encoder'].items()},
            **{'CE_'+k:v for k,v in r['CE'].items()},'source_file':'measurements.jsonl'})
    for filename,table in [('latency-summary.csv',summary),('latency-measurements.csv',details)]:
        names=list(dict.fromkeys(k for row in table for k in row))
        with (args.output/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(table)
        with (args.output/filename).open(encoding='utf-8-sig',newline='') as stream:
            if len(list(csv.DictReader(stream)))!=len(table):raise ValueError('CSV row count differs after readback')
    qa={'complete':True,'source_receipt_complete':True,'independent_summary_recompute_matches':True,
        'same_100_query_IDs_per_method':True,'measured_method_requests':len(records),
        'warmup_method_requests_excluded':len(warmup),'summary_CSV_rows':len(summary),
        'measurement_CSV_rows':len(details),'files':['latency-summary.csv','latency-measurements.csv'],
        'authoring':'flat UTF-8 CSV export; no workbook formulas or formatting'}
    write_json(args.output/'export-qa.json',qa);return qa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','run','export'),required=True)
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-'+k+'.jsonl') for k in ('exact','paraphrase')])
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/cost-benchmark'))
    p.add_argument('--frozen-recipe',type=Path)
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--index',type=Path,default=Path('data/index/hidate'))
    p.add_argument('--prepared',type=Path,default=Path('data/processed/hidate'))
    p.add_argument('--source',type=Path,default=Path('data/raw/hidate/full'))
    p.add_argument('--original-tree',type=Path,default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    p.add_argument('--full-tree',type=Path,default=Path('runs/hidate/revision-aware/native-development/models/lambdamart_revision.txt'))
    p.add_argument('--ce-development',type=Path,default=Path('runs/hidate/revision-aware/ce-training'))
    p.add_argument('--ce-cost-model',choices=('lambdamart_revision20','gate_revision20'),default='lambdamart_revision20')
    args=p.parse_args()
    {'prepare':prepare,'run':run,'export':export}[args.phase](args)


if __name__=='__main__':main()
