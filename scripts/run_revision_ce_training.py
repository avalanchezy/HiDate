"""TRAIN-only CE-conditioned trees/gates on the fixed historical-RRF ten.

Preparation and scoring preserve all original splits. VALID is used only for
evaluation; the address head/pool is never refitted by this worker.
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
from hidate.lambdamart import FIXED_PARAMETERS, lightgbm_module
from hidate.reranking import PairScoreCache
from hidate.revision import REVISION_FEATURES, normalize_scores, fit_gate, gate_scores
from evaluate_final import structural_result
from run_core_mechanisms import check_pair_context, prefill, read, rows, save_row
from run_revision_development import deliverable_indices, ordered_indices
from run_revision_dualview import fill_missing, ALPHAS
from run_revision_history_stage import load_compact


CE_FEATURES = ('historical_CE_query_rank', 'current_CE_query_rank')


def check_context(meta,pin,args):
    backend=getattr(args,'required_backend','math')
    if backend=='math':return check_pair_context(meta,pin)
    saved=meta.get('model_pin') or meta.get('model_pins',{}).get('reranker')
    if (not pin['model_id'].startswith('Qwen/Qwen3-Reranker-') or not saved
            or any(saved[k]!=pin[k] for k in ('model_id','revision'))
            or meta.get('max_length')!=1024 or meta.get('sdpa_backend')!='default'):
        raise ValueError('Qwen CE cache must match the fixed Qwen pin/default backend/1024 length')
    # Original Japanese DEV used the unchanged factory task; French instructions
    # are deliberately not accepted for this TRAIN/VALID experiment.
    from hidate.qwen3 import DEFAULT_INSTRUCTION
    if pin.get('conventions',{}).get('instruction',DEFAULT_INSTRUCTION)!=DEFAULT_INSTRUCTION:
        raise ValueError('Preserve the original Japanese Qwen DEV instruction')
    if meta.get('effective_model_pin') is not None and meta['effective_model_pin']!=pin:
        raise ValueError('Existing own CE receipt has a different effective instruction/pin')


def feature_sets(protocol):
    base = [k for k in protocol['rank_features'] if k != 'available_target']
    definitions = {'base13': base + ['historical_query_coverage', 'current_query_coverage'] + list(CE_FEATURES),
                   'revision20': base + list(REVISION_FEATURES) + list(CE_FEATURES)}
    if [len(v) for v in definitions.values()] != [13, 20]:
        raise ValueError('Use the fixed same-information 13/20 feature definitions')
    return definitions


def query_map(paths):
    result = {}
    for path in paths:
        for q in rows(path):
            if q['split'] in ('train', 'valid'):
                if q['query_id'] in result: raise ValueError('Duplicate original TRAIN/VALID query ID')
                result[q['query_id']] = q
    return result


def load_cache(args, queries, chunks, pin):
    dev = read(args.dev_metadata)
    if not dev['complete']: raise ValueError('Original DEV score cache is incomplete')
    check_context(dev,pin,args)
    cache = PairScoreCache(None); provenance = {'original': prefill(cache, [args.dev_pairs]), 'compact': []}
    for directory in args.compact_caches:
        meta = read(directory/'scoring.json')
        if not meta['complete']: raise ValueError('Completed compact CE source required')
        check_context(meta,pin,args)
        provenance['compact'].append(load_compact(cache,directory/'new-ce-pair-scores.jsonl',queries,chunks))
    provenance['unique_pairs'] = len(cache.cache)
    return cache, provenance


def selected_row(row, parents):
    """Keep H0's own available/bound/parent-distinct ten and original labels."""
    addresses = row['addresses']
    scores = [a['features']['max_history_rrf'] for a in addresses]
    indices = deliverable_indices(addresses, ordered_indices(addresses, scores), parents)
    return {'query_id':row['query']['query_id'], 'split':row['query']['split'],
        'candidates':[{'address':addresses[i]['address'],
            'historical_chunk_id':addresses[i]['historical_representative_chunk_id'],
            'current_chunk_id':addresses[i]['current_best_chunk_id'],
            'features':addresses[i]['features'], 'label':int(row['labels'][i]),
            'selection_rank':rank} for rank,i in enumerate(indices,1)]}


def prepared_rows(args, splits=('train','valid')):
    for split in splits:
        yield from rows(args.output/'prepared'/(split+'.jsonl'))


def requested_pairs(records):
    return [(r['query_id'], c[view]) for r in records for c in r['candidates']
            for view in ('historical_chunk_id','current_chunk_id')]


def prepare(args, *, expected_counts=None):
    path = args.output/'preparation.json'
    if path.exists() and read(path)['complete']: return read(path)
    started=time.perf_counter(); expected_counts=expected_counts or {'train':14030,'valid':2490}
    meta=read(args.development/'features/preparation.json')
    if not meta['complete'] or meta['seed']!=42 or meta['queries']!=expected_counts:
        raise ValueError('Use complete original TRAIN/VALID feature preparation')
    definitions=feature_sets(read(args.protocol)); queries=query_map(args.queries)
    if dict(Counter(q['split'] for q in queries.values()))!=expected_counts:
        raise ValueError('Original TRAIN/VALID query counts differ')
    chunks={c['chunk_id']:c for c in rows(args.index/'chunks.jsonl')}
    parents={cid:c['parent_passage_id'] for cid,c in chunks.items()}
    pin=read(args.stack)['models']['reranker'];cache,cache_sources=load_cache(args,queries,chunks,pin)
    out=args.output/'prepared';out.mkdir(parents=True,exist_ok=True)
    seen=set(); counts=Counter();statistics=Counter(); references=[]
    for split in ('train','valid'):
        with (out/(split+'.jsonl')).open('w',encoding='utf-8') as stream:
            for row in rows(args.development/'features'/(split+'.jsonl')):
                qid=row['query']['query_id'];q=queries[qid]
                if qid in seen or row['query']['split']!=split or q['split']!=split:
                    raise ValueError('Duplicate or changed original split')
                if any(row['query'][k]!=q[k] for k in ('family_id','query_variant','stratum','target_status','target_passage_id')):
                    raise ValueError('Feature metadata differs from original query')
                seen.add(qid);counts[split]+=1
                if len(row['addresses'])!=len(row['labels']):raise ValueError('Address labels must align')
                chosen=selected_row(row,parents)
                if any(c['historical_chunk_id'] is None for c in chosen['candidates']):
                    raise ValueError('A retrieved address lacks a historical representative')
                references.extend(requested_pairs([chosen]));save_row(stream,chosen)
                if split=='train':
                    statistics['original_address_rows']+=len(row['addresses'])
                    statistics['original_address_positives']+=sum(row['labels'])
                    statistics['original_positive_unavailable']+=sum(label and a['status']!='available'
                        for a,label in zip(row['addresses'],row['labels']))
                    statistics['selected_candidate_rows']+=len(chosen['candidates'])
                    statistics['selected_positive_rows']+=sum(c['label'] for c in chosen['candidates'])
                    statistics['selected_queries_without_positive']+=not any(c['label'] for c in chosen['candidates'])
                if len(seen)%1000==0:print(json.dumps({'phase':'prepare','query_rows':len(seen)}),flush=True)
    if seen!=set(queries) or dict(counts)!=expected_counts:raise ValueError('Incomplete original TRAIN/VALID coverage')
    missing={};unique=set();logical_hits=Counter()
    for qid,cid in references:
        key=(queries[qid]['query_text'],chunks[cid]['text']);unique.add(key)
        if key in cache.cache:logical_hits[queries[qid]['split']]+=1
        elif key not in missing:missing[key]={'query_id':qid,'chunk_id':cid,'split':queries[qid]['split']}
    with (args.output/'missing-pair-refs.jsonl').open('w',encoding='utf-8') as stream:
        for ref in missing.values():save_row(stream,ref)
    receipt={'complete':True,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'queries':dict(counts),
        'training_candidate_statistics':dict(statistics),'feature_sets':definitions,
        'logical_view_pairs':len(references),'unique_requested_pairs':len(unique),
        'logical_cache_hits_by_split':dict(logical_hits),'unique_missing_pairs':len(missing),
        'unique_missing_pairs_by_split':dict(Counter(r['split'] for r in missing.values())),
        'cache_sources':cache_sources,'candidate_policy':'fixed H0 maximum historical-RRF ten available/bound distinct parents',
        'representative':'highest retrieved historical RRF hit, chunk-ID tie break; original feature cache',
        'group_policy':'only deliverable TRAIN groups with a positive address enter final-stage fitting; original address heads unchanged',
        'new_TEST_rows':0,'training_runs':0,'new_neural_calls':0,'model_pin':{k:pin[k] for k in ('model_id','revision')},
        'required_backend':getattr(args,'required_backend','math'),'effective_model_pin':pin,
        'wall_seconds':time.perf_counter()-started,'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(path,receipt);return receipt


def add_ce_features(candidates):
    h=normalize_scores([c['historical_ce']['reranker_logit'] for c in candidates])
    c=normalize_scores([c['current_ce']['reranker_logit'] for c in candidates])
    features=[{**item['features'],CE_FEATURES[0]:float(hh),CE_FEATURES[1]:float(cc)}
              for item,hh,cc in zip(candidates,h,c)]
    return features,h,c


def training_arrays(records, feature_names):
    X=[];y=[];history=[];current=[];sizes=[];stats=Counter()
    for row in records:
        if row['split']!='train':raise ValueError('Final-stage models fit TRAIN only')
        labels=[c['label'] for c in row['candidates']]
        stats['all_train_queries']+=1
        if not any(labels):stats['no_deliverable_positive_skipped']+=1;continue
        features,h,c=add_ce_features(row['candidates'])
        X.extend([[f[k] for k in feature_names] for f in features]);y.extend(labels)
        history.extend(h);current.extend(c);sizes.append(len(labels))
        stats['fit_queries']+=1;stats['fit_candidate_rows']+=len(labels);stats['positive_rows']+=sum(labels)
        stats['pairwise_active_queries']+=any(label==0 for label in labels)
    return (np.asarray(X),np.asarray(y,dtype=np.int32),np.asarray(history),np.asarray(current),
            np.asarray(sizes,dtype=np.int32),dict(stats))


def score(args):
    receipt=read(args.output/'preparation.json')
    if not receipt['complete']:raise ValueError('Complete CPU preparation required before scoring')
    scoring_path=args.output/'scoring.json'
    if scoring_path.exists() and read(scoring_path)['complete']:return read(scoring_path)
    queries=query_map(args.queries);chunks={c['chunk_id']:c for c in rows(args.index/'chunks.jsonl')}
    pin=read(args.stack)['models']['reranker']
    if scoring_path.exists():check_context(read(scoring_path),pin,args)
    scoring={'complete':False,'job_id':os.environ.get('PBS_JOBID'),'seed':42,
        'model_pin':{k:pin[k] for k in ('model_id','revision')},'max_length':1024,
        'sdpa_backend':getattr(args,'required_backend','math'),'effective_model_pin':pin,
        'input_ID_arrays_saved':False,'training_runs':0,'new_TEST_rows':0}
    write_json(scoring_path,scoring);cache,provenance=load_cache(args,queries,chunks,pin)
    records=list(prepared_rows(args));references=requested_pairs(records)
    scoring['source_caches']=provenance
    scoring['work']=fill_missing(cache,references,queries,chunks,pin,args.output,batch_size=args.batch_size,
        required_backend=getattr(args,'required_backend','math'))
    enriched=args.output/'scored';enriched.mkdir(exist_ok=True)
    for split in ('train','valid'):
        with (enriched/(split+'.jsonl')).open('w',encoding='utf-8') as stream:
            for row in records:
                if row['split']!=split:continue
                q=queries[row['query_id']]
                for c in row['candidates']:
                    for view in ('historical','current'):
                        c[view+'_ce']=cache.cache[(q['query_text'],chunks[c[view+'_chunk_id']]['text'])]
                save_row(stream,row)
    scoring['complete']=True;write_json(scoring_path,scoring);return scoring


def fit(args):
    out=args.output/'models';out.mkdir(exist_ok=True);path=out/'fit.json'
    if path.exists() and read(path)['complete']:return read(path)
    if not read(args.output/'scoring.json')['complete']:raise ValueError('Complete actual CE scores required')
    definitions=feature_sets(read(args.protocol));records=list(rows(args.output/'scored/train.jsonl'))
    lgb=lightgbm_module();fits={}
    for variant,names in definitions.items():
        X,y,h,c,sizes,stats=training_arrays(records,names)
        for kind in ('lambdamart','gate'):
            name=kind+'_'+variant;receipt=out/(name+'.json')
            if receipt.exists() and read(receipt)['complete']:
                fits[name]=read(receipt);continue
            started=time.perf_counter()
            if kind=='lambdamart':
                params={**FIXED_PARAMETERS,'lambdarank_norm':True}
                model=lgb.LGBMRanker(**params);model.fit(X,y,group=sizes,feature_name=names)
                model.booster_.save_model(str(out/(name+'.txt')))
                result={'parameters':params,'actual_trees':model.booster_.num_trees(),'feature_names':names}
            else:
                result=fit_gate(X,h,c,y,sizes,names,regularization=.001)
            result.update(complete=True,seed=42,feature_names=names,statistics=stats,
                fit_seconds=time.perf_counter()-started,validation_used_for_fitting=False,
                fits=1,job_id=os.environ.get('PBS_JOBID'),experts='query-rank-normalized historical/current actual CE logits')
            write_json(receipt,result);fits[name]=result
            print(json.dumps({'phase':'fit','model':name,'fit_seconds':result['fit_seconds'],
                'converged':result.get('converged')}),flush=True)
    receipt={'complete':True,'seed':42,'models':fits,'fit_runs':4,'hyperparameter_search':False,
        'training_scope':'only positive-containing fixed-RRF deliverable10 TRAIN groups',
        'original_address_heads_refitted':False,'validation_used_for_fitting':False,
        'new_TEST_rows':0,'job_id':os.environ.get('PBS_JOBID')}
    write_json(path,receipt);return receipt


def fixed_scores(candidates):
    h=np.asarray([c['historical_ce']['reranker_logit'] for c in candidates])
    c=np.asarray([c['current_ce']['reranker_logit'] for c in candidates])
    result={'Historical_CE':np.asarray([c['historical_ce']['reranker_score'] for c in candidates]),
        'Current_CE':np.asarray([c['current_ce']['reranker_score'] for c in candidates]),
        'RRF_native':np.asarray([c['features']['max_history_rrf'] for c in candidates])}
    for mode in ('rank','minmax'):
        hh,cc=normalize_scores(h,mode),normalize_scores(c,mode)
        for alpha in ALPHAS:result[f'HC_{mode}_a{alpha:g}']=alpha*hh+(1-alpha)*cc
    return result


def evaluate(args):
    path=args.output/'evaluation.json'
    if path.exists() and read(path)['complete']:return read(path)
    started=time.perf_counter(); fits=read(args.output/'models/fit.json')
    if not fits['complete']:raise ValueError('Four complete fixed TRAIN fits required')
    lgb=lightgbm_module();trees={name:lgb.Booster(model_file=str(args.output/'models'/(name+'.txt')))
        for name in fits['models'] if name.startswith('lambdamart')}
    queries=query_map(args.queries);valid={k:q for k,q in queries.items() if q['split']=='valid'}
    chunks={c['chunk_id']:c for c in rows(args.index/'chunks.jsonl')}
    cells=defaultdict(lambda:defaultdict(list));panels=defaultdict(list);weights=defaultdict(list);seen=set()
    with (args.output/'valid-predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows(args.output/'scored/valid.jsonl'):
            q=valid[row['query_id']];cc=row['candidates']
            if row['split']!='valid' or q['query_id'] in seen:raise ValueError('Original VALID rows required once')
            seen.add(q['query_id']);features,h,c=add_ce_features(cc);scores=fixed_scores(cc)
            for name,model in fits['models'].items():
                X=np.asarray([[f[k] for k in model['feature_names']] for f in features])
                if name in trees:scores[name]=trees[name].predict(X,raw_score=True,num_threads=4)
                else:
                    scores[name],gate=gate_scores(model,X,h,c);weights[name].extend(gate.tolist())
            metrics={};selected={}
            for name,values in scores.items():
                order=sorted(range(len(cc)),key=lambda i:(-float(values[i]),cc[i]['selection_rank'],cc[i]['current_chunk_id']))[:4]
                selected[name]=order;m=structural_result(q,[chunks[cc[i]['current_chunk_id']] for i in order],'natural')
                metrics[name]={k:m[k] for k in ('target_mrr','target_recall_at_four','top_parent_correct','top_revision_correct','top_address_correct')}
                if q['target_status']=='available':
                    cells[name][q['query_variant']+'|'+q['stratum']].append(metrics[name])
                    panels[name+'|'+q['query_variant']+'|available'].append(metrics[name])
                    if q.get('source_quote_absent_from_target') is True:
                        panels[name+'|'+q['query_variant']+'|quote_absent'].append(metrics[name])
            save_row(stream,{**{k:q[k] for k in ('query_id','family_id','query_variant','stratum','target_status')},
                'split':'valid','quote_absent':q.get('source_quote_absent_from_target'),
                'current_chunk_ids':[c['current_chunk_id'] for c in cc],
                'conditions':{name:{'structural':m,'selected_common_indices':selected[name]} for name,m in metrics.items()}})
    if seen!=set(valid):raise ValueError('Original VALID evaluation coverage incomplete')
    def mean(v):return {'n':len(v),**{k:float(np.mean([r[k] for r in v]))
        for k in ('target_mrr','target_recall_at_four','top_parent_correct')}}
    heads={name:{'cells':{k:mean(v) for k,v in group.items()},
        'macro_mrr':float(np.mean([mean(v)['target_mrr'] for v in group.values()])),
        'macro_recall_at4':float(np.mean([mean(v)['target_recall_at_four'] for v in group.values()]))} for name,group in cells.items()}
    prior_path=getattr(args,'reference_evaluation',None)
    if prior_path is None and args.compact_caches:prior_path=args.compact_caches[0]/'evaluation.json'
    if prior_path is not None:
        prior=read(prior_path)
        for new,old in [('Historical_CE','Historical_CE_before_binding'),('Current_CE','Current_CE'),('RRF_native','RRF_native')]:
            if not math.isclose(heads[new]['macro_mrr'],prior['heads'][old]['macro_mrr'],abs_tol=1e-12):
                raise ValueError('Fixed common-ten original VALID reference changed: '+new)
    elif getattr(args,'native_reference',None) is not None:
        prior=read(args.native_reference)
        if not prior['complete'] or not math.isclose(heads['RRF_native']['macro_mrr'],
                prior['conditions']['history_rrf_native|own_pool']['mrr4'],abs_tol=1e-12):
            raise ValueError('Own-stack original VALID RRF native reference changed')
    else:raise ValueError('An own-stack original VALID reference is required')
    if any(len(h['cells'])!=4 for h in heads.values()):raise ValueError('Four available original VALID cells required')
    result={'complete':True,'seed':42,'split':'valid','queries':len(seen),'heads':heads,
        'panels':{k:mean(v) for k,v in panels.items()},'fit':fits,
        'gate_historical_weight_quantiles':{k:np.quantile(v,[0,.1,.5,.9,1]).tolist() for k,v in weights.items()},
        'candidate_policy':'same fixed historical-RRF ten; same requested-date current text output',
        'actual_CE_experts':'both query-rank-normalized CE logits, no cosine substitution',
        'fixed_baselines':'raw historical/current CE plus rank/minmax fixed five-alpha mixtures, all reported',
        'training_runs':4,'original_address_heads_refitted':False,'new_TEST_rows':0,
        'selection':'original VALID development; this worker does not freeze or run a TEST recipe',
        'job_id':os.environ.get('PBS_JOBID'),'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_wall_seconds':time.perf_counter()-started}
    write_json(path,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);root=Path('runs/hidate/revision-aware')
    p.add_argument('--phase',choices=('prepare','score','fit','evaluate','all'),required=True)
    p.add_argument('--output',type=Path,default=root/'ce-training')
    p.add_argument('--development',type=Path,default=root/'native-development')
    p.add_argument('--compact-caches',nargs='*',type=Path,default=[root/'diagnostics/revision-dualview',root/'diagnostics/history-stage'])
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    p.add_argument('--index',type=Path,default=Path('data/index/hidate'))
    p.add_argument('--protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--dev-metadata',type=Path,default=Path('runs/hidate/dev/scoring.json'))
    p.add_argument('--dev-pairs',type=Path,default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    p.add_argument('--reference-evaluation',type=Path)
    p.add_argument('--native-reference',type=Path)
    p.add_argument('--required-backend',choices=('math','default'),default='math')
    p.add_argument('--batch-size',type=int,default=32);args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase in ('prepare','all'):
        d=prepare(args);print(json.dumps({'phase':'prepared','unique_missing_pairs':d['unique_missing_pairs'],
            'queries':d['queries'],'by_split':d['unique_missing_pairs_by_split']}),flush=True)
    if args.phase in ('score','all'):score(args)
    if args.phase in ('fit','all'):fit(args)
    if args.phase in ('evaluate','all'):
        d=evaluate(args);print(json.dumps({'complete':d['complete'],'queries':d['queries'],
            'macro_mrr':{k:v['macro_mrr'] for k,v in d['heads'].items()}}),flush=True)


if __name__=='__main__':main()
