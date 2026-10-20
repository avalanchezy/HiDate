"""Frozen-Ruri historical/current CE diagnostics on original VALID only."""
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
from hidate.reranking import PairScoreCache
from hidate.revision import normalize_scores
from evaluate_final import structural_result
from run_core_mechanisms import check_pair_context, prefill, rows, read, save_row


ALPHAS = (0., .25, .5, .75, 1.)
STRUCTURAL = ('target_recall_at_four', 'target_mrr', 'top_parent_correct',
              'top_revision_correct', 'top_address_correct')


def recipes(candidates):
    """All fixed recipes share the same ten; no alpha is chosen here."""
    h = np.array([c['historical_ce']['reranker_logit'] for c in candidates])
    c = np.array([c['current_ce']['reranker_logit'] for c in candidates])
    hp = np.array([c['historical_ce']['reranker_score'] for c in candidates])
    cp = np.array([c['current_ce']['reranker_score'] for c in candidates])
    rrf = np.array([c['rrf_score'] for c in candidates])
    result = {'Historical_CE_before_binding': hp, 'Current_CE': cp, 'RRF_native': rrf,
              'Historical_CE_before_binding_raw': h, 'Current_CE_raw': c}
    for mode, hh, cc in (('raw', h, c), ('sigmoid', hp, cp),
                         ('rank', normalize_scores(h), normalize_scores(c)),
                         ('minmax', normalize_scores(h, 'minmax'), normalize_scores(c, 'minmax'))):
        for alpha in ALPHAS:
            result[f'HC_{mode}_a{alpha:g}'] = alpha*hh+(1-alpha)*cc
    # Preserve the original CE sigmoid input for this scale-only RRF comparison.
    for mode in ('rank', 'minmax'):
        hh, cc = normalize_scores(rrf, mode), normalize_scores(cp, mode)
        for alpha in ALPHAS:
            result[f'RRF_CE_{mode}_a{alpha:g}'] = alpha*hh+(1-alpha)*cc
    return result


def fill_missing(cache, references, queries, chunks, pin, output, *, factory=None, batch_size=32,
                 required_backend='math'):
    """Persist compact reference/score/token receipts, never input-ID matrices."""
    source = output/'new-ce-pair-scores.jsonl'
    total = Counter(calls=0, pairs=0, input_tokens=0, padded_tokens=0, model_seconds=0.)
    if source.exists():
        for row in rows(source):
            result = row['result']
            for i, ref in enumerate(row['pair_refs']):
                key = (queries[ref['query_id']]['query_text'], chunks[ref['chunk_id']]['text'])
                cache.cache[key] = {'reranker_score': result['scores'][i],
                    'reranker_logit': result['logits'][i], 'pair_token_count': result['token_counts'][i]}
            total.update(calls=1, pairs=len(row['pair_refs']), input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1], model_seconds=result['seconds'])
    unique = {}
    for qid, cid in references:
        key = (queries[qid]['query_text'], chunks[cid]['text'])
        if key not in cache.cache and key not in unique:
            unique[key] = {'query_id': qid, 'chunk_id': cid}
    pending = list(unique.items()); this_job = Counter(calls=0,pairs=0,input_tokens=0,padded_tokens=0,model_seconds=0.)
    model = None
    with source.open('a', encoding='utf-8') as stream:
        for offset in range(0, len(pending), batch_size):
            selected = pending[offset:offset+batch_size]
            if model is None:
                if factory is None:
                    from hidate.models import create_reranker
                    model = create_reranker(pin, device='cuda', max_length=1024)
                else:
                    model = factory(pin)
            raw = model.score([key for key, ref in selected], include_inputs=False)
            result = {k: raw[k] for k in ('scores','logits','token_counts','total_tokens','input_shape','sdpa_backend','seconds')}
            if (result['sdpa_backend'] != required_backend or len(result['scores']) != len(selected)
                    or sum(result['token_counts']) != result['total_tokens']
                    or not all(math.isfinite(v) for v in result['scores']+result['logits'])):
                raise ValueError('Frozen CE compact receipt is inconsistent')
            if required_backend=='default':
                if (raw.get('score_format'),raw.get('logit_format'))!=('conditional_yes_no','yes_minus_no'):
                    raise ValueError('Qwen CE must preserve conditional yes/no score and margin semantics')
                result.update(score_format=raw['score_format'],logit_format=raw['logit_format'])
            save_row(stream, {'pair_refs': [ref for key, ref in selected], 'result': result})
            stream.flush()
            for i, (key, ref) in enumerate(selected):
                cache.cache[key] = {'reranker_score': result['scores'][i],
                    'reranker_logit': result['logits'][i], 'pair_token_count': result['token_counts'][i]}
            work = dict(calls=1, pairs=len(selected), input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1], model_seconds=result['seconds'])
            total.update(work); this_job.update(work)
            if this_job['calls'] % 20 == 0:
                print(json.dumps({'phase':'historical_CE','new_pairs':this_job['pairs'],'pending_pairs':len(pending)}),flush=True)
    return {'logical_view_pairs': len(references), 'unique_requested_pairs': len({
        (queries[qid]['query_text'], chunks[cid]['text']) for qid, cid in references}),
        'this_job': dict(this_job), 'all_new_pair_work': dict(total), 'model_loaded_this_job': model is not None}


def evaluate(args, *, factory=None, expected_valid=2490):
    started = time.perf_counter(); output = args.output_dir
    if (output/'evaluation.json').exists() and read(output/'evaluation.json')['complete']:
        return read(output/'evaluation.json')
    cpu = read(args.prepared_dir/'evaluation.json')
    if not cpu['complete'] or cpu['split'] != 'valid' or cpu['queries'] != expected_valid:
        raise ValueError('Complete original VALID common-H0 preparation is required')
    queries = {q['query_id']: q for path in args.queries for q in rows(path) if q['split'] == 'valid'}
    if len(queries) != expected_valid:
        raise ValueError('Use all original2490 VALID IDs')
    prepared = list(rows(args.prepared_dir/'ce-before-binding-inputs.jsonl'))
    if len(prepared) != len(queries) or {r['query_id'] for r in prepared} != set(queries):
        raise ValueError('Common-H0 candidate coverage differs from original VALID')
    for row in prepared:
        q = queries[row['query_id']]
        if row['split'] != 'valid' or row['candidate_pool'] != 'history_rrf_top10' or row['query_text'] != q['query_text']:
            raise ValueError('Use only original VALID raw queries and frozen H0 candidates')
        if any(row[k] != q[k] for k in ('family_id','query_variant','target_asof','target_status')):
            raise ValueError('Common-H0 metadata differs from original VALID')
    pin = read(args.stack)['models']['reranker']; meta = read(args.dev_metadata)
    if not meta['complete']: raise ValueError('Original DEV CE cache must be complete')
    check_pair_context(meta, pin)
    output.mkdir(parents=True,exist_ok=True)
    scoring_path = output/'scoring.json'
    if scoring_path.exists(): check_pair_context(read(scoring_path),pin)
    scoring = {'complete':False,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'split':'valid',
        'model_pin':{k:pin[k] for k in ('model_id','revision')},'max_length':1024,'sdpa_backend':'math',
        'source_cache':str(args.dev_pair_receipts),'cache_key':'exact raw query and complete chunk text',
        'training_runs':0,'new_TEST_rows':0,'input_ID_arrays_saved':False}
    write_json(scoring_path,scoring)
    cache = PairScoreCache(None); scoring['prefill'] = prefill(cache,[args.dev_pair_receipts])
    chunks = {c['chunk_id']:c for c in rows(args.chunks)}
    references = [(row['query_id'], candidate[view]) for row in prepared for candidate in row['candidates']
                  for view in ('historical_chunk_id','current_chunk_id')]
    scoring['work'] = fill_missing(cache,references,queries,chunks,pin,output,factory=factory,batch_size=args.batch_size)
    scoring['complete']=True;write_json(scoring_path,scoring)
    # Native/CE as-of references use their original own ten and no new scoring.
    asof = {}
    for row in rows(args.dev_reranked):
        qid=row['query_id']
        if qid not in queries or qid in asof or row['split']!='valid':
            raise ValueError('Original as-of reference cache must contain each VALID ID once')
        asof[qid]=[{k:c[k] for k in ('chunk_id','selection_rank','reranker_score')}
            for c in row['conditions']['asof_hybrid']['reranked_candidates']]
    if set(asof)!=set(queries):raise ValueError('Original as-of reference VALID coverage differs')
    cells=defaultdict(lambda:defaultdict(list));panels=defaultdict(list);complement=defaultdict(Counter)
    score_values=defaultdict(list);conditions_seen=set();n=0
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in prepared:
            q=queries[row['query_id']]; candidates=[]
            for c in row['candidates']:
                hkey=(q['query_text'],chunks[c['historical_chunk_id']]['text'])
                ckey=(q['query_text'],chunks[c['current_chunk_id']]['text'])
                candidates.append({**c,'historical_ce':cache.cache[hkey],'current_ce':cache.cache[ckey]})
                for view in ('historical','current'):
                    score_values[view+'_logit'].append(candidates[-1][view+'_ce']['reranker_logit'])
                    score_values[view+'_sigmoid'].append(candidates[-1][view+'_ce']['reranker_score'])
            vectors=recipes(candidates);methods={};selected={}
            for name,scores in vectors.items():
                order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),candidates[i]['selection_rank'],candidates[i]['current_chunk_id']))[:4]
                selected[name]=order
                methods[name]=[chunks[candidates[i]['current_chunk_id']] for i in order]
            for name,key in (('AsOf_hybrid_native',lambda c:c['selection_rank']),
                             ('AsOf_hybrid_existing_CE',lambda c:(-c['reranker_score'],c['selection_rank'],c['chunk_id']))):
                chosen=sorted(asof[q['query_id']],key=key)[:4]
                selected[name]=[c['chunk_id'] for c in chosen]
                methods[name]=[chunks[c['chunk_id']] for c in chosen]
            metrics={name:{k:structural_result(q,evidence,'natural')[k] for k in STRUCTURAL}
                     for name,evidence in methods.items()}
            conditions_seen.update(metrics)
            if q['target_status']=='available':
                for name,m in metrics.items():
                    cells[name][q['query_variant']+'|'+q['stratum']].append(m)
                    panels[name+'|'+q['query_variant']+'|available'].append(m)
                    if q.get('source_quote_absent_from_target') is True:
                        panels[name+'|'+q['query_variant']+'|quote_absent'].append(m)
                for left,right in (('Historical_CE_before_binding','Current_CE'),('RRF_native','Current_CE')):
                    state=('both' if metrics[left]['target_recall_at_four'] and metrics[right]['target_recall_at_four'] else
                        'left_only' if metrics[left]['target_recall_at_four'] else
                        'right_only' if metrics[right]['target_recall_at_four'] else 'neither')
                    complement[left+'__'+right+'|'+q['query_variant']][state]+=1
            compact=[{k:c[k] for k in ('address','historical_chunk_id','current_chunk_id','rrf_score','selection_rank')}
                | {'historical_ce':c['historical_ce'],'current_ce':c['current_ce']} for c in candidates]
            saved={k:q[k] for k in ('query_id','family_id','query_variant','stratum','target_asof','target_status')}
            save_row(stream,{**saved,'split':'valid','quote_absent':q.get('source_quote_absent_from_target'),
                'candidate_pool':'history_rrf_top10','candidates':compact,
                'conditions':{name:{'structural':metric,'selected_common_indices_or_asof_chunk_ids':selected[name]}
                              for name,metric in metrics.items()}})
            n+=1
            if n%200==0:stream.flush();print(json.dumps({'phase':'metrics','valid_queries':n}),flush=True)
    if n!=expected_valid:raise ValueError('VALID rows incomplete')
    def summarize(values):
        return {'n':len(values),'mrr_at4':sum(v['target_mrr'] for v in values)/len(values),
                'recall_at4':sum(v['target_recall_at_four'] for v in values)/len(values),
                'parent_at1':sum(v['top_parent_correct'] for v in values)/len(values)}
    heads={name:{'cells':{key:summarize(v) for key,v in group.items()},
        'macro_mrr':sum(summarize(v)['mrr_at4'] for v in group.values())/len(group)} for name,group in cells.items()}
    if expected_valid==2490:
        for new,old in (('RRF_native','RRF_native_common_rrf10'),('Current_CE','Current_CE_common_rrf10')):
            if not math.isclose(heads[new]['macro_mrr'],cpu['heads'][old]['macro_mrr'],abs_tol=1e-12):
                raise ValueError('Original common-H0 comparator changed: '+new)
        if any(len(head['cells'])!=4 for head in heads.values()):raise ValueError('Four available VALID cells required')
    result={'complete':True,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'split':'valid','queries':n,
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started,
        'heads':heads,'panels':{key:summarize(v) for key,v in panels.items()},
        'complementarity_available_only':{key:{**dict(v),'n_available':sum(v.values())} for key,v in complement.items()},
        'conditions':sorted(conditions_seen),'fixed_alphas':list(ALPHAS),'alpha_direction':'history weight',
        'normalization':{'HC_raw':'historical/current CE logits','HC_sigmoid':'original CE sigmoid scores',
            'HC_rank_minmax':'within-query transforms of CE logits',
            'RRF_CE_rank_minmax':'same within-query transform of RRF and original current CE sigmoid'},
        'score_distributions':{key:{'n':len(v),'min':min(v),'max':max(v),'q10_q50_q90':np.quantile(v,[.1,.5,.9]).tolist()} for key,v in score_values.items()},
        'common_pool':'original historical-RRF H0 ten bound parents; each historical CE score selects an address then delivers its fixed current chunk',
        'asof_references':'own original hybrid ten; native selection-rank or existing CE, no address binding/new scoring',
        'scoring':scoring,'training_runs':0,'new_TEST_rows':0,
        'selection':'all fixed grids reported; this diagnostic job does not select a new primary method',
        'gate_scope':'No CE gate trained; separate first-round gate uses RRF/current cosine only'}
    write_json(output/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path('runs/hidate/revision-aware/diagnostics')
    p.add_argument('--prepared-dir',type=Path,default=root/'rank-head-valid')
    p.add_argument('--output-dir',type=Path,default=root/'revision-dualview')
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    p.add_argument('--chunks',type=Path,default=Path('data/index/hidate/chunks.jsonl'))
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--dev-metadata',type=Path,default=Path('runs/hidate/dev/scoring.json'))
    p.add_argument('--dev-reranked',type=Path,default=Path('runs/hidate/dev/reranked.jsonl'))
    p.add_argument('--dev-pair-receipts',type=Path,default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    result=evaluate(args)
    print(json.dumps({'complete':result['complete'],'queries':result['queries'],
        'macro_mrr':{key:v['macro_mrr'] for key,v in result['heads'].items()}}),flush=True)


if __name__=='__main__':main()
