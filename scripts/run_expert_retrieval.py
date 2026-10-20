"""Frozen HiDate ranking comparisons on independently authored expert scenarios."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.lambdamart import lightgbm_module
from hidate.ranking import address_probabilities
from hidate.reranking import PairScoreCache
from hidate.revision import gate_scores
from hidate.scale import SharedCosines
from discover_final import vector_bank, query_segmenter, query_tokens
from evaluate_revision_confirmation import frozen, select_pools
from run_revision_scale import (read, rows, save_row, compact_chunks, scale_chunks,
                                    TimedDiscoveryEngine, compact_discovery, load_trees)
from run_revision_ce_training import add_ce_features, fixed_scores
from run_revision_dualview import fill_missing


ROOT = Path('runs/hidate/expert-benchmark')
DATA = Path('data/expert-benchmark')
REV = Path('runs/hidate/revision-aware')


def write_result(path, **values):
    result = dict(seed=42, job_id=os.environ.get('PBS_JOBID'), training_runs=0,
                  selection_updates=0, **values)
    write_json(path, result)
    return result


def discover(args):
    frozen(args)
    out = args.output/'discovery'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'discovery.json').exists() and read(out/'discovery.json')['complete']:
        return read(out/'discovery.json')
    queries = {q['query_id']: q for q in rows(args.queries)}
    index = read(args.index/'index.json')
    if not index['complete']:
        raise ValueError('Complete the shared expert-date corpus first')
    docs = vector_bank(args.index/'embeddings', 'chunk-ids.json')
    bank = vector_bank(args.query_index, 'query-ids.json')
    if not docs['manifest']['complete'] or not bank['manifest']['complete']:
        raise ValueError('Complete document and query encodings first')
    if any(docs['manifest']['model_pin'][key] != bank['manifest']['model_pin'][key]
           for key in ('model_id', 'revision')):
        raise ValueError('Document and query model pins differ')
    vector_rows = {qid: int(alias) for qid, alias in zip(bank['ids'], bank['aliases'])}
    if set(vector_rows) != set(queries):
        raise ValueError('Query encoding IDs differ')
    chunks = compact_chunks(args.index/'chunks.jsonl')
    if docs['ids'] != [c['chunk_id'] for c in chunks]:
        raise ValueError('Document vector order differs from chunk order')
    chunk_map = {c['chunk_id']: c for c in chunks}
    tokens = {r['chunk_id']: r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    targets = {tuple(r['address_key']): {'status': r['status'], 'passage_ids': r['passage_ids']}
               for r in rows(args.index/'target-addresses.jsonl')}
    bindings = {(r['law_id'], r['requested_asof']): {'status': r['status'], 'revision_id': r.get('revision_id')}
                for r in rows(args.index/'asof-bindings.jsonl')}
    law_ids = set(read(args.index/'nested-corpora.json')['1000'])
    requested_dates = {q['target_asof'] for q in queries.values()}
    if len(law_ids) != 1000 or any(bindings.get((law, day), {}).get('status') != 'collected'
                                 for law in law_ids for day in requested_dates):
        raise ValueError('Every requested date must cover the same 1000-law bank')
    shared = SharedCosines(docs['vectors'])
    engine = TimedDiscoveryEngine(scale_chunks(chunks, law_ids), tokens, shared,
                                  docs['aliases'], docs['ids'], targets, bindings, top_k=50, rrf_k=60)
    timing = Counter()
    started = time.perf_counter()
    segment = query_segmenter()
    total = 0
    with (out/'candidates.jsonl').open('w', encoding='utf-8') as stream:
        for q in sorted(queries.values(), key=lambda x: (x['target_asof'], x['query_id'])):
            terms = query_tokens(q['query_text'], segment)
            previous = engine.binding_seconds
            begin = time.perf_counter()
            d = engine.discover(q, terms, bank['vectors'][vector_rows[q['query_id']]])
            elapsed = time.perf_counter()-begin
            bound = engine.binding_seconds-previous
            begin = time.perf_counter()
            record = compact_discovery(q, d, chunk_map, tokens, terms, '1000')
            record['timing'] = dict(retrieval_seconds=elapsed-bound, binding_seconds=bound,
                                    feature_seconds=time.perf_counter()-begin)
            timing.update(record['timing'])
            save_row(stream, record)
            engine._cache.clear()
            total += 1
            if total % 20 == 0:
                stream.flush()
                print(json.dumps({'phase': 'discovery', 'completed': total, 'total': len(queries)}), flush=True)
    return write_result(out/'discovery.json', complete=total==len(queries), queries=total,
                        law_count=len(law_ids), target_dates=sorted(requested_dates),
                        index=str(args.index), gold_labels_read=False,
                        timing=dict(timing), wall_seconds=time.perf_counter()-started)


def prepare(args):
    frozen(args)
    out = args.output/'final'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'preparation.json').exists() and read(out/'preparation.json')['complete']:
        return read(out/'preparation.json')
    discovery = read(args.output/'discovery/discovery.json')
    if not discovery['complete']:
        raise ValueError('Complete the expert query discovery first')
    chunks = {c['chunk_id']: c for c in compact_chunks(args.index/'chunks.jsonl')}
    parents = {cid: c['parent_passage_id'] for cid, c in chunks.items()}
    trees, definitions = load_trees(args)
    lr = read(args.original_lr)
    count = 0
    with (out/'prepared.jsonl').open('w', encoding='utf-8') as stream:
        for d in rows(args.output/'discovery/candidates.jsonl'):
            addresses = d['addresses']
            scores = {'Original_LR': address_probabilities(lr, [a['features'] for a in addresses])}
            for name, model in trees.items():
                X = np.asarray([[a['features'][k] for k in definitions[name]['feature_names']] for a in addresses])
                scores[name] = model.predict(X, raw_score=True, num_threads=4) if len(X) else np.array([])
            pools = select_pools(addresses, scores, parents)
            needed = sorted(set(i for pool in pools.values() for i in pool))
            mapping = {old: new for new, old in enumerate(needed)}
            candidates = [{'address': addresses[i]['address'],
                           'historical_chunk_id': addresses[i]['historical_representative_chunk_id'],
                           'current_chunk_id': addresses[i]['current_best_chunk_id'],
                           'features': addresses[i]['features'],
                           'original_lr_score': float(scores['Original_LR'][i])} for i in needed]
            direct, seen = [], set()
            for hit in d['asof']['hybrid']:
                cid = hit['chunk_id']
                if parents[cid] in seen:
                    continue
                seen.add(parents[cid])
                direct.append(dict(current_chunk_id=cid, native_score=hit['score'], selection_rank=len(direct)+1))
                if len(direct) == 10:
                    break
            record = {k: d[k] for k in ('query_id', 'query_text', 'query_variant', 'target_asof')}
            record.update(candidates=candidates, pools={n: [mapping[i] for i in ids] for n, ids in pools.items()},
                          asof=direct, discovered_addresses=[a['address'] for a in addresses])
            save_row(stream, record)
            count += 1
    return write_result(out/'preparation.json', complete=count==discovery['queries'], queries=count,
                        models=definitions, gold_labels_read=False)


def score(args):
    out = args.output/'final'
    if (out/'scoring.json').exists() and read(out/'scoring.json')['complete']:
        return read(out/'scoring.json')
    if not read(out/'preparation.json')['complete']:
        raise ValueError('Complete expert candidate preparation first')
    prepared = list(rows(out/'prepared.jsonl'))
    queries = {q['query_id']: q for q in prepared}
    chunks = {c['chunk_id']: c for c in rows(args.index/'chunks.jsonl')}
    pin = read(args.stack)['models']['reranker']
    references = []
    for r in prepared:
        qid, cc = r['query_id'], r['candidates']
        references.extend((qid, c['historical_chunk_id']) for c in cc)
        references.extend((qid, cc[i]['current_chunk_id']) for i in set(r['pools']['RRF']+r['pools']['Original_LR']))
        references.extend((qid, c['current_chunk_id']) for c in r['asof'])
    cache = PairScoreCache(None)
    work = fill_missing(cache, references, queries, chunks, pin, out, batch_size=32)
    with (out/'scored.jsonl').open('w', encoding='utf-8') as stream:
        for r in prepared:
            text = r['query_text']
            for i, c in enumerate(r['candidates']):
                c['historical_ce'] = cache.cache[(text, chunks[c['historical_chunk_id']]['text'])]
                if i in set(r['pools']['RRF']+r['pools']['Original_LR']):
                    c['current_ce'] = cache.cache[(text, chunks[c['current_chunk_id']]['text'])]
            for c in r['asof']:
                c['current_ce'] = cache.cache[(text, chunks[c['current_chunk_id']]['text'])]
            save_row(stream, r)
    return write_result(out/'scoring.json', complete=True, queries=len(prepared), work=work,
                        model_pin={k: pin[k] for k in ('model_id', 'revision')},
                        gold_labels_read=False, max_length=1024, sdpa_backend='math')


def ranked_ids(candidates, scores):
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]),
                   candidates[i]['selection_rank'], candidates[i]['current_chunk_id']))
    return [candidates[i]['current_chunk_id'] for i in order[:4]]


def rank_conditions(r, chunks, fits, trees, alpha):
    conditions = {}
    for head, pool in r['pools'].items():
        cc = [{**r['candidates'][i], 'selection_rank': rank} for rank, i in enumerate(pool, 1)]
        conditions[head+'_native'] = ranked_ids(cc, -np.arange(len(cc)))
        conditions[head+'_historical_CE'] = ranked_ids(cc, [c['historical_ce']['reranker_score'] for c in cc])
    cc = [{**r['candidates'][i], 'selection_rank': rank} for rank, i in enumerate(r['pools']['RRF'], 1)]
    views = ['current_ce' if chunks[c['historical_chunk_id']]['parent_passage_id'] == chunks[c['current_chunk_id']]['parent_passage_id']
             else 'historical_ce' for c in cc]
    conditions['RRF_identity_view'] = ranked_ids(cc, [c[v]['reranker_score'] for c, v in zip(cc, views)])
    features, historical, current = add_ce_features(cc)
    for name, values in fixed_scores(cc).items():
        conditions['common_'+name] = ranked_ids(cc, values)
    for name, model in fits.items():
        X = np.asarray([[f[k] for k in model['feature_names']] for f in features])
        values = (trees[name].predict(X, raw_score=True, num_threads=4) if name in trees else
                  gate_scores(model, X, historical, current)[0]) if cc else np.array([])
        conditions['ce_'+name] = ranked_ids(cc, values)
    cc = [{**r['candidates'][i], 'selection_rank': rank} for rank, i in enumerate(r['pools']['Original_LR'], 1)]
    conditions['Original_HiDate_LR_CE'] = ranked_ids(cc, [alpha*c['original_lr_score']+(1-alpha)*c['current_ce']['reranker_score'] for c in cc])
    cc = r['asof']
    conditions['AsOf_hybrid_native'] = ranked_ids(cc, [c['native_score'] for c in cc])
    conditions['AsOf_hybrid_CE'] = ranked_ids(cc, [c['current_ce']['reranker_score'] for c in cc])
    return conditions


def evaluate(args):
    from hidate.expert_metrics import retrieval_metrics
    out = args.output/'final'
    decision = frozen(args)
    if not read(out/'scoring.json')['complete']:
        raise ValueError('Complete frozen expert CE scoring first')
    gold = {q['query_id']: q for q in rows(args.gold)}
    chunks = {c['chunk_id']: c for c in compact_chunks(args.index/'chunks.jsonl')}
    fits = read(args.ce_models/'fit.json')['models']
    lgb = lightgbm_module()
    trees = {name: lgb.Booster(model_file=str(args.ce_models/(name+'.txt')))
             for name in fits if name.startswith('lambdamart')}
    alpha = read(args.original_policy)['alpha']
    panels, records = defaultdict(Counter), 0
    with (out/'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for r in rows(out/'scored.jsonl'):
            q = gold[r['query_id']]
            ranked = rank_conditions(r, chunks, fits, trees, alpha)
            if decision['selected_method'] not in ranked:
                raise ValueError('Frozen method absent')
            result = {}
            for method, ids in ranked.items():
                metrics = retrieval_metrics(q, [chunks[cid] for cid in ids], k=4)
                result[method] = dict(evidence_chunk_ids=ids, structural=metrics)
                if (q['answerability'] == 'answerable'
                        and q.get('retrieval_target_type', 'provision_set') == 'provision_set'):
                    for group in ('answerable', 'category='+q['category']):
                        acc = panels[method, group]
                        acc['n'] += 1
                        for key, value in metrics.items():
                            if value is not None:
                                acc[key] += value
            save_row(stream, dict(query_id=r['query_id'], target_asof=r['target_asof'],
                                  family_id=q['family_id'], category=q['category'],
                                  answerability=q['answerability'],
                                  retrieval_target_type=q.get('retrieval_target_type', 'provision_set'),
                                  conditions=result))
            records += 1
    aggregates = defaultdict(dict)
    for (name, group), values in panels.items():
        aggregates[name][group] = {'n': values['n'], **{k: v/values['n'] for k, v in values.items() if k != 'n'}}
    return write_result(out/'evaluation.json', complete=records==len(gold), queries=records,
                        family_count=len({q['family_id'] for q in gold.values()}),
                        selected_method=decision['selected_method'], freeze=decision,
                        answerability=dict(Counter(q['answerability'] for q in gold.values())),
                        retrieval_eligibility=dict(Counter(
                            ('insufficient' if q['answerability'] != 'answerable' else
                             q.get('retrieval_target_type', 'provision_set')) for q in gold.values())),
                        panels=dict(aggregates), gold_labels_read=True,
                        metric_scope='supplied annotated provision sets; answer correctness is evaluated separately')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('discover', 'prepare', 'score', 'evaluate'))
    p.add_argument('--queries', type=Path, default=DATA/'unlabeled/queries.jsonl')
    p.add_argument('--gold', type=Path, default=DATA/'sealed/gold.jsonl')
    p.add_argument('--index', type=Path, default=Path('data/index/hidate/expert-benchmark/corpus'))
    p.add_argument('--query-index', type=Path, default=Path('data/index/hidate/expert-benchmark/queries'))
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--freeze', type=Path, default=REV/'frozen-recipe.json')
    p.add_argument('--development', type=Path, default=REV/'native-development')
    p.add_argument('--original-tree', type=Path, default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    p.add_argument('--original-lr', type=Path, default=Path('runs/hidate/final/ranker-used.json'))
    p.add_argument('--original-policy', type=Path, default=Path('runs/hidate/policy/frozen.json'))
    p.add_argument('--rank-protocol', type=Path, default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--ce-models', type=Path, default=REV/'ce-training/models')
    p.add_argument('--stack', type=Path, default=Path('models/ruri/model-stack.json'))
    args = p.parse_args()
    result = globals()[args.phase](args)
    print(json.dumps({'phase': args.phase, 'complete': result['complete']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
