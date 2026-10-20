"""CPU native heads on the original TEST queries in the existing scale bank.

Discovery persists all historical addresses and dated bindings for later CE.
No model is fitted, no neural model is loaded, and no method is selected here.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.discovery import DiscoveryEngine, chunk_address
from hidate.revision import enrich_addresses
from hidate.scale import SharedCosines
from discover_final import vector_bank, query_segmenter, query_tokens
from evaluate_final import structural_result


TREE_HEADS = ('Original_LM10', 'Dual_relevance_LM11', 'Full_LM18', 'Lean_noalignment_LM16')
METHODS = ('RRF_native', 'AsOf_hybrid_native') + TREE_HEADS
METRICS = ('top_address_correct', 'top_parent_correct', 'top_revision_correct',
           'target_recall_at_four', 'target_mrr')
CHUNK_FIELDS = ('chunk_id', 'parent_passage_id', 'law_id', 'revision_id', 'scope',
                'article_num', 'slot_id', 'source_role', 'source_kind', 'source_id',
                'asof_dates', 'family_id', 'url')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def receipt(**values):
    return {'seed': 42, 'job_id': os.environ.get('PBS_JOBID'),
        'at_utc': datetime.now(timezone.utc).isoformat(), 'training_runs': 0,
        'neural_model_calls': 0, 'encoder_calls': 0, 'reranker_calls': 0,
        'reader_calls': 0, 'dev_selections': 0, **values}


def compact_chunks(path):
    # Text remains in the frozen source index; native features use saved tokens.
    return [{k: row[k] for k in CHUNK_FIELDS if k in row} for row in rows(path)]


def scale_chunks(chunks, law_ids):
    # Static guidance has source_id, not law_id. Its existing role stays intact.
    return [c for c in chunks if c['source_role'] == 'full' and
            (c['source_kind'] == 'guidance' or c['law_id'] in law_ids)]


class TimedDiscoveryEngine(DiscoveryEngine):
    """Time the existing binding plus bound-current feature construction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding_seconds = 0.

    def _addresses(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._addresses(*args, **kwargs)
        self.binding_seconds += time.perf_counter()-started
        return result


def compact_discovery(query, discovery, chunks, tokens, terms, scale):
    """Enrich the full pool before any head selects or filters candidates."""
    addresses = enrich_addresses(discovery, terms, tokens)
    kept = ('address', 'source_kind', 'source_id', 'status', 'revision_id',
        'address_key', 'target_asof', 'parent_passage_ids', 'target_passage_ids',
        'current_best_chunk_id', 'current_feature_chunk_count', 'history_parent_passage_ids',
        'history_chunk_hits', 'historical_representative_chunk_id',
        'historical_normalized', 'current_normalized', 'features')
    compact = []
    for address in addresses:
        hid, cid = address['historical_representative_chunk_id'], address['current_best_chunk_id']
        compact.append({**{k: address[k] for k in kept if k in address},
            'historical_parent_passage_id': chunks[hid]['parent_passage_id'] if hid else None,
            'current_parent_passage_id': chunks[cid]['parent_passage_id'] if cid else None})
    metadata = {k:query[k] for k in ('family_id','stratum','target_status') if k in query}
    if 'source_quote_absent_from_target' in query:
        metadata['quote_absent'] = query['source_quote_absent_from_target']
    return {'scale': str(scale), **{k: query[k] for k in ('query_id', 'query_text',
        'query_variant', 'split', 'source_role', 'target_asof')}, **metadata,
        'addresses': compact, 'asof': discovery['asof'], 'history': discovery['history'],
        'bank_counts': discovery['bank_counts'],
        'current_feature_chunks': discovery['current_feature_chunks']}


def native_predictions(query, cached, chunks, tree_scores):
    """Every address head delivers the same already-bound canonical chunk."""
    addresses = cached['addresses']
    score_map = {'RRF_native': [a['features']['max_history_rrf'] for a in addresses], **tree_scores}
    conditions = {}
    for method, scores in score_map.items():
        if len(scores) != len(addresses):
            raise ValueError('One score per full discovered address is required')
        order = sorted(range(len(addresses)), key=lambda i: (-float(scores[i]), tuple(addresses[i]['address'])))
        candidates, seen = [], set()
        for i in order:
            address = addresses[i]
            cid = address['current_best_chunk_id']
            if address['status'] != 'available' or cid is None:
                continue
            parent = chunks[cid]['parent_passage_id']
            if parent in seen:
                continue
            seen.add(parent)
            candidates.append({'address_index': i, 'address': address['address'],
                'current_chunk_id': cid, 'current_parent_passage_id': parent,
                'historical_chunk_id': address['historical_representative_chunk_id'],
                'native_score': float(scores[i]), 'selection_rank': len(candidates)+1})
            if len(candidates) == 10:
                break
        evidence = [chunks[c['current_chunk_id']] for c in candidates[:4]]
        conditions[method] = {'candidate_pool': 'own_pool',
            'full_address_scores': [float(s) for s in scores], 'candidates': candidates,
            'evidence_chunk_ids': [c['chunk_id'] for c in evidence],
            'evidence_parent_ids': [c['parent_passage_id'] for c in evidence],
            'structural': structural_result(query, evidence, 'natural')}
    direct, seen = [], set()
    for hit in cached['asof']['hybrid']:
        chunk = chunks[hit['chunk_id']]
        parent = chunk['parent_passage_id']
        if parent in seen:
            continue
        seen.add(parent)
        direct.append({'address': list(chunk_address(chunk)), 'current_chunk_id': chunk['chunk_id'],
            'current_parent_passage_id': parent, 'native_score': hit['score'],
            'selection_rank': len(direct)+1})
        if len(direct) == 10:
            break
    evidence = [chunks[c['current_chunk_id']] for c in direct[:4]]
    conditions['AsOf_hybrid_native'] = {'candidate_pool': 'own_asof_hybrid', 'candidates': direct,
        'evidence_chunk_ids': [c['chunk_id'] for c in evidence],
        'evidence_parent_ids': [c['parent_passage_id'] for c in evidence],
        'structural': structural_result(query, evidence, 'natural')}
    return {k: cached[k] for k in ('scale', 'query_id', 'query_variant', 'family_id',
        'stratum', 'target_asof', 'target_status', 'quote_absent')} | {'conditions': conditions}


def original_queries(args, protocol):
    if (protocol['expected_queries_per_variant'], protocol['expected_test_queries'],
            protocol['discovery_top_k'], protocol['rrf_k']) != (1586,3172,50,60):
        raise ValueError('Keep the original 3172-query TEST and 50/60 discovery protocol')
    queries, kinds = {}, {}
    for kind, variant in (('exact', 'exact_quote'), ('paraphrase', 'generated_paraphrase')):
        selected = [q for q in rows(args.original_prepared/f'queries-{kind}.jsonl') if q['split']=='test']
        if len(selected) != protocol['expected_queries_per_variant']:
            raise ValueError('Use the unchanged 1586-query original TEST per variant')
        for q in selected:
            if q['query_variant'] != variant or q['source_role'] != 'full' or q['query_id'] in queries:
                raise ValueError('Original TEST identity, variant, or source role differs')
            queries[q['query_id']], kinds[q['query_id']] = q, kind
    if len(queries) != protocol['expected_test_queries']:
        raise ValueError('Use the original 3172-query TEST regression, not the new confirmatory set')
    return queries, kinds


def discover(args):
    started = time.perf_counter()
    protocol = read(args.protocol)
    queries, kinds = original_queries(args, protocol)
    scales = [str(s) for s in args.scales]
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    metadata = out/'discovery.json'
    if metadata.exists() and read(metadata)['complete']:
        saved = read(metadata)
        if saved['scales'] != scales or set(saved['query_ids']) != set(queries) or saved['index_path'] != str(args.index):
            raise ValueError('Existing discovery belongs to different scale/query/index inputs')
        return saved
    index = read(args.index/'index.json')
    if not index['complete'] or not index['original_chunk_and_lexical_prefix_unchanged']:
        raise ValueError('Use the completed original-prefix-preserving scale index')
    documents = vector_bank(args.index/'embeddings', 'chunk-ids.json')
    pin = read(args.stack)['models']['embedding']
    if any(documents['manifest']['model_pin'][k] != pin[k] for k in ('model_id','revision')):
        raise ValueError('Scale bank must use the original frozen Ruri encoder')
    chunks = compact_chunks(args.index/'chunks.jsonl')
    if documents['ids'] != [c['chunk_id'] for c in chunks]:
        raise ValueError('Merged chunk order differs from its saved vector aliases')
    chunk_map = {c['chunk_id']: c for c in chunks}
    tokens = {r['chunk_id']: r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    targets = {tuple(r['address_key']): {'status': r['status'], 'passage_ids': r['passage_ids']}
               for r in rows(args.index/'target-addresses.jsonl')}
    bindings = {(r['law_id'],r['requested_asof']): {'status': r['status'], 'revision_id': r.get('revision_id')}
                for r in rows(args.index/'asof-bindings.jsonl')}
    banks = {}
    for kind in ('exact','paraphrase'):
        bank = vector_bank(args.original_index/('queries-'+kind), 'query-ids.json')
        if any(bank['manifest']['model_pin'][k] != documents['manifest']['model_pin'][k] for k in ('model_id','revision')):
            raise ValueError('Original query vectors differ from the merged corpus encoder')
        bank['rows'] = {key: int(alias) for key, alias in zip(bank['ids'],bank['aliases'])}
        banks[kind] = bank
    nested = read(args.index/'nested-corpora.json')
    shared = SharedCosines(documents['vectors'])
    engines = {scale: TimedDiscoveryEngine(scale_chunks(chunks, set(nested[scale])), tokens,
        shared, documents['aliases'], documents['ids'], targets, bindings,
        top_k=protocol['discovery_top_k'], rrf_k=protocol['rrf_k']) for scale in scales}
    manifest = receipt(complete=False, scales=scales, query_ids=list(queries),
        expected_query_scale_rows=len(queries)*len(scales), completed_query_scale_rows=0,
        index_path=str(args.index), original_prepared=str(args.original_prepared),
        original_query_index=str(args.original_index), embedding_pin=documents['manifest']['model_pin'],
        discovery_top_k=50, rrf_k=60, source_chunks=str(args.index/'chunks.jsonl'),
        scale_chunk_counts={s: len(e.chunks) for s,e in engines.items()},
        outputs={'candidates': 'candidates.jsonl'},
        cache_contract='all discovered addresses; all historical hits; fixed bound-current chunks; full-pool revision normalization',
        memory_policy='clear exact-result memoization after each durable full-pool write; retain shared dense product and date-bank indices',
        timing_definition={'retrieval_seconds': 'DiscoveryEngine.discover excluding _addresses; includes dense/BM25/RRF and bank setup',
            'binding_seconds': '_addresses: date binding and existing bound-current base-feature construction',
            'feature_seconds': 'nine revision features and normalization over all discovered addresses'},
        original_test_only=True, new_confirmatory_queries=0, method_selection=False)
    write_json(metadata, manifest)
    timing, count = Counter(), 0
    segmenter = query_segmenter()
    with (out/'candidates.jsonl').open('w', encoding='utf-8') as stream:
        for q in sorted(queries.values(), key=lambda q: (q['target_asof'],q['query_id'])):
            start = time.perf_counter(); terms = query_tokens(q['query_text'], segmenter)
            timing['query_tokenization_seconds'] += time.perf_counter()-start
            bank = banks[kinds[q['query_id']]]
            vector = bank['vectors'][bank['rows'][q['query_id']]]
            for scale, engine in engines.items():
                bound_before = engine.binding_seconds
                start = time.perf_counter(); d = engine.discover(q, terms, vector)
                elapsed = time.perf_counter()-start
                binding = engine.binding_seconds-bound_before
                start = time.perf_counter(); record = compact_discovery(q,d,chunk_map,tokens,terms,scale)
                measured = {'retrieval_seconds': elapsed-binding, 'binding_seconds': binding,
                            'feature_seconds': time.perf_counter()-start}
                record['timing'] = measured; timing.update(measured)
                save_row(stream,record); count += 1
                # The durable full-pool record replaces the large in-memory result cache.
                engine._cache.clear()
            if count % 50 == 0:
                stream.flush(); manifest.update(completed_query_scale_rows=count,timing=dict(timing))
                write_json(metadata,manifest)
                print(json.dumps({'phase':'discover','query_scale_rows':count}),flush=True)
    manifest.update(complete=count==manifest['expected_query_scale_rows'], completed_query_scale_rows=count,
        timing=dict(timing), shared_dense_products=shared.products,
        retrieval_runs={s:e.retrieval_runs for s,e in engines.items()},
        wall_seconds=time.perf_counter()-started)
    write_json(metadata,manifest)
    return manifest


def load_trees(args):
    from hidate.lambdamart import lightgbm_module
    models = args.development/'models'
    fit, parts, original = read(models/'fit.json'), read(models/'component-controls.json'), read(args.original_tree.parent/'fit.json')
    if not all(r['complete'] and r['seed']==42 for r in (fit,parts,original)):
        raise ValueError('Use complete seed42 original TRAIN models')
    if fit['validation_used_for_fitting'] or parts['validation_used_for_fitting'] or original['validation_used']:
        raise ValueError('Scale heads must have been fitted on original TRAIN without VALID')
    definitions = {
        'Original_LM10': (args.original_tree, original['feature_names']),
        'Dual_relevance_LM11': (models/'lambdamart_dual_relevance.txt', parts['fits']['lambdamart_dual_relevance']['feature_names']),
        'Full_LM18': (models/'lambdamart_revision.txt', fit['fits']['lambdamart_revision']['feature_names']),
        'Lean_noalignment_LM16': (models/'lambdamart_no_alignment.txt', parts['fits']['lambdamart_no_alignment']['feature_names'])}
    if [len(names) for path,names in definitions.values()] != [10,11,18,16]:
        raise ValueError('Frozen head feature counts differ')
    if original['feature_names'] != read(args.rank_protocol)['rank_features']:
        raise ValueError('Original ten-feature order differs from the original protocol')
    lgb = lightgbm_module()
    trees = {name: lgb.Booster(model_file=str(path)) for name,(path,names) in definitions.items()}
    for name, tree in trees.items():
        if tree.feature_name() != definitions[name][1]:
            raise ValueError('Model columns differ from their saved fit receipt')
    return trees, {name:{'path':str(path),'feature_names':names} for name,(path,names) in definitions.items()}


def add_panels(panels, prediction):
    variant, family = prediction['query_variant'], prediction['family_id']
    groups = ['all', 'variant='+variant, 'family='+family, 'variant='+variant+'|family='+family]
    if prediction['target_status']=='available':
        groups += ['available', 'variant='+variant+'|available', 'family='+family+'|available',
                   'cell='+variant+'|'+prediction['stratum']]
        if prediction['quote_absent'] is not None:
            quote = 'quote_absent' if prediction['quote_absent'] else 'quote_present'
            groups += [quote, 'variant='+variant+'|'+quote, 'family='+family+'|'+quote]
    for method, condition in prediction['conditions'].items():
        for group in groups:
            acc = panels[(prediction['scale'],method,group)]
            acc['n'] += 1
            for metric in METRICS:
                acc[metric] += condition['structural'][metric]


def native(args):
    started = time.perf_counter(); protocol = read(args.protocol)
    queries, _ = original_queries(args,protocol)
    source = read(args.output/'discovery.json')
    scales = [str(s) for s in args.scales]
    if not source['complete'] or source['scales'] != scales or set(source['query_ids']) != set(queries):
        raise ValueError('Complete matching original-TEST scale discovery is required')
    if source['index_path'] != str(args.index):
        raise ValueError('The cached current/historical chunk IDs require their original scale index')
    trees, definitions = load_trees(args)
    chunks = {c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    manifest = receipt(complete=False, scales=scales, methods=list(METHODS), models=definitions,
        discovery_receipt=str(args.output/'discovery.json'), original_test_only=True,
        new_confirmatory_queries=0, expected_query_scale_rows=source['expected_query_scale_rows'],
        completed_query_scale_rows=0, source_timing=source['timing'],
        score_policy='RRF raw max_history_rrf; LM raw margin; descending score then address; no fusion or policy selection',
        support_policy='none; native structural ranking only',
        evidence_policy='same fixed bound-current chunk per historical address; ten distinct parents, top four delivered',
        outputs={'predictions':'predictions.jsonl'}, method_selection=False)
    write_json(args.output/'evaluation.json',manifest)
    panels, seen, timing = defaultdict(Counter), set(), Counter()
    with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for cached in rows(args.output/'candidates.jsonl'):
            key = (cached['scale'],cached['query_id'])
            if key in seen or key[0] not in scales or key[1] not in queries:
                raise ValueError('Duplicate or non-original scale/query cache row')
            q = queries[key[1]]
            if any(cached[k] != q[k] for k in ('query_text','split','source_role','target_asof','query_variant')):
                raise ValueError('Cached discovery differs from the original TEST query')
            start = time.perf_counter(); scores = {}
            for name,tree in trees.items():
                features = definitions[name]['feature_names']
                matrix = np.asarray([[a['features'][k] for k in features] for a in cached['addresses']], dtype=float)
                scores[name] = tree.predict(matrix,raw_score=True,num_threads=4) if len(matrix) else []
            timing['predict_seconds'] += time.perf_counter()-start
            start = time.perf_counter(); prediction = native_predictions(q,cached,chunks,scores)
            timing['native_materialization_and_metrics_seconds'] += time.perf_counter()-start
            save_row(stream,prediction); add_panels(panels,prediction); seen.add(key)
            if len(seen) % 100 == 0:
                stream.flush(); manifest.update(completed_query_scale_rows=len(seen),timing=dict(timing))
                write_json(args.output/'evaluation.json',manifest)
                print(json.dumps({'phase':'native','query_scale_rows':len(seen)}),flush=True)
    result = defaultdict(dict)
    for (scale,method,group),acc in panels.items():
        result[scale+'|'+method][group] = {'n':acc['n'], **{m:acc[m]/acc['n'] for m in METRICS}}
    macro = {}
    for key,groups in result.items():
        cells = [value for name,value in groups.items() if name.startswith('cell=')]
        macro[key] = {'available_cells':len(cells), **{m:float(np.mean([v[m] for v in cells])) for m in METRICS}}
    expected = {(scale,qid) for scale in scales for qid in queries}
    manifest.update(complete=seen==expected, completed_query_scale_rows=len(seen),
        method_records=len(seen)*len(METHODS), panels=dict(result), available_four_cell_macro=macro,
        timing={**source['timing'],**dict(timing)}, wall_seconds=time.perf_counter()-started)
    write_json(args.output/'evaluation.json',manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('all','discover','native'),default='all')
    p.add_argument('--scales',nargs='+',type=int,choices=(120,500,1000),default=[1000])
    p.add_argument('--protocol',type=Path,default=Path('configs/scale_protocol.json'))
    p.add_argument('--rank-protocol',type=Path,default=Path('configs/retrieval-protocol.json'))
    p.add_argument('--index',type=Path,default=Path('data/index/hidate/extended-benchmark/scale'))
    p.add_argument('--original-index',type=Path,default=Path('data/index/hidate'))
    p.add_argument('--original-prepared',type=Path,default=Path('data/processed/hidate'))
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--development',type=Path,default=Path('runs/hidate/revision-aware/native-development'))
    p.add_argument('--original-tree',type=Path,default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/transfer/ruri-scale-native'))
    args = p.parse_args()
    if len(set(args.scales)) != len(args.scales):
        p.error('Each declared scale is evaluated once')
    if args.phase in ('all','discover'): discover(args)
    if args.phase in ('all','native'): native(args)


if __name__=='__main__':
    main()
