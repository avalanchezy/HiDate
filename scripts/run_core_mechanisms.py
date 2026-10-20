"""Fit and evaluate the frozen Ruri as-of/history × fixed/learned matrix.

A0/A1 use the CPU-prepared as-of address pool and canonical bound chunks.
H0/H1 are read from the original completed natural retrieval, never rerun.
Only uncached exact raw query/document pairs reach the GPU reranker.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from hidate.io import write_json
from hidate.ablations import compact_dev_row
from hidate.discovery import chunk_address
from hidate.ranking import fit_address_ranker, training_examples
from hidate.reranking import PairScoreCache, rerank_query
from hidate.selection import choose_policy, selected_evidence
from evaluate_final import compact_candidate, structural_result


FIXED_ALPHA = .75
EXPECTED_COUNTS = {'train': 14030, 'valid': 2490, 'test': 3172}
METRICS = ('top_address_correct', 'top_parent_correct', 'top_revision_correct',
           'target_mrr', 'target_recall_at_four')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs(args):
    protocol, stack = read(args.protocol), read(args.stack)
    if not stack['complete']:
        raise ValueError('The frozen model stack must be complete')
    if protocol['seed'] != 42 or len(protocol['rank_features']) != 10:
        raise ValueError('Core conditions retain seed42 and the original ten features')
    if [protocol[k] for k in ('discovery_top_k', 'reranker_top_k', 'reader_max_passages')] != [50, 10, 4]:
        raise ValueError('Core conditions retain the original50/10/4 budgets')
    queries = {}
    for path in args.queries:
        for query in rows(path):
            if query['query_id'] in queries:
                raise ValueError('Duplicate original query ID')
            queries[query['query_id']] = query
    for path in args.asof_discovery:
        meta = read(Path(path).parent / 'discovery.json')
        if not meta['complete'] or meta['candidate_pool'] != 'asof' or meta['seed'] != 42:
            raise ValueError('A0/A1 need complete seed42 as-of-only discovery')
        if set(meta['features']) != set(protocol['rank_features']):
            raise ValueError('As-of discovery feature schema differs')
        if any(meta['model_pin'][k] != stack['models']['embedding'][k] for k in ('model_id', 'revision')):
            raise ValueError('As-of discovery embedding pin differs from the stack')
    return queries, protocol, stack


def discoveries(paths, queries, split):
    expected = {key for key, q in queries.items() if q['split'] == split}
    seen = set()
    for path in paths:
        for row in rows(path):
            query = queries[row['query_id']]
            if query['split'] != split:
                continue
            if row['query_id'] in seen:
                raise ValueError('Duplicate as-of discovery query: ' + row['query_id'])
            if row.get('candidate_pool') != 'asof' or 'history' in row:
                raise ValueError('A0/A1 must not consume historical discovery')
            if any(row[k] != query[k] for k in ('split', 'query_variant', 'target_asof', 'source_role')):
                raise ValueError('As-of discovery metadata differs from the original query')
            seen.add(row['query_id'])
            yield row
    if seen != expected:
        raise ValueError(f'Incomplete original {split} coverage: {len(seen)}/{len(expected)}')


def fit_core(args, *, expected_train=14030):
    queries, protocol, _ = inputs(args)
    train = {key: q for key, q in queries.items() if q['split'] == 'train'}
    if len(train) != expected_train:
        raise ValueError('The original complete TRAIN allocation is required')
    output = Path(args.output_dir)
    path = output / 'ranker/a1.json'
    provenance = [str(Path(p).resolve()) for p in args.asof_discovery]
    if path.exists():
        model = read(path)
        if (not model['complete'] or model['feature_names'] != protocol['rank_features']
                or model['training_statistics']['query_count'] != expected_train
                or model['discovery_paths'] != provenance):
            raise ValueError('Existing A1 fit differs; do not refit over it')
        return read(output / 'fitting.json')
    started = time.perf_counter()
    # Keep only the fields used by the unchanged TRAIN fitter.
    data = [{'query_id': d['query_id'], 'addresses': [
        {k: a[k] for k in ('address', 'features', 'status')} for a in d['addresses']]}
        for d in discoveries(args.asof_discovery, queries, 'train')]
    X, y, stats = training_examples(data, train, protocol['rank_features'],
        required_variants=('exact_quote', 'generated_paraphrase'))
    model = fit_address_ranker(X, y, protocol['rank_features'])
    model.update(complete=True, condition='A1', candidate_pool='asof', fixed_alpha=FIXED_ALPHA,
        training_statistics=stats, discovery_paths=provenance, fitted_at_utc=now(),
        job_id=os.environ.get('PBS_JOBID'))
    write_json(path, model)
    receipt = {'complete': True, 'seed': 42, 'training_runs': 1, 'condition': 'A1',
        'training_query_count': len(train), 'training_statistics': stats,
        'ranker_path': str(path), 'fixed_alpha': FIXED_ALPHA, 'new_splits': 0,
        'finished_at_utc': now(), 'wall_seconds': time.perf_counter()-started}
    write_json(output / 'fitting.json', receipt)
    return receipt


def check_pair_context(meta, pin):
    saved = meta.get('model_pin') or meta.get('model_pins', {}).get('reranker')
    if not saved or any(saved[k] != pin[k] for k in ('model_id', 'revision')):
        raise ValueError('Pair-receipt model pin differs from the Ruri stack')
    if meta.get('max_length', meta.get('reranker_max_length')) != 1024:
        raise ValueError('Pair-receipt max length differs from1024')
    if meta.get('sdpa_backend', meta.get('reranker_sdpa_backend')) != 'math':
        raise ValueError('Pair receipts require the original Ruri math attention backend')


def prefill(cache, paths):
    sources = []
    for path in paths:
        count = 0
        for row in rows(path):
            pairs, result = row['pairs'], row['result']
            if not all(len(result[k]) == len(pairs) for k in ('scores', 'logits', 'token_counts')):
                raise ValueError('Pair receipt arrays are not aligned')
            for i, pair in enumerate(pairs):
                score, logit = float(result['scores'][i]), float(result['logits'][i])
                if not math.isfinite(score) or not math.isfinite(logit):
                    raise ValueError('Nonfinite saved pair score')
                cache.cache[(pair['query_text'], pair['document_text'])] = {
                    'reranker_score': score, 'reranker_logit': logit,
                    'pair_token_count': int(result['token_counts'][i])}
                count += 1
        sources.append({'path': str(path), 'pair_records': count})
    return {'sources': sources, 'unique_pairs': len(cache.cache),
            'precedence': 'later receipt wins: original FINAL after DEV, then own prior misses'}


class LazyReranker:
    def __init__(self, pin, device, factory):
        self.pin, self.device, self.factory, self.model = pin, device, factory, None

    def score(self, pairs, include_inputs=False):
        if self.model is None:
            if self.factory is None:
                from hidate.models import create_reranker
                self.model = create_reranker(self.pin, device=self.device, max_length=1024)
            else:
                self.model = self.factory(self.pin)
        return self.model.score(pairs, include_inputs=include_inputs)


def reachability(query, discovery, condition, evidence=None):
    gold = (query['law_id'], query['scope'], query['article_num'])
    pool = discovery['addresses']
    known = all('status' in a and 'current_best_chunk_id' in a for a in pool)
    bound = [a for a in pool if a['status'] == 'available' and a['current_best_chunk_id'] is not None] if known else None
    chosen = {chunk_address(c) for c in condition['reranked_candidates']}
    evidence = condition.get('evidence', []) if evidence is None else evidence
    available = query['target_status'] == 'available'
    return {'gold_address_in_pool': any(tuple(a['address']) == gold for a in pool),
        'gold_address_bindable': any(tuple(a['address']) == gold for a in bound) if known else None,
        'gold_address_in_preselect10': gold in chosen, 'pool_address_count': len(pool),
        'gold_address_in_final4': any(chunk_address(c) == gold for c in evidence),
        'gold_parent_in_preselect10': any(c['parent_passage_id'] == query['target_passage_id']
            for c in condition['reranked_candidates']) if available else None,
        'gold_parent_in_final4': any(c['parent_passage_id'] == query['target_passage_id']
            for c in evidence) if available else None,
        'eligible_address_count': len(pool), 'bindable_address_count': len(bound) if known else None,
        'bindable_count_known': known, 'preselect_parent_count': len(condition['reranked_candidates'])}


def saved_condition(query, discovery, condition, alpha, *, fixed_head=False):
    if fixed_head:
        condition = {**condition, 'reranked_candidates': [
            {**c, 'learned_probability': 0.} for c in condition['reranked_candidates']],
            'address_diagnostics': [{**a, 'learned_probability': 0.}
                                    for a in condition['address_diagnostics']]}
    evidence = selected_evidence(condition, alpha=alpha)
    return {'alpha': alpha, 'head': 'constant_zero/max_RRF_preselect' if fixed_head else 'TRAIN_asof_LR',
        'requested_pair_count': condition['requested_pair_count'],
        'requested_input_tokens': condition['requested_input_tokens'],
        'reranked_candidates': [compact_candidate(c) for c in condition['reranked_candidates']],
        'evidence': [compact_candidate(c) for c in evidence],
        'evidence_chunk_ids': [c['chunk_id'] for c in evidence],
        'evidence_parent_ids': [c['parent_passage_id'] for c in evidence],
        'address_diagnostics': condition['address_diagnostics'],
        'stage_reachability': reachability(query, discovery, condition, evidence),
        'structural': structural_result(query, evidence, 'natural'),
        'support_calibration': 'not evaluated in the fixed-policy retrieval matrix'}


def old_history(args, queries):
    expected = {key for key, q in queries.items() if q['split'] == 'test'}
    saved = {}
    for row in rows(args.full_retrieval):
        if row['evaluation_scope'] != 'natural':
            continue
        key = row['query_id']
        if key not in expected or key in saved:
            raise ValueError('Original natural TEST identities differ')
        q = queries[key]
        if any(row[k] != q[k] for k in ('query_variant', 'target_asof', 'source_role', 'target_key')):
            raise ValueError('Original natural TEST metadata differs')
        if row['conditions']['history_learned']['alpha'] != FIXED_ALPHA:
            raise ValueError('H1 must use the original full alpha.75')
        saved[key] = {label: {**row['conditions'][method], 'reused_original': True,
            'origin_method': method, 'matrix_fixed_alpha': FIXED_ALPHA}
            for label, method in (('H0', 'history_bound'), ('H1', 'history_learned'))}
        for condition in saved[key].values():
            condition['stage_reachability'] = reachability(q,
                {'addresses': condition['address_diagnostics']}, condition)
            condition['stage_reachability_source'] = 'saved address_diagnostics/reranked_candidates/evidence'
    if set(saved) != expected:
        raise ValueError('Original H0/H1 natural TEST is incomplete')
    return saved


def metric_panels(cells):
    return {key: {'n': len(values), **{m: sum(float(v[m]) for v in values)/len(values)
            for m in METRICS}} for key, values in cells.items()}


def evaluate_core(args, *, expected_counts=None, reranker_factory=None):
    started = time.perf_counter()
    expected_counts = expected_counts or EXPECTED_COUNTS
    queries, protocol, stack = inputs(args)
    if dict(Counter(q['split'] for q in queries.values())) != expected_counts:
        raise ValueError('Core evaluation needs the complete unchanged TRAIN/VALID/TEST counts')
    full, dev = read(args.full_evaluation), read(args.dev_metadata)
    pin = stack['models']['reranker']
    if pin['model_id'] != 'cl-nagoya/ruri-v3-reranker-310m':
        raise ValueError('This matrix uses the original Ruri stack')
    if not full['complete'] or not dev['complete'] or full['frozen_policy']['alpha'] != FIXED_ALPHA:
        raise ValueError('Original completed full evaluation and frozen alpha.75 are required')
    check_pair_context(dev, pin); check_pair_context(full, pin)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    ranker = read(output / 'ranker/a1.json')
    if (not ranker['complete'] or ranker['feature_names'] != protocol['rank_features']
            or ranker['training_statistics']['query_count'] != expected_counts['train']):
        raise ValueError('Complete original-TRAIN A1 fit is required')
    chunks = {c['chunk_id']: c for c in rows(args.chunks)}
    history = old_history(args, queries)
    lazy = LazyReranker(pin, args.device, reranker_factory)
    receipt_path = output / 'model-pair-receipts.jsonl'
    cache = PairScoreCache(lazy)
    receipt_sources = [args.dev_pair_receipts, args.final_pair_receipts]
    if receipt_path.exists():
        check_pair_context(read(output / 'evaluation.json'), pin)
        receipt_sources.append(receipt_path)
    loaded = prefill(cache, receipt_sources)
    summary = {'complete': False, 'started_at_utc': now(), 'job_id': os.environ.get('PBS_JOBID'),
        'seed': 42, 'fixed_alpha': FIXED_ALPHA, 'matrix_conditions': ['A0', 'A1', 'H0', 'H1'],
        'expected_query_counts': expected_counts, 'natural_test_rows': 0, 'valid_rows': 0,
        'model_pin': {k: pin[k] for k in ('model_id', 'revision')}, 'max_length': 1024,
        'sdpa_backend': 'math', 'cache_prefill': loaded, 'pair_budget': 10, 'final_parent_budget': 4,
        'ranker_path': str(output / 'ranker/a1.json'), 'full_retrieval_path': str(args.full_retrieval),
        'asof_discovery_paths': [str(p) for p in args.asof_discovery],
        'H0_reuse': 'source alpha0 CE order equals fixed alpha.75 with constant-zero learned term',
        'new_training_runs': 0, 'reused_history_queries': len(history), 'retuned_policy': None}
    write_json(output / 'evaluation.json', summary)
    cells = defaultdict(list)
    with receipt_path.open('a', encoding='utf-8') as receipts:
        def save_receipt(row):
            save_row(receipts, row); receipts.flush()
        cache.receipt_callback = save_receipt
        # Always retain VALID A1 ten-candidate scores, enabling later CPU-only policy selection.
        valid = []
        with (output / 'valid-reranked.jsonl').open('w', encoding='utf-8') as stream:
            for discovery in discoveries(args.asof_discovery, queries, 'valid'):
                q = queries[discovery['query_id']]
                result = rerank_query(q, discovery, chunks, cache, conditions=('history_learned',), ranker=ranker)
                valid.append(compact_dev_row(result))
                save_row(stream, {'query_id': q['query_id'], 'query_variant': q['query_variant'],
                    'split': 'valid', 'family_id': q['family_id'], 'conditions': {
                    'A1': saved_condition(q, discovery, result['conditions']['history_learned'], FIXED_ALPHA)}})
        summary['valid_rows'] = len(valid)
        policy = None
        if args.retune_a1:
            policy = choose_policy(valid, queries, protocol['dev_selection']['learned_reranker_mix_grid'])
            policy.update(complete=True, selection_split='valid', condition='A1_valid_retuned',
                          excluded_from_primary_matrix=True)
            write_json(output / 'a1-valid-policy.json', policy)
            summary['retuned_policy'] = policy
        del valid
        with (output / 'retrieval.jsonl').open('w', encoding='utf-8') as stream:
            for discovery in discoveries(args.asof_discovery, queries, 'test'):
                q = queries[discovery['query_id']]
                result = rerank_query(q, discovery, chunks, cache,
                    conditions=('history_bound', 'history_learned'), ranker=ranker)['conditions']
                conditions = {'A0': saved_condition(q, discovery, result['history_bound'], FIXED_ALPHA, fixed_head=True),
                              'A1': saved_condition(q, discovery, result['history_learned'], FIXED_ALPHA),
                              **history[q['query_id']]}
                if policy is not None:
                    conditions['A1_valid_retuned'] = saved_condition(q, discovery, result['history_learned'], policy['alpha'])
                flag = q.get('source_quote_absent_from_target')
                record = {k: q[k] for k in ('query_id', 'query_variant', 'family_id', 'stratum',
                          'target_status', 'target_asof', 'source_role', 'target_key')}
                record.update(split='test', variant=q['query_variant'], quote_absent=flag,
                              evaluation_scope='natural', conditions=conditions)
                save_row(stream, record)
                subsets = ['all_queries']
                if q['target_status'] == 'available':
                    subsets.append('available')
                    if flag is not None:
                        subsets.append('quote_absent' if flag else 'quote_present')
                else:
                    subsets.append('unavailable')
                for name, condition in conditions.items():
                    for subset in subsets:
                        cells['|'.join((name, q['query_variant'], subset))].append(condition['structural'])
                summary['natural_test_rows'] += 1
                if summary['natural_test_rows'] % 100 == 0:
                    stream.flush()
                    summary.update(physical_pair_count=cache.physical_pair_count,
                                   logical_pair_requests=cache.logical_pair_requests)
                    write_json(output / 'evaluation.json', summary)
                    print(json.dumps({'test_rows': summary['natural_test_rows'],
                                      'new_pairs': cache.physical_pair_count}), flush=True)
    summary.update(complete=True, finished_at_utc=now(), wall_seconds=time.perf_counter()-started,
        panels=metric_panels(cells), physical_pair_count=cache.physical_pair_count,
        logical_pair_requests=cache.logical_pair_requests, model_calls=cache.model_call_count,
        encoded_input_tokens=cache.encoded_input_tokens, padded_token_slots=cache.padded_token_slots,
        model_seconds=cache.model_wall_seconds_with_cuda_sync, model_loaded=lazy.model is not None,
        actual_new_pair_sdpa_backend=cache.sdpa_backend,
        cache_reused_requests=cache.logical_pair_requests-cache.physical_pair_count)
    write_json(output / 'evaluation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fit', 'evaluate'), required=True)
    parser.add_argument('--queries', nargs='+', type=Path, default=[Path('data/processed/hidate/queries-exact.jsonl'), Path('data/processed/hidate/queries-paraphrase.jsonl')])
    base = Path('runs/hidate/mechanism-analysis')
    parser.add_argument('--asof-discovery', nargs='+', type=Path, default=[base/'asof-discovery/exact/candidates.jsonl', base/'asof-discovery/paraphrase/candidates.jsonl'])
    parser.add_argument('--protocol', type=Path, default=Path('configs/retrieval-protocol.json'))
    parser.add_argument('--stack', type=Path, default=Path('models/ruri/model-stack.json'))
    parser.add_argument('--chunks', type=Path, default=Path('data/index/hidate/chunks.jsonl'))
    parser.add_argument('--full-retrieval', type=Path, default=Path('runs/hidate/final/retrieval.jsonl'))
    parser.add_argument('--full-evaluation', type=Path, default=Path('runs/hidate/final/evaluation.json'))
    parser.add_argument('--dev-metadata', type=Path, default=Path('runs/hidate/dev/scoring.json'))
    parser.add_argument('--dev-pair-receipts', type=Path, default=Path('runs/hidate/dev/model-pair-receipts.jsonl'))
    parser.add_argument('--final-pair-receipts', type=Path, default=Path('runs/hidate/final/model-pair-receipts.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=base/'core-2x2')
    parser.add_argument('--retune-a1', action='store_true', help='Also save a separate VALID-selected A1 competitor; primary alpha remains.75')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = fit_core(args) if args.phase == 'fit' else evaluate_core(args)
    print(json.dumps({k: result[k] for k in ('complete', 'fixed_alpha')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
