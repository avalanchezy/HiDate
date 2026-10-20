"""Original TRAIN/VALID revision-feature development on a PBS CPU allocation."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate.lambdamart import FIXED_PARAMETERS, lightgbm_module
from hidate.ranking import address_probabilities
from hidate.revision import (REVISION_FEATURES, enrich_addresses, address_labels,
    normalize_scores, fit_gate, gate_scores)
from discover_candidates import query_segmenter, query_tokens


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def prepare(args):
    out = args.output / 'features'; out.mkdir(parents=True, exist_ok=True)
    completed = out / 'preparation.json'
    if completed.exists() and read(completed)['complete']:
        return read(completed)
    started = time.perf_counter()
    queries = {q['query_id']: q for p in args.queries for q in rows(p)
               if q['split'] in ('train', 'valid')}
    tokens = {r['chunk_id']: r['tokens'] for r in rows(args.index / 'lexical-tokens.jsonl')}
    segment = query_segmenter()
    seen, counts = set(), Counter()
    with (out / 'train.jsonl').open('w', encoding='utf-8') as train, \
         (out / 'valid.jsonl').open('w', encoding='utf-8') as valid:
        for path in args.discovery:
            for d in rows(path):
                qid = d['query_id']
                if qid not in queries:
                    continue
                q = queries[qid]
                if qid in seen:
                    raise ValueError('Duplicate TRAIN/VALID discovery query')
                seen.add(qid)
                addresses = enrich_addresses(d, query_tokens(q['query_text'], segment), tokens)
                compact = [{key: a[key] for key in ('address', 'status', 'current_best_chunk_id',
                    'historical_representative_chunk_id', 'features',
                    'historical_normalized', 'current_normalized')} for a in addresses]
                record = {'query': {k: q[k] for k in ('query_id', 'split', 'family_id',
                    'query_variant', 'stratum', 'target_status', 'target_passage_id',
                    'law_id', 'scope', 'article_num')}, 'addresses': compact,
                    'labels': address_labels(q, addresses)}
                write_row(train if q['split'] == 'train' else valid, record)
                counts[q['split']] += 1
                if len(seen) % 1000 == 0:
                    print(json.dumps({'phase': 'features', 'queries': len(seen)}), flush=True)
    if seen != set(queries):
        raise ValueError('TRAIN/VALID discovery is incomplete')
    receipt = {'complete': True, 'queries': dict(counts), 'seed': 42,
        'job_id': os.environ.get('PBS_JOBID'), 'wall_seconds': time.perf_counter() - started,
        'split_policy': 'original TRAIN/VALID only', 'test_rows': 0,
        'representative': 'highest retrieved historical RRF hit; chunk-ID tie break',
        'query_revision_features': list(REVISION_FEATURES),
        'text_tokens': 'existing Sudachi normalized chunk tokens including natural headers',
        'source_quote_or_gold_used_for_features': False,
        'unavailable_comparison': 'neutral zero revision features; retain original address label'}
    write_json(completed, receipt)
    return receipt


def training_arrays(path, feature_names):
    X, y, h, c, sizes = [], [], [], [], []
    stats = Counter()
    for row in rows(path):
        if row['query']['split'] != 'train':
            raise ValueError('Fit accepts TRAIN only')
        size = len(row['addresses']); sizes.append(size)
        for a, label in zip(row['addresses'], row['labels']):
            X.append([a['features'][k] for k in feature_names]); y.append(label)
            h.append(a['historical_normalized']); c.append(a['current_normalized'])
            stats['positive_unavailable'] += label and a['status'] != 'available'
        stats['queries'] += 1; stats['candidates'] += size
        stats['positives'] += sum(row['labels'])
        stats['zero_positive_queries'] += not any(row['labels'])
    return (np.asarray(X), np.asarray(y, dtype=np.int32), np.asarray(h), np.asarray(c),
            np.asarray(sizes), dict(stats))


def fit(args):
    out = args.output / 'models'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'fit.json').exists() and read(out / 'fit.json')['complete']:
        return read(out / 'fit.json')
    base = [k for k in read(args.protocol)['rank_features'] if k != 'available_target']
    names = base + list(REVISION_FEATURES)
    X, y, h, c, sizes, stats = training_arrays(args.output / 'features/train.jsonl', names)
    lgb = lightgbm_module()
    fits = {}
    for name, chosen in [('base', base), ('revision', names)]:
        indices = [names.index(k) for k in chosen]
        matrix = X[:, indices]
        started = time.perf_counter()
        params = {**FIXED_PARAMETERS, 'lambdarank_norm': True}
        model = lgb.LGBMRanker(**params)
        model.fit(matrix, y, group=sizes, feature_name=chosen)
        model.booster_.save_model(str(out / ('lambdamart_' + name + '.txt')))
        fits['lambdamart_' + name] = {'feature_names': chosen, 'parameters': params,
            'fit_seconds': time.perf_counter() - started, 'actual_trees': model.booster_.num_trees()}
        started = time.perf_counter()
        gate = fit_gate(matrix, h, c, y, sizes, chosen)
        write_json(out / ('gate_' + name + '.json'), gate)
        fits['gate_' + name] = {'feature_names': chosen, 'fit_seconds': time.perf_counter() - started,
                               'converged': gate['converged'], 'iterations': gate['iterations']}
        print(json.dumps({'phase': 'fit', 'variant': name, 'fits': fits}), flush=True)
    result = {'complete': True, 'seed': 42, 'job_id': os.environ.get('PBS_JOBID'),
        'statistics': stats, 'fits': fits, 'validation_used_for_fitting': False,
        'baseline_feature_policy': 'original feature set minus available_target',
        'training_scope': 'full original retrieved address groups, including unavailable positives',
        'gate_experts': 'within-query rank-normalized historical RRF and bound-current cosine',
        'gate_loss': 'mean pairwise logistic loss per query; uninformative no-positive groups skipped',
        'tree_loss': 'query-grouped LambdaRank with lambdarank_norm enabled',
        'neural_model_calls': 0}
    write_json(out / 'fit.json', result)
    return result


def ordered_indices(addresses, scores):
    return sorted(range(len(addresses)), key=lambda i: (-float(scores[i]), tuple(addresses[i]['address'])))


def fit_component_controls(args):
    """Separate two-view matching gains from information about the revision."""
    out = args.output / 'models'
    path = out / 'component-controls.json'
    if path.exists() and read(path)['complete']:
        return read(path)
    base = [k for k in read(args.protocol)['rank_features'] if k != 'available_target']
    names = base + list(REVISION_FEATURES)
    X, y, _, _, sizes, stats = training_arrays(args.output / 'features/train.jsonl', names)
    variants = {
        'lambdamart_dual_relevance': base + ['historical_query_coverage', 'current_query_coverage'],
        'lambdamart_no_alignment': [k for k in names if k not in ('focused_edit_fraction', 'focused_alignment_missing')],
        'lambdamart_no_query_change': [k for k in names if k not in (
            'query_terms_lost', 'query_terms_gained', 'query_terms_retained', 'query_numeric_changed')],
    }
    lgb = lightgbm_module(); fits = {}
    for name, chosen in variants.items():
        started = time.perf_counter()
        params = {**FIXED_PARAMETERS, 'lambdarank_norm': True}
        model = lgb.LGBMRanker(**params)
        model.fit(X[:, [names.index(k) for k in chosen]], y, group=sizes, feature_name=chosen)
        model.booster_.save_model(str(out / (name + '.txt')))
        fits[name] = {'feature_names': chosen, 'parameters': params,
                     'fit_seconds': time.perf_counter() - started, 'actual_trees': model.booster_.num_trees()}
    receipt = {'complete': True, 'seed': 42, 'fits': fits, 'statistics': stats,
        'job_id': os.environ.get('PBS_JOBID'), 'validation_used_for_fitting': False,
        'fixed_policy': 'same native ranking and tree parameters; no fusion retuning',
        'purpose': 'isolate revision comparison from extra historical/current lexical relevance'}
    write_json(path, receipt)
    return receipt


def deliverable_indices(addresses, order, parents, limit=10):
    result, seen = [], set()
    for index in order:
        a = addresses[index]; cid = a['current_best_chunk_id']
        if a['status'] != 'available' or cid is None:
            continue
        parent = parents[cid]
        if parent not in seen:
            seen.add(parent); result.append(index)
        if len(result) == limit:
            break
    return result


def evaluate(args):
    started = time.perf_counter()
    lgb = lightgbm_module(); models = args.output / 'models'
    fits = read(models / 'fit.json')['fits']
    controls = models / 'component-controls.json'
    if controls.exists():
        fits = {**fits, **read(controls)['fits']}
    trees = {k: lgb.Booster(model_file=str(models / (k + '.txt')))
             for k in fits if k.startswith('lambdamart')}
    gates = {k: read(models / (k + '.json')) for k in fits if k.startswith('gate')}
    original_tree = lgb.Booster(model_file=str(args.original_tree))
    lr = read(args.original_lr); original_features = read(args.protocol)['rank_features']
    parents = {r['chunk_id']: r['parent_passage_id'] for r in rows(args.index / 'chunks.jsonl')}
    panels = defaultdict(list); address_panels = defaultdict(list); complement = Counter()
    output = args.output / args.evaluation_name; output.mkdir(parents=True, exist_ok=True)
    distributions = defaultdict(list)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        query_count = 0
        for row in rows(args.output / 'features/valid.jsonl'):
            q, addresses = row['query'], row['addresses']
            if q['split'] != 'valid':
                raise ValueError('Development evaluation requires original VALID')
            h = np.array([a['historical_normalized'] for a in addresses])
            c = np.array([a['current_normalized'] for a in addresses])
            score_map = {'history_rrf_native': h, 'current_cosine_native': c,
                'original_lr_native': address_probabilities(lr, [a['features'] for a in addresses]),
                'original_lambdamart_native': original_tree.predict(np.array([
                    [a['features'][k] for k in original_features] for a in addresses]), num_threads=4)}
            for key, model in trees.items():
                X = np.array([[a['features'][k] for k in fits[key]['feature_names']] for a in addresses])
                score_map[key] = model.predict(X, num_threads=4)
            for key, model in gates.items():
                X = np.array([[a['features'][k] for k in model['feature_names']] for a in addresses])
                score_map[key], gate = gate_scores(model, X, h, c)
                distributions[key + '_historical_weight'].extend(gate.tolist())
            for mode in ('rank', 'minmax'):
                hh = normalize_scores([a['features']['max_history_rrf'] for a in addresses], mode)
                cc = normalize_scores([a['features']['current_best_cosine'] for a in addresses], mode)
                for alpha in (0., .25, .5, .75, 1.):
                    score_map[f'fixed_{mode}_{alpha:g}'] = alpha * hh + (1-alpha) * cc
            common = deliverable_indices(addresses, ordered_indices(addresses, h), parents)
            condition_results = {}
            for name, scores in score_map.items():
                order = ordered_indices(addresses, scores)
                gold_position = next((r for r, i in enumerate(order, 1) if row['labels'][i]), None)
                address_mrr = 1 / gold_position if gold_position is not None and gold_position <= 4 else 0.
                address_panels[name].append(address_mrr)
                for scope, candidate_ids in [('own_pool', deliverable_indices(addresses, order, parents)),
                    ('common_rrf10', sorted(common, key=lambda i: (-float(scores[i]), tuple(addresses[i]['address']))))]:
                    chosen = candidate_ids[:4]
                    hit = next((r for r, i in enumerate(chosen, 1)
                        if parents[addresses[i]['current_best_chunk_id']] == q['target_passage_id']), None)
                    value = {'recall4': float(hit is not None), 'mrr4': 1 / hit if hit else 0.}
                    key = name + '|' + scope
                    condition_results[key] = {**value,
                        'chunk_ids': [addresses[i]['current_best_chunk_id'] for i in chosen]}
                    if q['target_status'] == 'available':
                        panels[(key, q['query_variant'] + '|' + q['stratum'])].append(value)
            h_ok = condition_results['history_rrf_native|common_rrf10']['recall4']
            c_ok = condition_results['current_cosine_native|common_rrf10']['recall4']
            if q['target_status'] == 'available':
                complement[f'history={int(h_ok)},current={int(c_ok)}'] += 1
            write_row(stream, {'query_id': q['query_id'], 'family_id': q['family_id'],
                'query_variant': q['query_variant'], 'stratum': q['stratum'],
                'target_status': q['target_status'], 'conditions': condition_results})
            query_count += 1
    cell_results = defaultdict(dict)
    for (key, cell), values in panels.items():
        cell_results[key][cell] = {'count': len(values), **{
            metric: float(np.mean([v[metric] for v in values])) for metric in ('recall4', 'mrr4')}}
    aggregates = {key: {'cells': cells, **{metric: float(np.mean([v[metric] for v in cells.values()]))
        for metric in ('recall4', 'mrr4')}} for key, cells in cell_results.items()}
    ranked = sorted([(k, v) for k, v in aggregates.items() if k.endswith('|own_pool')],
                    key=lambda pair: (-pair[1]['mrr4'], pair[0]))
    report = {'complete': True, 'split': 'valid', 'queries': query_count, 'seed': 42,
        'job_id': os.environ.get('PBS_JOBID'), 'wall_seconds': time.perf_counter() - started,
        'metric_policy': 'original four-cell macro target-parent MRR@4; available queries',
        'conditions': aggregates, 'address_mrr4_all_queries': {k: float(np.mean(v)) for k,v in address_panels.items()},
        'best_development_recipe': ranked[0][0],
        'ordered_own_pool_recipes': [{'condition': k, 'mrr4': v['mrr4'], 'recall4': v['recall4']} for k,v in ranked],
        'common_rrf10_success_complementarity_available_queries': dict(complement),
        'gate_weight_quantiles': {k: np.quantile(v, [0,.1,.5,.9,1]).tolist() for k,v in distributions.items()},
        'test_rows': 0, 'neural_model_calls': 0,
        'stage': 'native score development; historical/current CE diagnosis is a separate experiment',
        'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'evaluation.json', report)
    print(json.dumps({'phase': 'valid', 'ordered_recipes': report['ordered_own_pool_recipes']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all','prepare','fit','evaluate','component-controls'), default='all')
    parser.add_argument('--evaluation-name', default='valid')
    parser.add_argument('--queries', nargs='+', type=Path, default=[Path('data/processed/hidate/queries-exact.jsonl'), Path('data/processed/hidate/queries-paraphrase.jsonl')])
    parser.add_argument('--discovery', nargs='+', type=Path, default=[Path('runs/hidate/discovery/exact-train-valid/candidates.jsonl'), Path('runs/hidate/discovery/paraphrase-train-valid/candidates.jsonl')])
    parser.add_argument('--index', type=Path, default=Path('data/index/hidate'))
    parser.add_argument('--protocol', type=Path, default=Path('configs/retrieval-protocol.json'))
    parser.add_argument('--original-lr', type=Path, default=Path('runs/hidate/final/ranker-used.json'))
    parser.add_argument('--original-tree', type=Path, default=Path('runs/hidate/mechanism-analysis/lambdamart/ranker/model.txt'))
    parser.add_argument('--output', type=Path, default=Path('runs/hidate/revision-aware/native-development'))
    args = parser.parse_args()
    if args.phase in ('all', 'prepare'): prepare(args)
    if args.phase in ('all', 'fit'): fit(args)
    if args.phase == 'component-controls': fit_component_controls(args)
    if args.phase in ('all', 'evaluate', 'component-controls'): evaluate(args)


if __name__ == '__main__':
    main()
