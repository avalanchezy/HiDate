"""Fit feature ablations on TRAIN, freeze VALID mixtures, evaluate natural TEST."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from hidate.io import write_json
from hidate.ablations import feature_sets, compact_dev_row
from hidate.models import create_reranker
from hidate.ranking import training_examples, fit_address_ranker
from hidate.reranking import PairScoreCache, rerank_query
from hidate.selection import choose_policy, selected_evidence, support_score
from evaluate_final import structural_result


def rows(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fit(root, output):
    output.mkdir(parents=True, exist_ok=True)
    queries = {r['query_id']: r for p in ('queries-exact.jsonl', 'queries-paraphrase.jsonl')
               for r in rows(Path('data/processed/hidate') / p) if r['split'] == 'train'}
    names = json.loads(Path('configs/retrieval-protocol.json').read_text())['rank_features']
    discoveries = [r for variant in ('exact-train-valid', 'paraphrase-train-valid')
                   for r in rows(root / variant / 'candidates.jsonl') if r['query_id'] in queries]
    if len(discoveries) != len(queries) or {r['query_id'] for r in discoveries} != set(queries):
        raise ValueError('Ablation fitting needs the complete unchanged TRAIN set')
    records = {}
    for label, features in feature_sets(names).items():
        path = output / (label + '.json')
        if path.exists():
            records[label] = json.loads(path.read_text())
            continue
        X, y, statistics = training_examples(discoveries, queries, features,
                            required_variants=('exact_quote', 'generated_paraphrase'))
        model = fit_address_ranker(X, y, features)
        model.update(complete=True, intervention=label, training_statistics=statistics,
                     job_id=os.environ.get('PBS_JOBID'), fitted_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, model)
        records[label] = model
        print(json.dumps({'fitted': label, 'features': features, 'queries': len(queries)}), flush=True)
    write_json(output / 'fitting.json', {'complete': True, 'seed': 42, 'models': list(records),
              'training_query_count': len(queries), 'new_seeds': 0, 'new_splits': 0})


def evaluate(root, discovery_root, final_discovery_root, stack_path, output):
    output.mkdir(parents=True, exist_ok=True)
    queries = {r['query_id']: r for p in ('queries-exact.jsonl', 'queries-paraphrase.jsonl')
               for r in rows(Path('data/processed/hidate') / p)}
    expected_dev = {key for key, row in queries.items() if row['split'] == 'valid'}
    expected_test = {key for key, row in queries.items() if row['split'] == 'test'}
    chunks = {r['chunk_id']: r for r in rows('data/index/hidate/chunks.jsonl')}
    pin = json.loads(stack_path.read_text())['models']['reranker']
    model = create_reranker(pin, device='cuda', max_length=1024)
    cache = PairScoreCache(model)
    summary = {'complete': False, 'seed': 42, 'job_id': os.environ.get('PBS_JOBID'), 'models': {}}
    write_json(output / 'evaluation.json', summary)
    protocol = json.loads(Path('configs/retrieval-protocol.json').read_text())
    started = time.perf_counter()
    for label in feature_sets(protocol['rank_features']):
        ranker = json.loads((root / 'rankers' / (label + '.json')).read_text())
        dev, dev_ids = [], set()
        for variant in ('exact-train-valid', 'paraphrase-train-valid'):
            for discovery in rows(discovery_root / variant / 'candidates.jsonl'):
                query = queries[discovery['query_id']]
                if query['split'] != 'valid':
                    continue
                if query['query_id'] in dev_ids:
                    raise ValueError('Duplicate VALID query in ablation discovery')
                dev_ids.add(query['query_id'])
                result = rerank_query(query, discovery, chunks, cache, conditions=('history_learned',), ranker=ranker)
                dev.append(compact_dev_row(result))
        if dev_ids != expected_dev:
            raise ValueError('Ablations require every unchanged VALID query')
        policy = choose_policy(dev, queries, protocol['dev_selection']['learned_reranker_mix_grid'])
        policy.update(complete=True, selection_split='valid', intervention=label)
        write_json(output / (label + '-policy.json'), policy)
        del dev
        cells, test_ids = defaultdict(list), set()
        with (output / (label + '-test.jsonl')).open('w', encoding='utf8') as stream:
            for variant in ('exact', 'paraphrase'):
                for discovery in rows(final_discovery_root / ('test-natural-' + variant) / 'candidates.jsonl'):
                    query = queries[discovery['query_id']]
                    if query['query_id'] not in expected_test or query['query_id'] in test_ids:
                        raise ValueError('TEST discovery must contain each held-out query exactly once')
                    test_ids.add(query['query_id'])
                    result = rerank_query(query, discovery, chunks, cache, conditions=('history_learned',), ranker=ranker)
                    condition = result['conditions']['history_learned']
                    evidence = selected_evidence(condition, alpha=policy['alpha'])
                    scores = structural_result(query, evidence, 'natural')
                    confidence = support_score(evidence, condition['address_diagnostics'])
                    item = {'query_id': query['query_id'], 'family_id': query['family_id'],
                            'query_variant': query['query_variant'], 'stratum': query['stratum'],
                            'quote_absent': query.get('source_quote_absent_from_target', False),
                            **scores, 'support_confidence': confidence,
                            'support_accepted': confidence >= policy['support_threshold']['threshold'],
                            'top_evidence': [{k: r.get(k) for k in ('chunk_id', 'parent_passage_id', 'law_id',
                                              'revision_id', 'scope', 'article_num')} for r in evidence]}
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                    subset = ('unavailable_or_unknown' if query['target_status'] != 'available' or item['quote_absent'] is None
                              else 'quote_absent' if item['quote_absent'] else 'quote_present')
                    for subset in ('all', subset):
                        cells[(query['query_variant'], subset)].append(scores)
        if test_ids != expected_test:
            raise ValueError('Ablations require every unchanged natural TEST query')
        panel = {}
        for cell, values in cells.items():
            panel['|'.join(cell)] = {'n': len(values), **{metric: sum(r[metric] for r in values) / len(values)
                  for metric in ('top_parent_correct', 'top_revision_correct', 'target_mrr', 'target_recall_at_four')}}
        summary['models'][label] = {'alpha': policy['alpha'], 'panels': panel}
        write_json(output / 'evaluation.json', summary)
        print(json.dumps({'intervention': label, 'panels': panel}), flush=True)
    summary.update(complete=True, wall_seconds=time.perf_counter()-started,
                   physical_pair_count=cache.physical_pair_count, model_calls=cache.model_call_count,
                   model_seconds=cache.model_wall_seconds_with_cuda_sync)
    write_json(output / 'evaluation.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fit', 'evaluate'), required=True)
    parser.add_argument('--discovery', type=Path, default=Path('runs/hidate/discovery'))
    parser.add_argument('--final-discovery', type=Path, default=Path('runs/hidate/final-discovery'))
    parser.add_argument('--output', type=Path, default=Path('runs/hidate/extended-benchmark/ablations'))
    parser.add_argument('--stack', type=Path, default=Path('models/ruri/model-stack.json'))
    args = parser.parse_args()
    if args.phase == 'fit':
        fit(args.discovery, args.output / 'rankers')
    else:
        evaluate(args.output, args.discovery, args.final_discovery, args.stack, args.output)
