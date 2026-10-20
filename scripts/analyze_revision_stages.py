"""Summarize saved discovery -> binding -> own top10 -> final4, without models."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import time


ROOT = Path('runs/hidate/revision-aware')
DATA = Path('data/processed/hidate-holdout/confirmatory')
METRICS = ('discovery_address_hit', 'bindable_parent_hit',
           'candidate_parent_at10', 'final_recall_at4')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def aggregate(records):
    cells = defaultdict(list)
    for r in records:
        cells[r['query_variant'] + '|' + r['stratum']].append(r)
    def mean(values):
        return {'n': len(values), **{k: sum(float(v[k]) for v in values) / len(values) for k in METRICS}}
    if len(cells) != 4:
        raise ValueError('Four original variant/date cells are required')
    cell_means = {k: mean(v) for k, v in sorted(cells.items())}
    return {'available': mean(records), 'cells': cell_means,
        'available_four_cell_macro': {'n': len(records),
            **{k: sum(v[k] for v in cell_means.values()) / 4 for k in METRICS}}}


def qwen_preselection(prepared, query):
    gold = (query['target_key'][0], query['target_key'][2], query['target_key'][3])
    target = query['target_passage_id']
    addresses = prepared['addresses']
    return {'discovery_address_hit': any(tuple(a['address']) == gold for a in addresses),
        'bindable_parent_hit': any(a['status'] == 'available' and
            a['current_parent_passage_id'] == target for a in addresses),
        'pools': {**{head: any(c['current_parent_passage_id'] == target for c in cc)
                   for head, cc in prepared['heads'].items()},
                  'AsOf': any(c['current_parent_passage_id'] == target for c in prepared['asof'])}}


def method_stages(prediction, qwen_stage, asof_stage):
    result = {}
    for name, condition in prediction['conditions'].items():
        asof = condition.get('head') == 'AsOf' or name.startswith('AsOf_')
        source = asof_stage if asof else (qwen_stage if qwen_stage is not None else prediction)
        candidate = (condition['candidate_parent_at10'] if 'candidate_parent_at10' in condition
                     else qwen_stage['pools'][condition['head']])
        result[name] = {'discovery_bank': 'target_date' if asof else 'history',
            'discovery_address_hit': int(source['discovery_address_hit']),
            'bindable_parent_hit': int(source['bindable_parent_hit']),
            'candidate_parent_at10': int(candidate),
            'final_recall_at4': int(condition['structural']['target_recall_at_four'])}
    return result


def query_map(confirmation=False):
    if not confirmation:
        return {q['query_id']: q for path in [Path('data/processed/hidate/queries-exact.jsonl'),
                Path('data/processed/hidate/queries-paraphrase.jsonl')]
                for q in rows(path) if q['split'] == 'test'}
    # Called only after the final evaluation receipt is complete. No selection.
    gold = {q['query_id']: q for q in rows(DATA / 'sealed/queries-exact.jsonl')}
    queries = {}
    for name in ['queries-exact-unlabeled.jsonl', 'queries-paraphrase-unlabeled.jsonl']:
        for public in rows(DATA / 'unlabeled' / name):
            q = gold[public.get('base_query_id', public['query_id'])]
            queries[public['query_id']] = {**q, **{k: public[k] for k in
                ('query_id', 'query_variant', 'query_text', 'target_asof')}}
    return queries


def asof_discovery(paths, queries, index):
    from hidate.discovery import chunk_address
    chunks = {c['chunk_id']: (chunk_address(c), c['parent_passage_id'])
              for c in rows(index / 'chunks.jsonl')}
    result = {}
    for path in paths:
        for d in rows(path):
            qid = d['query_id']; q = queries[qid]
            gold = (q['target_key'][0], q['target_key'][2], q['target_key'][3])
            hits = [chunks[h['chunk_id']] for h in d['asof']['hybrid']]
            if qid in result:
                raise ValueError('Duplicate saved as-of discovery query')
            result[qid] = {'discovery_address_hit': any(a == gold for a, parent in hits),
                'bindable_parent_hit': any(parent == q['target_passage_id'] for a, parent in hits)}
    if set(result) != set(queries):
        raise ValueError('Saved as-of fifty-hit discovery coverage differs')
    return result


def cohort_spec(name, root):
    if name == 'qwen120':
        output = root / 'transfer/qwen-test'
        meta = read(output / 'preparation.json')
        return {'directory': output, 'extra': [output / 'ce-controls'], 'qwen': True,
            'discovery': [Path(p) for p in meta['discovery_paths']], 'index': Path(meta['index_path'])}
    directory, discovery, index = {
        'ruri120': ('transfer/ruri-original-final', 'transfer/ruri-original-native', 'data/index/hidate'),
        'ruri1000': ('transfer/ruri-scale-final', 'transfer/ruri-scale-native',
                     'data/index/hidate/extended-benchmark/scale'),
        'confirmation': ('confirmation/final', 'confirmation/discovery',
                         'data/index/hidate/extended-benchmark/scale')}[name]
    return {'directory': root / directory, 'extra': [], 'qwen': False,
        'discovery': [root / discovery / 'candidates.jsonl'], 'index': Path(index)}


def analyze_cohort(name, root):
    started = time.perf_counter(); spec = cohort_spec(name, root)
    directories = [spec['directory']] + spec['extra']
    evaluations = [read(d / 'evaluation.json') for d in directories]
    expected = 14912 if name == 'confirmation' else 3172
    if not all(e['complete'] and e['queries'] == expected for e in evaluations):
        raise ValueError('Complete saved final evaluation required before stage analysis')
    queries = query_map(name == 'confirmation')
    if len(queries) != expected:
        raise ValueError('Fixed original/confirmation query allocation differs')
    asof = asof_discovery(spec['discovery'], queries, spec['index'])
    prepared = {}
    if spec['qwen']:
        for row in rows(spec['directory'] / 'prepared.jsonl'):
            prepared[row['query_id']] = qwen_preselection(row, queries[row['query_id']])
        if set(prepared) != set(queries):
            raise ValueError('Own-Qwen saved preselection coverage differs')
    values = defaultdict(list); banks = {}; violations = Counter(); source_counts = []
    for directory in directories:
        seen = set()
        for pred in rows(directory / 'predictions.jsonl'):
            qid = pred['query_id']; q = queries[qid]
            if qid in seen or any(pred[k] != q[k] for k in ('query_variant', 'stratum', 'target_status')):
                raise ValueError('Saved prediction identity/metadata differs from fixed allocation')
            seen.add(qid)
            if pred['target_status'] != 'available':
                continue
            for method, stages in method_stages(pred, prepared.get(qid), asof[qid]).items():
                banks[method] = stages['discovery_bank']
                record = {**{k: pred[k] for k in ('query_variant', 'stratum')}, **stages}
                values[method].append(record)
                if not (stages['discovery_address_hit'] >= stages['bindable_parent_hit'] >=
                        stages['candidate_parent_at10'] >= stages['final_recall_at4']):
                    violations[method] += 1
        if seen != set(queries):
            raise ValueError('Saved final prediction coverage differs')
        source_counts.append({'path': str(directory / 'predictions.jsonl'), 'queries': len(seen)})
    methods = {method: {'discovery_bank': banks[method], **aggregate(records)}
               for method, records in values.items()}
    for evaluation in evaluations:
        for method, old in evaluation['available_four_cell_macro'].items():
            actual = methods[method]['available_four_cell_macro']['final_recall_at4']
            if abs(actual - old['target_recall_at_four']) > 1e-12:
                raise ValueError('Saved final Recall@4 replay differs: ' + method)
    return {'complete': True, 'queries': expected,
        'families': len({q['family_id'] for q in queries.values()}),
        'available_families': len({q['family_id'] for q in queries.values() if q['target_status'] == 'available'}),
        'available_queries':
        sum(q['target_status'] == 'available' for q in queries.values()),
        'source_predictions': source_counts, 'discovery_paths': [str(p) for p in spec['discovery']],
        'index': str(spec['index']), 'methods': methods, 'nested_stage_violations': dict(violations),
        'wall_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohorts', nargs='+', choices=('ruri120', 'ruri1000', 'qwen120', 'confirmation'),
        default=['ruri120', 'ruri1000', 'qwen120'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/stage-reachability/regression')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'complete': False, 'seed': 42, 'training_runs': 0, 'model_calls': 0,
        'selection_updates': 0, 'job_id': os.environ.get('PBS_JOBID'), 'cohorts': {},
        'definitions': {'available': 'query-weighted mean over available targets',
            'available_four_cell_macro': 'equal mean of four variant x date-stratum cell means',
            'discovery_address_hit': 'gold scoped address in its own bank fused top50',
            'bindable_parent_hit': 'bound target parent reachable in that discovered pool',
            'candidate_parent_at10': 'target parent in this head own deliverable top10',
            'final_recall_at4': 'target parent in saved returned top4',
            'n': 'number of query observations; macro uses equal cell means, not pooled n weights'}}
    for name in args.cohorts:
        result['cohorts'][name] = analyze_cohort(name, args.root)
        print(json.dumps({'cohort': name, 'complete': True, 'methods':
            len(result['cohorts'][name]['methods'])}), flush=True)
    result.update(complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
    (args.output / 'stage-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (args.output / 'stage-metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort', 'method', 'discovery_bank', 'scope', 'n'] + list(METRICS))
        writer.writeheader()
        for cohort, data in result['cohorts'].items():
            for method, data in data['methods'].items():
                scopes = {k: data[k] for k in ('available', 'available_four_cell_macro')}
                scopes.update({'cell=' + k: v for k, v in data['cells'].items()})
                for scope, metrics in scopes.items():
                    writer.writerow({'cohort': cohort, 'method': method, 'discovery_bank': data['discovery_bank'],
                        'scope': scope, **metrics})
    print(json.dumps({'complete': True, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
