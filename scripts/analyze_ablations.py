"""Read back complete saved ablations against the original frozen full model."""
from collections import defaultdict
import json
from pathlib import Path

from hidate.io import write_json
from hidate.ablations import REMOVED_FEATURES
from hidate.statistics import paired_bootstrap


def rows(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


root = Path('runs/hidate/extended-benchmark/ablations')
queries = {r['query_id']: r for name in ('queries-exact.jsonl', 'queries-paraphrase.jsonl')
           for r in rows(Path('data/processed/hidate') / name) if r['split'] == 'test'}
expected = set(queries)
baseline = {}
for row in rows('runs/hidate/final/retrieval.jsonl'):
    if row['evaluation_scope'] == 'natural':
        key = row['query_id']
        if key in baseline or key not in expected:
            raise ValueError('Original frozen baseline TEST identities disagree')
        baseline[key] = row['conditions']['history_learned']['structural']
if set(baseline) != expected:
    raise ValueError('Original baseline requires the same complete3172 natural TEST rows')
methods = {'full_frozen': baseline}
coverage = {}
for label in REMOVED_FEATURES:
    data = list(rows(root / (label + '-test.jsonl')))
    actual = [row['query_id'] for row in data]
    policy = json.loads((root / (label + '-policy.json')).read_text())
    ranker = json.loads((root / 'rankers' / (label + '.json')).read_text())
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Ablation TEST identity mismatch: ' + label)
    if policy['queries'] != 2490 or ranker['training_statistics']['query_count'] != 14030 or ranker['random_state'] != 42:
        raise ValueError('Ablation TRAIN/VALID counts or fixed seed differ: ' + label)
    coverage[label] = {'TRAIN': 14030, 'VALID': 2490, 'TEST': len(actual),
                       'unique_TEST_ids': len(set(actual)), 'alpha': policy['alpha']}
    methods[label] = {row['query_id']: row for row in data}
panels = defaultdict(list)
for key, query in queries.items():
    flag = query.get('source_quote_absent_from_target')
    subset = ('unavailable_or_unknown' if query['target_status'] != 'available' or flag is None
              else 'quote_absent' if flag else 'quote_present')
    for label in ('all', subset):
        panels[(query['query_variant'], label)].append(key)
metrics = ('top_parent_correct', 'top_revision_correct', 'target_mrr', 'target_recall_at_four')
summary, effects = {}, {}
for panel, ids in panels.items():
    name = '|'.join(panel)
    summary[name] = {'n': len(ids), 'families': len({queries[key]['family_id'] for key in ids}), 'methods': {}}
    for label, data in methods.items():
        summary[name]['methods'][label] = {m: sum(data[key][m] for key in ids) / len(ids) for m in metrics}
        if label == 'full_frozen' or panel[1] != 'quote_absent':
            continue
        effects[name + '|' + label] = {}
        for metric in metrics:
            left = [{'id': key, 'group_id': queries[key]['family_id'], metric: data[key][metric]} for key in ids]
            right = [{'id': key, 'group_id': queries[key]['family_id'], metric: baseline[key][metric]} for key in ids]
            effects[name + '|' + label][metric] = paired_bootstrap(left, right, metric, seed=42, n_resamples=2000)
record = {'complete': True, 'evaluation_job': '5100217.linux1.dg.creatis.insa-lyon.fr',
          'original_full_baseline': 'runs/hidate/final/retrieval.jsonl; history_learned; natural; original frozen alpha0.75',
          'coverage_readback': coverage, 'panels': dict(summary), 'full_minus_ablation': effects,
          'new_model_forward_calls': 0, 'new_training_runs': 0}
write_json(root / 'paired-analysis.json', record)
lines = ['# HiDate feature-group ablations', '',
         'Four single-seed TRAIN fits; unchanged2490 VALID and3172 natural TEST identities. '
         'Each mixture is selected on VALID; the original full model and alpha0.75 are reused unchanged.', '',
         '| Query variant | Removed feature group | Recall@4 | Full Recall@4 | Full minus ablation pp [95% family CI] |',
         '|---|---|---:|---:|---:|']
for variant in ('exact_quote', 'generated_paraphrase'):
    panel = variant + '|quote_absent'
    full = summary[panel]['methods']['full_frozen']['target_recall_at_four']
    for label in REMOVED_FEATURES:
        value = summary[panel]['methods'][label]['target_recall_at_four']
        effect = effects[panel + '|' + label]['target_recall_at_four']
        low, high = effect['ci95']
        lines.append(f'| {variant} | {label} | {value*100:.2f} | {full*100:.2f} | '
                     f'{effect["delta_b_minus_a"]*100:+.2f} [{low*100:+.2f}, {high*100:+.2f}] |')
lines += ['', 'The source-quote-absent stratum contains263 paired queries across13 observed TEST families per variant. '
          'Present and unavailable/unknown target panels are kept separate in paired-analysis.json.', '']
report = Path('reports/feature-ablations.md')
report.write_text('\n'.join(lines), encoding='utf8')
print('\n'.join(lines))
