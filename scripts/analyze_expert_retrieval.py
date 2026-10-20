"""Family-paired statistics of complete saved expert-scenario retrieval results."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path('runs/hidate/expert-benchmark')
DATA = Path('data/expert-benchmark')
SELECTED = 'Full_LM18_historical_CE'
PRIMARY = 'revision_correct_set_complete_at_k'
METRICS = (PRIMARY, 'first_complete_set_reciprocal_rank_at_k',
           'annotated_address_recall_at_k', 'revision_correct_set_coverage_at_k')
CATEGORIES = ('old_terms', 'real_world', 'changed_conditions')
EXPECTED = dict(queries=320, answerable=260, provision_eligible=259, temporal=1,
                families=40, pairs=80, eligible_pairs=79, temporal_pairs=1, conditions=33)
PAIR_METRIC = 'both_dates_retrieval_complete_at4'


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def provision_eligible(row):
    return (row['answerability'] == 'answerable' and
            row.get('retrieval_target_type', 'provision_set') == 'provision_set')


def validate(evaluation, predictions, pairs, expected):
    if not evaluation.get('complete') or evaluation.get('selected_method') != SELECTED:
        raise ValueError('A complete evaluation with the frozen selected method is required')
    indexed = {}
    for row in predictions:
        if row['query_id'] in indexed:
            raise ValueError('Duplicate query: '+row['query_id'])
        indexed[row['query_id']] = row
    if len(indexed) != expected['queries'] or evaluation['queries'] != expected['queries']:
        raise ValueError('Full query coverage differs from the fixed expert allocation')
    answerable = [r for r in predictions if r['answerability'] == 'answerable']
    eligible = [r for r in predictions if provision_eligible(r)]
    temporal = [r for r in predictions if
                r.get('retrieval_target_type', 'provision_set') == 'temporal_nonapplicability']
    families = {r['family_id'] for r in predictions}
    if (len(answerable) != expected['answerable'] or len(families) != expected['families']
            or evaluation['family_count'] != len(families)
            or {r['family_id'] for r in answerable} != families):
        raise ValueError('Answerable/family coverage differs from the fixed expert allocation')
    if len(eligible) != expected['provision_eligible'] or len(temporal) != expected['temporal']:
        raise ValueError('Provision-eligible/temporal coverage differs from the fixed allocation')
    if dict(Counter(r['answerability'] for r in predictions)) != evaluation['answerability']:
        raise ValueError('Saved answerability counts differ from predictions')
    if {r['category'] for r in answerable} != set(CATEGORIES):
        raise ValueError('The three answerable categories must all be retained')
    methods = sorted(evaluation['panels'])
    if len(methods) != expected['conditions'] or SELECTED not in methods:
        raise ValueError('Saved condition coverage differs from the fixed allocation')
    for row in predictions:
        target_type = row.get('retrieval_target_type', 'provision_set')
        if target_type not in ('provision_set', 'temporal_nonapplicability'):
            raise ValueError('Unknown retrieval target type: '+target_type)
        if target_type == 'temporal_nonapplicability' and row['answerability'] != 'answerable':
            raise ValueError('Temporal nonapplicability remains an answerable case')
        if set(row['conditions']) != set(methods):
            raise ValueError('Per-query condition coverage differs: '+row['query_id'])
        for name in methods:
            metrics = row['conditions'][name]['structural']
            if set(metrics) != set(METRICS):
                raise ValueError('Per-query metric coverage differs: '+row['query_id']+'/'+name)
            if row['answerability'] != 'answerable':
                if any(value is not None for value in metrics.values()):
                    raise ValueError('Insufficient cases must have None retrieval metrics')
            elif target_type == 'temporal_nonapplicability':
                if any(value is not None for value in metrics.values()):
                    raise ValueError('Temporal nonapplicability cases must have None retrieval metrics')
            elif any(value is None or not isinstance(value, (int, float))
                     or not np.isfinite(value) or not 0 <= value <= 1
                     for value in metrics.values()):
                raise ValueError('Answerable cases need a finite [0,1] metric: '+row['query_id'])
            elif metrics[PRIMARY] not in (0, 1):
                raise ValueError('Revision-correct set complete must be binary')

    pair_ids, used_queries = set(), set()
    for pair in pairs:
        if pair['pair_id'] in pair_ids:
            raise ValueError('Duplicate pair: '+pair['pair_id'])
        pair_ids.add(pair['pair_id'])
        before, after = pair['before_question_id'], pair['after_question_id']
        if before not in indexed or after not in indexed:
            raise ValueError('Pair query absent from complete saved predictions')
        if before == after or {before, after} & used_queries:
            raise ValueError('A date pair reuses a question')
        used_queries.update((before, after))
        if (pair['verified_answer_relation'] not in ('changed', 'unchanged')
                or pair['pair_status'] != 'verified'
                or pair['distinct_versions_verified'] != 'yes'
                or pair['same_facts_verified'] != 'yes'):
            raise ValueError('All date pairs need verified changed/unchanged relations and confirmations')
        b, a = indexed[before], indexed[after]
        if (b['family_id'] != a['family_id'] or b['category'] != a['category']
                or b['answerability'] != 'answerable' or a['answerability'] != 'answerable'
                or b['target_asof'] >= a['target_asof']):
            raise ValueError('Date pair family/category/answerability/date identity differs')
    if len(pair_ids) != expected['pairs']:
        raise ValueError('Full date pair coverage differs from the fixed expert allocation')
    eligible_pairs = sum(all(provision_eligible(indexed[p[k]]) for k in
                         ('before_question_id', 'after_question_id')) for p in pairs)
    if (eligible_pairs != expected['eligible_pairs'] or
            len(pairs)-eligible_pairs != expected['temporal_pairs']):
        raise ValueError('Provision-eligible/temporal pair coverage differs from the fixed allocation')
    return indexed, answerable, sorted(eligible, key=lambda r: r['query_id']), methods


def summarize(values, families, methods, metric_names):
    """Query-weighted means, resampling entire law families with identical draws."""
    family_names = sorted(set(families))
    sums = np.asarray([values[np.asarray(families) == name].sum(axis=0)
                       for name in family_names])
    counts = np.asarray([families.count(name) for name in family_names])
    draws = np.random.default_rng(42).integers(0, len(family_names),
                                               size=(2000, len(family_names)))
    samples = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:, None, None]
    means = values.mean(axis=0)
    selected_index = methods.index(SELECTED)
    differences = samples[:, selected_index:selected_index+1, :] - samples
    ci = np.percentile(samples, [2.5, 97.5], axis=0)
    diff_ci = np.percentile(differences, [2.5, 97.5], axis=0)
    return {'n': len(values), 'families': len(family_names), 'methods': {
        method: {metric: {'mean': float(means[i, j]),
                          'ci95': ci[:, i, j].tolist(),
                          'selected_minus_reference': float(means[selected_index, j]-means[i, j]),
                          'selected_minus_reference_ci95': diff_ci[:, i, j].tolist()}
                 for j, metric in enumerate(metric_names)}
        for i, method in enumerate(methods)}}


def analyze(evaluation, predictions, pairs, *, expected=EXPECTED):
    indexed, answerable, eligible, methods = validate(evaluation, predictions, pairs, expected)
    panels = {}
    replay_error = 0.
    for scope, category in [('answerable', None)]+[('category='+c, c) for c in CATEGORIES]:
        rows = [r for r in eligible if category is None or r['category'] == category]
        values = np.asarray([[[r['conditions'][m]['structural'][k] for k in METRICS]
                              for m in methods] for r in rows], dtype=float)
        panel = summarize(values, [r['family_id'] for r in rows], methods, METRICS)
        full_count = sum(category is None or r['category'] == category for r in answerable)
        panel.update(population='provision_eligible_answerable',
                     full_answerable_count=full_count, provision_eligible_count=len(rows),
                     temporal_nonapplicability_count=full_count-len(rows))
        for name in methods:
            saved = evaluation['panels'][name][scope]
            if saved['n'] != len(rows):
                raise ValueError('Saved evaluation replay n differs: '+name+'/'+scope)
            for metric in METRICS:
                error = abs(panel['methods'][name][metric]['mean']-saved[metric])
                if not np.isfinite(error) or error > 1e-12:
                    raise ValueError('Saved evaluation replay mean differs: '+name+'/'+scope+'/'+metric)
                replay_error = max(replay_error, error)
        panels[scope] = panel

    pair_panels = {}
    for scope, relation in [('all_verified_pairs', None), ('verified_changed', 'changed'),
                            ('verified_unchanged', 'unchanged')]:
        full = [p for p in pairs if relation is None or p['verified_answer_relation'] == relation]
        selected = sorted([p for p in full if all(provision_eligible(indexed[p[k]]) for k in
                           ('before_question_id', 'after_question_id'))], key=lambda p: p['pair_id'])
        if not selected:
            raise ValueError('Date pair changed/unchanged stratum must not be empty')
        values, families = [], []
        for pair in selected:
            b, a = (indexed[pair[k]] for k in ('before_question_id', 'after_question_id'))
            families.append(b['family_id'])
            values.append([[float(b['conditions'][m]['structural'][PRIMARY] == 1 and
                                  a['conditions'][m]['structural'][PRIMARY] == 1)] for m in methods])
        pair_panels[scope] = summarize(np.asarray(values), families, methods, [PAIR_METRIC])
        pair_panels[scope].update(full_pair_count=len(full), eligible_pair_count=len(selected),
                                 temporal_pair_count=len(full)-len(selected))

    eligible_pairs = [p for p in pairs if all(provision_eligible(indexed[p[k]]) for k in
                      ('before_question_id', 'after_question_id'))]
    temporal_queries = [r['query_id'] for r in answerable if not provision_eligible(r)]

    return {'complete': True, 'selected_method': SELECTED, 'primary_metric': PRIMARY,
        'training_runs': 0, 'model_calls': 0, 'selection_updates': 0,
        'bootstrap': {'unit': 'law_family', 'draws': 2000, 'rng_seed': 42,
            'interval': '95% percentile', 'point_estimator': 'query-weighted mean',
            'resampling': 'all member observations retained each time a law family is sampled',
            'contrast': 'Full_LM18_historical_CE minus reference; identical family draws',
            'role': 'uncertainty estimation of saved predictions, not repeated training'},
        'definitions': {'retrieval_scope': 'expert-annotated scoped-address/version sets, k=4',
            'alternative_sets': 'AND within a gold set; OR between alternative sufficient sets',
            'answerable_panel': 'means over provision-eligible answerable cases; n excludes temporal nonapplicability',
            'insufficient_cases': 'retained in coverage; None metrics excluded from provision means',
            'temporal_nonapplicability': 'answerable negative based on official timing records; no applicable '
                                       'provision target, so four metrics are None and excluded from provision means',
            PAIR_METRIC: 'both before and after have revision-correct set complete@4; '
                         'only pairs with two provision-eligible dates enter the denominator; '
                         'this does not score whether the answer changed correctly'},
        'coverage': {'queries': len(predictions), 'answerable_queries': len(answerable),
            'provision_eligible_queries': len(eligible),
            'temporal_nonapplicability_queries': len(temporal_queries),
            'temporal_nonapplicability_query_ids': sorted(temporal_queries),
            'insufficient_queries': len(predictions)-len(answerable),
            'families': len({r['family_id'] for r in predictions}), 'conditions': len(methods),
            'category_counts_answerable': dict(Counter(r['category'] for r in answerable)),
            'category_counts_provision_eligible': dict(Counter(r['category'] for r in eligible)),
            'answerability': dict(Counter(r['answerability'] for r in predictions)),
            'date_pairs': len(pairs),
            'full_pair_count': len(pairs), 'eligible_pair_count': len(eligible_pairs),
            'temporal_pair_count': len(pairs)-len(eligible_pairs),
            'verified_pair_relations': dict(Counter(p['verified_answer_relation'] for p in pairs)),
            'eligible_pair_relations': dict(Counter(p['verified_answer_relation'] for p in eligible_pairs)),
            'query_ids': sorted(indexed), 'pair_ids': sorted(p['pair_id'] for p in pairs),
            'condition_names': methods, 'saved_evaluation_max_absolute_replay_error': replay_error},
        'panels': panels, 'date_pair_panels': pair_panels}


def save_analysis(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    fields = ['scope', 'condition', 'metric', 'metric_role', 'n', 'families', 'mean',
              'ci95_low', 'ci95_high', 'selected_method', 'selected_minus_reference',
              'selected_minus_reference_ci95_low', 'selected_minus_reference_ci95_high']
    for filename, panels in [('retrieval-metrics.csv', result['panels']),
                              ('date-pair-retrieval.csv', result['date_pair_panels'])]:
        denominator_fields = (['full_answerable_count', 'provision_eligible_count', 'temporal_nonapplicability_count']
                              if filename == 'retrieval-metrics.csv' else
                              ['full_pair_count', 'eligible_pair_count', 'temporal_pair_count'])
        with (output/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields+denominator_fields)
            writer.writeheader()
            for scope, panel in panels.items():
                for condition, metrics in panel['methods'].items():
                    for metric, values in metrics.items():
                        writer.writerow(dict(scope=scope, condition=condition, metric=metric,
                            metric_role='primary' if metric == PRIMARY else 'secondary',
                            n=panel['n'], families=panel['families'], mean=values['mean'],
                            ci95_low=values['ci95'][0], ci95_high=values['ci95'][1],
                            selected_method=SELECTED,
                            selected_minus_reference=values['selected_minus_reference'],
                            selected_minus_reference_ci95_low=values['selected_minus_reference_ci95'][0],
                            selected_minus_reference_ci95_high=values['selected_minus_reference_ci95'][1],
                            **{k: panel[k] for k in denominator_fields}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=ROOT/'final/evaluation.json')
    parser.add_argument('--predictions', type=Path, default=ROOT/'final/predictions.jsonl')
    parser.add_argument('--date-pairs', type=Path, default=DATA/'sealed/date-pairs.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'analysis')
    args = parser.parse_args()
    result = analyze(json.loads(args.evaluation.read_text(encoding='utf-8')),
                     read_rows(args.predictions), read_rows(args.date_pairs))
    result.update(job_id=os.environ.get('PBS_JOBID'), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  sources={k: str(getattr(args, k)) for k in ('evaluation', 'predictions', 'date_pairs')})
    save_analysis(result, args.output)
    print(json.dumps({'complete': True, 'queries': result['coverage']['queries'],
                      'conditions': result['coverage']['conditions'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
