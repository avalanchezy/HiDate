"""Small saved-output fixtures for complete, family-paired expert statistics."""

from copy import deepcopy
import csv

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_expert_retrieval import analyze, save_analysis, SELECTED, METRICS


def fixture(temporal=False):
    methods = [SELECTED, 'reference', 'weak']
    spec = [
        ('A_B', 'A', 'old_terms', 1, 1),
        ('A_A', 'A', 'old_terms', 1, 0),
        ('A_S', 'A', 'old_terms', 0, 1),
        ('B_B', 'B', 'real_world', 1, 0),
        ('B_A', 'B', 'real_world', 0, 0),
        ('C_B', 'C', 'changed_conditions', 0, 1),
        ('C_A', 'C', 'changed_conditions', 0, 1),
    ]
    predictions = []
    for qid, family, category, selected, reference in spec:
        predictions.append(dict(query_id=qid, family_id=family, category=category,
            target_asof='2025-01-01' if qid.endswith('_B') else '2025-06-01',
            answerability='answerable', conditions={method: {'structural': {
                metric: float(value)/2 if metric.startswith('first_complete') else float(value)
                for metric in METRICS}}
                for method, value in zip(methods, [selected, reference, 0])}))
    for family in ['A', 'B']:
        predictions.append(dict(query_id=family+'_I', family_id=family,
            category='insufficient_evidence', target_asof='2025-06-01',
            answerability='insufficient_facts', conditions={method: {
                'structural': dict.fromkeys(METRICS, None)} for method in methods}))
    if temporal:
        # A generic answerable negative with no applicable provision, not an ID rule.
        predictions[0]['retrieval_target_type'] = 'temporal_nonapplicability'
        for condition in predictions[0]['conditions'].values():
            condition['structural'] = dict.fromkeys(METRICS, None)
    evaluation = dict(complete=True, queries=9, family_count=3, selected_method=SELECTED,
        answerability={'answerable': 7, 'insufficient_facts': 2}, panels={})
    for method in methods:
        evaluation['panels'][method] = {}
        for panel, category in [('answerable', None), ('category=old_terms', 'old_terms'),
                                ('category=real_world', 'real_world'),
                                ('category=changed_conditions', 'changed_conditions')]:
            rr = [r for r in predictions if r['answerability']=='answerable'
                  and r.get('retrieval_target_type', 'provision_set') == 'provision_set'
                  and (category is None or r['category']==category)]
            evaluation['panels'][method][panel] = {'n': len(rr), **{
                metric: sum(r['conditions'][method]['structural'][metric] for r in rr)/len(rr)
                for metric in METRICS}}
    pairs = [dict(pair_id='DP'+family, scenario_id=family,
                  before_question_id=family+'_B', after_question_id=family+'_A',
                  pair_status='verified', distinct_versions_verified='yes',
                  same_facts_verified='yes', verified_answer_relation=relation)
             for family, relation in [('A', 'changed'), ('B', 'unchanged'), ('C', 'changed')]]
    expected = dict(queries=9, answerable=7, families=3, pairs=3, conditions=3,
                    provision_eligible=6 if temporal else 7, temporal=1 if temporal else 0,
                    eligible_pairs=2 if temporal else 3, temporal_pairs=1 if temporal else 0)
    return evaluation, predictions, pairs, expected


def run(data):
    evaluation, predictions, pairs, expected = data
    return analyze(evaluation, predictions, pairs, expected=expected)


def test_query_means_and_family_paired_differences_use_same_bootstrap_draws():
    result = run(fixture())
    assert result['complete']
    assert result['coverage']['answerable_queries'] == 7
    metric = METRICS[0]
    panel = result['panels']['answerable']
    assert panel['families'] == 3
    assert panel['methods'][SELECTED][metric]['mean'] == pytest.approx(3/7)
    reference = panel['methods']['reference'][metric]
    assert reference['mean'] == pytest.approx(4/7)
    assert reference['selected_minus_reference'] == pytest.approx(-1/7)
    # All-A family draws give 0 contrast, all-B .5, all-C -1.
    # Family bootstrap preserves A's three observations together.
    assert reference['selected_minus_reference_ci95'] == [-1., .5]
    assert panel['methods'][SELECTED][metric]['selected_minus_reference_ci95'] == [0., 0.]
    assert result['panels']['category=old_terms']['n'] == 3
    assert result['panels']['category=real_world']['n'] == 2
    assert result == run(fixture())


def test_date_pair_outcome_means_both_dates_retrieval_complete_not_answer_change():
    result = run(fixture())['date_pair_panels']
    key = 'both_dates_retrieval_complete_at4'
    assert result['all_verified_pairs']['n'] == 3
    assert result['all_verified_pairs']['methods'][SELECTED][key]['mean'] == pytest.approx(1/3)
    assert result['verified_changed']['n'] == 2
    assert result['verified_changed']['methods'][SELECTED][key]['mean'] == .5
    assert result['verified_unchanged']['n'] == 1
    assert result['verified_unchanged']['methods'][SELECTED][key]['mean'] == 0.


@pytest.mark.parametrize('change,match', [
    ('missing_query', 'query coverage'),
    ('duplicate_query', 'Duplicate query'),
    ('missing_condition', 'condition coverage'),
    ('missing_metric', 'metric coverage'),
    ('null_answerable', 'finite.*metric'),
    ('zero_insufficient', 'Insufficient.*None'),
    ('partial_receipt', 'complete evaluation'),
    ('wrong_saved_mean', 'Saved evaluation replay'),
])
def test_incomplete_or_inconsistent_inputs_fail_instead_of_dropping_records(change, match):
    data = fixture()
    evaluation, predictions, pairs, expected = data
    if change == 'missing_query': predictions.pop()
    elif change == 'duplicate_query': predictions[-1] = deepcopy(predictions[0])
    elif change == 'missing_condition': predictions[-1]['conditions'].pop('weak')
    elif change == 'missing_metric': predictions[0]['conditions']['reference']['structural'].pop(METRICS[0])
    elif change == 'null_answerable': predictions[0]['conditions']['reference']['structural'][METRICS[0]] = None
    elif change == 'zero_insufficient': predictions[-1]['conditions']['weak']['structural'][METRICS[0]] = 0.
    elif change == 'partial_receipt': evaluation['complete'] = False
    elif change == 'wrong_saved_mean': evaluation['panels']['reference']['answerable'][METRICS[0]] = .99
    with pytest.raises(ValueError, match=match): run(data)


@pytest.mark.parametrize('change,match', [
    ('missing_pair', 'pair coverage'),
    ('duplicate_pair', 'Duplicate pair'),
    ('missing_pair_query', 'Pair query'),
    ('reused_date_query', 'reuses a question'),
    ('unverified_relation', 'verified changed/unchanged'),
])
def test_date_pair_coverage_is_checked_before_aggregation(change, match):
    data = fixture()
    pairs = data[2]
    if change == 'missing_pair': pairs.pop()
    elif change == 'duplicate_pair': pairs[-1] = deepcopy(pairs[0])
    elif change == 'missing_pair_query': pairs[0]['before_question_id'] = 'absent'
    elif change == 'reused_date_query': pairs[1]['before_question_id'] = pairs[0]['before_question_id']
    elif change == 'unverified_relation': pairs[0]['verified_answer_relation'] = 'not_evaluable'
    with pytest.raises(ValueError, match=match): run(data)


def test_csv_exports_keep_every_condition_metric_and_category(tmp_path):
    result = run(fixture())
    save_analysis(result, tmp_path)
    with (tmp_path/'retrieval-metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3*4*4
    assert {r['condition'] for r in rows} == {SELECTED, 'reference', 'weak'}
    assert sum(r['metric_role']=='primary' for r in rows) == 3*4
    with (tmp_path/'date-pair-retrieval.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 3*3
    assert {r['metric'] for r in pairs} == {'both_dates_retrieval_complete_at4'}


def test_temporal_instance_and_pair_remain_in_coverage_but_not_provision_denominators(tmp_path):
    result = run(fixture(temporal=True))
    coverage = result['coverage']
    assert coverage['queries'] == 9 and coverage['answerable_queries'] == 7
    assert coverage['provision_eligible_queries'] == 6
    assert coverage['temporal_nonapplicability_queries'] == 1
    assert coverage['insufficient_queries'] == 2
    assert coverage['full_pair_count'] == 3
    assert coverage['eligible_pair_count'] == 2 and coverage['temporal_pair_count'] == 1
    assert coverage['verified_pair_relations'] == {'changed': 2, 'unchanged': 1}
    assert coverage['eligible_pair_relations'] == {'unchanged': 1, 'changed': 1}
    panel = result['panels']['answerable']
    assert panel['n'] == panel['provision_eligible_count'] == 6
    assert panel['full_answerable_count'] == 7
    assert panel['temporal_nonapplicability_count'] == 1
    assert panel['methods'][SELECTED][METRICS[0]]['mean'] == pytest.approx(2/6)
    pairs = result['date_pair_panels']
    assert pairs['all_verified_pairs']['n'] == 2
    assert pairs['all_verified_pairs']['full_pair_count'] == 3
    assert pairs['verified_changed']['n'] == 1
    assert pairs['verified_changed']['full_pair_count'] == 2
    assert pairs['verified_changed']['temporal_pair_count'] == 1
    assert pairs['verified_unchanged']['n'] == 1
    assert pairs['verified_unchanged']['temporal_pair_count'] == 0
    assert pairs['all_verified_pairs']['methods'][SELECTED]['both_dates_retrieval_complete_at4']['mean'] == 0.
    save_analysis(result, tmp_path)
    with (tmp_path/'date-pair-retrieval.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {r['n'] for r in rows if r['scope']=='all_verified_pairs'} == {'2'}
    assert {r['full_pair_count'] for r in rows if r['scope']=='all_verified_pairs'} == {'3'}


def test_temporal_target_with_a_numeric_metric_is_rejected_instead_of_scored():
    data = fixture(temporal=True)
    data[1][0]['conditions'][SELECTED]['structural'][METRICS[0]] = 0.
    with pytest.raises(ValueError, match='Temporal.*None'): run(data)


def test_temporal_pair_is_still_required_for_full_coverage_validation():
    data = fixture(temporal=True)
    data[2].pop(0)
    with pytest.raises(ValueError, match='pair coverage'): run(data)


def test_320_question_80_pair_allocation_keeps_full_relations_and_eligible_denominators():
    # Synthetic bookkeeping fixture: no actual packet fields or model outputs.
    evaluation, examples, _, expected = fixture()
    predictions, pairs = [], []
    categories = ['old_terms', 'real_world', 'changed_conditions']
    for i in range(80):
        for side, example in [('B', examples[0]), ('A', examples[1])]:
            row = deepcopy(example)
            row.update(query_id=f'PAIR{i}_{side}', family_id=f'LAW{i % 40}',
                       category=categories[i % 3])
            if i == 0 and side == 'B':
                row['retrieval_target_type'] = 'temporal_nonapplicability'
                for condition in row['conditions'].values():
                    condition['structural'] = dict.fromkeys(METRICS, None)
            predictions.append(row)
        pairs.append(dict(pair_id=f'DATEPAIR{i}', before_question_id=f'PAIR{i}_B',
            after_question_id=f'PAIR{i}_A', verified_answer_relation='changed' if i < 40 else 'unchanged',
            pair_status='verified', distinct_versions_verified='yes', same_facts_verified='yes'))
    for i in range(100):
        row = deepcopy(examples[2])
        row.update(query_id=f'SINGLE{i}', family_id=f'LAW{i % 40}', category=categories[i % 3])
        predictions.append(row)
    for i in range(60):
        row = deepcopy(examples[-1])
        row.update(query_id=f'INSUFFICIENT{i}', family_id=f'LAW{i % 40}')
        predictions.append(row)
    evaluation.update(queries=320, family_count=40,
                      answerability={'answerable': 260, 'insufficient_facts': 60})
    for method, panels in evaluation['panels'].items():
        for scope in panels:
            rows = [r for r in predictions if r['answerability'] == 'answerable'
                    and r.get('retrieval_target_type', 'provision_set') == 'provision_set'
                    and (scope == 'answerable' or r['category'] == scope.split('=', 1)[1])]
            panels[scope] = {'n': len(rows), **{metric: sum(
                r['conditions'][method]['structural'][metric] for r in rows)/len(rows)
                for metric in METRICS}}
    expected.update(queries=320, answerable=260, provision_eligible=259, temporal=1,
                    families=40, pairs=80, eligible_pairs=79, temporal_pairs=1)
    result = analyze(evaluation, predictions, pairs, expected=expected)
    coverage = result['coverage']
    assert coverage['queries'] == 320 and coverage['answerable_queries'] == 260
    assert coverage['provision_eligible_queries'] == 259
    assert coverage['temporal_nonapplicability_queries'] == 1
    assert coverage['insufficient_queries'] == 60
    assert coverage['verified_pair_relations'] == {'changed': 40, 'unchanged': 40}
    assert coverage['eligible_pair_relations'] == {'changed': 39, 'unchanged': 40}
    assert result['panels']['answerable']['n'] == 259
    for scope, full, eligible, temporal in [('all_verified_pairs', 80, 79, 1),
                                           ('verified_changed', 40, 39, 1),
                                           ('verified_unchanged', 40, 40, 0)]:
        panel = result['date_pair_panels'][scope]
        assert (panel['full_pair_count'], panel['eligible_pair_count'], panel['temporal_pair_count']) == (full, eligible, temporal)
        assert panel['n'] == eligible
