import importlib.util
from pathlib import Path

import pytest


def worker():
    path = Path(__file__).parents[1] / 'scripts/analyze_revision_stages.py'
    spec = importlib.util.spec_from_file_location('stage_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_cell_macro_does_not_weight_large_same_date_cell_more():
    m = worker()
    records = []
    for variant in ('exact_quote', 'generated_paraphrase'):
        for stratum in ('same_date', 'changed_cross_date'):
            count = 2 if (variant, stratum) == ('exact_quote', 'same_date') else 1
            for _ in range(count):
                records.append({'query_variant': variant, 'stratum': stratum,
                    'discovery_address_hit': 1, 'bindable_parent_hit': 1,
                    'candidate_parent_at10': 1,
                    'final_recall_at4': int((variant, stratum) == ('exact_quote', 'same_date'))})
    result = m.aggregate(records)
    assert result['available']['n'] == 5
    assert result['available']['final_recall_at4'] == pytest.approx(2 / 5)
    assert result['available_four_cell_macro']['final_recall_at4'] == pytest.approx(1 / 4)
    assert result['cells']['exact_quote|same_date']['n'] == 2


def test_qwen_head_stage_uses_own_parents_and_asof_uses_own_discovery():
    m = worker()
    query = {'target_key': ['law', 'date', 'MainProvision', '2'],
        'target_passage_id': 'gold-parent', 'target_status': 'available'}
    prepared = {'addresses': [
        {'address': ['law', 'MainProvision', '2'], 'status': 'available',
         'current_parent_passage_id': 'gold-parent'}],
        'heads': {'RRF_native': [{'current_parent_passage_id': 'other'}],
                  'Full_LM18': [{'current_parent_passage_id': 'gold-parent'}]},
        'asof': [{'current_parent_passage_id': 'other'}]}
    stages = m.qwen_preselection(prepared, query)
    pred = {'conditions': {
        'RRF_native__historical_ce': {'head': 'RRF_native', 'structural': {'target_recall_at_four': False}},
        'Full_LM18__historical_ce': {'head': 'Full_LM18', 'structural': {'target_recall_at_four': True}},
        'AsOf_hybrid_native': {'head': 'AsOf', 'structural': {'target_recall_at_four': False}}}}
    values = m.method_stages(pred, stages, {'discovery_address_hit': False, 'bindable_parent_hit': False})
    assert values['RRF_native__historical_ce']['discovery_address_hit']
    assert values['RRF_native__historical_ce']['candidate_parent_at10'] == 0
    assert values['Full_LM18__historical_ce']['candidate_parent_at10'] == 1
    assert values['Full_LM18__historical_ce']['final_recall_at4'] == 1
    assert values['AsOf_hybrid_native']['discovery_bank'] == 'target_date'
    assert values['AsOf_hybrid_native']['discovery_address_hit'] == 0
