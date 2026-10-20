from hidate.ablations import feature_sets, compact_dev_row


def test_feature_groups_intervene_only_declared_information():
    names = ['max_history_rrf', 'available_target', 'current_best_cosine', 'history_latest_anchor_gap_years']
    groups = feature_sets(names)
    assert groups['without_current_text'] == ['max_history_rrf', 'available_target', 'history_latest_anchor_gap_years']
    assert groups['without_revision_gap'] == ['max_history_rrf', 'available_target', 'current_best_cosine']
    assert names == ['max_history_rrf', 'available_target', 'current_best_cosine', 'history_latest_anchor_gap_years']


def test_compact_development_preserves_ranking_values_without_document_repetition():
    candidate = dict(parent_passage_id='p', learned_probability=.7, reranker_score=.9,
                     selection_rank=1, chunk_id='c', text='large text')
    row = {'query_id': 'v', 'conditions': {'history_learned': {'reranked_candidates': [candidate],
                'address_diagnostics': [{'status': 'available', 'learned_probability': .7, 'features': {}}]}}}
    compact = compact_dev_row(row)['conditions']['history_learned']
    assert 'text' not in compact['reranked_candidates'][0]
    assert compact['reranked_candidates'][0]['learned_probability'] == .7
