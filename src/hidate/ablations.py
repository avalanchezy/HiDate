"""Feature-group interventions for the existing single-split HiDate ranker."""

REMOVED_FEATURES = {
    'without_history': ('max_history_rrf', 'mean_history_rrf', 'distinct_revision_count',
                        'history_best_rank_rrf', 'history_best_rank_dense', 'history_best_rank_bm25'),
    'without_current_text': ('current_best_cosine', 'max_lexical_overlap_current'),
    'without_revision_gap': ('history_latest_anchor_gap_years',),
    'without_availability': ('available_target',),
}


def feature_sets(names):
    return {label: [name for name in names if name not in removed]
            for label, removed in REMOVED_FEATURES.items()}


def compact_dev_row(row):
    condition = row['conditions']['history_learned']
    fields = ('parent_passage_id', 'learned_probability', 'reranker_score', 'selection_rank', 'chunk_id')
    return {'query_id': row['query_id'], 'conditions': {'history_learned': {
        'reranked_candidates': [{key: value[key] for key in fields} for value in condition['reranked_candidates']],
        'address_diagnostics': [{key: value[key] for key in ('status', 'learned_probability')}
                                for value in condition['address_diagnostics']]}}}
