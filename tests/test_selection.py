"""DEV mixture selection uses dated evidence and recomputes its ordering."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("hidate.selection")
    except ModuleNotFoundError:
        pytest.fail("HiDate DEV selector is not implemented")


def candidate(identity, reranker, learned):
    return {"chunk_id": identity, "parent_passage_id": identity, "selection_rank": 1,
            "reranker_score": reranker, "learned_probability": learned}


def test_mix_reorders_all_saved_candidates_and_requires_the_target_revision_parent():
    row = {"reranked_candidates": [candidate("old_revision", .9, .1), candidate("current_revision", .1, .9)],
           "address_diagnostics": []}
    zero = module().selected_evidence(row, alpha=0.)
    one = module().selected_evidence(row, alpha=1.)
    assert zero[0]["parent_passage_id"] == "old_revision"
    assert one[0]["parent_passage_id"] == "current_revision"
    q = {"target_status": "available", "target_passage_id": "current_revision"}
    assert module().target_mrr(q, zero) == .5 and module().target_mrr(q, one) == 1.


def test_precision_threshold_uses_positive_confidence_groups_and_maximum_coverage():
    result = module().precision_threshold([(.9, 1), (.8, 1), (.7, 0), (0., 0)], target_precision=.9)
    assert result["threshold"] == .8 and result["accepted"] == 2
    assert result["coverage"] == .5 and result["precision"] == 1.


def test_support_score_rejects_an_unavailable_higher_ranked_address():
    evidence = [candidate("present", .9, .5)]
    diagnostics = [{"status": "absent", "learned_probability": .9}]
    assert module().support_score(evidence, diagnostics) == 0.
    assert module().support_score(evidence, []) == .5


def test_baseline_support_uses_its_own_scores_and_only_bound_method_checks_missing_address():
    evidence = [candidate("present", .9, .05)]
    diagnostics = [{"status": "absent", "learned_probability": .99}]
    assert module().baseline_support_score("asof_hybrid", evidence, diagnostics) == .9
    assert module().baseline_support_score("history_unbound", evidence, diagnostics) == .9
    assert module().baseline_support_score("history_bound", evidence, diagnostics) == 0.


def test_dev_policy_balances_strata_and_recomputes_final_four_passages():
    queries, rows = {}, []
    for i in range(4):
        identity = str(i)
        queries[identity] = {"split": "valid", "target_status": "available", "target_passage_id": "gold",
                             "query_variant": "exact_quote", "stratum": "same_date" if i < 3 else "changed_cross_date"}
        candidates = ([candidate("gold", .9, .1), candidate("wrong", .1, .9)] if i < 3 else
                      [candidate(str(j), .9 - j * .1, .1) for j in range(4)] + [candidate("gold", 0., 1.)])
        rows.append({"query_id": identity, "conditions": {"history_learned": {
            "reranked_candidates": candidates, "address_diagnostics": []}}})
    policy = module().choose_policy(rows, queries, [0., 1.])
    assert policy["alpha"] == 1.
    assert [row["macro_mrr"] for row in policy["alpha_grid_results"]] == [.5, .75]
