import numpy as np

from hidate import scale


def law(identity):
    return {"law_info": {"law_id": identity}, "family_id": identity}


def test_scale_prefix_keeps_original_order_and_four_anchors_exclude_expert_dates():
    original = [law("b"), law("a")]
    selected = original + [law("c")]
    banks = scale.nested_law_banks(selected, original, sizes=(2, 3))
    assert banks == {"2": ["b", "a"], "3": ["b", "a", "c"]}
    plan = scale.snapshot_plan(selected, original, ["2018-09-30", "2026-09-30"])
    assert plan == [("c", "2018-09-30"), ("c", "2026-09-30")]


def test_same_revision_passage_accumulates_anchors_without_duplicating_body():
    corpus = {}
    base = {"passage_id": "p", "visible_text": "本文", "requested_asof": "2018-09-30", "raw_path": "early.json"}
    scale.add_passage(corpus, base)
    scale.add_passage(corpus, {**base, "requested_asof": "2021-09-30", "raw_path": "later.json"})
    assert len(corpus) == 1
    assert corpus["p"]["asof_dates"] == ["2018-09-30", "2021-09-30"]
    assert "requested_asof" not in corpus["p"]


def test_shared_cosines_reuse_one_actual_product_across_three_nested_banks():
    matrix = scale.SharedCosines(np.array([[1., 0.], [0., 1.]]))
    for _ in range(3):
        assert (matrix @ np.array([1., 0.])).tolist() == [1., 0.]
    assert matrix.products == 1
    assert (matrix @ np.array([0., 1.])).tolist() == [0., 1.]
    assert matrix.products == 2


def test_quote_panel_does_not_count_unavailable_as_quote_present():
    rows = [{"query_variant": "exact_quote", "stratum": "unavailable_target", "quote_absent": None,
             "top_parent_correct": False, "top_revision_correct": False, "target_mrr": 0.,
             "target_recall_at_four": False, "accepted": False}]
    panel = scale.metric_panels(rows)
    assert "exact_quote|quote_present" not in panel
    assert panel["exact_quote|all"]["n"] == 1


def test_existing_discovery_engine_can_share_global_aliases_across_nested_banks():
    from hidate.discovery import DiscoveryEngine

    def chunk(identity, law_id):
        return {"chunk_id": identity, "parent_passage_id": identity + "-parent", "law_id": law_id,
                "article_num": "1", "scope": "MainProvision", "revision_id": "r1", "source_role": "full",
                "source_kind": "statute", "asof_dates": ["2020-01-01"]}

    chunks = [chunk("c1", "law1"), chunk("c2", "law2")]
    shared = scale.SharedCosines(np.array([[1., 0.], [0., 1.]]))
    targets = {("law1", "2020-01-01", "MainProvision", "1"):
               {"status": "available", "passage_ids": ["c1-parent"]}}
    bindings = {("law1", "2020-01-01"): {"status": "collected", "revision_id": "r1"}}
    tokens = {"c1": ["target"], "c2": ["other"]}
    query = {"query_text": "target", "target_asof": "2020-01-01", "source_role": "full",
             "query_id": "q", "query_variant": "exact_quote", "split": "test"}
    for bank in (chunks[:1], chunks):
        engine = DiscoveryEngine(bank, tokens, shared, [0, 1], ["c1", "c2"], targets, bindings)
        result = engine.discover(query, ["target"], np.array([1., 0.]))
        assert result["asof"]["dense"][0]["chunk_id"] == "c1"
        address = next(a for a in result["addresses"] if a["address"][0] == "law1")
        assert address["parent_passage_ids"] == ["c1-parent"]
    assert shared.products == 1
