"""Small CPU fixtures for common budgets, binding diagnostics and exact pair reuse."""

import copy
import importlib
import json
import math
import runpy
from pathlib import Path

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "src/hidate/reranking.py"
    assert path.exists(), "HiDate reranking primitives are not implemented"
    return vars(importlib.import_module("hidate.reranking"))


def fixture():
    chunks = {}
    for number in range(1, 13):
        chunk_id = f"c{number}"
        chunks[chunk_id] = {"chunk_id": chunk_id, "parent_passage_id": f"P{number}",
            "text": f"現在の条文本文{number}。", "source_kind": "statute", "law_id": "LAW",
            "scope": "MainProvision", "article_num": str(number), "revision_id": "CURRENT",
            "source_role": "full", "asof_dates": ["2026-09-30"]}
    chunks["c1b"] = {**chunks["c1"], "chunk_id": "c1b", "text": "現在の第一条の別の断片。"}
    chunks["h1"] = {**chunks["c1"], "chunk_id": "h1", "parent_passage_id": "OLD1", "text": "昔の第一条本文。", "revision_id": "OLD"}
    chunks["h1b"] = {**chunks["h1"], "chunk_id": "h1b", "text": "昔の第一条の別の断片。"}
    chunks["h2"] = {**chunks["c2"], "chunk_id": "h2", "parent_passage_id": "OLD2", "text": "昔の第二条本文。", "revision_id": "OLD"}
    asof = [{"chunk_id": chunk_id, "score": 1 / (i + 1)} for i, chunk_id in enumerate(["c1", "c1b", *[f"c{i}" for i in range(2, 13)]])]
    history = [{"chunk_id": "h1", "score": .9}, {"chunk_id": "h1b", "score": .8}, {"chunk_id": "h2", "score": .7}]

    def address(number, rrf, cosine, status="available"):
        return {"address": ["LAW", "MainProvision", str(number)], "status": status,
            "features": {"max_history_rrf": rrf, "current_best_cosine": cosine,
                         "max_lexical_overlap_current": .5, "available_target": float(status == "available")},
            "current_best_chunk_id": f"c{number}" if status == "available" else None,
            "target_passage_ids": [f"P{number}"] if status == "available" else [],
            "parent_passage_ids": [f"P{number}"] if status == "available" else [],
            "historical_best_chunk_ids": {"OLD": f"h{number}"},
            "history_parent_passage_ids": [f"OLD{number}"], "history_chunk_hits": []}

    discovery = {"query_id": "QUERY", "query_variant": "exact_quote", "split": "valid",
        "target_asof": "2026-09-30", "source_role": "full",
        "asof": {method: copy.deepcopy(asof) for method in ("bm25", "dense", "hybrid")},
        "history": {method: copy.deepcopy(history) for method in ("bm25", "dense", "hybrid")},
        "addresses": [address(99, .95, -1., "absent"), address(1, .9, .1), address(2, .7, .8)]}
    ranker = {"feature_names": ["current_best_cosine"], "mean": [0.], "scale": [1.], "coefficient": [1.], "intercept": 0.}
    query = {"query_id": "QUERY", "query_text": "検索する内容。", "query_variant": "exact_quote", "split": "valid",
        "target_asof": "2026-09-30", "source_role": "full", "labels": {"passage_id": "SECRET_GOLD_PARENT"}, "law_id": "SECRET_GOLD_LAW"}
    return chunks, discovery, ranker, query


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, pairs, include_inputs):
        assert include_inputs is True
        self.calls.append(pairs)
        scores = [.8 if document == "現在の条文本文2。" else .4 for query, document in pairs]
        return {"scores": scores, "logits": [math.log(score / (1 - score)) for score in scores],
            "token_counts": [3] * len(pairs), "total_tokens": 3 * len(pairs), "input_shape": [len(pairs), 4],
            "input_ids": [[10, 11, 12, 0]] * len(pairs), "attention_mask": [[1, 1, 1, 0]] * len(pairs),
            "seconds": .25, "sdpa_backend": "math"}


def test_candidates_share_budget_deduplicate_parents_and_preserve_missing_diagnostics():
    chunks, discovery, ranker, _ = fixture()
    original = copy.deepcopy(discovery)
    candidates = module()["select_candidates"](discovery, chunks, ranker=ranker)
    for method in ("asof_bm25", "asof_dense", "asof_hybrid"):
        assert [row["chunk_id"] for row in candidates[method]["candidates"]] == ["c1", *[f"c{i}" for i in range(2, 11)]]
        assert len(candidates[method]["candidates"]) == 10
    assert [row["chunk_id"] for row in candidates["history_unbound"]["candidates"]] == ["h1", "h2"]
    assert [row["chunk_id"] for row in candidates["history_bound"]["candidates"]] == ["c1", "c2"]
    assert [row["chunk_id"] for row in candidates["history_current"]["candidates"]] == ["c2", "c1"]
    assert [row["chunk_id"] for row in candidates["history_learned"]["candidates"]] == ["c2", "c1"]
    missing = candidates["history_bound"]["address_diagnostics"][0]
    assert missing["status"] == "absent" and missing["address"][2] == "99"
    assert missing["current_best_chunk_id"] is None
    assert candidates["history_learned"]["candidates"][0]["learned_probability"] == pytest.approx(1 / (1 + math.exp(-.8)))
    assert discovery == original
    with pytest.raises(ValueError, match="ranker"):
        module()["select_candidates"](discovery, chunks, conditions=["history_learned"])


def test_pair_cache_saves_actual_scores_work_and_keeps_distinct_condition_requests():
    chunks, discovery, ranker, query = fixture()
    functions = module()
    model, receipts = Reranker(), []
    cache = functions["PairScoreCache"](model, receipt_callback=receipts.append)
    result = functions["rerank_query"](query, discovery, chunks, cache, ranker=ranker)
    assert cache.physical_pair_count == 12 and cache.logical_pair_requests == 38
    assert cache.encoded_input_tokens == 36 and cache.padded_token_slots == 48
    assert result["conditions"]["asof_hybrid"].get("requested_input_tokens") == 30
    assert result["conditions"]["history_learned"].get("requested_input_tokens") == 6
    assert len(receipts) == 1 and receipts[0]["result"]["input_ids"][0] == [10, 11, 12, 0]
    assert all(pair[0] == "検索する内容。" and "SECRET" not in pair[1] for pair in model.calls[0])
    hybrid = result["conditions"]["asof_hybrid"]
    assert hybrid["requested_pair_count"] == 10
    assert hybrid["final_passage_ids"] == ["P2", "P1", "P3", "P4"]
    assert hybrid["reranked_candidates"][0]["reranker_score"] == .8
    assert hybrid["reranked_candidates"][0]["reranker_logit"] == pytest.approx(math.log(4))
    assert hybrid["reranked_candidates"][0]["pair_token_count"] == 3
    learned = result["conditions"]["history_learned"]["reranked_candidates"]
    assert all("features" in row and "learned_probability" in row for row in learned)
    alias = {**query, "query_id": "ALIAS", "query_variant": "generated_paraphrase"}
    repeated = functions["rerank_query"](alias, discovery, chunks, cache, ranker=ranker)
    assert cache.physical_pair_count == 12 and cache.logical_pair_requests == 76
    assert cache.encoded_input_tokens == 36
    assert (hybrid["requested_input_tokens"] + repeated["conditions"]["asof_hybrid"]["requested_input_tokens"]) == 60
    assert len(model.calls) == 1
    functions["rerank_query"]({**query, "query_text": "別の検索依頼。"}, discovery, chunks, cache, ranker=ranker)
    assert cache.physical_pair_count == 24 and len(model.calls) == 2


def test_development_driver_runs_all_declared_valid_rows_without_using_gold(tmp_path):
    script = ROOT / "scripts/rerank_development.py"
    assert script.exists(), "HiDate development reranking driver is not implemented"
    chunks, discovery, _, query = fixture()
    paths = {name: tmp_path / name for name in ("queries.jsonl", "discovery.jsonl", "chunks.jsonl", "stack.json")}
    write_jsonl(paths["queries.jsonl"], [query])
    write_jsonl(paths["discovery.jsonl"], [discovery])
    write_jsonl(paths["chunks.jsonl"], chunks.values())
    write_json(paths["stack.json"], {"complete": True, "models": {"reranker": {"model_id": "cl-nagoya/ruri-v3-reranker-310m",
        "revision": "FIXED_PIN", "local_path": "fixture", "download_complete": True}}})
    model = Reranker()
    output = tmp_path / "output"
    result = runpy.run_path(str(script))["rerank_development"]([paths["queries.jsonl"]], [paths["discovery.jsonl"]],
        paths["chunks.jsonl"], paths["stack.json"], output, reranker=model)
    assert result["complete"] is True and result["query_rows"] == 1
    assert result["logical_pair_requests"] == 36 and result["physical_pair_count"] == 12
    assert result.get("condition_requested_input_tokens") == {
        "asof_bm25": 30, "asof_dense": 30, "asof_hybrid": 30,
        "history_unbound": 6, "history_bound": 6, "history_current": 6}
    assert "device" not in result
    saved = load_jsonl(output / "reranked.jsonl")
    assert len(saved) == 1 and "history_learned" not in saved[0]["conditions"]
    assert len(load_jsonl(output / "model-pair-receipts.jsonl")) == 1
    with pytest.raises(ValueError, match="valid"):
        runpy.run_path(str(script))["rerank_development"]([paths["queries.jsonl"]], [paths["discovery.jsonl"]],
            paths["chunks.jsonl"], paths["stack.json"], output, reranker=model, splits=["test"])
