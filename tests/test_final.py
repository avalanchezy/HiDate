"""Small saved-input fixtures for frozen final retrieval and forced-choice QA."""

import copy
import json
import math
import runpy
from pathlib import Path

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.evidence import external_queries
from hidate.reranking import ALL_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]


def driver():
    path = ROOT / "scripts/evaluate_final.py"
    assert path.exists(), "The frozen final evaluator has not been implemented"
    return runpy.run_path(str(path))["evaluate_final"]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(row["content"]) for row in messages) + 3))


class Reranker:
    def __init__(self, events):
        self.events, self.pairs = events, []

    def score(self, pairs, include_inputs):
        assert include_inputs
        self.events.append("reranker_forward")
        self.pairs.extend(pairs)
        scores = [.55 if document == "第一条\n現在の正しい本文。" else .85 for _, document in pairs]
        return {"scores": scores, "logits": [math.log(s / (1 - s)) for s in scores],
            "token_counts": [3] * len(pairs), "total_tokens": len(pairs) * 3,
            "input_shape": [len(pairs), 4], "input_ids": [[10, 11, 12, 0]] * len(pairs),
            "attention_mask": [[1, 1, 1, 0]] * len(pairs), "seconds": .1,
            "sdpa_backend": "default"}


class Reader:
    tokenizer = Tokenizer()

    def __init__(self, events):
        self.events, self.messages = events, []

    def score_choices(self, messages, include_inputs):
        assert include_inputs
        self.events.append("reader_forward")
        self.messages.append(messages)
        n = len(self.tokenizer.apply_chat_template(messages))
        return {"prediction": "c", "choice_logprobs": {"a": -4., "b": -3., "c": -.1, "d": -2.},
            "choice_token_ids": {"a": 1, "b": 2, "c": 3, "d": 4}, "choice_position": n - 1,
            "prompt_tokens": n, "token_counts": [n], "total_tokens": n, "input_shape": [1, n],
            "input_ids": [list(range(n))], "attention_mask": [[1] * n], "seconds": .2,
            "output_shape": [1, n, 5]}


def fixture(tmp_path):
    paths = {name: tmp_path / name for name in ("exact.jsonl", "paraphrase.jsonl", "lawqa.jsonl",
        "lawqa-queries.jsonl", "chunks.jsonl", "stack.json", "policy.json", "ranker.json")}
    chunks = [
        {"chunk_id": "a", "parent_passage_id": "SECRET_CURRENT_A", "text": "第一条\n現在の正しい本文。",
            "source_kind": "statute", "law_id": "LAW", "revision_id": "SECRET_REV", "scope": "MainProvision",
            "article_num": "1", "source_role": "full", "asof_dates": ["2026-09-30"]},
        {"chunk_id": "b", "parent_passage_id": "OTHER_B", "text": "第二条\n別の本文。",
            "source_kind": "statute", "law_id": "LAW", "revision_id": "REV_B", "scope": "MainProvision",
            "article_num": "2", "source_role": "full", "asof_dates": ["2026-09-30"]}]
    exact = [{"query_id": "available", "query_text": "原文と同じ検索内容。", "query_variant": "exact_quote",
        "split": "test", "source_role": "full", "target_asof": "2026-09-30", "stratum": "changed_cross_date",
        "target_key": ["LAW", "2026-09-30", "MainProvision", "1"], "target_status": "available",
        "target_revision_id": "SECRET_REV", "target_passage_id": "SECRET_CURRENT_A",
        "labels": {"passage_id": "SECRET_CURRENT_A"}},
        {"query_id": "unavailable", "query_text": "自然に廃止された条文の検索。", "query_variant": "exact_quote",
        "split": "test", "source_role": "full", "target_asof": "2026-09-30", "stratum": "unavailable_target",
        "target_key": ["LAW", "2026-09-30", "MainProvision", "99"], "target_status": "deleted",
        "target_revision_id": "SECRET_REV", "target_passage_id": None, "labels": {"passage_id": None}}]
    paraphrase = [{**q, "base_query_id": q["query_id"], "query_id": q["query_id"] + "|paraphrase",
        "query_variant": "generated_paraphrase"} for q in exact]
    qa = {"ファイル名": "external-one", "問題文": "原問題全文。", "選択肢": "a 選択一\nb 選択二\nc 選択三\nd 選択四",
        "コンテキスト": "公開された元の文脈。", "output": "c", "references": ["official-original-url"]}
    lawqa_queries = external_queries([qa], ["2025-11-04", "2026-09-30"])
    for key, rows in (("exact.jsonl", exact), ("paraphrase.jsonl", paraphrase), ("lawqa.jsonl", [qa]),
                      ("lawqa-queries.jsonl", lawqa_queries), ("chunks.jsonl", chunks)):
        write_jsonl(paths[key], rows)
    write_json(paths["stack.json"], {"complete": True, "models": {role: {
        "model_id": role, "revision": "FROZEN_PIN", "download_complete": True, "local_path": "fixture"}
        for role in ("reranker", "reader")}})
    write_json(paths["ranker.json"], {"complete": True, "feature_names": ["current_best_cosine"],
        "mean": [0.], "scale": [1.], "coefficient": [1.], "intercept": 0.})
    write_json(paths["policy.json"], {"complete": True, "alpha": 1., "ranker_path": str(paths["ranker.json"]),
        "method_thresholds": {method: {"threshold": .9, "accepted": 0, "coverage": 0., "precision": None}
                              for method in ALL_CONDITIONS}})

    def discovery(query, controlled=False):
        hits = [{"chunk_id": key, "score": .9 - i / 10} for i, key in enumerate(["b"] if controlled else ["a", "b"])]

        def address(key, number, cosine, status="available"):
            return {"address": ["LAW", "MainProvision", number], "status": status,
                "current_best_chunk_id": key if status == "available" else None,
                "target_passage_ids": [chunks[0 if key == "a" else 1]["parent_passage_id"]] if status == "available" else [],
                "parent_passage_ids": [chunks[0 if key == "a" else 1]["parent_passage_id"]] if status == "available" else [],
                "features": {"max_history_rrf": .8, "current_best_cosine": cosine,
                             "max_lexical_overlap_current": .5, "available_target": float(status == "available")}}

        addresses = [address("a", "1", -2. if controlled else 2., "corpus_support_unavailable" if controlled else "available"),
                     address("b", "2", 1.), address(None, "99", -3., "deleted")]
        if controlled:
            addresses[0]["original_target_binding"] = {"status": "available", "revision_id": "SECRET_REV",
                                                        "parent_passage_ids": ["SECRET_CURRENT_A"]}
        return {**{key: query[key] for key in ("query_id", "query_variant", "split", "target_asof", "source_role")},
            "evaluation_scope": "controlled" if controlled else "lawqa" if query["split"] == "external" else "natural",
            "asof": {m: copy.deepcopy(hits) for m in ("bm25", "dense", "hybrid")},
            "history": {m: copy.deepcopy(hits) for m in ("bm25", "dense", "hybrid")}, "addresses": addresses}

    discovery_root = tmp_path / "discovery"
    for name, queries, controlled in (("test-natural-exact", exact, False), ("test-natural-paraphrase", paraphrase, False),
        ("test-controlled-exact", exact[:1], True), ("test-controlled-paraphrase", paraphrase[:1], True),
        ("lawqa", lawqa_queries, False)):
        directory = discovery_root / name
        write_jsonl(directory / "candidates.jsonl", [discovery(q, controlled) for q in queries])
        write_json(directory / "discovery.json", {"complete": True, "query_rows": len(queries), "completed_query_rows": len(queries)})
    return paths, discovery_root, qa


def invoke(tmp_path, paths, discovery_root, events):
    models = {}

    def make_reranker(pin):
        events.append("load_reranker")
        models["reranker"] = Reranker(events)
        return models["reranker"]

    def make_reader(pin):
        events.append("load_reader")
        models["reader"] = Reader(events)
        return models["reader"]

    result = driver()([paths["exact.jsonl"], paths["paraphrase.jsonl"]], paths["lawqa-queries.jsonl"],
        paths["lawqa.jsonl"], discovery_root, paths["chunks.jsonl"], paths["stack.json"],
        paths["policy.json"], tmp_path / "output", reranker_factory=make_reranker,
        reader_factory=make_reader, release_reranker=lambda model: events.append("release_reranker"))
    return result, models


def test_complete_final_scopes_frozen_alpha_and_forced_choice_even_when_withheld(tmp_path):
    paths, root, qa = fixture(tmp_path)
    events = []
    result, models = invoke(tmp_path, paths, root, events)
    assert result["complete"] and result["training_runs"] == 0
    assert result["retrieval_rows_by_scope"] == {"natural": 4, "controlled": 2, "lawqa": 2}
    rows = load_jsonl(tmp_path / "output/retrieval.jsonl")
    natural = next(r for r in rows if r["query_id"] == "available" and r["evaluation_scope"] == "natural")
    control = next(r for r in rows if r["query_id"] == "available" and r["evaluation_scope"] == "controlled")
    baseline, learned = natural["conditions"]["asof_hybrid"], natural["conditions"]["history_learned"]
    assert baseline["alpha"] == 0. and baseline["evidence"][0]["parent_passage_id"] == "OTHER_B"
    assert baseline["evidence"][0]["address"] == ["LAW", "MainProvision", "2"]
    assert baseline["support_confidence"] == .85
    assert learned["alpha"] == 1. and learned["evidence"][0]["parent_passage_id"] == "SECRET_CURRENT_A"
    assert learned["support_confidence"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert learned["structural"]["top_parent_correct"] is True
    assert control["target_status"] == "corpus_support_unavailable"
    assert control["original_gold"]["target_passage_id"] == "SECRET_CURRENT_A"
    assert not any(c["structural"]["top_parent_correct"] for c in control["conditions"].values())
    unavailable = next(r for r in rows if r["query_id"] == "unavailable" and r["evaluation_scope"] == "natural")
    assert unavailable["target_status"] == "deleted"
    assert "prediction" not in natural and "correct_answer" not in natural
    assert events.index("release_reranker") < events.index("load_reader")
    assert max(i for i, e in enumerate(events) if e == "reranker_forward") < events.index("release_reranker")
    answers = load_jsonl(tmp_path / "output/answers.jsonl")
    assert len(answers) == 2 * 7 + 2 == len(models["reader"].messages)
    assert sum(r["method"] == "closed_book" for r in answers) == 1
    assert sum(r["method"] == "released_oracle" for r in answers) == 1
    retrieval_answers = [r for r in answers if r["evaluation_scope"] == "lawqa"]
    assert len(retrieval_answers) == 14 and all(not r["accepted"] for r in retrieval_answers)
    assert all(r["prediction"] == "c" and r["correct"] for r in answers)
    assert all(qa["問題文"] in m[-1]["content"] and qa["選択肢"] in m[-1]["content"] for m in models["reader"].messages)
    closed = next(r for r in answers if r["method"] == "closed_book")
    oracle = next(r for r in answers if r["method"] == "released_oracle")
    assert closed["evidence"] == [] and oracle["evidence"][0]["text"] == qa["コンテキスト"]
    assert "target_asof" not in closed and "target_asof" not in oracle
    assert all("SECRET" not in q and "SECRET" not in doc for q, doc in models["reranker"].pairs)
    assert result["physical_pair_count"] == 6  # three distinct query texts, two documents each
    assert result["logical_pair_requests"] > result["physical_pair_count"]
    assert result["condition_requested_input_tokens"]["asof_hybrid"] == 42
    assert len(load_jsonl(tmp_path / "output/model-reader-receipts.jsonl")) == 16
    pair_receipt = load_jsonl(tmp_path / "output/model-pair-receipts.jsonl")[0]
    assert pair_receipt["result"]["attention_mask"][0] == [1, 1, 1, 0]
    assert "devices" not in result  # injected fixture does not invent GPU metadata


@pytest.mark.parametrize("broken", ["policy", "discovery", "missing_row"])
def test_frozen_gate_and_complete_inputs_precede_any_model_load(tmp_path, broken):
    paths, root, _ = fixture(tmp_path)
    if broken == "policy":
        policy = json.loads(paths["policy.json"].read_text())
        policy["complete"] = False
        write_json(paths["policy.json"], policy)
    elif broken == "discovery":
        write_json(root / "lawqa/discovery.json", {"complete": False})
    else:
        write_jsonl(root / "test-natural-exact/candidates.jsonl", [])
    events = []
    with pytest.raises(ValueError, match="complete|exactly one|frozen"):
        invoke(tmp_path, paths, root, events)
    assert events == []


def test_invalid_reader_forward_stops_without_inventing_or_skipping_an_answer(tmp_path, monkeypatch):
    paths, root, _ = fixture(tmp_path)
    original = Reader.score_choices

    def invalid(self, messages, include_inputs):
        result = original(self, messages, include_inputs)
        result["choice_logprobs"]["c"] = float("nan")
        return result

    monkeypatch.setattr(Reader, "score_choices", invalid)
    events = []
    with pytest.raises(ValueError, match="invalid forced-choice"):
        invoke(tmp_path, paths, root, events)
    saved = json.loads((tmp_path / "output/evaluation.json").read_text(encoding="utf-8"))
    assert saved["complete"] is False and saved["phase"] == "reader"
    assert saved["retrieval_rows_by_scope"] == {"natural": 4, "controlled": 2, "lawqa": 2}
    assert events.count("reader_forward") == 1 and load_jsonl(tmp_path / "output/answers.jsonl") == []
