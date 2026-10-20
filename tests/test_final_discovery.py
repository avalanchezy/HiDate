"""Final discovery keeps frozen IDs and cannot recover removed text aliases."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from hidate.io import load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/discover_final.py"
DATE = "2026-09-30"
LAWQA_DATES = ["2024-09-30", "2025-11-04", DATE]


def function():
    assert SCRIPT.exists(), "Frozen final HiDate discovery is not implemented"
    return runpy.run_path(str(SCRIPT))["discover_final"]


def chunk(identity, parent, revision, dates, *, law="L", scope="MainProvision", number="3_2", role="full"):
    return {"chunk_id": identity, "parent_passage_id": parent, "revision_id": revision,
        "source_kind": "statute", "law_id": law, "scope": scope, "article_num": number,
        "slot_id": f"{law}:{scope}:{number}", "asof_dates": dates, "source_role": role,
        "family_id": law, "split": "train" if law == "TRAIN" else "test", "text": "契約 手続"}


def embeddings(path, pin, identities, aliases, vectors):
    write_json(path / "encoding.json", {"complete": True, "model_pin": pin})
    write_json(path / "query-ids.json", identities)
    np.save(path / "vector-row-indices.npy", np.asarray(aliases, dtype=np.int64))
    np.save(path / "vectors.npy", np.asarray(vectors, dtype=np.float32))


@pytest.fixture
def frozen_inputs(tmp_path):
    index, prepared, source, controls, output = [tmp_path / name for name in
        ("index", "prepared", "source", "controls", "output")]
    policy = tmp_path / "frozen.json"
    write_json(policy, {"complete": True, "alpha": .5, "training_runs": 0})
    write_json(index / "index.json", {"complete": True})
    write_json(prepared / "preparation.json", {"complete": True, "seed": 42})
    write_json(prepared / "family-split.json", {"seed": 42, "train": ["TRAIN"], "test": ["L"]})
    chunks = [chunk("current", "NOW", "r2", [DATE]),
        chunk("same_old", "ALIAS", "r1", ["2021-09-30"]),
        chunk("different_old", "OLD", "r0", ["2018-09-30"]),
        chunk("train_current", "TRAIN_PARENT", "r2", [DATE], law="TRAIN"),
        chunk("other_scope", "SUPPL", "r2", [DATE], scope="SupplProvision:1"),
        chunk("external", "EXTERNAL_PARENT", "rx", LAWQA_DATES, law="X", role="external"),
        {"chunk_id": "guide", "parent_passage_id": "GUIDE", "source_id": "GUIDE",
         "source_kind": "guidance", "source_role": "external", "text": "契約 手続"},
        chunk("future", "FUTURE", "r3", ["2027-09-30"])]
    write_jsonl(index / "chunks.jsonl", chunks)
    write_jsonl(index / "lexical-tokens.jsonl", [{"chunk_id": c["chunk_id"], "tokens": ["契約", "手続"]} for c in chunks])
    pin = {"model_id": "fixture-ruri", "revision": "frozen"}
    doc = index / "embeddings"
    write_json(doc / "encoding.json", {"complete": True, "model_pin": pin})
    write_json(doc / "chunk-ids.json", [c["chunk_id"] for c in chunks])
    np.save(doc / "vectors.npy", np.array([[.9, np.sqrt(.19)], [.2, np.sqrt(.96)], [1., 0.], [.8, .6]], dtype=np.float32))
    np.save(doc / "vector-row-indices.npy", np.array([2, 2, 0, 1, 2, 3, 2, 2], dtype=np.int64))
    key = ["L", DATE, "MainProvision", "3_2"]
    targets = [{"address_key": key, "status": "available", "passage_ids": ["NOW"], "revision_id": "r2"},
        {"address_key": ["TRAIN", DATE, "MainProvision", "3_2"], "status": "available", "passage_ids": ["TRAIN_PARENT"], "revision_id": "r2"}]
    targets.extend({"address_key": ["X", date, "MainProvision", "3_2"], "status": "available",
                    "passage_ids": ["EXTERNAL_PARENT"], "revision_id": "rx"} for date in LAWQA_DATES)
    write_jsonl(prepared / "target-addresses.jsonl", targets)
    write_jsonl(source / "asof-bindings.jsonl", [{"law_id": row["address_key"][0],
        "requested_asof": row["address_key"][1], "status": "collected", "revision_id": row["revision_id"]} for row in targets])
    exact = [{"query_id": "E1", "query_text": "契約", "query_variant": "exact_quote", "target_asof": DATE,
        "source_role": "full", "split": "test", "target_status": "available", "target_key": key,
        "law_id": "ORACLE", "source_body_text": "GOLD MUST NOT ENTER DISCOVERY"},
        {"query_id": "E2", "query_text": "手続", "query_variant": "exact_quote", "target_asof": DATE,
         "source_role": "full", "split": "test", "target_status": "absent"},
        {"query_id": "TRAIN_QUERY", "query_text": "訓練", "query_variant": "exact_quote", "target_asof": DATE,
         "source_role": "full", "split": "train", "target_status": "available"}]
    paraphrase = [{**q, "base_query_id": q["query_id"], "query_id": q["query_id"] + "|paraphrase",
                  "query_variant": "generated_paraphrase"} for q in exact]
    write_jsonl(prepared / "queries-exact.jsonl", exact)
    write_jsonl(prepared / "queries-paraphrase.jsonl", paraphrase)
    original = [{"ファイル名": f"LAWQA{i:03}", "問題文": "契約", "選択肢": "a b c d", "output": "a"} for i in range(140)]
    write_jsonl(prepared / "lawqa-external.jsonl", original)
    lawqa = [{"query_id": q["ファイル名"] + "|corpus_asof=" + date, "original_question_id": q["ファイル名"],
        "query_text": q["問題文"], "query_variant": "natural_question", "target_asof": date,
        "source_role": "external", "split": "external"} for q in original for date in LAWQA_DATES]
    write_jsonl(prepared / "lawqa-retrieval-queries.jsonl", lawqa)
    query_dirs = {name: index / ("queries-" + name) for name in ("exact", "paraphrase", "lawqa")}
    for name, rows in (("exact", exact), ("paraphrase", paraphrase), ("lawqa", lawqa)):
        embeddings(query_dirs[name], pin, [q["query_id"] for q in rows], [0] * len(rows), [[1., 0.]])
    write_json(controls / "controls.json", {"complete": True, "eligible_queries": 1,
        "counts_by_date": {DATE: {"eligible_queries": 1}}})
    write_json(controls / (DATE + ".json"), {"target_date": DATE, "removed_chunk_ids": ["current", "same_old"],
        "eligible_base_query_ids": ["E1"], "target_overrides": [{**targets[0], "status": "corpus_support_unavailable"}]})
    return {"index_dir": index, "prepared_dir": prepared, "source_dir": source, "controls_dir": controls,
        "policy_path": policy, "query_embeddings_dirs": query_dirs, "output_dir": output,
        "segmenter": lambda text: text.split()}


def test_unfrozen_policy_stops_before_any_final_artifact_loading(tmp_path):
    policy = tmp_path / "frozen.json"
    write_json(policy, {"complete": False})
    with pytest.raises(ValueError, match="frozen"):
        function()(tmp_path / "missing-index", tmp_path / "missing-prepared", tmp_path / "missing-source",
            tmp_path / "missing-controls", policy, {}, tmp_path / "output", segmenter=lambda text: text.split())
    assert not (tmp_path / "output").exists()


def test_common_control_removes_all_aliases_preserves_old_wording_and_original_maps(frozen_inputs):
    inputs = frozen_inputs
    originals = {name: (inputs["prepared_dir"] / name).read_bytes() for name in
        ("queries-exact.jsonl", "queries-paraphrase.jsonl", "target-addresses.jsonl", "family-split.json")}
    summary = function()(**inputs)
    out = inputs["output_dir"]
    natural = load_jsonl(out / "test-natural-exact/candidates.jsonl")
    controlled = load_jsonl(out / "test-controlled-exact/candidates.jsonl")
    generated = load_jsonl(out / "test-controlled-paraphrase/candidates.jsonl")
    assert {q["query_id"] for q in natural} == {"E1", "E2"}
    assert [q["query_id"] for q in controlled] == ["E1"]
    assert [q["query_id"] for q in generated] == ["E1|paraphrase"]
    assert controlled[0]["evaluation_scope"] == generated[0]["evaluation_scope"] == "controlled"
    assert controlled[0]["base_query_id"] == generated[0]["base_query_id"] == "E1"
    assert controlled[0]["history"] == generated[0]["history"]
    history = {h["chunk_id"]: h["score"] for h in controlled[0]["history"]["dense"]}
    assert set(history) == {"different_old", "other_scope", "train_current"}
    assert history["different_old"] == pytest.approx(.9)  # Original vector row 0 survives masking, not row 2.
    assert "current" in {h["chunk_id"] for h in natural[0]["asof"]["dense"]}
    address = next(row for row in controlled[0]["addresses"] if row["address"] == ["L", "MainProvision", "3_2"])
    assert address["status"] == "corpus_support_unavailable" and address["parent_passage_ids"] == []
    assert address["target_passage_ids"] == ["NOW"] and address["current_best_chunk_id"] is None
    assert address["original_target_binding"]["status"] == "available"
    assert summary["complete"] is True and summary["query_rows"] == summary["completed_query_rows"] == 426
    assert summary["unique_document_vectors"] == 4 and summary["corpus_chunk_rows"] == 8
    assert summary["training_runs"] == summary["model_forward_calls"] == 0 and summary["seed"] == 42
    assert summary["outputs"]["test-natural-paraphrase"]["query_rows"] == 2
    for name, original in originals.items():
        assert (inputs["prepared_dir"] / name).read_bytes() == original


def test_all140_questions_survive_three_external_corpus_conditions(frozen_inputs):
    function()(**frozen_inputs)
    rows = load_jsonl(frozen_inputs["output_dir"] / "lawqa/candidates.jsonl")
    assert len(rows) == 420 and len({row["query_id"] for row in rows}) == 420
    assert len({row["original_question_id"] for row in rows}) == 140
    assert {row["target_asof"] for row in rows} == set(LAWQA_DATES)
    for row in rows:
        assert row["evaluation_scope"] == "lawqa" and row["source_role"] == row["split"] == "external"
        assert row["corpus_condition_asof"] == row["target_asof"]
        assert {hit["chunk_id"] for hit in row["history"]["dense"]} == {"external", "guide"}
    manifest = json.loads((frozen_inputs["output_dir"] / "lawqa/discovery.json").read_text())
    assert manifest["query_rows"] == manifest["completed_query_rows"] == 420
    assert manifest["retrieval_runs"] == 3 and manifest["cache_hits"] == 417


def test_missing_vector_id_is_rejected_without_silently_dropping_a_test_query(frozen_inputs):
    path = frozen_inputs["query_embeddings_dirs"]["paraphrase"] / "query-ids.json"
    write_json(path, ["WRONG", "E2|paraphrase", "TRAIN_QUERY|paraphrase"])
    with pytest.raises(ValueError, match="query.*vector|vector.*query"):
        function()(**frozen_inputs)
    assert not frozen_inputs["output_dir"].exists()


def test_control_base_id_coverage_must_match_all_available_test_queries(frozen_inputs):
    path = frozen_inputs["controls_dir"] / (DATE + ".json")
    audit = json.loads(path.read_text())
    audit["eligible_base_query_ids"] = []
    write_json(path, audit)
    with pytest.raises(ValueError, match="control.*coverage|coverage.*control"):
        function()(**frozen_inputs)
    assert not frozen_inputs["output_dir"].exists()
