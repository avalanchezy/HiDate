import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate import discovery as discovery


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/discover_candidates.py"
DATE = "2024-09-30"


class UniqueVectorBank:
    def __init__(self, vectors):
        self.values = np.asarray(vectors, dtype=np.float32)
        self.shape = self.values.shape
        self.dot_calls = 0

    def __matmul__(self, query):
        self.dot_calls += 1
        return self.values @ query

    def __getitem__(self, key):
        pytest.fail("Discovery must not expand aliased chunk vectors")


def chunk(chunk_id, parent, revision, law="A", num="1", **extra):
    return {"chunk_id": chunk_id, "parent_passage_id": parent, "revision_id": revision,
        "source_kind": "statute", "law_id": law, "scope": "MainProvision", "article_num": num,
        "slot_id": f"{law}:MainProvision:{num}", "text": "契約 手続", "body_text": "契約 手続",
        "asof_dates": ["2018-09-30"], "source_role": "full", "split": "train", "family_id": law, **extra}


def query(query_id="Q1", **extra):
    return {"query_id": query_id, "query_text": "契約 手続", "query_variant": "exact_quote",
            "target_asof": DATE, "source_role": "full", "split": "train", "law_id": "A",
            "scope": "MainProvision", "article_num": "1", "labels": {"passage_id": "CUR_A"}, **extra}


@pytest.fixture
def inputs():
    chunks = [chunk("OLD_A0", "OLD_A", "RA0", asof_dates=["2018-09-30", "2021-09-30"]),
        chunk("OLD_A1", "OLD_A", "RA0", asof_dates=["2018-09-30", "2021-09-30"]),
        chunk("CUR_A0", "CUR_A", "RA1", asof_dates=[DATE], text="手続 新規"),
        chunk("OTHER_B0", "CUR_B", "RB1", law="B", asof_dates=[DATE], split="test", text="登記 契約"),
        chunk("DELETED_A0", "OLD_DEL", "RA0", num="3", text="契約 解除"),
        chunk("SUPPL_A0", "SUPPL_A", "RA0", scope="SupplProvision:original", slot_id="A:SupplProvision:original:1"),
        chunk("FUTURE_A0", "FUT_A", "RA2", asof_dates=["2026-09-30"]),
        chunk("EXTERNAL_X0", "CUR_X", "RX1", law="X", asof_dates=[DATE], source_role="external", split="external"),
        {"chunk_id": "GUIDE0", "parent_passage_id": "guidance-01", "source_id": "guidance-01",
         "source_kind": "guidance", "source_role": "external", "split": "external", "text": "契約 手続",
         "url": "https://agency.example/guide.pdf"}]
    tokens = {c["chunk_id"]: c["text"].split() for c in chunks}
    vectors = [[1., 0.], [.8, .6], [.9, np.sqrt(.19)], [.4, np.sqrt(.84)], [.7, np.sqrt(.51)]]
    aliases = np.array([0, 0, 1, 2, 3, 0, 0, 4, 0], dtype=np.int64)
    targets = {("A", DATE, "MainProvision", "1"): {"status": "available", "passage_ids": ["CUR_A"], "revision_id": "RA1"},
               ("A", DATE, "MainProvision", "3"): {"status": "deleted", "passage_ids": ["DEL_A"], "revision_id": "RA1"},
               ("B", DATE, "MainProvision", "1"): {"status": "available", "passage_ids": ["CUR_B"], "revision_id": "RB1"},
               ("X", DATE, "MainProvision", "1"): {"status": "available", "passage_ids": ["CUR_X"], "revision_id": "RX1"}}
    bindings = {(law, DATE): {"status": "collected", "revision_id": revision}
                for law, revision in [("A", "RA1"), ("B", "RB1"), ("X", "RX1")]}
    return chunks, tokens, vectors, aliases, targets, bindings


def engine(inputs, vectors=None):
    chunks, tokens, values, aliases, targets, bindings = inputs
    return discovery.DiscoveryEngine(chunks, tokens, vectors if vectors is not None else np.asarray(values),
        aliases, [c["chunk_id"] for c in chunks], targets, bindings)


def test_banks_keep_all_full_families_filter_future_and_score_unique_vectors_once(inputs):
    bank = UniqueVectorBank(inputs[2])
    result = engine(inputs, bank).discover(query(), ["契約", "手続"], np.array([1., 0.]))
    assert bank.dot_calls == 1
    asof = {h["chunk_id"] for h in result["asof"]["dense"]}
    history = {h["chunk_id"] for h in result["history"]["dense"]}
    assert asof == {"CUR_A0", "OTHER_B0"}  # B's TEST family remains in a TRAIN query bank.
    assert {"OLD_A0", "OLD_A1", "SUPPL_A0", "DELETED_A0", "CUR_A0", "OTHER_B0"} == history
    assert "FUTURE_A0" not in history and "GUIDE0" not in history and "EXTERNAL_X0" not in history
    assert [h["chunk_id"] for h in result["asof"]["dense"]] == ["OTHER_B0", "CUR_A0"]
    assert all(len(result[mode][method]) <= 50 for mode in ("asof", "history") for method in ("bm25", "dense", "hybrid"))


def test_features_use_revision_votes_current_binding_and_unavailable_defaults(inputs):
    result = engine(inputs).discover(query(), ["契約", "手続"], np.array([1., 0.]))
    rows = {tuple(r["address"]): r for r in result["addresses"]}
    main = rows[("A", "MainProvision", "1")]
    assert main["status"] == "available" and main["parent_passage_ids"] == ["CUR_A"]
    assert main["current_best_chunk_id"] == "CUR_A0"
    assert main["current_feature_chunk_count"] == 1 and result["current_feature_chunks"] == 2
    assert main["features"]["current_best_cosine"] == pytest.approx(.8)
    assert main["features"]["max_lexical_overlap_current"] == .5
    assert main["features"]["distinct_revision_count"] == 2.
    assert main["features"]["available_target"] == 1.
    assert main["features"]["history_latest_anchor_gap_years"] == 0.
    assert set(main["historical_best_chunk_ids"].values()) <= {"OLD_A0", "OLD_A1", "CUR_A0"}
    deleted = rows[("A", "MainProvision", "3")]
    assert deleted["status"] == "deleted" and deleted["target_passage_ids"] == ["DEL_A"]
    assert deleted["features"]["current_best_cosine"] == -1.
    assert deleted["features"]["max_lexical_overlap_current"] == 0.
    assert deleted["features"]["available_target"] == 0.
    assert deleted["features"]["history_latest_anchor_gap_years"] == pytest.approx(2192 / 365.25)
    assert rows[("A", "SupplProvision:original", "1")]["status"] == "absent"
    assert len(discovery.FEATURE_NAMES) == 10
    assert tuple(sorted(discovery.FEATURE_NAMES)) == discovery.FEATURE_NAMES
    assert all(tuple(r["features"]) == discovery.FEATURE_NAMES for r in rows.values())
    assert all(np.isfinite(list(r["features"].values())).all() for r in rows.values())


def test_cache_preserves_each_query_id_and_gold_metadata_does_not_change_features(inputs):
    bank = UniqueVectorBank(inputs[2])
    discover = engine(inputs, bank)
    first = discover.discover(query(), ["契約", "手続"], np.array([1., 0.]))
    second = discover.discover(query("Q2", split="valid", law_id="WRONG_HINT", labels={"passage_id": "ORACLE"},
                               source_body_text="注入された参照文。"), ["契約", "手続"], np.array([1., 0.]))
    assert first["query_id"] == "Q1" and second["query_id"] == "Q2" and second["split"] == "valid"
    assert first["addresses"] == second["addresses"] and first["asof"] == second["asof"]
    assert bank.dot_calls == 1 and discover.retrieval_runs == 1 and discover.cache_hits == 1


def test_external_bank_keeps_static_guidance_and_excludes_full_families(inputs):
    result = engine(inputs).discover(query("LAWQA", source_role="external", split="external"),
                                    ["契約", "手続"], np.array([1., 0.]))
    assert {h["chunk_id"] for h in result["asof"]["hybrid"]} == {"GUIDE0", "EXTERNAL_X0"}
    assert {h["chunk_id"] for h in result["history"]["hybrid"]} == {"GUIDE0", "EXTERNAL_X0"}
    guide = next(r for r in result["addresses"] if r["address"] == ["guidance", "guidance-01"])
    assert guide["status"] == "available" and guide["current_best_chunk_id"] == "GUIDE0"
    assert guide["features"]["history_latest_anchor_gap_years"] == 0.
    assert "target_asof" not in guide and "address_key" not in guide


def test_missing_history_rank_uses51_when_address_only_enters_bm25_top50():
    chunks = [chunk(f"N{i:02d}", f"P{i}", "R", num=str(i), asof_dates=[DATE]) for i in range(60)]
    tokens = {c["chunk_id"]: ["契約"] if i >= 50 else ["別語"] for i, c in enumerate(chunks)}
    values = np.array([[1 - i / 100, np.sqrt(1 - (1 - i / 100) ** 2)] for i in range(60)])
    targets = {("A", DATE, "MainProvision", str(i)): {"status": "available", "revision_id": "R", "passage_ids": [f"P{i}"]} for i in range(60)}
    discover = discovery.DiscoveryEngine(chunks, tokens, values, np.arange(60),
        [c["chunk_id"] for c in chunks], targets, {("A", DATE): {"status": "collected", "revision_id": "R"}})
    result = discover.discover(query(), ["契約"], np.array([1., 0.]))
    row = next(r for r in result["addresses"] if r["address"] == ["A", "MainProvision", "50"])
    assert row["features"]["history_best_rank_dense"] == 51.
    assert row["features"]["history_best_rank_bm25"] == 1.


def discovery_files(tmp_path, inputs):
    assert SCRIPT.exists(), "HiDate discovery CLI is not implemented"
    chunks, tokens, vectors, aliases, targets, bindings = inputs
    index, prepared, source, qvectors = [tmp_path / name for name in ("index", "prepared", "source", "query-vectors")]
    write_json(index / "index.json", {"complete": True})
    write_jsonl(index / "chunks.jsonl", chunks)
    write_jsonl(index / "lexical-tokens.jsonl", [{"chunk_id": cid, "tokens": ts} for cid, ts in tokens.items()])
    embedding = index / "embeddings"
    embedding.mkdir()
    pin = {"model_id": "fixture-ruri", "revision": "frozen"}
    write_json(embedding / "encoding.json", {"complete": True, "model_pin": pin})
    write_json(embedding / "chunk-ids.json", [c["chunk_id"] for c in chunks])
    np.save(embedding / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    np.save(embedding / "vector-row-indices.npy", aliases)
    write_json(prepared / "preparation.json", {"complete": True})
    queries = [query(), query("Q2", split="valid"), query("QTEST", split="test", query_text="登記")]
    write_jsonl(prepared / "queries-exact.jsonl", queries)
    write_jsonl(prepared / "target-addresses.jsonl", [{"address_key": list(key), **row} for key, row in targets.items()])
    write_jsonl(source / "asof-bindings.jsonl", [{"law_id": key[0], "requested_asof": key[1], **row} for key, row in bindings.items()])
    write_json(qvectors / "encoding.json", {"complete": True, "model_pin": pin})
    write_json(qvectors / "query-ids.json", [q["query_id"] for q in queries])
    np.save(qvectors / "vectors.npy", np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    np.save(qvectors / "vector-row-indices.npy", np.array([0, 0, 1], dtype=np.int64))
    return index, prepared, source, qvectors


def test_cli_runs_local_discovery_with_explicit_paths(tmp_path, inputs, monkeypatch):
    index, prepared, source, qvectors = discovery_files(tmp_path, inputs)
    output = tmp_path / "local-discovery"
    functions = runpy.run_path(str(SCRIPT))
    discover = functions["discover_candidates"]
    monkeypatch.setitem(functions["main"].__globals__, "discover_candidates",
        lambda *args, **kwargs: discover(*args, **kwargs, segmenter=lambda text: text.split()))
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT),
        "--index-dir", str(index), "--queries", str(prepared / "queries-exact.jsonl"),
        "--query-embeddings-dir", str(qvectors), "--prepared-dir", str(prepared),
        "--source-dir", str(source), "--output-dir", str(output),
        "--protocol", str(ROOT / "configs/retrieval-protocol.json"), "--splits", "test"])
    assert functions["main"]() == 0
    rows = load_jsonl(output / "candidates.jsonl")
    assert [row["query_id"] for row in rows] == ["QTEST"]
    manifest = json.loads((output / "discovery.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True and manifest["splits"] == ["test"]
    assert manifest["training_runs"] == manifest["model_inference_runs"] == 0


def test_tiny_file_orchestration_reuses_aliases_writes_all_selected_ids_and_no_fit(tmp_path, inputs):
    index, prepared, source, qvectors = discovery_files(tmp_path, inputs)
    output = tmp_path / "discovery"
    functions = runpy.run_path(str(SCRIPT))
    manifest = functions["discover_candidates"](index, prepared / "queries-exact.jsonl", qvectors,
        prepared, source, output, splits=("train", "valid"), segmenter=lambda text: text.split())
    rows = load_jsonl(output / "candidates.jsonl")
    assert [r["query_id"] for r in rows] == ["Q1", "Q2"]
    assert {r["split"] for r in rows} == {"train", "valid"}
    assert all(any(h["chunk_id"] == "OTHER_B0" for h in r["asof"]["dense"]) for r in rows)
    assert manifest["complete"] is True and manifest["queries"] == 2
    assert manifest["retrieval_runs"] == 1 and manifest["cache_hits"] == 1
    assert manifest["current_chunk_scoring_count"] == 2
    assert manifest["training_runs"] == manifest["model_inference_runs"] == 0
    assert manifest["features"] == list(discovery.FEATURE_NAMES)
    assert (ROOT / "tools/pbs/discovery.pbs").exists()
