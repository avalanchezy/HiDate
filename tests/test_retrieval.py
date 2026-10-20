import math
import re

import numpy as np
import pytest

from hidate import retrieval as retrieval


class OffsetTokenizer:
    """Tiny supplied tokenizer: non-whitespace characters are individual tokens."""
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation):
        assert not add_special_tokens and return_offsets_mapping and not truncation
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def passage(parent="OLD|LAW:MainProvision:12_2|1.0", **extra):
    return {"passage_id": parent, "law_id": "LAW", "revision_id": "OLD", "scope": "MainProvision",
            "article_num": "12_2", "slot_id": "LAW:MainProvision:12_2", "law_title": "試験法",
            "title": "第十二条の二", "caption": "（規定）", "visible_text": "甲 乙丙\n丁 戊己庚辛",
            "text": "原文全体。", "paragraphs": [{"text": "原文全体。"}], "visible_paragraphs": [],
            "deleted": False, "hidden": False, "asof_dates": ["2018-09-30", "2021-09-30"],
            "source_role": "full", "split": "train", "family_id": "FAMILY", "raw_paths": ["saved.json"], **extra}


def chunk(chunk_id, revision="OLD", **extra):
    record = {"chunk_id": chunk_id, "parent_passage_id": revision + "|LAW:MainProvision:12_2|1.0",
              "law_id": "LAW", "scope": "MainProvision", "article_num": "12_2",
              "slot_id": "LAW:MainProvision:12_2", "revision_id": revision,
              "source_kind": "statute", "asof_dates": ["2018-09-30", "2021-09-30"],
              "source_role": "full", "split": "train", **extra}
    return record


def test_chunk_windows_use_literal_offsets_and_keep_natural_header_separate():
    record = passage()
    chunks = retrieval.chunk_passages([record], OffsetTokenizer(), max_body_tokens=4, overlap_tokens=1)
    assert [(c["token_start"], c["token_end"]) for c in chunks] == [(0, 4), (3, 7), (6, 8)]
    assert [(c["char_start"], c["char_end"]) for c in chunks] == [(0, 6), (5, 10), (9, 11)]
    assert [c["body_text"] for c in chunks] == ["甲 乙丙\n丁", "丁 戊己庚", "庚辛"]
    assert [c["body_token_count"] for c in chunks] == [4, 4, 2]
    assert [c["chunk_id"] for c in chunks] == [record["passage_id"] + f"|chunk{i}" for i in range(3)]
    assert chunks[0]["header_text"] == "試験法 第十二条の二 （規定）"
    assert record["law_id"] not in chunks[0]["text"] and record["passage_id"] not in chunks[0]["text"]
    assert chunks[0]["parent_passage_id"] == record["passage_id"]
    assert chunks[0]["raw_paths"] == ["saved.json"] and chunks[0]["asof_dates"] == record["asof_dates"]
    assert chunks[0]["source_role"] == "full" and chunks[0]["split"] == "train"
    assert "paragraphs" not in chunks[0] and "visible_text" not in chunks[0]
    assert record["text"] == "原文全体。"


def test_guidance_chunks_keep_url_and_capture_provenance_without_legal_date():
    document = {"source_id": "guidance-01", "title": "監督指針", "url": "https://agency.example/guide.pdf",
                "text": "公式資料の本文。", "pdf_path": "guide.pdf", "text_path": "guide.txt",
                "archive_capture": "2025-04-01", "source_role": "external", "split": "external"}
    chunks = retrieval.chunk_passages([document], OffsetTokenizer(), max_body_tokens=4, overlap_tokens=1)
    assert chunks[0]["parent_passage_id"] == "guidance-01"
    assert chunks[0]["chunk_id"] == "guidance-01|chunk0" and chunks[0]["source_kind"] == "guidance"
    assert chunks[0]["url"] == document["url"] and chunks[0]["archive_capture"] == "2025-04-01"
    assert "asof_dates" not in chunks[0] and "requested_asof" not in chunks[0] and "law_id" not in chunks[0]
    assert chunks[0]["body_text"] == document["text"][chunks[0]["char_start"]:chunks[0]["char_end"]]


def test_chunking_excludes_unavailable_visible_material_and_rejects_nonadvancing_windows():
    records = [passage(deleted=True), passage(hidden=True), passage(visible_text="")]
    assert retrieval.chunk_passages(records, OffsetTokenizer()) == []
    with pytest.raises(ValueError, match="overlap"):
        retrieval.chunk_passages([passage()], OffsetTokenizer(), max_body_tokens=4, overlap_tokens=4)


def test_bm25_uses_document_frequency_and_term_frequency_in_japanese_tokens():
    documents = {"C": ["登記", "会社"], "B": ["契約", "会社"], "A": ["契約", "契約"]}
    hits = retrieval.bm25_rank(["契約"], documents)
    assert [h["chunk_id"] for h in hits] == ["A", "B", "C"]
    assert hits[0]["score"] == pytest.approx(math.log(1 + 1.5 / 2.5) * 1.375)
    assert hits[1]["score"] == pytest.approx(math.log(1 + 1.5 / 2.5))
    assert hits[2]["score"] == 0
    assert retrieval.bm25_rank(["契約"], {}, top_k=50) == []


@pytest.mark.parametrize("query", [["契約"], ["契約", "契約", "登記"], [], ["未知語"]])
def test_cached_bm25_matches_reference_for_repeated_terms_and_zero_fill(query):
    documents = {"D": ["登記", "会社", "会社"], "B": ["契約", "会社"],
                 "C": [], "A": ["契約", "契約", "登記", "契約"]}
    cached = retrieval.BM25Index(documents)
    actual = cached.rank(query, top_k=4)
    reference = retrieval.bm25_rank(query, documents, top_k=4)
    assert [r["chunk_id"] for r in actual] == [r["chunk_id"] for r in reference]
    assert [r["score"] for r in actual] == pytest.approx([r["score"] for r in reference], rel=1e-12, abs=1e-12)
    # Rankings use frozen indexed statistics after construction.
    documents["A"][:] = ["変更後"]
    assert cached.rank(query, top_k=4) == actual


def test_cached_bm25_cutoff_ties_follow_chunk_id_and_keep_zero_score_budget():
    documents = {"Z": ["会社"], "C": ["契約"], "A": ["契約"], "B": ["契約"], "Y": []}
    cached = retrieval.BM25Index(documents)
    assert [r["chunk_id"] for r in cached.rank(["契約"], top_k=2)] == ["A", "B"]
    hits = cached.rank(["契約"], top_k=4)
    assert [r["chunk_id"] for r in hits] == ["A", "B", "C", "Y"]
    assert hits[-1]["score"] == 0.
    assert cached.rank([], top_k=2) == [{"chunk_id": "A", "score": 0.}, {"chunk_id": "B", "score": 0.}]
    assert retrieval.BM25Index({}).rank(["契約"]) == []


def test_cosine_topk_has_deterministic_ties_and_empty_bank():
    matrix = np.array([[1., 0.], [0., 1.], [1., 0.]])
    hits = retrieval.cosine_topk(np.array([1., 0.]), matrix, ["C", "B", "A"], top_k=2)
    assert hits == [{"chunk_id": "A", "score": 1.}, {"chunk_id": "C", "score": 1.}]
    assert retrieval.cosine_topk(np.array([1., 0.]), np.empty((0, 2)), []) == []


def test_rrf_uses_constant60_and_counts_duplicate_hits_once_per_ranklist():
    lexical = [{"chunk_id": "A", "score": 10}, {"chunk_id": "A", "score": 9}, {"chunk_id": "B", "score": 8}]
    dense = [{"chunk_id": "B", "score": .9}, {"chunk_id": "C", "score": .8}]
    fused = retrieval.rrf([lexical, dense])
    assert [h["chunk_id"] for h in fused] == ["B", "A", "C"]
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1]["score"] == pytest.approx(1 / 61)
    assert fused[2]["score"] == pytest.approx(1 / 62)


def test_dated_eligibility_keeps_static_guidance_filters_future_and_deduplicates_chunks():
    old = chunk("OLD0")
    present = chunk("CURRENT0", "CURRENT", asof_dates=["2024-09-30", "2026-09-30"])
    future = chunk("FUTURE0", "FUTURE", asof_dates=["2026-09-30"])
    guidance = {"chunk_id": "G0", "source_id": "guidance-01", "source_kind": "guidance", "source_role": "external"}
    corpus = [old, dict(old), present, future, guidance]
    assert [r["chunk_id"] for r in retrieval.eligible_chunks(corpus, "2024-09-30", mode="asof")] == ["CURRENT0", "G0"]
    assert [r["chunk_id"] for r in retrieval.eligible_chunks(corpus, "2024-09-30", mode="history")] == ["OLD0", "CURRENT0", "G0"]
    # Caller selects the supervision role; the static external guide is excluded before temporal filtering.
    full = [r for r in corpus if r["source_role"] == "full"]
    assert all(r["source_role"] == "full" for r in retrieval.eligible_chunks(full, "2024-09-30", "history"))


def test_history_aggregates_chunks_once_per_scoped_address_and_distinct_revision():
    chunks = {"OLD0": chunk("OLD0"), "OLD1": chunk("OLD1"), "NEW0": chunk("NEW0", "NEW"),
              "SUPPL0": chunk("SUPPL0", scope="SupplProvision:original", slot_id="LAW:SupplProvision:original:12_2")}
    hits = [{"chunk_id": "OLD0", "score": .2}, {"chunk_id": "OLD1", "score": .8},
            {"chunk_id": "OLD1", "score": .8}, {"chunk_id": "NEW0", "score": .4},
            {"chunk_id": "SUPPL0", "score": .9}]
    rows = retrieval.aggregate_history(hits, chunks)
    main = next(r for r in rows if r["scope"] == "MainProvision")
    assert main["address"] == ("LAW", "MainProvision", "12_2")
    assert main["revision_scores"] == {"OLD": .8, "NEW": .4}
    assert main["max_score"] == .8 and main["mean_score"] == pytest.approx(.6)
    assert main["support_count"] == 2
    assert len(rows) == 2 and next(r for r in rows if r["scope"] != "MainProvision")["support_count"] == 1


def test_binding_resolves_actual_target_parent_and_preserves_unavailable_and_failure_statuses():
    rows = [{"address": ("LAW", "MainProvision", num), "source_kind": "statute", "law_id": "LAW", "scope": "MainProvision", "article_num": num}
            for num in ("12_2", "13", "14", "15", "16", "17")]
    date = "2024-09-30"
    targets = {("LAW", date, "MainProvision", "12_2"): {"status": "available", "revision_id": "ACTUAL_TARGET", "passage_ids": ["CURRENT_PARENT"]},
               ("LAW", date, "MainProvision", "13"): {"status": "deleted", "revision_id": "ACTUAL_TARGET", "passage_ids": ["DELETED_PARENT"]},
               ("LAW", date, "MainProvision", "14"): {"status": "hidden", "revision_id": "ACTUAL_TARGET", "passage_ids": ["HIDDEN_PARENT"]},
               ("LAW", date, "MainProvision", "15"): {"status": "no_substantive_text", "revision_id": "ACTUAL_TARGET", "passage_ids": ["BLANK_PARENT"]},
               ("LAW", date, "MainProvision", "16"): {"status": "ambiguous", "revision_id": "ACTUAL_TARGET", "passage_ids": ["P1", "P2"]}}
    bindings = {("LAW", date): {"status": "collected", "revision_id": "ACTUAL_TARGET"}}
    bound = retrieval.bind_addresses(rows, date, targets, bindings)
    assert [r["status"] for r in bound] == ["available", "deleted", "hidden", "no_substantive_text", "ambiguous", "absent"]
    assert bound[0]["parent_passage_ids"] == ["CURRENT_PARENT"] and bound[0]["revision_id"] == "ACTUAL_TARGET"
    assert bound[0]["address_key"] == ("LAW", date, "MainProvision", "12_2")
    assert all(r["parent_passage_ids"] == [] for r in bound[1:])
    assert bound[1]["target_passage_ids"] == ["DELETED_PARENT"]
    failed = retrieval.bind_addresses(rows[:1], date, targets, {("LAW", date): {"status": "collection_failure"}})
    assert failed[0]["status"] == "collection_failure" and failed[0]["parent_passage_ids"] == []
    assert retrieval.bind_addresses(rows[:1], date, targets, {})[0]["status"] == "collection_failure"


def test_guidance_aggregation_and_binding_remain_static_sources():
    chunks = {"G0": {"chunk_id": "G0", "source_id": "guidance-01", "source_kind": "guidance",
                      "parent_passage_id": "guidance-01", "url": "https://agency.example/guide.pdf"},
              "G1": {"chunk_id": "G1", "source_id": "guidance-01", "source_kind": "guidance",
                      "parent_passage_id": "guidance-01", "url": "https://agency.example/guide.pdf"}}
    rows = retrieval.aggregate_history([{"chunk_id": "G0", "score": .3}, {"chunk_id": "G1", "score": .8}], chunks)
    assert len(rows) == 1 and rows[0]["source_id"] == "guidance-01" and rows[0]["max_score"] == .8
    bound = retrieval.bind_addresses(rows, "2024-09-30", {}, {})[0]
    assert bound["status"] == "available" and bound["parent_passage_ids"] == ["guidance-01"]
    assert "law_id" not in bound and "target_asof" not in bound and "address_key" not in bound
