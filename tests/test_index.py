"""Tiny tokenizer/segmenter boundaries exercise CPU index output and provenance."""

import json
import re
import runpy
from pathlib import Path

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_index.py"


def builder_functions():
    assert SCRIPT.exists(), "HiDate CPU index builder is not implemented"
    return runpy.run_path(str(SCRIPT))


class OffsetTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
        assert truncation is False
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S", text)]
        encoded = {"input_ids": list(range(len(spans) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            assert add_special_tokens is False
            encoded["offset_mapping"] = spans
        return encoded


@pytest.fixture
def inputs(tmp_path):
    prepared = tmp_path / "prepared"
    passage = {"passage_id": "OLD|LAW:MainProvision:12_2|1.0", "law_id": "LAW", "revision_id": "OLD",
        "scope": "MainProvision", "article_num": "12_2", "slot_id": "LAW:MainProvision:12_2",
        "law_title": "試験法", "title": "第十二条の二", "caption": "（規定）",
        "visible_text": "甲 乙丙\n丁 戊己庚辛", "text": "原文全体。", "paragraphs": [], "visible_paragraphs": [],
        "deleted": False, "hidden": False, "tree_path": [1, 0],
        "asof_dates": ["2018-09-30", "2021-09-30"], "source_role": "full", "split": "train",
        "family_id": "FAMILY", "raw_paths": ["snapshots/old.json"]}
    external = {**passage, "passage_id": "EXT|RESERVED:MainProvision:1|0.1", "law_id": "RESERVED",
        "slot_id": "RESERVED:MainProvision:1", "revision_id": "EXT", "visible_text": "外部条文。",
        "source_role": "external", "split": "external", "family_id": "RESERVED"}
    guidance = {"source_id": "guidance-02", "title": "公式指針", "text": "公式資料の本文。",
        "url": "https://www.fsa.go.jp/common/law/kaiji/250221_kaiji.pdf",
        "acquisition_url": "https://warp.ndl.go.jp/captured-original.pdf",
        "archive_capture": "2025-04-01T13:12:32+09:00", "pdf_path": "guide.pdf", "text_path": "guide.txt",
        "source_role": "external", "split": "external"}
    write_jsonl(prepared / "corpus-passages.jsonl", [passage, external, {**passage, "passage_id": "DELETED", "deleted": True}])
    write_jsonl(prepared / "guidance-documents.jsonl", [guidance])
    write_json(prepared / "preparation.json", {"complete": True})
    stack = tmp_path / "stack.json"
    write_json(stack, {"complete": True, "models": {"embedding": {"model_id": "cl-nagoya/ruri-v3-310m",
        "revision": "frozen-ruri-revision", "local_path": "fixture-local-tokenizer", "download_complete": True,
        "conventions": {"document_prefix": "検索文書: "}}}})
    return prepared, stack, tmp_path / "index", passage, guidance


def test_build_indexes_literal_windows_external_guidance_and_measured_prefixed_lengths(inputs):
    prepared, stack, output, passage, guidance = inputs
    functions = builder_functions()
    record = functions["build_index"](prepared, stack, output, tokenizer=OffsetTokenizer(),
        segmenter=lambda text: [*list(text), "付属"],
        package_versions={"SudachiPy": "fixture", "SudachiDict-core": "fixture", "transformers": "fixture"},
        max_body_tokens=4, overlap_tokens=1)
    assert record["complete"] is True
    chunks = load_jsonl(output / "chunks.jsonl")
    full = [row for row in chunks if row["source_role"] == "full"]
    assert [row["body_text"] for row in full] == ["甲 乙丙\n丁", "丁 戊己庚", "庚辛"]
    assert [(row["token_start"], row["token_end"]) for row in full] == [(0, 4), (3, 7), (6, 8)]
    assert full[0]["ruri_document_tokens"] == 24  # prefix5 + header13 + body4 + special2
    assert full[0]["raw_paths"] == ["snapshots/old.json"]
    assert full[0]["asof_dates"] == ["2018-09-30", "2021-09-30"]
    guides = [row for row in chunks if row["source_kind"] == "guidance"]
    assert len(guides) == 3
    assert all(row["url"] == guidance["url"] and row["archive_capture"] == guidance["archive_capture"] for row in guides)
    assert all("law_id" not in row and "asof_dates" not in row for row in guides)
    lexical = load_jsonl(output / "lexical-tokens.jsonl")
    assert [row["chunk_id"] for row in lexical] == [row["chunk_id"] for row in chunks]
    assert lexical[0]["tokens"][-1] == "付属"
    assert "（" not in lexical[0]["tokens"] and "\n" not in lexical[0]["tokens"]
    assert record["statistics"]["chunks"] == 8
    assert record["statistics"]["chunks_by_role"] == {"full": 3, "external": 5}
    assert record["statistics"]["chunks_by_kind"] == {"statute": 5, "guidance": 3}
    assert record["statistics"]["chunks_by_parent"][passage["passage_id"]] == 3
    assert record["tokenizer"]["revision"] == "frozen-ruri-revision"
    assert record["ruri_input_limit"] == 1024
    assert record["lexical_tokenization"]["split_mode"] == "A"
    assert record["package_versions"]["SudachiDict-core"] == "fixture"
    assert record["training_runs"] == record["model_inference_runs"] == 0


def test_same_normalized_token_filter_can_be_used_for_queries():
    tokens = builder_functions()["japanese_tokens"]("任意の質問", lambda text: ["会社", " \t", "。", "付属", "(", ")", "＋", "2026"])
    assert tokens == ["会社", "付属", "＋", "2026"]


def test_index_reports_oversized_prefixed_header_instead_of_silent_truncation(inputs):
    prepared, stack, output, _, _ = inputs
    rows = load_jsonl(prepared / "corpus-passages.jsonl")
    rows[0]["law_title"] = "甲" * 1024
    write_jsonl(prepared / "corpus-passages.jsonl", rows)
    with pytest.raises(ValueError, match="1024"):
        builder_functions()["build_index"](prepared, stack, output, tokenizer=OffsetTokenizer(),
            segmenter=lambda text: [text], package_versions={"SudachiPy": "fixture"},
            max_body_tokens=4, overlap_tokens=1)
    assert json.loads((output / "index.json").read_text())["complete"] is False
