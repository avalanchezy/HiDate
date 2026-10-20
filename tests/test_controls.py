"""Date-wise missing-support banks cannot hydrate removed revision aliases."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("hidate.controls")
    except ModuleNotFoundError:
        pytest.fail("HiDate controlled evidence removal is not implemented")


def parent(identity, revision, text, scope="MainProvision"):
    return {"passage_id": identity, "law_id": "L", "scope": scope, "article_num": "3_2",
            "revision_id": revision, "visible_text": text, "source_role": "full"}


def fixtures():
    parents = [parent("now", "r2", "今の 条文"), parent("alias", "r1", "今の条文"),
               parent("old", "r0", "過去の異なる条文"),
               parent("scope", "r2", "今の条文", "SupplProvision:1")]
    chunks = [{"chunk_id": p["passage_id"] + "|chunk0", "parent_passage_id": p["passage_id"]} for p in parents]
    key = ("L", "2026-09-30", "MainProvision", "3_2")
    targets = {key: {"status": "available", "passage_ids": ["now"], "revision_id": "r2"}}
    queries = [{"split": "test", "target_asof": "2026-09-30", "target_key": list(key)}]
    return chunks, parents, targets, queries, key


def test_remove_current_and_exact_historical_aliases_but_keep_distinct_history_and_scope():
    chunks, parents, targets, queries, key = fixtures()
    result = module().controlled_date_bank(chunks, parents, targets, queries, "2026-09-30")
    assert [r["parent_passage_id"] for r in result["chunks"]] == ["old", "scope"]
    assert result["target_index"][key]["status"] == "corpus_support_unavailable"
    assert result["target_index"][key]["passage_ids"] == ["now"]
    assert targets[key]["status"] == "available"
    assert result["audit"]["extra_exact_text_aliases"] == ["alias"]


def test_other_dates_or_train_queries_never_define_the_test_removal_bank():
    chunks, parents, targets, queries, key = fixtures()
    queries[0]["split"] = "train"
    result = module().controlled_date_bank(chunks, parents, targets, queries, "2026-09-30")
    assert result["chunks"] == chunks and result["target_index"][key]["status"] == "available"
    queries[0]["split"] = "test"
    other = module().controlled_date_bank(chunks, parents, targets, queries, "2024-09-30")
    assert other["chunks"] == chunks
