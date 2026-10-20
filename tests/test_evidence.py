"""Reader evidence budgets preserve the complete released choice question."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("hidate.evidence")
    except ModuleNotFoundError:
        pytest.fail("HiDate evidence packing is not implemented")


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(row["content"]) for row in messages) + 3))


def qa():
    return {"ファイル名": "元の問題", "問題文": "問題全文を維持してください。",
            "選択肢": "a 選択肢一\nb 選択肢二\nc 選択肢三\nd 選択肢四", "output": "c",
            "コンテキスト": "元の文脈", "references": ["https://example.org/law/L"]}


def test_external_query_dates_are_corpus_conditions_and_never_query_text():
    rows = module().external_queries([qa()], ["2025-11-04", "2026-09-30"])
    assert len(rows) == 2 and rows[0]["query_id"] != rows[1]["query_id"]
    assert rows[0]["query_text"] == qa()["問題文"] == rows[1]["query_text"]
    assert rows[0]["source_role"] == "external" and rows[0]["split"] == "external"
    assert rows[0]["original_question_id"] == qa()["ファイル名"]
    assert "law_id" not in rows[0] and "gold_answer" not in rows[0]


def test_budget_packing_truncates_only_evidence_and_preserves_question_and_choices():
    tokenizer = CharacterTokenizer()
    base = module().pack_choice_evidence(qa(), [], tokenizer, prompt_budget=1000)
    passage = {"chunk_id": "c", "text": "長い参考文" * 80}
    result = module().pack_choice_evidence(qa(), [passage], tokenizer,
        prompt_budget=base["prompt_tokens"] + 60)
    user = result["messages"][-1]["content"]
    assert qa()["問題文"] in user and qa()["選択肢"] in user
    assert result["prompt_tokens"] <= base["prompt_tokens"] + 60
    assert result["evidence"][0]["truncated"] and result["evidence"][0]["text"]
    assert passage["text"].startswith(result["evidence"][0]["text"])


def test_packing_respects_four_passages_and_does_not_inject_gold_answer():
    passages = [{"chunk_id": str(i), "text": "参考資料"} for i in range(6)]
    result = module().pack_choice_evidence(qa(), passages, CharacterTokenizer(), prompt_budget=1000)
    assert len(result["evidence"]) == 4
    assert [row["chunk_id"] for row in result["evidence"]] == ["0", "1", "2", "3"]
    different = qa()
    different["output"] = "a"
    assert module().pack_choice_evidence(different, passages, CharacterTokenizer(), prompt_budget=1000) == result
