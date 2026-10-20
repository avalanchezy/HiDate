"""Prepare complete external LawQA corpus conditions without temporal gold."""

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.evidence import external_queries, pack_choice_evidence


def main():
    prepared = Path("data/processed/hidate")
    original = load_jsonl(prepared / "lawqa-external.jsonl")
    dates = ["2024-09-30", "2025-11-04", "2026-09-30"]
    queries = external_queries(original, dates)
    write_jsonl(prepared / "lawqa-retrieval-queries.jsonl", queries)
    stack = json.loads(Path("models/ruri/model-stack.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(stack["models"]["reader"]["local_path"], local_files_only=True)
    prompt_records = []
    for question in original:
        closed = pack_choice_evidence(question, [], tokenizer)
        oracle = pack_choice_evidence(question, [{"chunk_id": question["ファイル名"] + "|released_context",
                                                 "text": question["コンテキスト"]}], tokenizer)
        prompt_records.append({"original_question_id": question["ファイル名"],
            "closed_book_prompt_tokens": closed["prompt_tokens"],
            "released_context_prompt_tokens": oracle["prompt_tokens"],
            "released_context_truncated": any(row["truncated"] for row in oracle["evidence"]),
            "released_context_original_characters": len(question["コンテキスト"]),
            "released_context_included_characters": sum(row["included_characters"] for row in oracle["evidence"])})
    report = {"complete": True, "job_id": os.environ.get("PBS_JOBID"), "questions": len(original),
        "queries": len(queries), "corpus_dates": dates, "invented_question_dates": 0,
        "query_input": "original 問題文 only; no context, options or released answer",
        "reader_prompt_budget": 4096, "reader_max_passages": 4,
        "oracle_contexts_truncated": sum(row["released_context_truncated"] for row in prompt_records),
        "max_closed_book_tokens": max(row["closed_book_prompt_tokens"] for row in prompt_records),
        "max_released_context_prompt_tokens": max(row["released_context_prompt_tokens"] for row in prompt_records),
        "reader_pin": {key: stack["models"]["reader"][key] for key in ["model_id", "revision"]},
        "prompts": prompt_records, "model_forward_calls": 0, "training_runs": 0}
    write_json(Path("runs/hidate/pbs/lawqa-prepare/preparation.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "prompts"}), flush=True)


if __name__ == "__main__":
    main()
