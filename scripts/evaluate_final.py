"""Run frozen final HiDate retrieval, then complete external forced-choice QA."""

import argparse
import gc
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from hidate.io import write_json
from hidate.discovery import chunk_address
from hidate.evidence import pack_choice_evidence
from hidate.reranking import ALL_CONDITIONS, PairScoreCache, rerank_query
from hidate.selection import baseline_support_score, selected_evidence, support_score


DISCOVERY_GROUPS = (
    ("test-natural-exact", "natural", "exact_quote"),
    ("test-natural-paraphrase", "natural", "generated_paraphrase"),
    ("test-controlled-exact", "controlled", "exact_quote"),
    ("test-controlled-paraphrase", "controlled", "generated_paraphrase"),
    ("lawqa", "lawqa", "natural_question"),
)


def json_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def compact_candidate(row):
    # Actual model text is recorded once per physical pair in pair receipts.
    return {**{key: value for key, value in row.items()
               if key not in {"text", "body_text", "visible_text", "visible_paragraphs", "paragraphs"}},
            "address": list(chunk_address(row))}


def structural_result(query, evidence, scope):
    available = scope == "natural" and query["target_status"] == "available"
    target = query["target_key"]
    address = (target[0], target[2], target[3])
    top = evidence[0] if evidence else None
    rank = next((i for i, item in enumerate(evidence, 1)
                 if available and item["parent_passage_id"] == query["target_passage_id"]), None)
    return {"target_available": available,
        "top_address_correct": bool(top and chunk_address(top) == address),
        "top_revision_correct": bool(available and top and chunk_address(top) == address
                                     and top.get("revision_id") == query["target_revision_id"]),
        "top_parent_correct": bool(available and top and top["parent_passage_id"] == query["target_passage_id"]),
        "target_parent_rank": rank, "target_mrr": 1. / rank if rank else 0.,
        "target_recall_at_four": rank is not None,
        "task": "source-derived addressed retrieval and citation; no semantic QA gold"}


def frozen_inputs(queries_paths, lawqa_queries_path, questions_path, discovery_root, policy_path, stack_path,
                  ranker_path=None, preflight_path=None, expected_counts=None):
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if not policy["complete"]:
        raise ValueError("The final frozen policy must be complete before any model load or forward")
    if set(policy["method_thresholds"]) != set(ALL_CONDITIONS) or not 0. <= policy["alpha"] <= 1.:
        raise ValueError("The frozen policy must specify all seven thresholds and its chosen alpha")
    ranker_path = Path(ranker_path or policy["ranker_path"])
    ranker = json.loads(ranker_path.read_text(encoding="utf-8"))
    if not ranker["complete"]:
        raise ValueError("The frozen one-TRAIN address ranker must be complete")
    queries = {}
    for path in queries_paths:
        for query in json_rows(path):
            if query["split"] == "test":
                if query["query_id"] in queries:
                    raise ValueError("Duplicate final structural query ID")
                queries[query["query_id"]] = query
    external, questions = {}, {}
    for row in json_rows(lawqa_queries_path):
        if row["query_id"] in external or row["split"] != "external" or row["source_role"] != "external":
            raise ValueError("Each complete external query requires one unique ID and its original external role")
        external[row["query_id"]] = row
    for row in json_rows(questions_path):
        if row["ファイル名"] in questions:
            raise ValueError("Every released external question requires one unique original question ID")
        questions[row["ファイル名"]] = row
    if set(queries).intersection(external):
        raise ValueError("Structural and external query IDs must differ")
    if {row["original_question_id"] for row in external.values()} != set(questions):
        raise ValueError("Every released external question requires its complete retrieval inputs")
    queries.update(external)
    groups, counts = [], Counter()
    for name, scope, variant in DISCOVERY_GROUPS:
        directory = Path(discovery_root) / name
        report = json.loads((directory / "discovery.json").read_text(encoding="utf-8"))
        if not report["complete"]:
            raise ValueError(f"Final discovery must be complete: {name}")
        expected = {key for key, q in queries.items() if q["query_variant"] == variant
                    and (scope != "controlled" or q["target_status"] == "available")}
        seen = set()
        for row in json_rows(directory / "candidates.jsonl"):
            key = row["query_id"]
            if key in seen or key not in expected:
                raise ValueError(f"Each final input requires exactly one discovery row: {name}")
            query = queries[key]
            if row["evaluation_scope"] != scope or any(row[field] != query[field] for field in
                    ("query_variant", "split", "target_asof", "source_role")):
                raise ValueError(f"Final discovery metadata disagrees with its query map: {name}")
            seen.add(key)
        if seen != expected:
            raise ValueError(f"Each final input requires exactly one complete discovery row: {name}")
        groups.append((directory, scope, variant))
        counts[scope] += len(seen)
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError(f"Complete final row counts disagree: {dict(counts)} != {expected_counts}")
    if expected_counts is not None:
        expected_dates = {"2024-09-30", "2025-11-04", "2026-09-30"}
        actual_pairs = [(q["original_question_id"], q["target_asof"]) for q in external.values()]
        if len(questions) != 140 or len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != {
                (identity, date) for identity in questions for date in expected_dates}:
            raise ValueError("Complete final LawQA inputs require all 140 original questions at the three frozen corpus dates")
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    pins = {role: {key: stack["models"][role][key] for key in ("model_id", "revision")}
            for role in ("reranker", "reader")}
    if not stack["complete"] or not all(stack["models"][role]["download_complete"] for role in pins):
        raise ValueError("Both frozen model acquisitions must be complete")
    if preflight_path is not None:
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        verified = {role: {key: preflight["model_pins"][role][key] for key in ("model_id", "revision")}
                    for role in pins}
        if preflight["status"] != "PASS" or verified != pins:
            raise ValueError("Final model pins must match the passed GPU preflight")
    return policy, ranker, ranker_path, queries, questions, groups, dict(counts), stack, pins


def evaluate_final(queries_paths, lawqa_queries_path, questions_path, discovery_root, chunks_path,
                   stack_path, policy_path, output_dir, *, ranker_path=None, preflight_path=None,
                   expected_counts=None, reranker_factory=None, reader_factory=None, release_reranker=None):
    start = time.perf_counter()
    policy, ranker, ranker_path, queries, questions, groups, counts, stack, pins = frozen_inputs(
        queries_paths, lawqa_queries_path, questions_path, discovery_root, policy_path, stack_path,
        ranker_path, preflight_path, expected_counts)
    chunks = {row["chunk_id"]: row for row in json_rows(chunks_path)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "started_at_utc": utc_now(), "job_id": os.environ.get("PBS_JOBID"),
        "phase": "reranking", "training_runs": 0, "dev_updates": 0,
        "policy_path": str(Path(policy_path).resolve()), "frozen_policy": policy,
        "ranker_path": str(ranker_path.resolve()), "model_pins": pins,
        "stack_model_pins": {role: {key: item[key] for key in ("model_id", "revision")}
                             for role, item in stack["models"].items()},
        "model_local_paths": {role: stack["models"][role]["local_path"] for role in pins},
        "query_paths": [str(Path(path).resolve()) for path in queries_paths],
        "lawqa_queries_path": str(Path(lawqa_queries_path).resolve()),
        "questions_path": str(Path(questions_path).resolve()), "chunks_path": str(Path(chunks_path).resolve()),
        "discovery_root": str(Path(discovery_root).resolve()), "expected_retrieval_rows_by_scope": counts,
        "retrieval_rows_by_scope": {}, "reader_calls": 0,
        "external_questions": len(questions), "expected_reader_calls": counts["lawqa"] * len(ALL_CONDITIONS) + 2 * len(questions),
        "conditions": list(ALL_CONDITIONS), "pair_budget": 10, "final_parent_budget": 4,
        "reranker_max_length": 1024, "physical_pair_batch_size": 32,
        "reader_prompt_budget": 4096, "reader_max_passages": 4,
        "cache_policy": "exact raw (query_text, chunk.text) within this complete execution only",
        "structural_task": "address/revision retrieval and citation only; no semantic QA labels",
        "controlled_target_status": "corpus_support_unavailable",
        "reader_policy": "forced-choice for every external input, including withheld; closed book/oracle once per question",
        "outputs": {"retrieval": "retrieval.jsonl", "answers": "answers.jsonl",
                    "pair_receipts": "model-pair-receipts.jsonl", "reader_receipts": "model-reader-receipts.jsonl"}}
    write_json(output / "evaluation.json", manifest)
    write_json(output / "ranker-used.json", ranker)
    real_reranker = reranker_factory is None
    if real_reranker:
        import torch
        from hidate.models import create_reranker
        reranker = create_reranker(stack["models"]["reranker"], device="cuda", max_length=1024)
        manifest["devices"] = {"reranker": {"name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "dtype": str(next(reranker.model.parameters()).dtype)}}
    else:
        reranker = reranker_factory(stack["models"]["reranker"])
    completed, pair_requests, input_tokens, final_counts = Counter(), Counter(), Counter(), Counter()
    scoped_pair_requests, scoped_input_tokens = Counter(), Counter()
    structural_counts = defaultdict(Counter)
    with (output / "model-pair-receipts.jsonl").open("w", encoding="utf-8") as receipts, \
            (output / "retrieval.jsonl").open("w", encoding="utf-8") as records:
        def save_pair_receipt(row):
            save_row(receipts, row)
            receipts.flush()
        cache = PairScoreCache(reranker, receipt_callback=save_pair_receipt)
        for directory, scope, _ in groups:
            for discovery in json_rows(directory / "candidates.jsonl"):
                query = queries[discovery["query_id"]]
                reranked = rerank_query(query, discovery, chunks, cache, ranker=ranker, conditions=ALL_CONDITIONS)
                record = {key: reranked[key] for key in ("query_id", "query_variant", "split", "target_asof", "source_role")}
                record.update(evaluation_scope=scope, corpus_condition_asof=query["target_asof"], conditions={})
                if scope == "lawqa":
                    record["original_question_id"] = query["original_question_id"]
                else:
                    record.update(base_query_id=query.get("base_query_id", query["query_id"]),
                        target_status="corpus_support_unavailable" if scope == "controlled" else query["target_status"],
                        target_key=query["target_key"], stratum=query["stratum"],
                        original_gold={key: query[key] for key in
                            ("target_status", "target_key", "target_revision_id", "target_passage_id", "labels")})
                for method, condition in reranked["conditions"].items():
                    alpha = policy["alpha"] if method == "history_learned" else 0.
                    evidence = selected_evidence(condition, alpha=alpha)
                    confidence = (support_score(evidence, condition["address_diagnostics"]) if method == "history_learned"
                                  else baseline_support_score(method, evidence, condition["address_diagnostics"]))
                    threshold = policy["method_thresholds"][method]["threshold"]
                    saved = {"alpha": alpha, "support_confidence": confidence, "support_threshold": threshold,
                        "accepted": confidence >= threshold, "requested_pair_count": condition["requested_pair_count"],
                        "requested_input_tokens": condition["requested_input_tokens"],
                        "evidence": [compact_candidate(row) for row in evidence],
                        "evidence_chunk_ids": [row["chunk_id"] for row in evidence],
                        "evidence_parent_ids": [row["parent_passage_id"] for row in evidence],
                        "reranked_candidates": [compact_candidate(row) for row in condition["reranked_candidates"]],
                        "address_diagnostics": condition["address_diagnostics"]}
                    if scope != "lawqa":
                        saved["structural"] = structural_result(query, evidence, scope)
                        cell = structural_counts["|".join((scope, query["query_variant"], query["stratum"], method))]
                        cell.update(queries=1, accepted=int(saved["accepted"]),
                            top_address_correct=int(saved["structural"]["top_address_correct"]),
                            top_revision_correct=int(saved["structural"]["top_revision_correct"]),
                            top_parent_correct=int(saved["structural"]["top_parent_correct"]),
                            accepted_correct_support=int(saved["accepted"] and saved["structural"]["top_parent_correct"]),
                            recall_at_four=int(saved["structural"]["target_recall_at_four"]))
                        cell["mrr_sum"] += saved["structural"]["target_mrr"]
                    record["conditions"][method] = saved
                    pair_requests[method] += condition["requested_pair_count"]
                    input_tokens[method] += condition["requested_input_tokens"]
                    scoped_pair_requests["|".join((scope, query["query_variant"], method))] += condition["requested_pair_count"]
                    scoped_input_tokens["|".join((scope, query["query_variant"], method))] += condition["requested_input_tokens"]
                    final_counts[method] += len(evidence)
                save_row(records, record)
                completed[scope] += 1
                if sum(completed.values()) % 100 == 0:
                    records.flush()
                    manifest.update(retrieval_rows_by_scope=dict(completed), physical_pair_count=cache.physical_pair_count,
                                    logical_pair_requests=cache.logical_pair_requests)
                    write_json(output / "evaluation.json", manifest)
                    print(json.dumps({"phase": "reranking", "completed": dict(completed),
                                      "physical_pairs": cache.physical_pair_count}), flush=True)
    manifest.update(retrieval_rows_by_scope=dict(completed), logical_pair_requests=cache.logical_pair_requests,
        physical_pair_count=cache.physical_pair_count, exact_pair_cache_reuses=cache.logical_pair_requests - cache.physical_pair_count,
        reranker_model_call_count=cache.model_call_count, reranker_encoded_input_tokens=cache.encoded_input_tokens,
        reranker_padded_token_slots=cache.padded_token_slots,
        reranker_wall_seconds_with_cuda_sync=cache.model_wall_seconds_with_cuda_sync,
        reranker_sdpa_backend=cache.sdpa_backend, condition_pair_requests=dict(pair_requests),
        condition_requested_input_tokens=dict(input_tokens), scope_variant_condition_pair_requests=dict(scoped_pair_requests),
        scope_variant_condition_requested_input_tokens=dict(scoped_input_tokens), final_parent_counts=dict(final_counts),
        structural_counts={key: dict(value) for key, value in structural_counts.items()})
    # Remove the cache's GPU model reference before loading the 7B reader.
    cache.reranker = None
    if release_reranker is not None:
        release_reranker(reranker)
    del reranker
    if real_reranker:
        gc.collect()
        torch.cuda.empty_cache()
    manifest.update(phase="reader", reranker_released_before_reader=True)
    write_json(output / "evaluation.json", manifest)
    if reader_factory is None:
        import torch
        from hidate.models import QwenReader
        reader = QwenReader(stack["models"]["reader"]["local_path"], device="cuda", prompt_budget=4096)
        manifest.setdefault("devices", {})["reader"] = {"name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "dtype": str(next(reader.model.parameters()).dtype),
            "attention_implementation": reader.model.config._attn_implementation}
    else:
        reader = reader_factory(stack["models"]["reader"])
    answer_counts, reader_work = defaultdict(Counter), Counter()
    with (output / "answers.jsonl").open("w", encoding="utf-8") as answers, \
            (output / "model-reader-receipts.jsonl").open("w", encoding="utf-8") as receipts:
        def answer(question, method, passages, metadata):
            packed = pack_choice_evidence(question, passages, reader.tokenizer, prompt_budget=4096, max_passages=4)
            result = reader.score_choices(packed["messages"], include_inputs=True)
            if (set(result["choice_logprobs"]) != set("abcd") or result["prediction"] not in "abcd"
                    or not all(math.isfinite(value) for value in result["choice_logprobs"].values())):
                raise ValueError("Qwen returned invalid forced-choice scores")
            if result["prompt_tokens"] != packed["prompt_tokens"] or result["total_tokens"] != sum(result["token_counts"]):
                raise ValueError("Reader actual token receipt disagrees with the packed prompt")
            gold = question["output"].strip().lower()
            if gold not in ("a", "b", "c", "d"):
                raise ValueError("Released LawQA answer must be a single choice label")
            manifest["reader_calls"] += 1
            identity = {**metadata, "method": method, "original_question_id": question["ファイル名"]}
            save_row(receipts, {**identity, "call_number": manifest["reader_calls"],
                "messages": packed["messages"], "result": result})
            record = {**identity, "prediction": result["prediction"], "released_answer": gold,
                "correct": result["prediction"] == gold, "choice_logprobs": result["choice_logprobs"],
                "evidence": packed["evidence"], "prompt_tokens": result["prompt_tokens"],
                "reader_receipt_call_number": manifest["reader_calls"],
                "evidence_is_corpus_condition": metadata["evaluation_scope"] == "lawqa"}
            save_row(answers, record)
            cell = answer_counts["|".join((metadata["evaluation_scope"], method, metadata.get("target_asof", "once_per_question")))]
            cell.update(calls=1, correct=int(record["correct"]), accepted=int(metadata["accepted"]),
                        accepted_correct=int(metadata["accepted"] and record["correct"]))
            cell.update(input_tokens=result["total_tokens"],
                        padded_token_slots=result["input_shape"][0] * result["input_shape"][1])
            cell["wall_seconds_with_cuda_sync"] += result["seconds"]
            reader_work.update(input_tokens=result["total_tokens"],
                padded_token_slots=result["input_shape"][0] * result["input_shape"][1])
            reader_work["wall_seconds_with_cuda_sync"] += result["seconds"]
            if manifest["reader_calls"] % 100 == 0:
                answers.flush()
                receipts.flush()
                write_json(output / "evaluation.json", manifest)
                print(json.dumps({"phase": "reader", "calls": manifest["reader_calls"],
                                  "expected": manifest["expected_reader_calls"]}), flush=True)

        for record in json_rows(output / "retrieval.jsonl"):
            if record["evaluation_scope"] != "lawqa":
                continue
            question = questions[record["original_question_id"]]
            for method, condition in record["conditions"].items():
                passages = [chunks[key] for key in condition["evidence_chunk_ids"]]
                answer(question, method, passages, {"query_id": record["query_id"],
                    "query_variant": record["query_variant"], "evaluation_scope": "lawqa", "target_asof": record["target_asof"],
                    "source_role": record["source_role"], "accepted": condition["accepted"],
                    "support_confidence": condition["support_confidence"], "support_threshold": condition["support_threshold"],
                    "selected_evidence_chunk_ids": condition["evidence_chunk_ids"],
                    "selected_evidence_parent_ids": condition["evidence_parent_ids"]})
        for question in questions.values():
            for method, passages in (("closed_book", []), ("released_oracle", [{
                    "chunk_id": question["ファイル名"] + "|released_context", "text": question["コンテキスト"]}])):
                answer(question, method, passages, {"evaluation_scope": "lawqa_reference", "accepted": True,
                    "acceptance_policy": "unconditional reference; no retrieval support threshold"})
    manifest.update(phase="complete", answer_counts={key: dict(value) for key, value in answer_counts.items()},
        reader_work=dict(reader_work), finished_at_utc=utc_now(), wall_seconds=time.perf_counter() - start)
    if dict(completed) != counts or manifest["reader_calls"] != manifest["expected_reader_calls"]:
        raise ValueError("Final execution did not complete every declared retrieval and reader input")
    manifest["complete"] = True
    write_json(output / "evaluation.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", type=Path, default=[Path("data/processed/hidate/queries-exact.jsonl"),
                                                                    Path("data/processed/hidate/queries-paraphrase.jsonl")])
    parser.add_argument("--lawqa-queries", type=Path, default=Path("data/processed/hidate/lawqa-retrieval-queries.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/processed/hidate/lawqa-external.jsonl"))
    parser.add_argument("--discovery-root", type=Path, default=Path("runs/hidate/final-discovery"))
    parser.add_argument("--chunks", type=Path, default=Path("data/index/hidate/chunks.jsonl"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--policy", type=Path, default=Path("runs/hidate/policy/frozen.json"))
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidate/final"))
    args = parser.parse_args()
    evaluate_final(args.queries, args.lawqa_queries, args.questions, args.discovery_root, args.chunks,
        args.stack, args.policy, args.output_dir, ranker_path=args.ranker, preflight_path=args.preflight,
        expected_counts={"natural": 3172, "controlled": 3160, "lawqa": 420})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
