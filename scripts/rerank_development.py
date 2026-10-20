"""Rerank complete declared VALID discovery with identical frozen pair budgets."""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hidate.io import load_jsonl, write_json
from hidate.reranking import ALL_CONDITIONS, BASELINE_CONDITIONS, PairScoreCache, rerank_query


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rerank_development(queries_paths, discovery_paths, chunks_path, stack_path, output_dir, *,
                       reranker=None, ranker_path=None, conditions=None, splits=("valid",), preflight_path=None):
    if set(splits) != {"valid"}:
        raise ValueError("Development reranking permits only valid; final TEST requires a frozen later execution")
    started = time.perf_counter()
    ranker = json.loads(Path(ranker_path).read_text(encoding="utf-8")) if ranker_path is not None else None
    conditions = tuple(conditions) if conditions is not None else (ALL_CONDITIONS if ranker is not None else BASELINE_CONDITIONS)
    if "history_learned" in conditions and ranker is None:
        raise ValueError("history_learned requires --ranker with the frozen one-TRAIN fit")
    if ranker is not None and not ranker["complete"]:
        raise ValueError("The supplied TRAIN address ranker is not complete")
    queries = {query["query_id"]: query for path in queries_paths for query in load_jsonl(path) if query["split"] in splits}
    discoveries = [row for path in discovery_paths for row in load_jsonl(path) if row["query_id"] in queries]
    ids = [row["query_id"] for row in discoveries]
    if len(ids) != len(set(ids)) or set(ids) != set(queries):
        raise ValueError("Every declared VALID query must have exactly one complete discovery row")
    chunks = {chunk["chunk_id"]: chunk for chunk in load_jsonl(chunks_path)}
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    pin = stack["models"]["reranker"]
    if not stack["complete"] or not pin["download_complete"]:
        raise ValueError("Frozen Ruri reranker acquisition must be complete")
    model_pin = {key: pin[key] for key in ("model_id", "revision")}
    if preflight_path is not None:
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        verified_pin = {key: preflight["model_pins"]["reranker"][key] for key in ("model_id", "revision")}
        if preflight["status"] != "PASS" or verified_pin != model_pin:
            raise ValueError("Reranking pin must match the passed GPU preflight")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "started_at_utc": utc_now(), "splits": list(splits),
        "query_paths": [str(Path(path).resolve()) for path in queries_paths],
        "discovery_paths": [str(Path(path).resolve()) for path in discovery_paths],
        "chunks_path": str(Path(chunks_path).resolve()), "model_pin": model_pin,
        "model_local_path": pin["local_path"], "ranker_path": str(Path(ranker_path).resolve()) if ranker_path else None,
        "conditions": list(conditions), "query_rows": len(queries), "completed_query_rows": 0,
        "variants": dict(Counter(query["query_variant"] for query in queries.values())),
        "pair_budget": 10, "final_parent_budget": 4, "max_length": 1024, "physical_batch_size": 32,
        "candidate_policy": "deduplicate parent before ten from the same fifty; bound methods use one current chunk/address",
        "cache_policy": "exact (query_text, chunk.text) equality within this execution",
        "training_runs": 0, "outputs": {"reranked": "reranked.jsonl", "pair_receipts": "model-pair-receipts.jsonl"}}
    write_json(output / "scoring.json", manifest)
    if reranker is None:
        import torch
        from hidate.models import create_reranker
        reranker = create_reranker(pin, device="cuda", max_length=1024)
        manifest["device"] = {"name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                              "dtype": str(next(reranker.model.parameters()).dtype)}
        write_json(output / "scoring.json", manifest)
    condition_requests, condition_input_tokens, final_counts = Counter(), Counter(), Counter()
    with (output / "model-pair-receipts.jsonl").open("w", encoding="utf-8") as receipts, \
            (output / "reranked.jsonl").open("w", encoding="utf-8") as records:
        def save_receipt(record):
            receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
            receipts.flush()
        cache = PairScoreCache(reranker, receipt_callback=save_receipt)
        for discovery in discoveries:
            row = rerank_query(queries[discovery["query_id"]], discovery, chunks, cache,
                               conditions=conditions, ranker=ranker)
            records.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.flush()
            for condition, result in row["conditions"].items():
                condition_requests[condition] += result["requested_pair_count"]
                condition_input_tokens[condition] += result["requested_input_tokens"]
                final_counts[condition] += len(result["final_evidence"])
            manifest.update(completed_query_rows=manifest["completed_query_rows"] + 1,
                logical_pair_requests=cache.logical_pair_requests, physical_pair_count=cache.physical_pair_count,
                exact_pair_cache_reuses=cache.logical_pair_requests - cache.physical_pair_count,
                model_call_count=cache.model_call_count, encoded_input_tokens=cache.encoded_input_tokens,
                padded_token_slots=cache.padded_token_slots,
                model_wall_seconds_with_cuda_sync=cache.model_wall_seconds_with_cuda_sync,
                sdpa_backend=cache.sdpa_backend, condition_pair_requests=dict(condition_requests),
                condition_requested_input_tokens=dict(condition_input_tokens),
                final_parent_counts=dict(final_counts))
            write_json(output / "scoring.json", manifest)
            if manifest["completed_query_rows"] % 100 == 0 or manifest["completed_query_rows"] == len(queries):
                print(json.dumps({key: manifest[key] for key in
                    ("completed_query_rows", "query_rows", "logical_pair_requests", "physical_pair_count")}), flush=True)
    manifest.update(complete=manifest["completed_query_rows"] == len(queries), finished_at_utc=utc_now(),
                    wall_seconds=time.perf_counter() - started)
    write_json(output / "scoring.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", type=Path, required=True)
    parser.add_argument("--discovery", nargs="+", type=Path, required=True)
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS)
    parser.add_argument("--splits", nargs="+", choices=("valid",), default=["valid"])
    parser.add_argument("--chunks", type=Path, default=Path("data/index/hidate/chunks.jsonl"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidate/dev"))
    args = parser.parse_args()
    rerank_development(args.queries, args.discovery, args.chunks, args.stack, args.output_dir,
        ranker_path=args.ranker, conditions=args.conditions, splits=args.splits, preflight_path=args.preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
