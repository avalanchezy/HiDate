"""Unlabeled adapters for the original HiDate paraphraser and Ruri query encoder."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

from generate_paraphrases import generate_paraphrases
from encode_queries import encode_queries


ROOT = Path("data/processed/hidate-holdout/confirmatory")
RUNS = Path("runs/hidate/revision/test-queries")
INDEX = Path("data/index/hidate/holdout-test/queries")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_inputs(public_queries, output):
    """Use only released query text, dates and opaque IDs, never target labels."""
    with Path(public_queries).open(encoding="utf-8") as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    exact_path = output / "queries-exact-unlabeled.jsonl"
    with exact_path.open("w", encoding="utf-8") as stream:
        for row in source:
            if row["query_variant"] != "exact_quote" or not row["query_id"].startswith("HOLDOUT-"):
                raise ValueError("Use the new frozen confirmatory exact-query inputs")
            adapted = {k: row[k] for k in ("query_id", "query_text", "query_variant", "target_asof")}
            adapted.update(source_quote_text=row["query_text"], source_role="full", split="test",
                           evaluation_role="confirmatory_test")
            stream.write(json.dumps(adapted, ensure_ascii=False) + "\n")
    result = {"complete": True, "query_rows": len(source),
              "unique_source_quotes": len({r["query_text"] for r in source}),
              "source": str(public_queries), "output": str(exact_path), "seed": 42,
              "allowed_fields": ["query_id", "query_text", "query_variant", "target_asof", "source_quote_text", "source_role", "split", "evaluation_role"],
              "gold_labels_read": False, "sealed_files_read": False,
              "training_runs": 0, "model_inference_runs": 0, "evaluation_runs": 0}
    write(output / "input-preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "paraphrase", "encode"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, default=RUNS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    args = parser.parse_args()
    output = args.root / "unlabeled"
    if args.phase == "prepare":
        print(json.dumps(prepare_inputs(args.root / "queries-inputs.jsonl", output), indent=2))
        return
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    exact = output / "queries-exact-unlabeled.jsonl"
    paraphrase = output / "queries-paraphrase-unlabeled.jsonl"
    if args.phase == "paraphrase":
        prepare_inputs(args.root / "queries-inputs.jsonl", output)
        result = generate_paraphrases(exact, args.stack, paraphrase, args.run_dir / "paraphrases",
                                     resume=True, preflight_path=args.preflight)
        complete = result["complete"]
        summary = {k: result[k] for k in ("exact_queries", "unique_source_quotes", "completed_unique_quotes",
                   "generation_calls_this_run", "mapped_queries", "empty_outputs", "model_pin",
                   "prompt_tokens", "generated_tokens", "generation_wall_seconds_with_cuda_sync")}
    else:
        generation = read(args.run_dir / "paraphrases/generation.json")
        if not generation["complete"]:
            raise ValueError("Complete the unchanged greedy paraphrase protocol before encoding both variants")
        result = encode_queries([exact, paraphrase], args.stack, args.index,
                                variant="all", preflight_path=args.preflight)
        complete = result["complete"]
        summary = {k: result[k] for k in ("query_rows", "unique_model_texts", "encoded_unique_texts",
                   "variant_counts", "encoded_input_tokens", "padded_token_slots", "batch_count",
                   "model_pin", "vectors_shape", "encoder_wall_seconds_with_cuda_sync")}
    receipt = {"complete": complete, "phase": args.phase, "job_id": os.environ.get("PBS_JOBID"),
               "at_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
               "sealed_files_read": False, "gold_labels_read": False,
               "training_runs": 0, "retrieval_runs": 0, "evaluation_runs": 0, **summary}
    write(output / (args.phase + "-receipt.json"), receipt)
    write(output / "pipeline-paths.json", {"public_source_inputs": str(args.root / "queries-inputs.jsonl"),
        "exact_queries_unlabeled": str(exact), "paraphrase_queries_unlabeled": str(paraphrase),
        "generation_run_dir": str(args.run_dir / "paraphrases"), "query_embedding_index": str(args.index),
        "embedding_interface": ["vectors.npy", "query-ids.json", "query-texts.json", "vector-row-indices.npy", "encoding.json"],
        "retrieval_or_evaluation_started": False})
    print(json.dumps(receipt, indent=2), flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
