"""Encode every frozen query ID, reusing only exactly identical query text."""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hidate.io import load_jsonl, write_json


BATCH_SIZE = 32
MAX_LENGTH = 1024
DIMENSION = 768


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encode_queries(queries_paths, stack_path, output_dir, *, encoder=None, variant="all", preflight_path=None):
    started = time.perf_counter()
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    pin = stack["models"]["embedding"]
    dimension = pin.get("conventions", {}).get("embedding_dimension", DIMENSION)
    if not stack["complete"] or not pin["download_complete"]:
        raise ValueError("HiDate frozen query encoder acquisition must be complete")
    model_pin = {key: pin[key] for key in ("model_id", "revision")}
    if preflight_path is not None:
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        verified_pin = {key: preflight["model_pins"]["embedding"][key] for key in ("model_id", "revision")}
        if preflight["status"] != "PASS" or verified_pin != model_pin:
            raise ValueError("HiDate query encoder pin must match its passed GPU preflight")
    queries = [query for path in queries_paths for query in load_jsonl(path)
               if variant == "all" or query["query_variant"] == variant]
    if not queries:
        raise ValueError("HiDate query encoding has no rows for the selected variant")
    if any(not query["query_text"].strip() for query in queries):
        raise ValueError("HiDate query text is empty; preserve the generation error instead of substituting text")
    unique_texts, text_rows, vector_rows, query_metadata = [], {}, [], []
    for query in queries:
        text = query["query_text"]
        if text not in text_rows:
            text_rows[text] = len(unique_texts)
            unique_texts.append(text)
        row = text_rows[text]
        vector_rows.append(row)
        query_metadata.append({"query_id": query["query_id"], "base_query_id": query.get("base_query_id"),
            "query_variant": query["query_variant"], "source_role": query["source_role"],
            "split": query["split"], "vector_row": row})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "query-ids.json", [query["query_id"] for query in queries])
    write_json(output / "query-texts.json", unique_texts)
    np.save(output / "vector-row-indices.npy", np.asarray(vector_rows, dtype=np.int64), allow_pickle=False)
    manifest = {"complete": False, "started_at_utc": utc_now(), "variant": variant,
        "queries_paths": [str(Path(path).resolve()) for path in queries_paths],
        "stack_path": str(Path(stack_path).resolve()),
        "preflight_path": str(Path(preflight_path).resolve()) if preflight_path else None,
        "model_pin": model_pin, "model_local_path": pin["local_path"], "query_rows": len(queries),
        "unique_model_texts": len(unique_texts), "reused_exact_text_rows": len(queries) - len(unique_texts),
        "variant_counts": dict(Counter(query["query_variant"] for query in queries)),
        "source_role_counts": dict(Counter(query["source_role"] for query in queries)),
        "split_counts": dict(Counter(query["split"] for query in queries)), "query_metadata": query_metadata,
        "cache_policy": "exact query_text equality within this execution; first occurrence vector order",
        "input_kind": "query", "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "dimension": dimension,
        "batch_count": 0, "encoded_unique_texts": 0, "encoded_input_tokens": 0, "padded_token_slots": 0,
        "encoder_wall_seconds_with_cuda_sync": 0.0, "training_runs": 0, "query_encoding_runs": 1,
        "outputs": {"vectors": "vectors.npy", "query_ids": "query-ids.json", "query_texts": "query-texts.json",
                    "vector_row_indices": "vector-row-indices.npy"}}
    write_json(output / "encoding.json", manifest)
    if encoder is None:
        import torch
        from hidate.models import create_encoder
        encoder = create_encoder(pin, device="cuda", max_length=MAX_LENGTH)
        manifest["device"] = {"name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                              "dtype": str(next(encoder.model.parameters()).dtype)}
    vectors = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype=np.float32,
                                       shape=(len(unique_texts), dimension))
    for offset in range(0, len(unique_texts), BATCH_SIZE):
        end = min(offset + BATCH_SIZE, len(unique_texts))
        result = encoder.encode(unique_texts[offset:end], kind="query", include_inputs=False)
        batch = np.asarray(result["embeddings"], dtype=np.float32)
        if batch.shape != (end - offset, dimension) or not np.isfinite(batch).all():
            raise ValueError("Query encoder returned an invalid shape or nonfinite vector values")
        if len(result["token_counts"]) != end - offset or sum(result["token_counts"]) != result["total_tokens"]:
            raise ValueError("Query encoder token receipt does not match the encoded rows")
        vectors[offset:end] = batch
        vectors.flush()
        manifest["batch_count"] += 1
        manifest["encoded_unique_texts"] = end
        manifest["encoded_input_tokens"] += result["total_tokens"]
        manifest["padded_token_slots"] += int(result["input_shape"][0] * result["input_shape"][1])
        manifest["encoder_wall_seconds_with_cuda_sync"] += result["seconds"]
        manifest["sdpa_backend"] = result["sdpa_backend"]
        write_json(output / "encoding.json", manifest)
        if manifest["batch_count"] % 100 == 0 or end == len(unique_texts):
            print(json.dumps({key: manifest[key] for key in
                ("encoded_unique_texts", "unique_model_texts", "encoded_input_tokens", "sdpa_backend")}), flush=True)
    del vectors
    manifest.update(complete=True, finished_at_utc=utc_now(), wall_seconds=time.perf_counter() - started,
                    vectors_shape=[len(unique_texts), dimension])
    write_json(output / "encoding.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, nargs="+")
    parser.add_argument("--variant", choices=("all", "exact_quote", "generated_paraphrase"), default="all")
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/hidate/queries"))
    args = parser.parse_args()
    paths = args.queries or ([Path("data/processed/hidate/queries-exact.jsonl")] if args.variant == "exact_quote"
        else [Path("data/processed/hidate/queries-paraphrase.jsonl")] if args.variant == "generated_paraphrase"
        else [Path("data/processed/hidate/queries-exact.jsonl"), Path("data/processed/hidate/queries-paraphrase.jsonl")])
    encode_queries(paths, args.stack, args.output_dir, variant=args.variant, preflight_path=args.preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
