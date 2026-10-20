"""Encode the frozen HiDate corpus once, retaining every scoped chunk identity."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json


BATCH_SIZE = 32
MAX_LENGTH = 1024
DIMENSION = 768


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encode_corpus(index_dir, stack_path, preflight_path, output_dir=None, *, encoder=None):
    """Reuse only identical complete chunk.text strings in this execution.

    Unique vectors are written directly to a NumPy memmap batch by batch. The
    corpus row mapping keeps the original chunks, scopes, dates, roles and IDs.
    """
    start = time.perf_counter()
    index = Path(index_dir)
    output = Path(output_dir) if output_dir is not None else index / "embeddings"
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    source_index = json.loads((index / "index.json").read_text(encoding="utf-8"))
    pin = stack["models"]["embedding"]
    dimension = pin.get("conventions", {}).get("embedding_dimension", DIMENSION)
    if not stack["complete"] or not pin["download_complete"] or not source_index["complete"]:
        raise ValueError("Frozen HiDate index and model acquisition must be complete")
    if preflight["status"] != "PASS":
        raise ValueError("HiDate real-model preflight must PASS before corpus encoding")
    verified_pin = preflight["model_pins"]["embedding"]
    if (verified_pin["model_id"], verified_pin["revision"]) != (pin["model_id"], pin["revision"]):
        raise ValueError("Corpus encoder model revision differs from the verified preflight")

    unique_texts, text_rows, chunk_ids, vector_rows, roles = [], {}, [], [], Counter()
    with (index / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = chunk["text"]
            if text not in text_rows:
                text_rows[text] = len(unique_texts)
                unique_texts.append(text)
            chunk_ids.append(chunk["chunk_id"])
            vector_rows.append(text_rows[text])
            roles[chunk["source_role"]] += 1
    if not unique_texts:
        raise ValueError("Frozen HiDate index has no chunks to encode")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "chunk-ids.json", chunk_ids)
    np.save(output / "vector-row-indices.npy", np.asarray(vector_rows, dtype=np.int64), allow_pickle=False)
    manifest = {"complete": False, "started_at_utc": utc_now(),
        "index_path": str((index / "index.json").resolve()), "chunks_path": str((index / "chunks.jsonl").resolve()),
        "stack_path": str(Path(stack_path).resolve()), "preflight_path": str(Path(preflight_path).resolve()),
        "preflight_job_id": preflight.get("job_id"),
        "model_pin": {key: pin[key] for key in ("model_id", "revision", "local_path")},
        "corpus_rows": len(chunk_ids), "unique_model_texts": len(unique_texts),
        "reused_exact_text_rows": len(chunk_ids) - len(unique_texts), "source_role_counts": dict(roles),
        "cache_policy": "exact complete chunk.text equality within this execution; first occurrence order",
        "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "dimension": dimension,
        "vectors_dtype": "float32", "vector_row_indices_dtype": "int64",
        "batch_count": 0, "encoded_unique_texts": 0, "encoded_input_tokens": 0,
        "padded_token_slots": 0, "encoder_wall_seconds_with_cuda_sync": 0.0,
        "outputs": {"vectors": "vectors.npy", "chunk_ids": "chunk-ids.json", "vector_row_indices": "vector-row-indices.npy"},
        "training_runs": 0, "query_encoding_runs": 0}
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
        result = encoder.encode(unique_texts[offset:end], kind="document")
        batch = np.asarray(result["embeddings"], dtype=np.float32)
        if batch.shape != (end - offset, dimension) or not np.isfinite(batch).all():
            raise ValueError("Corpus encoder returned an invalid vector batch shape or nonfinite values")
        if len(result["token_counts"]) != end - offset or sum(result["token_counts"]) != result["total_tokens"]:
            raise ValueError("Corpus encoder token receipt does not match the encoded batch")
        vectors[offset:end] = batch
        vectors.flush()
        manifest["batch_count"] += 1
        manifest["encoded_unique_texts"] = end
        manifest["encoded_input_tokens"] += result["total_tokens"]
        manifest["padded_token_slots"] += int(result["input_shape"][0] * result["input_shape"][1])
        manifest["encoder_wall_seconds_with_cuda_sync"] += result["seconds"]
        manifest["sdpa_backend"] = result.get("sdpa_backend")
        write_json(output / "encoding.json", manifest)
        if manifest["batch_count"] % 100 == 0 or end == len(unique_texts):
            print(json.dumps({"encoded_unique_texts": end, "unique_model_texts": len(unique_texts),
                "encoded_input_tokens": manifest["encoded_input_tokens"],
                "encoder_wall_seconds_with_cuda_sync": manifest["encoder_wall_seconds_with_cuda_sync"]}), flush=True)
        del batch, result
    del vectors
    manifest.update(complete=True, finished_at_utc=utc_now(), wall_seconds=time.perf_counter() - start,
                    vectors_shape=[len(unique_texts), dimension])
    write_json(output / "encoding.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    encode_corpus(args.index_dir, args.stack, args.preflight, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
