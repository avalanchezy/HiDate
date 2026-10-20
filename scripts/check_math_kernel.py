"""Check Ruri numerical kernels on a batch from the source corpus."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from hidate.models import RuriEncoder, RuriReranker


def main():
    stack = json.loads(Path("models/ruri/model-stack.json").read_text())
    texts, chunk_ids, seen = [], [], set()
    with Path("data/index/hidate/chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            texts.append(row["text"])
            chunk_ids.append(row["chunk_id"])
            if len(texts) == 32:
                break
    report = {"status": "RUNNING", "job_id": os.environ.get("PBS_JOBID"),
              "device": torch.cuda.get_device_name(0), "chunk_ids": chunk_ids,
              "embedding_pin": stack["models"]["embedding"]["revision"],
              "reranker_pin": stack["models"]["reranker"]["revision"]}
    output = Path("runs/hidate/pbs/math-kernel-check/check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = RuriEncoder(stack["models"]["embedding"]["local_path"])
    encoded = encoder.encode(texts, include_inputs=True)
    single = encoder.encode(texts[:1], include_inputs=True)
    vectors = np.asarray(encoded.pop("embeddings"))
    single_vector = np.asarray(single.pop("embeddings"))[0]
    report["encoder"] = {**encoded, "finite_rows": int(np.isfinite(vectors).all(axis=1).sum()),
        "norm_range": [float(x) for x in (np.linalg.norm(vectors, axis=1).min(),
                                         np.linalg.norm(vectors, axis=1).max())],
        "first_row_single_cosine": float(vectors[0] @ single_vector)}
    report["single_encoder"] = single
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    reranker = RuriReranker(stack["models"]["reranker"]["local_path"])
    # Natural short queries with the same long/short document padding trigger.
    scored = reranker.score([[text[:40], text] for text in texts], include_inputs=True)
    report["reranker"] = {**scored, "finite_scores": int(np.isfinite(scored["scores"]).sum())}
    passed = (vectors.shape == (32, 768) and np.isfinite(vectors).all()
              and np.isfinite(scored["scores"]).all() and scored["output_shape"] == [32, 1]
              and abs(np.linalg.norm(vectors, axis=1) - 1).max() < 1e-5)
    report["status"] = "PASS" if passed else "FAIL"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "encoder_finite_rows": report["encoder"]["finite_rows"],
        "encoder_backend": encoded.get("sdpa_backend"),
        "reranker_finite_scores": report["reranker"]["finite_scores"],
        "reranker_backend": scored.get("sdpa_backend")}), flush=True)
    if not passed:
        raise RuntimeError("The exact failed batch still contains invalid model outputs")


if __name__ == "__main__":
    main()
