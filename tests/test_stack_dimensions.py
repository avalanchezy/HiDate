"""Encoding drivers must persist the dimension declared by their model stack."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


class FourDimensionalEncoder:
    def encode(self, texts, **kwargs):
        return {"embeddings": [[1., 0., 0., 0.] for _ in texts], "token_counts": [3] * len(texts),
                "total_tokens": 3 * len(texts), "seconds": .1,
                "input_shape": [len(texts), 3], "sdpa_backend": "default"}


@pytest.mark.parametrize("driver", ["corpus", "queries"])
def test_encoding_driver_uses_declared_dimension_for_persisted_vectors(tmp_path, driver):
    pin = {"model_id": "Qwen/Qwen3-Embedding-0.6B", "local_path": "fixture", "revision": "fixed",
           "download_complete": True, "conventions": {"embedding_dimension": 4}}
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"complete": True, "models": {"embedding": pin}}), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"status": "PASS", "model_pins": {"embedding": pin}}), encoding="utf-8")
    output = tmp_path / "output"
    if driver == "corpus":
        index = tmp_path / "index"
        index.mkdir()
        (index / "index.json").write_text('{"complete": true}', encoding="utf-8")
        (index / "chunks.jsonl").write_text(json.dumps({"chunk_id": "C1", "text": "条文。", "source_role": "full"}) + "\n", encoding="utf-8")
        encode = runpy.run_path(str(ROOT / "scripts/encode_corpus.py"))["encode_corpus"]
        result = encode(index, stack, preflight, output, encoder=FourDimensionalEncoder())
    else:
        queries = tmp_path / "queries.jsonl"
        queries.write_text(json.dumps({"query_id": "Q1", "query_variant": "exact_quote", "query_text": "質問。",
                                       "source_role": "full", "split": "valid"}) + "\n", encoding="utf-8")
        encode = runpy.run_path(str(ROOT / "scripts/encode_queries.py"))["encode_queries"]
        result = encode([queries], stack, output, encoder=FourDimensionalEncoder(), preflight_path=preflight)
    assert result["complete"] and result["dimension"] == 4
    assert result["vectors_shape"] == [1, 4]
    assert np.load(output / "vectors.npy").tolist() == [[1., 0., 0., 0.]]
