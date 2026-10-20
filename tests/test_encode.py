"""Fake encoder fixtures verify exact-text reuse and persisted corpus mapping."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/encode_corpus.py"


def encode():
    assert SCRIPT.exists(), "HiDate corpus encoder is not implemented"
    return runpy.run_path(str(SCRIPT))["encode_corpus"]


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, kind):
        assert kind == "document"
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * 768
            vector[0], vector[1] = float(len(text)), float(ord(text[-1]))
            vectors.append(vector)
        return {"embeddings": vectors, "token_counts": [7] * len(texts),
                "total_tokens": 7 * len(texts), "seconds": 0.5,
                "input_shape": [len(texts), 10]}


def source_files(tmp_path, chunks, preflight_status="PASS"):
    index = tmp_path / "index"
    index.mkdir()
    (index / "chunks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks), encoding="utf-8")
    (index / "index.json").write_text(json.dumps({"complete": True, "chunk_count": len(chunks)}), encoding="utf-8")
    pin = {"model_id": "cl-nagoya/ruri-v3-310m", "revision": "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
           "local_path": "/fixed/models/ruri/embedding", "download_complete": True,
           "conventions": {"embedding_dimension": 768}}
    stack = tmp_path / "model-stack.json"
    stack.write_text(json.dumps({"complete": True, "models": {"embedding": pin}}), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"status": preflight_status, "job_id": "preflight-fixture",
                                    "model_pins": {"embedding": pin}}), encoding="utf-8")
    return index, stack, preflight


def test_identical_model_text_encoded_once_without_merging_chunk_identity_or_role(tmp_path):
    chunks = [
        {"chunk_id": "R2|LAW:MainProvision:1|chunk0", "text": "試験法 第一条\n本文。", "source_role": "full", "split": "train"},
        {"chunk_id": "guidance-01|chunk0", "text": "指針\n案内。", "source_role": "external", "split": "external"},
        {"chunk_id": "R1|LAW:MainProvision:1|chunk0", "text": "試験法 第一条\n本文。", "source_role": "full", "split": "test"},
        {"chunk_id": "OTHER|LAW:MainProvision:1|chunk0", "text": "試験法 第一条\n本文。 ", "source_role": "external", "split": "external"},
    ]
    index, stack, preflight = source_files(tmp_path, chunks)
    original = (index / "chunks.jsonl").read_bytes()
    model = FakeEncoder()
    result = encode()(index, stack, preflight, encoder=model)
    output = index / "embeddings"
    assert model.calls == [["試験法 第一条\n本文。", "指針\n案内。", "試験法 第一条\n本文。 "]]
    ids = json.loads((output / "chunk-ids.json").read_text(encoding="utf-8"))
    assert ids == [row["chunk_id"] for row in chunks]
    mapping = np.load(output / "vector-row-indices.npy")
    assert mapping.tolist() == [0, 1, 0, 2]
    assert np.issubdtype(mapping.dtype, np.integer)
    vectors = np.load(output / "vectors.npy", mmap_mode="r")
    assert isinstance(vectors, np.memmap) and vectors.shape == (3, 768)
    assert vectors.dtype == np.float32
    assert np.array_equal(vectors[mapping[0]], vectors[mapping[2]])
    assert not np.array_equal(vectors[mapping[0]], vectors[mapping[3]])
    assert [(row["chunk_id"], row["source_role"], row["split"]) for row in chunks] == [
        (ids[0], "full", "train"), (ids[1], "external", "external"),
        (ids[2], "full", "test"), (ids[3], "external", "external")]
    assert (index / "chunks.jsonl").read_bytes() == original
    stored = json.loads((output / "encoding.json").read_text(encoding="utf-8"))
    assert result["complete"] is stored["complete"] is True
    assert stored["corpus_rows"] == 4 and stored["unique_model_texts"] == 3
    assert stored["encoded_input_tokens"] == 21 and stored["encoder_wall_seconds_with_cuda_sync"] == 0.5
    assert stored["padded_token_slots"] == 30
    assert stored["model_pin"]["revision"] == "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"


def test_fixed_batches_write_incrementally_and_sum_actual_encoder_receipts(tmp_path):
    chunks = [{"chunk_id": f"law|chunk{i}", "text": f"本文{i}。", "source_role": "full"} for i in range(35)]
    chunks.append({"chunk_id": "external-copy|chunk0", "text": "本文0。", "source_role": "external"})
    index, stack, preflight = source_files(tmp_path, chunks)
    model = FakeEncoder()
    result = encode()(index, stack, preflight, encoder=model)
    assert list(map(len, model.calls)) == [32, 3]
    assert result["batch_size"] == 32 and result["max_length"] == 1024
    assert result["batch_count"] == 2 and result["encoded_unique_texts"] == 35
    assert result["encoded_input_tokens"] == 245 and result["encoder_wall_seconds_with_cuda_sync"] == 1.0
    assert result["padded_token_slots"] == 350
    assert result["source_role_counts"] == {"full": 35, "external": 1}
    output = index / "embeddings"
    assert np.load(output / "vectors.npy", mmap_mode="r").shape == (35, 768)
    assert np.load(output / "vector-row-indices.npy").tolist() == list(range(35)) + [0]


def test_failed_or_different_preflight_stops_before_encoder_call(tmp_path):
    chunks = [{"chunk_id": "law|chunk0", "text": "本文。", "source_role": "full"}]
    index, stack, preflight = source_files(tmp_path, chunks, preflight_status="FAIL")
    model = FakeEncoder()
    with pytest.raises(ValueError, match="preflight"):
        encode()(index, stack, preflight, encoder=model)
    assert model.calls == []
    report = json.loads(preflight.read_text(encoding="utf-8"))
    report["status"] = "PASS"
    report["model_pins"]["embedding"]["revision"] = "a different public revision"
    preflight.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        encode()(index, stack, preflight, encoder=model)
    assert model.calls == []
