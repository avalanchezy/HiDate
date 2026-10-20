"""Read back all literal HiDate chunks and encoded identity aliases on a PBS CPU."""

import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    folder = Path("data/index/hidate")
    encoding = json.loads((folder / "embeddings/encoding.json").read_text())
    index = json.loads((folder / "index.json").read_text())
    assert encoding["complete"] and index["complete"]
    assert encoding["sdpa_backend"] == "math"
    prepared = Path("data/processed/hidate")
    parents = {row["passage_id"]: row for row in records(prepared / "corpus-passages.jsonl")}
    parents.update({row["source_id"]: row for row in records(prepared / "guidance-documents.jsonl")})
    tokenizer = AutoTokenizer.from_pretrained(index["tokenizer"]["local_path"], local_files_only=True)
    ids = json.loads((folder / "embeddings/chunk-ids.json").read_text())
    aliases = np.load(folder / "embeddings/vector-row-indices.npy", allow_pickle=False)
    vectors = np.load(folder / "embeddings/vectors.npy", mmap_mode="r", allow_pickle=False)
    lexical = list(records(folder / "lexical-tokens.jsonl"))
    text_rows, unique_texts, used_parents = {}, [], set()
    total_tokens = max_length = count = 0
    token_lengths = {}
    for row_number, chunk in enumerate(records(folder / "chunks.jsonl")):
        parent = parents[chunk["parent_passage_id"]]
        guidance = chunk["source_kind"] == "guidance"
        body = parent["text"] if guidance else parent["visible_text"]
        assert chunk["body_text"] == body[chunk["char_start"]:chunk["char_end"]]
        fields = ("title",) if guidance else ("law_title", "title", "caption")
        header = " ".join(parent.get(key, "") for key in fields if parent.get(key))
        assert chunk["header_text"] == header
        assert chunk["text"] == (header + "\n" + chunk["body_text"] if header else chunk["body_text"])
        assert chunk["token_end"] - chunk["token_start"] == chunk["body_token_count"] <= 448
        assert ids[row_number] == chunk["chunk_id"] == lexical[row_number]["chunk_id"]
        text = chunk["text"]
        if text not in text_rows:
            text_rows[text] = len(unique_texts)
            unique_texts.append(text)
            token_lengths[text] = len(tokenizer("検索文書: " + text, add_special_tokens=True,
                                                truncation=False)["input_ids"])
        assert int(aliases[row_number]) == text_rows[text]
        assert token_lengths[text] == chunk["ruri_document_tokens"] <= 1024
        max_length = max(max_length, token_lengths[text])
        total_tokens += len(lexical[row_number]["tokens"])
        used_parents.add(chunk["parent_passage_id"])
        count += 1
    assert count == len(ids) == len(aliases) == len(lexical) == encoding["corpus_rows"]
    assert vectors.shape == (len(unique_texts), 768) == tuple(encoding["vectors_shape"])
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    assert abs(norms - 1).max() < 1e-5
    assert encoding["encoded_input_tokens"] == sum(token_lengths.values())
    assert encoding["encoded_unique_texts"] == len(unique_texts)
    assert count == index["statistics"]["chunks"]
    assert total_tokens == index["statistics"]["lexical_tokens"]
    assert len(used_parents) == index["statistics"]["parents"]
    report = {"status": "PASS", "job_id": os.environ.get("PBS_JOBID"),
        "chunks": count, "parents": len(used_parents), "unique_model_texts": len(unique_texts),
        "all_literal_spans_and_headers_match": True, "all_ids_and_text_aliases_match": True,
        "lexical_tokens": total_tokens, "max_prefixed_tokens": max_length,
        "encoded_input_tokens": sum(token_lengths.values()), "all_vectors_finite": True,
        "vector_norm_range": [float(norms.min()), float(norms.max())], "sdpa_backend": "math"}
    output = Path("runs/hidate/pbs/index-readback/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
