"""The second encoder must use the already frozen chunk identities and spans."""

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_index_keeps_original_chunk_text_and_reports_its_own_token_limit(tmp_path):
    script = ROOT / "scripts/prepare_qwen3_index.py"
    assert script.exists(), "Qwen3 fixed-chunk index staging is not implemented"
    prepare = runpy.run_path(str(script))["prepare_index"]
    source = tmp_path / "original"
    source.mkdir()
    original = {"complete": True, "tokenizer": {"model_id": "cl-nagoya/ruri-v3-310m"},
                "chunking": {"max_body_tokens": 448, "overlap_tokens": 64}}
    (source / "index.json").write_text(json.dumps(original), encoding="utf-8")
    chunks = [{"chunk_id": "c1", "text": "短い。", "char_start": 0, "char_end": 3},
              {"chunk_id": "c2", "text": "長" * 1025, "char_start": 0, "char_end": 1025},
              {"chunk_id": "c3", "text": "短い。", "char_start": 0, "char_end": 3}]
    chunk_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks).encode("utf-8")
    (source / "chunks.jsonl").write_bytes(chunk_bytes)
    (source / "lexical-tokens.jsonl").write_text('{"chunk_id":"c1","tokens":["短い"]}\n', encoding="utf-8")
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"complete": True, "models": {"embedding": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B", "local_path": "fixture", "revision": "fixed"}}}), encoding="utf-8")

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [[1] * len(text) for text in texts]}

    output = tmp_path / "new-index"
    report = prepare(source, stack, output, tokenizer=Tokenizer())
    assert report["complete"]
    assert report["chunk_rows"] == 3 and report["unique_model_texts"] == 2
    assert report["over_limit_unique_texts"] == report["over_limit_chunk_rows"] == 1
    assert report["maximum_untruncated_tokens"] == 1025
    assert report["untruncated_unique_input_tokens"] == 1028
    assert report["expected_encoded_unique_input_tokens"] == 1027
    assert (source / "chunks.jsonl").read_bytes() == (output / "chunks.jsonl").read_bytes() == chunk_bytes
    staged = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert staged["tokenizer"] == original["tokenizer"] and staged["chunking"] == original["chunking"]
