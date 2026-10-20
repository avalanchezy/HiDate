"""Stage the original HiDate chunks unchanged and count Qwen3 truncation exposure."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def prepare_index(source_dir, stack_path, output_dir, *, tokenizer=None):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    original = json.loads((source / "index.json").read_text(encoding="utf-8"))
    if not stack["complete"] or not original["complete"]:
        raise ValueError("Both the original fixed index and Qwen3 acquisition must be complete")
    pin = stack["models"]["embedding"]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pin["local_path"], local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("chunks.jsonl", "lexical-tokens.jsonl"):
        target = output / name
        if not target.exists():
            os.link(source / name, target)
    staged = {**original, "fixed_chunk_source": str(source),
              "embedding_stack_path": str(Path(stack_path).resolve()),
              "chunk_definition": "original Ruri tokenizer boundaries, IDs, spans and lexical tokens retained verbatim"}
    (output / "index.json").write_text(json.dumps(staged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter()
    with (source / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                counts[json.loads(line)["text"]] += 1
    texts = list(counts)
    lengths = []
    for offset in range(0, len(texts), 256):
        batch = tokenizer(texts[offset:offset+256], padding=False, truncation=False,
                          return_attention_mask=False, return_token_type_ids=False)
        lengths.extend(len(ids) for ids in batch["input_ids"])
    limit = 1024
    report = {"complete": True, "job_id": os.environ.get("PBS_JOBID"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_chunk_source": str(source), "index_dir": str(output),
        "model_pin": {key: pin[key] for key in ("model_id", "revision")},
        "chunk_rows": sum(counts.values()), "unique_model_texts": len(texts),
        "maximum_untruncated_tokens": max(lengths), "model_input_limit": limit,
        "over_limit_unique_texts": sum(length > limit for length in lengths),
        "over_limit_chunk_rows": sum(counts[text] for text, length in zip(texts, lengths) if length > limit),
        "untruncated_unique_input_tokens": sum(lengths),
        "expected_encoded_unique_input_tokens": sum(min(length, limit) for length in lengths),
        "encoding_over_limit_policy": "Qwen3 tokenizer truncates to the same fixed 1024-token encoder limit; original chunk text is retained",
        "training_runs": 0, "model_forward_calls": 0}
    (output / "qwen3-token-lengths.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--stack", type=Path, default=Path("models/qwen3/model-stack.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/hidate-qwen3"))
    args = parser.parse_args()
    print(json.dumps(prepare_index(args.source_index, args.stack, args.output_dir)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
