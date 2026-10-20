"""Build the HiDate chunk and Japanese lexical index on CPU without model inference."""

import argparse
import json
import platform
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import chain
from pathlib import Path

from hidate.io import write_json
from hidate.retrieval import chunk_passages


def jsonl_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sudachi_segmenter():
    """The same normalized-form boundary is used for index and query tokenization."""
    from sudachipy import Dictionary, SplitMode

    dictionary = Dictionary(dict="core")
    tokenizer = dictionary.tokenizer()

    def normalized_forms(text):
        return [morpheme.normalized_form() for morpheme in tokenizer.tokenize(text, SplitMode.A)]

    return normalized_forms


def japanese_tokens(text, segmenter):
    return [token for token in segmenter(text)
            if any(not character.isspace() and not unicodedata.category(character).startswith("P")
                   for character in token)]


def installed_versions():
    return {**{package: version(package) for package in
        ("SudachiPy", "SudachiDict-core", "transformers", "tokenizers", "numpy")},
        "python": platform.python_version()}


def build_index(prepared_dir, stack_path, output_dir, *, tokenizer=None, segmenter=None,
                package_versions=None, max_body_tokens=448, overlap_tokens=64):
    prepared, output = Path(prepared_dir), Path(output_dir)
    preparation = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    if not preparation["complete"] or not stack["complete"]:
        raise ValueError("HiDate preparation and model acquisition must be complete before indexing")
    embedding = stack["models"]["embedding"]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(embedding["local_path"], use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("HiDate chunking requires a fast tokenizer with source character offsets")
    segmenter = segmenter or sudachi_segmenter()
    package_versions = package_versions if package_versions is not None else installed_versions()
    prefix = embedding["conventions"]["document_prefix"]
    limit = 1024
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared.resolve()), "stack_path": str(Path(stack_path).resolve()),
        "source_preparation_path": str((prepared / "preparation.json").resolve()),
        "source_corpus_paths": [str((prepared / name).resolve()) for name in
            ("corpus-passages.jsonl", "guidance-documents.jsonl")],
        "tokenizer": {"model_id": embedding["model_id"], "revision": embedding["revision"],
            "local_path": embedding["local_path"], "is_fast": tokenizer.is_fast},
        "chunking": {"max_body_tokens": max_body_tokens, "overlap_tokens": overlap_tokens,
            "offsets": "fast tokenizer; no special tokens; literal source body spans"},
        "document_prefix": prefix, "ruri_input_limit": limit,
        "lexical_tokenization": {"implementation": "SudachiPy", "dictionary": "SudachiDict-core",
            "split_mode": "A", "form": "normalized_form", "filter": "omit whitespace/punctuation-only tokens",
            "document_text": "natural header plus literal chunk body; no Ruri prefix",
            "query_interface": "japanese_tokens(query_text, sudachi_segmenter())",
            "documentation_url": "https://github.com/WorksApplications/sudachi.rs/tree/develop/python"},
        "package_versions": package_versions, "training_runs": 0, "model_inference_runs": 0,
        "outputs": {"chunks": "chunks.jsonl", "lexical_tokens": "lexical-tokens.jsonl"}}
    write_json(output / "index.json", manifest)
    records = chain(jsonl_records(prepared / "corpus-passages.jsonl"),
                    jsonl_records(prepared / "guidance-documents.jsonl"))
    chunks = chunk_passages(records, tokenizer, max_body_tokens=max_body_tokens, overlap_tokens=overlap_tokens)
    roles, splits, kinds, parents = Counter(), Counter(), Counter(), Counter()
    max_prefixed_length, lexical_token_count, over_limit = 0, 0, 0
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as chunk_stream, \
            (output / "lexical-tokens.jsonl").open("w", encoding="utf-8") as lexical_stream:
        for chunk in chunks:
            input_ids = tokenizer(prefix + chunk["text"], add_special_tokens=True, truncation=False)["input_ids"]
            chunk["ruri_document_tokens"] = len(input_ids)
            max_prefixed_length = max(max_prefixed_length, len(input_ids))
            over_limit += len(input_ids) > limit
            tokens = japanese_tokens(chunk["text"], segmenter)
            lexical_token_count += len(tokens)
            chunk_stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            lexical_stream.write(json.dumps({"chunk_id": chunk["chunk_id"], "tokens": tokens}, ensure_ascii=False) + "\n")
            roles[chunk["source_role"]] += 1
            splits[chunk["split"]] += 1
            kinds[chunk["source_kind"]] += 1
            parents[chunk["parent_passage_id"]] += 1
    manifest.update(statistics={"chunks": len(chunks), "parents": len(parents),
        "chunks_by_role": dict(roles), "chunks_by_split": dict(splits), "chunks_by_kind": dict(kinds),
        "chunks_by_parent": dict(parents), "lexical_tokens": lexical_token_count,
        "max_ruri_prefixed_document_tokens": max_prefixed_length, "chunks_over_ruri_limit": over_limit},
        all_chunks_fit_ruri_limit=over_limit == 0, finished_at_utc=datetime.now(timezone.utc).isoformat())
    manifest["complete"] = over_limit == 0
    write_json(output / "index.json", manifest)
    if over_limit:
        raise ValueError(f"{over_limit} HiDate chunks exceed the Ruri {limit}-token prefixed input limit")
    print(json.dumps({key: value for key, value in manifest["statistics"].items()
                      if key != "chunks_by_parent"}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/hidate"))
    args = parser.parse_args()
    build_index(args.prepared_dir, args.stack, args.output_dir)


if __name__ == "__main__":
    main()
