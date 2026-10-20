"""Generate frozen HiDate discovery candidates on an allocated CPU node."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import unicodedata

import numpy as np

from hidate.io import load_jsonl, write_json
from hidate.discovery import DiscoveryEngine, FEATURE_NAMES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def query_segmenter():
    from sudachipy import Dictionary, SplitMode
    tokenizer = Dictionary(dict="core").tokenizer()
    return lambda text: [morpheme.normalized_form() for morpheme in tokenizer.tokenize(text, SplitMode.A)]


def query_tokens(text, segmenter):
    # Same normalized-form and punctuation boundary as build_index.py.
    return [token for token in segmenter(text)
            if any(not character.isspace() and not unicodedata.category(character).startswith("P")
                   for character in token)]


def discover_candidates(index_dir, query_path, query_embeddings_dir, prepared_dir, source_dir,
                        output_dir, splits=("train", "valid"), *, segmenter=None,
                        protocol_path=Path("configs/retrieval-protocol.json")):
    start = time.perf_counter()
    index, prepared, source = Path(index_dir), Path(prepared_dir), Path(source_dir)
    query_embeddings, output = Path(query_embeddings_dir), Path(output_dir)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    index_manifest = json.loads((index / "index.json").read_text(encoding="utf-8"))
    preparation = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    encoding = json.loads((index / "embeddings/encoding.json").read_text(encoding="utf-8"))
    query_encoding = json.loads((query_embeddings / "encoding.json").read_text(encoding="utf-8"))
    if not all(record["complete"] for record in (index_manifest, preparation, encoding, query_encoding)):
        raise ValueError("Frozen HiDate index, preparation and both embedding artifacts must be complete")
    if any(encoding["model_pin"][key] != query_encoding["model_pin"][key] for key in ("model_id", "revision")):
        raise ValueError("Document and query embeddings must use the same frozen model revision")
    if set(protocol["rank_features"]) != set(FEATURE_NAMES):
        raise ValueError("Discovery features differ from the frozen retrieval protocol")
    chunks = load_jsonl(index / "chunks.jsonl")
    tokens = {row["chunk_id"]: row["tokens"] for row in load_jsonl(index / "lexical-tokens.jsonl")}
    ids = json.loads((index / "embeddings/chunk-ids.json").read_text(encoding="utf-8"))
    aliases = np.load(index / "embeddings/vector-row-indices.npy", mmap_mode="r", allow_pickle=False)
    vectors = np.load(index / "embeddings/vectors.npy", mmap_mode="r", allow_pickle=False)
    if ids != [chunk["chunk_id"] for chunk in chunks] or len(ids) != len(aliases):
        raise ValueError("Encoded corpus chunk identities do not match the frozen index")
    query_ids = json.loads((query_embeddings / "query-ids.json").read_text(encoding="utf-8"))
    query_aliases = np.load(query_embeddings / "vector-row-indices.npy", mmap_mode="r", allow_pickle=False)
    query_vectors = np.load(query_embeddings / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if len(query_ids) != len(query_aliases) or vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Query embedding identities or vector dimensions disagree")
    query_rows = {query_id: int(row) for query_id, row in zip(query_ids, query_aliases)}
    target_index = {tuple(row["address_key"]): row for row in load_jsonl(prepared / "target-addresses.jsonl")}
    bindings = {(row["law_id"], row["requested_asof"]): row for row in load_jsonl(source / "asof-bindings.jsonl")}
    engine = DiscoveryEngine(chunks, tokens, vectors, aliases, ids, target_index, bindings,
                             top_k=protocol["discovery_top_k"], rrf_k=protocol["rrf_k"])
    segmenter = segmenter if segmenter is not None else query_segmenter()
    selected = [row for row in load_jsonl(query_path) if row["split"] in splits]
    groups = defaultdict(list)
    for query in selected:
        groups[(query["source_role"], query["target_asof"])].append(query)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "started_at_utc": utc_now(), "job_id": os.environ.get("PBS_JOBID"),
        "index_dir": str(index.resolve()), "query_path": str(Path(query_path).resolve()),
        "query_embeddings_dir": str(query_embeddings.resolve()), "prepared_dir": str(prepared.resolve()),
        "bindings_path": str((source / "asof-bindings.jsonl").resolve()), "splits": list(splits),
        "model_pin": encoding["model_pin"], "protocol_path": str(Path(protocol_path).resolve()),
        "features": list(FEATURE_NAMES), "discovery_top_k": protocol["discovery_top_k"], "rrf_k": protocol["rrf_k"],
        "bm25": {"k1": 1.2, "b": .75, "idf": "positive Robertson"},
        "lexical_tokenization": index_manifest.get("lexical_tokenization"),
        "cache_key": ["query_text", "target_asof", "source_role"],
        "query_processing_order": "source_role,target_asof; original input order within each bank group",
        "query_vectors": "unique vectors with query-ID aliases; no model calls in discovery",
        "corpus_vectors": "unique vectors scored once per query evaluation; chunk aliases map scores only",
        "training_runs": 0, "model_inference_runs": 0, "queries": 0, "retrieval_runs": 0, "cache_hits": 0,
        "outputs": {"candidates": "candidates.jsonl"}}
    write_json(output / "discovery.json", manifest)
    address_counts, split_counts, variant_counts, banks = Counter(), Counter(), Counter(), {}
    with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for pool, queries in sorted(groups.items()):
            for query in queries:
                vector = query_vectors[query_rows[query["query_id"]]]
                row = engine.discover(query, query_tokens(query["query_text"], segmenter), vector)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest["queries"] += 1
                address_counts.update(address["status"] for address in row["addresses"])
                split_counts[row["split"]] += 1
                variant_counts[row["query_variant"]] += 1
                banks["|".join(pool)] = row["bank_counts"]
                if manifest["queries"] % 100 == 0:
                    stream.flush()
                    print(json.dumps({"queries": manifest["queries"], "retrieval_runs": engine.retrieval_runs,
                                      "cache_hits": engine.cache_hits}), flush=True)
    manifest.update(complete=True, finished_at_utc=utc_now(), wall_seconds=time.perf_counter() - start,
        retrieval_runs=engine.retrieval_runs, cache_hits=engine.cache_hits,
        current_chunk_scoring_count=engine.current_chunk_scoring_count,
        queries_by_split=dict(split_counts), queries_by_variant=dict(variant_counts),
        candidate_status_counts=dict(address_counts), banks=banks)
    write_json(output / "discovery.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--queries", type=Path, default=Path("data/processed/hidate/queries-exact.jsonl"))
    parser.add_argument("--query-embeddings-dir", type=Path, default=Path("data/index/hidate/queries-exact"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidate/discovery/exact-train-valid"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = parser.parse_args()
    discover_candidates(args.index_dir, args.queries, args.query_embeddings_dir, args.prepared_dir,
        args.source_dir, args.output_dir, splits=tuple(args.splits), protocol_path=args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
