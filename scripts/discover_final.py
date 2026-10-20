"""Discover every frozen TEST/control/LawQA query after DEV policy selection."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def vector_bank(directory, identity_file):
    directory = Path(directory)
    manifest = read_json(directory / "encoding.json")
    if not manifest["complete"]:
        raise ValueError("Frozen corpus/query vector acquisition must be complete")
    identities = read_json(directory / identity_file)
    aliases = np.load(directory / "vector-row-indices.npy", mmap_mode="r", allow_pickle=False)
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if len(identities) != len(aliases) or len(identities) != len(set(identities)):
        raise ValueError("Frozen vector identities must have one alias row per unique ID")
    return {"manifest": manifest, "ids": identities, "aliases": aliases, "vectors": vectors}


def query_segmenter():
    from sudachipy import Dictionary, SplitMode
    tokenizer = Dictionary(dict="core").tokenizer()
    return lambda text: [m.normalized_form() for m in tokenizer.tokenize(text, SplitMode.A)]


def query_tokens(text, segmenter):
    return [token for token in segmenter(text)
            if any(not c.isspace() and not unicodedata.category(c).startswith("P") for c in token)]


def discover_final(index_dir, prepared_dir, source_dir, controls_dir, policy_path,
                   query_embeddings_dirs, output_dir, *, segmenter=None,
                   protocol_path=Path("configs/retrieval-protocol.json")):
    started = time.perf_counter()
    policy = read_json(policy_path)
    if not policy["complete"]:
        raise ValueError("The DEV policy must be frozen before final TEST/LawQA discovery")
    index, prepared, source, controls, output = map(Path,
        (index_dir, prepared_dir, source_dir, controls_dir, output_dir))
    protocol = read_json(protocol_path)
    index_manifest = read_json(index / "index.json")
    preparation = read_json(prepared / "preparation.json")
    control_manifest = read_json(controls / "controls.json")
    if not all(row["complete"] for row in (index_manifest, preparation, control_manifest)):
        raise ValueError("Final discovery requires complete index, preparation and source controls")
    if set(protocol["rank_features"]) != set(FEATURE_NAMES):
        raise ValueError("Discovery features differ from the frozen protocol")

    chunks = load_jsonl(index / "chunks.jsonl")
    tokens = {row["chunk_id"]: row["tokens"] for row in load_jsonl(index / "lexical-tokens.jsonl")}
    documents = vector_bank(index / "embeddings", "chunk-ids.json")
    if documents["ids"] != [chunk["chunk_id"] for chunk in chunks]:
        raise ValueError("Corpus vector identities must match the frozen chunk row order")
    pin = documents["manifest"]["model_pin"]
    query_banks = {name: vector_bank(query_embeddings_dirs[name], "query-ids.json")
                   for name in ("exact", "paraphrase", "lawqa")}
    for bank in query_banks.values():
        if any(bank["manifest"]["model_pin"][key] != pin[key] for key in ("model_id", "revision")):
            raise ValueError("Final query vectors must use the frozen document embedding pin")
        if bank["vectors"].shape[1] != documents["vectors"].shape[1]:
            raise ValueError("Final query and document vector dimensions disagree")
        bank["rows"] = {identity: int(row) for identity, row in zip(bank["ids"], bank["aliases"])}

    exact = [q for q in load_jsonl(prepared / "queries-exact.jsonl") if q["split"] == "test"]
    paraphrase = [q for q in load_jsonl(prepared / "queries-paraphrase.jsonl") if q["split"] == "test"]
    lawqa = load_jsonl(prepared / "lawqa-retrieval-queries.jsonl")
    exact_ids = {q["query_id"] for q in exact}
    if len(exact_ids) != len(exact) or len({q["base_query_id"] for q in paraphrase}) != len(paraphrase) or {
            q["base_query_id"] for q in paraphrase} != exact_ids:
        raise ValueError("Both final TEST query variants must cover the same complete base IDs")
    original_ids = {q["ファイル名"] for q in load_jsonl(prepared / "lawqa-external.jsonl")}
    expected_lawqa = {(identity, date) for identity in original_ids for date in protocol["LawQA_corpus_conditions"]}
    if len(lawqa) != len(expected_lawqa) or {(q["original_question_id"], q["target_asof"]) for q in lawqa} != expected_lawqa:
        raise ValueError("LawQA must retain every original question at every declared corpus condition")
    if any(q["source_role"] != "external" or q["split"] != "external" for q in lawqa):
        raise ValueError("LawQA discovery uses the external source-role bank")
    if any(q["source_role"] != "full" for q in exact + paraphrase):
        raise ValueError("Structural TEST discovery uses the full source-role bank")

    available = [q for q in exact if q["target_status"] == "available"]
    available_ids = {q["query_id"] for q in available}
    audits = {}
    for date in sorted({q["target_asof"] for q in available}):
        audit = read_json(controls / (date + ".json"))
        expected = {q["query_id"] for q in available if q["target_asof"] == date}
        if audit["target_date"] != date or set(audit["eligible_base_query_ids"]) != expected:
            raise ValueError("Source-control base ID coverage must match every available TEST query")
        if not set(audit["removed_chunk_ids"]).issubset(documents["ids"]):
            raise ValueError("Source controls reference chunks outside the frozen corpus")
        audits[date] = audit
    if control_manifest["eligible_queries"] != len(available):
        raise ValueError("Source-control query coverage disagrees with the frozen TEST queries")

    datasets = {
        "test-natural-exact": ("natural", "exact", exact),
        "test-natural-paraphrase": ("natural", "paraphrase", paraphrase),
        "test-controlled-exact": ("controlled", "exact", available),
        "test-controlled-paraphrase": ("controlled", "paraphrase", [q for q in paraphrase if q["base_query_id"] in available_ids]),
        "lawqa": ("lawqa", "lawqa", lawqa)}
    for _, kind, queries in datasets.values():
        ids = [q["query_id"] for q in queries]
        if len(ids) != len(set(ids)) or not set(ids).issubset(query_banks[kind]["rows"]):
            raise ValueError("Each final query ID requires exactly one saved query vector")
    target_index = {tuple(row["address_key"]): row for row in load_jsonl(prepared / "target-addresses.jsonl")}
    bindings = {(row["law_id"], row["requested_asof"]): row for row in load_jsonl(source / "asof-bindings.jsonl")}
    segmenter = query_segmenter() if segmenter is None else segmenter

    manifests, groups, expected_ids, seen_ids = {}, defaultdict(list), {}, defaultdict(set)
    for name, (scope, kind, queries) in datasets.items():
        expected_ids[name] = {q["query_id"] for q in queries}
        manifests[name] = {"complete": False, "evaluation_scope": scope, "query_rows": len(queries),
            "completed_query_rows": 0, "started_at_utc": utc_now(), "job_id": os.environ.get("PBS_JOBID"),
            "policy_path": str(Path(policy_path).resolve()), "policy_frozen_at_utc": policy.get("frozen_at_utc"),
            "model_pin": pin, "index_dir": str(index.resolve()), "prepared_dir": str(prepared.resolve()),
            "query_embeddings_dir": str(Path(query_embeddings_dirs[kind]).resolve()),
            "controls_dir": str(controls.resolve()) if scope == "controlled" else None,
            "features": list(FEATURE_NAMES), "top_k": protocol["discovery_top_k"], "rrf_k": protocol["rrf_k"],
            "seed": protocol["seed"], "family_split_path": str((prepared / "family-split.json").resolve()),
            "retrieval_runs": 0, "cache_hits": 0, "current_chunk_scoring_count": 0,
            "queries_by_variant": dict(Counter(q["query_variant"] for q in queries)),
            "queries_by_split": dict(Counter(q["split"] for q in queries)), "banks": {},
            "training_runs": 0, "model_forward_calls": 0, "model_inference_runs": 0,
            "outputs": {"candidates": "candidates.jsonl"}}
        for query in queries:
            groups[(scope, query["source_role"], query["target_asof"])].append((name, kind, query))
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as streams:
        writers = {}
        for name, manifest in manifests.items():
            write_json(output / name / "discovery.json", manifest)
            writers[name] = streams.enter_context((output / name / "candidates.jsonl").open("w", encoding="utf-8"))
        for (scope, role, date), queries in sorted(groups.items()):
            bank_chunks, ids, aliases, targets = chunks, documents["ids"], documents["aliases"], target_index
            overrides = {}
            if scope == "controlled":
                audit = audits[date]
                removed = set(audit["removed_chunk_ids"])
                positions = [i for i, identity in enumerate(ids) if identity not in removed]
                bank_chunks = [chunks[i] for i in positions]
                ids = [documents["ids"][i] for i in positions]
                aliases = documents["aliases"][positions]
                overrides = {tuple(row["address_key"]): row for row in audit["target_overrides"]}
                targets = {**target_index, **overrides}
            engine = DiscoveryEngine(bank_chunks, tokens, documents["vectors"], aliases, ids, targets, bindings,
                top_k=protocol["discovery_top_k"], rrf_k=protocol["rrf_k"])
            for name, kind, query in queries:
                vector = query_banks[kind]["vectors"][query_banks[kind]["rows"][query["query_id"]]]
                before = (engine.retrieval_runs, engine.cache_hits, engine.current_chunk_scoring_count)
                # Query labels are used only for frozen sample allocation above.
                request = {key: query[key] for key in ("query_id", "query_text", "query_variant", "split", "target_asof", "source_role")}
                row = {**engine.discover(request, query_tokens(query["query_text"], segmenter), vector),
                       "evaluation_scope": scope, "corpus_condition_asof": date}
                if scope == "lawqa":
                    row["original_question_id"] = query["original_question_id"]
                else:
                    row["base_query_id"] = query.get("base_query_id", query["query_id"])
                if overrides:
                    row["addresses"] = [{**address, "original_target_binding": target_index[tuple(address["address_key"])]}
                        if tuple(address.get("address_key", ())) in overrides else address for address in row["addresses"]]
                writers[name].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                seen_ids[name].add(query["query_id"])
                manifest = manifests[name]
                manifest["completed_query_rows"] += 1
                for field, after, previous in zip(("retrieval_runs", "cache_hits", "current_chunk_scoring_count"),
                        (engine.retrieval_runs, engine.cache_hits, engine.current_chunk_scoring_count), before):
                    manifest[field] += after - previous
                manifest["banks"][role + "|" + date] = row["bank_counts"]
                if manifest["completed_query_rows"] % 100 == 0:
                    writers[name].flush()
                    print(json.dumps({"dataset": name, "completed_query_rows": manifest["completed_query_rows"],
                                      "query_rows": manifest["query_rows"]}), flush=True)
            del engine
    for name, manifest in manifests.items():
        if seen_ids[name] != expected_ids[name] or manifest["completed_query_rows"] != manifest["query_rows"]:
            raise ValueError("Final discovery output must preserve every declared sample ID exactly once")
        manifest.update(complete=True, finished_at_utc=utc_now(), shared_execution_wall_seconds=time.perf_counter() - started)
        write_json(output / name / "discovery.json", manifest)
    summary = {"complete": True, "job_id": os.environ.get("PBS_JOBID"), "finished_at_utc": utc_now(),
        "policy_path": str(Path(policy_path).resolve()), "model_pin": pin, "seed": protocol["seed"],
        "corpus_chunk_rows": len(chunks), "unique_document_vectors": int(documents["vectors"].shape[0]),
        "query_rows": sum(m["query_rows"] for m in manifests.values()),
        "completed_query_rows": sum(m["completed_query_rows"] for m in manifests.values()),
        "retrieval_runs": sum(m["retrieval_runs"] for m in manifests.values()),
        "cache_hits": sum(m["cache_hits"] for m in manifests.values()),
        "current_chunk_scoring_count": sum(m["current_chunk_scoring_count"] for m in manifests.values()),
        "wall_seconds": time.perf_counter() - started, "training_runs": 0, "model_forward_calls": 0,
        "original_data_changed": False, "outputs": manifests}
    write_json(output / "discovery-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--controls-dir", type=Path, default=Path("runs/hidate/controlled-removal"))
    parser.add_argument("--policy", type=Path, default=Path("runs/hidate/policy/frozen.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidate/final-discovery"))
    parser.add_argument("--exact-query-vectors", type=Path, default=Path("data/index/hidate/queries-exact"))
    parser.add_argument("--paraphrase-query-vectors", type=Path, default=Path("data/index/hidate/queries-paraphrase"))
    parser.add_argument("--lawqa-query-vectors", type=Path, default=Path("data/index/hidate/queries-lawqa"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    args = parser.parse_args()
    summary = discover_final(args.index_dir, args.prepared_dir, args.source_dir, args.controls_dir,
        args.policy, {"exact": args.exact_query_vectors, "paraphrase": args.paraphrase_query_vectors,
                      "lawqa": args.lawqa_query_vectors}, args.output_dir, protocol_path=args.protocol)
    print(json.dumps({key: value for key, value in summary.items() if key != "outputs"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
