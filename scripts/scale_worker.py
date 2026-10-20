"""Prepare, extend-encode, and evaluate nested law banks without changing HiDate TRAIN/VALID."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from hidate.io import write_json
from hidate.scale import SharedCosines, add_passage, metric_panels, nested_law_banks, snapshot_plan


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def row_write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def receipt(**values):
    return {"job_id": os.environ.get("PBS_JOBID"), "seed": 42,
            "at_utc": datetime.now(timezone.utc).isoformat(), "training_runs": 0,
            "ranker_fits": 0, "dev_selections": 0, "reader_calls": 0, **values}


def prepare(args, protocol):
    from transformers import AutoTokenizer
    from build_index import installed_versions, japanese_tokens, sudachi_segmenter
    from prepare_data import target_status
    from hidate.data import visible_provision
    from hidate.retrieval import chunk_passages
    from hidate.sources import extract_provisions

    output, extension = args.index, args.extension
    output.mkdir(parents=True, exist_ok=True)
    original = read(args.original_source / "selected-laws.json")
    selected = read(extension / "selected-laws.json")
    banks = nested_law_banks(selected, original, sizes=protocol["law_scales"])
    plan = snapshot_plan(selected, original, protocol["anchor_dates"])
    if len(original) != 120 or len(selected) != 1000 or len(plan) != 880 * 4:
        raise ValueError("Scale preparation requires the unchanged 120-law prefix and 880x4 new snapshots")
    missing = [f"{law_id}/{anchor}" for law_id, anchor in plan
               if not (extension / "snapshots" / law_id / (anchor + ".json")).is_file()]
    if missing:
        raise ValueError(f"Required scale snapshots not yet acquired: {len(missing)}; first={missing[:5]}")
    old_index, stack = read(args.original_index / "index.json"), read(args.stack)
    pin = stack["models"]["embedding"]
    if not old_index["complete"] or not stack["complete"] or any(old_index["tokenizer"][key] != pin[key] for key in ("model_id", "revision")):
        raise ValueError("Use the original complete Ruri index and its unchanged tokenizer/model pin")
    tokenizer = AutoTokenizer.from_pretrained(pin["local_path"], local_files_only=True, use_fast=True)
    segmenter = sudachi_segmenter()
    prefix = pin["conventions"]["document_prefix"]
    # Preserve original chunk IDs, literal offsets, boundaries, lexical values,
    # and original reserved/external rows byte-for-byte before appending extras.
    for name in ("chunks.jsonl", "lexical-tokens.jsonl"):
        shutil.copyfile(args.original_index / name, output / name)
    shutil.copyfile(args.original_prepared / "target-addresses.jsonl", output / "target-addresses.jsonl")
    shutil.copyfile(args.original_source / "asof-bindings.jsonl", output / "asof-bindings.jsonl")
    shutil.copyfile(args.original_source / "source-exclusions.json", output / "source-exclusions.json")
    shutil.copyfile(args.original_prepared / "family-split.json", output / "original-family-split.json")
    write_json(output / "nested-corpora.json", banks)
    original_ids = {row["law_info"]["law_id"] for row in original}
    old_bindings = {(r["law_id"], r["requested_asof"]): r for r in rows(output / "asof-bindings.jsonl")}
    if any(old_bindings.get((identity, anchor), {}).get("status") != "collected"
           for identity in original_ids for anchor in protocol["anchor_dates"]):
        raise ValueError("Original 120-law four-anchor source bindings must be retained and complete")
    stats = Counter()
    manifest = receipt(complete=False, phase="parse_new_laws", protocol=protocol,
        original_source=str(args.original_source), extension_source=str(extension),
        original_index=str(args.original_index), original_prepared=str(args.original_prepared),
        tokenizer={k: pin[k] for k in ("model_id", "revision", "local_path")},
        original_chunk_rows=old_index["statistics"]["chunks"], package_versions=installed_versions())
    write_json(output / "index.json", manifest)
    with (output / "chunks.jsonl").open("a", encoding="utf-8") as chunks_out, \
            (output / "lexical-tokens.jsonl").open("a", encoding="utf-8") as lexical_out, \
            (output / "target-addresses.jsonl").open("a", encoding="utf-8") as targets_out, \
            (output / "asof-bindings.jsonl").open("a", encoding="utf-8") as bindings_out, \
            (output / "extra-corpus-passages.jsonl").open("w", encoding="utf-8") as corpus_out:
        for law in selected:
            law_id = law["law_info"]["law_id"]
            if law_id in original_ids:
                continue
            corpus = {}
            for anchor in protocol["anchor_dates"]:
                path = extension / "snapshots" / law_id / (anchor + ".json")
                snapshot = read(path)
                if snapshot["law_info"]["law_id"] != law_id:
                    raise ValueError("Snapshot returned a different selected law")
                revision = snapshot["revision_info"]["law_revision_id"]
                raw_path = path.relative_to(extension).as_posix()
                binding = {"law_id": law_id, "requested_asof": anchor, "family_id": law["family_id"],
                           "source_role": "full", "status": "collected", "revision_id": revision,
                           "raw_path": raw_path, "raw_source_root": str(extension)}
                by_address = defaultdict(list)
                for record in extract_provisions(snapshot):
                    record.update(requested_asof=anchor, family_id=law["family_id"], source_role="full",
                                  split="distractor_only", raw_path=raw_path)
                    visible = visible_provision(record, snapshot)
                    add_passage(corpus, visible)
                    number = visible["article_num"] if visible["article_num"] is not None else visible["slot_id"].rsplit(":", 1)[-1]
                    by_address[visible["scope"], number].append(visible)
                    stats["new_dated_provisions"] += 1
                for (scope, number), values in by_address.items():
                    row = values[0]
                    row_write(targets_out, {"address_key": [law_id, anchor, scope, number], "law_id": law_id,
                        "requested_asof": anchor, "scope": scope, "address_num": number, "article_num": row["article_num"],
                        "slot_id": row["slot_id"], "status": target_status(values), "revision_id": revision,
                        "passage_ids": [r["passage_id"] for r in values], "raw_path": raw_path,
                        "raw_source_root": str(extension), "family_id": law["family_id"], "source_role": "full",
                        "split": "distractor_only"})
                    stats["new_target_addresses"] += 1
                binding["provision_count"] = sum(map(len, by_address.values()))
                row_write(bindings_out, binding)
                stats["new_snapshots"] += 1
            for passage in corpus.values():
                passage["asof_dates"].sort()
                passage["raw_paths"].sort()
                row_write(corpus_out, passage)
            new_chunks = chunk_passages(corpus.values(), tokenizer,
                max_body_tokens=protocol["chunk_body_tokens"], overlap_tokens=protocol["chunk_overlap_tokens"])
            for chunk in new_chunks:
                count = len(tokenizer(prefix + chunk["text"], add_special_tokens=True, truncation=False)["input_ids"])
                if count > 1024:
                    raise ValueError("A new literal chunk exceeds the unchanged Ruri prefixed input limit")
                chunk["ruri_document_tokens"] = count
                row_write(chunks_out, chunk)
                row_write(lexical_out, {"chunk_id": chunk["chunk_id"], "tokens": japanese_tokens(chunk["text"], segmenter)})
            stats.update(new_laws=1, new_unique_passages=len(corpus), new_chunks=len(new_chunks))
            if stats["new_laws"] % 20 == 0:
                manifest["statistics"] = dict(stats)
                write_json(output / "index.json", manifest)
                print(json.dumps(dict(stats)), flush=True)
    manifest.update(complete=True, phase="complete", statistics=dict(stats),
        merged_chunk_rows=manifest["original_chunk_rows"] + stats["new_chunks"],
        expert_dates_excluded=True, original_chunk_and_lexical_prefix_unchanged=True)
    write_json(output / "index.json", manifest)


def encode(args, protocol):
    from hidate.models import create_encoder
    output = args.index / "embeddings"
    output.mkdir(parents=True, exist_ok=True)
    index, stack = read(args.index / "index.json"), read(args.stack)
    old = args.original_index / "embeddings"
    old_encoding, preflight = read(old / "encoding.json"), read(args.preflight)
    pin = stack["models"]["embedding"]
    if not all(r["complete"] for r in (index, stack, old_encoding)) or preflight["status"] != "PASS":
        raise ValueError("Use complete original/new indexes, original vectors and passed original Ruri preflight")
    for record in (old_encoding["model_pin"], preflight["model_pins"]["embedding"]):
        if any(record[k] != pin[k] for k in ("model_id", "revision")):
            raise ValueError("New encoding must use the exact original Ruri revision")
    old_ids = read(old / "chunk-ids.json")
    old_aliases = np.load(old / "vector-row-indices.npy", mmap_mode="r")
    old_vectors = np.load(old / "vectors.npy", mmap_mode="r")
    text_rows, chunk_ids, aliases, new_texts = {}, [], [], []
    for position, chunk in enumerate(rows(args.original_index / "chunks.jsonl")):
        if old_ids[position] != chunk["chunk_id"]:
            raise ValueError("Original vectors and original chunk identities differ")
        text_rows.setdefault(chunk["text"], int(old_aliases[position]))
    for position, chunk in enumerate(rows(args.index / "chunks.jsonl")):
        if position < len(old_ids) and chunk["chunk_id"] != old_ids[position]:
            raise ValueError("Scale merged chunks must retain the complete original chunk prefix")
        if chunk["text"] not in text_rows:
            text_rows[chunk["text"]] = len(old_vectors) + len(new_texts)
            new_texts.append(chunk["text"])
        chunk_ids.append(chunk["chunk_id"])
        aliases.append(text_rows[chunk["text"]])
    write_json(output / "chunk-ids.json", chunk_ids)
    np.save(output / "vector-row-indices.npy", np.asarray(aliases, dtype=np.int64))
    dimension = old_vectors.shape[1]
    shape = (len(old_vectors) + len(new_texts), dimension)
    vectors = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype=np.float32, shape=shape)
    vectors[:len(old_vectors)] = old_vectors
    vectors.flush()
    manifest = receipt(complete=False, model_pin={k: pin[k] for k in ("model_id", "revision", "local_path")},
        corpus_rows=len(chunk_ids), vectors_shape=list(shape), dimension=dimension,
        original_unique_vectors_copied=len(old_vectors), new_unique_texts=len(new_texts), encoded_new_texts=0,
        old_vector_values_unchanged=True, original_encoding=str(old / "encoding.json"))
    write_json(output / "encoding.json", manifest)
    encoder = create_encoder(pin, device="cuda", max_length=1024) if new_texts else None
    work = Counter()
    for offset in range(0, len(new_texts), args.batch_size):
        batch_text = new_texts[offset:offset + args.batch_size]
        result = encoder.encode(batch_text, kind="document")
        batch = np.asarray(result["embeddings"], dtype=np.float32)
        if batch.shape != (len(batch_text), dimension) or not np.isfinite(batch).all():
            raise ValueError("New Ruri vector shape/values do not match the frozen embedding bank")
        vectors[len(old_vectors) + offset:len(old_vectors) + offset + len(batch)] = batch
        vectors.flush()
        work.update(batches=1, encoded_input_tokens=result["total_tokens"])
        work["model_seconds"] += result["seconds"]
        manifest.update(encoded_new_texts=offset + len(batch), work=dict(work))
        write_json(output / "encoding.json", manifest)
        if work["batches"] % 100 == 0:
            print(json.dumps({"encoded_new_texts": manifest["encoded_new_texts"], "new_unique_texts": len(new_texts)}), flush=True)
    manifest.update(complete=True, encoded_new_texts=len(new_texts), work=dict(work))
    write_json(output / "encoding.json", manifest)


def evaluate(args, protocol):
    from build_index import japanese_tokens, sudachi_segmenter
    from discover_final import vector_bank
    from evaluate_final import structural_result
    from hidate.discovery import DiscoveryEngine
    from hidate.models import create_reranker
    from hidate.reranking import ALL_CONDITIONS, PairScoreCache, rerank_query
    from hidate.selection import baseline_support_score, selected_evidence, support_score

    args.output.mkdir(parents=True, exist_ok=True)
    policy, ranker, stack = read(args.policy), read(args.ranker), read(args.stack)
    index = read(args.index / "index.json")
    if not all(record["complete"] for record in (policy, ranker, stack, index)):
        raise ValueError("Scale evaluation reuses the completed original ranker, VALID policy, stack, and new index")
    if tuple(protocol["conditions"]) != ALL_CONDITIONS:
        raise ValueError("Scale comparison requires all original seven HiDate conditions")
    documents = vector_bank(args.index / "embeddings", "chunk-ids.json")
    chunks = list(rows(args.index / "chunks.jsonl"))
    if documents["ids"] != [c["chunk_id"] for c in chunks]:
        raise ValueError("Merged chunk identities must match the vector alias bank")
    chunk_index = {c["chunk_id"]: c for c in chunks}
    tokens = {r["chunk_id"]: r["tokens"] for r in rows(args.index / "lexical-tokens.jsonl")}
    targets = {tuple(r["address_key"]): r for r in rows(args.index / "target-addresses.jsonl")}
    bindings = {(r["law_id"], r["requested_asof"]): r for r in rows(args.index / "asof-bindings.jsonl")}
    query_banks, queries = {}, []
    for name, filename in (("exact", "queries-exact.jsonl"), ("paraphrase", "queries-paraphrase.jsonl")):
        bank = vector_bank(args.original_index / ("queries-" + name), "query-ids.json")
        if any(bank["manifest"]["model_pin"][k] != documents["manifest"]["model_pin"][k] for k in ("model_id", "revision")):
            raise ValueError("Reuse only original query vectors from the same Ruri encoder")
        bank["rows"] = {identity: int(alias) for identity, alias in zip(bank["ids"], bank["aliases"])}
        query_banks[name] = bank
        subset = [q for q in rows(args.original_prepared / filename) if q["split"] == "test"]
        if len(subset) != protocol["expected_queries_per_variant"]:
            raise ValueError("Every scale requires both unchanged 1586-query TEST variants")
        queries.extend((name, q) for q in subset)
    identities = [q["query_id"] for _, q in queries]
    if len(set(identities)) != protocol["expected_test_queries"]:
        raise ValueError("The exact same 3172 unique natural TEST identities must be present")
    preflight = read(args.preflight)
    if preflight["status"] != "PASS":
        raise ValueError("Original Ruri preflight must pass before scale model evaluation")
    for role in ("embedding", "reranker"):
        if any(preflight["model_pins"][role][k] != stack["models"][role][k] for k in ("model_id", "revision")):
            raise ValueError("Scale evaluation must retain both original Ruri model pins")
    if any(documents["manifest"]["model_pin"][k] != stack["models"]["embedding"][k] for k in ("model_id", "revision")):
        raise ValueError("The evaluated stack must match the merged document embeddings")
    shared = SharedCosines(documents["vectors"])
    engines, scale_banks = {}, read(args.index / "nested-corpora.json")
    for scale in protocol["law_scales"]:
        law_ids = set(scale_banks[str(scale)])
        selected = [c for c in chunks if c["source_role"] == "full" and c["law_id"] in law_ids]
        engines[str(scale)] = DiscoveryEngine(selected, tokens, shared, documents["aliases"], documents["ids"],
            targets, bindings, top_k=protocol["discovery_top_k"], rrf_k=protocol["rrf_k"])
    reranker = create_reranker(stack["models"]["reranker"], device="cuda", max_length=1024)
    cache = PairScoreCache(reranker)
    segmenter = sudachi_segmenter()
    result_path = args.output / "retrieval.jsonl"
    existing = list(rows(result_path)) if result_path.exists() else []
    seen = {(r["scale"], r["query_id"], r["method"]) for r in existing}
    expected = {(str(scale), identity, method) for scale in protocol["law_scales"]
                for identity in identities for method in ALL_CONDITIONS}
    if not seen.issubset(expected) or len(seen) != len(existing):
        raise ValueError("Resume rows must belong uniquely to this frozen 3172x3x7 comparison")
    summaries = defaultdict(list)
    for record in existing:
        summaries[record["scale"], record["method"]].append(record)
    manifest = receipt(complete=False, protocol=protocol, policy_path=str(args.policy), ranker_path=str(args.ranker),
        alpha=policy["alpha"], frozen_policy=policy, model_pins={role: {k: stack["models"][role][k] for k in ("model_id", "revision")}
        for role in ("embedding", "reranker")}, expected_records=len(expected), completed_records=len(seen),
        same_query_ids=identities, index_path=str(args.index), scale_full_chunk_counts={k: len(e.chunks) for k, e in engines.items()})
    write_json(args.output / "evaluation.json", manifest)
    started = time.perf_counter()
    query_steps = 0
    with result_path.open("a", encoding="utf-8") as stream:
        for variant, query in sorted(queries, key=lambda pair: (pair[1]["target_asof"], pair[1]["query_id"])):
            bank = query_banks[variant]
            vector = bank["vectors"][bank["rows"][query["query_id"]]]
            query_tokens = japanese_tokens(query["query_text"], segmenter)
            for scale, engine in engines.items():
                pending = [method for method in ALL_CONDITIONS if (scale, query["query_id"], method) not in seen]
                if not pending:
                    continue
                discovery = engine.discover(query, query_tokens, vector)
                result = rerank_query(query, discovery, chunk_index, cache, conditions=pending, ranker=ranker)
                for method in pending:
                    condition = result["conditions"][method]
                    evidence = selected_evidence(condition, alpha=policy["alpha"] if method == "history_learned" else 0.)
                    confidence = (support_score(evidence, condition["address_diagnostics"]) if method == "history_learned"
                                  else baseline_support_score(method, evidence, condition["address_diagnostics"]))
                    threshold = policy["method_thresholds"][method]["threshold"]
                    record = {"scale": scale, "query_id": query["query_id"], "method": method,
                        "query_variant": query["query_variant"], "family_id": query["family_id"], "stratum": query["stratum"],
                        "target_asof": query["target_asof"], "quote_absent": query.get("source_quote_absent_from_target"),
                        **structural_result(query, evidence, "natural"), "accepted": confidence >= threshold,
                        "support_confidence": confidence, "support_threshold": threshold,
                        "requested_pairs": condition["requested_pair_count"],
                        "evidence_chunk_ids": [e["chunk_id"] for e in evidence],
                        "evidence_parent_ids": [e["parent_passage_id"] for e in evidence],
                        "bank_counts": discovery["bank_counts"]}
                    row_write(stream, record)
                    summaries[scale, method].append(record)
                    seen.add((scale, query["query_id"], method))
            stream.flush()
            query_steps += 1
            if query_steps % 50 == 0:
                manifest.update(completed_records=len(seen), physical_pairs_this_job=cache.physical_pair_count)
                write_json(args.output / "evaluation.json", manifest)
                print(json.dumps({"completed_records": len(seen), "expected_records": len(expected), "physical_pairs": cache.physical_pair_count}), flush=True)
    manifest.update(complete=seen == expected, completed_records=len(seen),
        panels={f"{scale}|{method}": metric_panels(values) for (scale, method), values in summaries.items()},
        logical_pairs_this_job=cache.logical_pair_requests, physical_pairs_this_job=cache.physical_pair_count,
        reranker_calls_this_job=cache.model_call_count, reranker_seconds_this_job=cache.model_wall_seconds_with_cuda_sync,
        shared_dense_products_this_job=shared.products, wall_seconds_this_job=time.perf_counter() - started)
    write_json(args.output / "evaluation.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "encode", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/scale_protocol.json"))
    parser.add_argument("--extension", type=Path, default=Path("data/raw/hidate/extended-benchmark/extension"))
    parser.add_argument("--original-source", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--original-prepared", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--original-index", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--index", type=Path, default=Path("data/index/hidate/extended-benchmark/scale"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--ranker", type=Path, default=Path("runs/hidate/ranker/address-model.json"))
    parser.add_argument("--policy", type=Path, default=Path("runs/hidate/policy/frozen.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/extended-benchmark/scale"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    globals()[args.phase](args, read(args.protocol))


if __name__ == "__main__":
    main()
