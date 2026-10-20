"""Independently audit HiDate discovery rows, with sampled bank-wide dense ordering."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import unicodedata

import numpy as np

from hidate.io import load_jsonl, write_json


ATOL, RTOL, UNIT_TOLERANCE = 2e-6, 2e-5, 1e-4


def address(chunk):
    if chunk["source_kind"] == "guidance":
        return ("guidance", chunk["source_id"])
    number = chunk["article_num"]
    return (chunk["law_id"], chunk["scope"], number if number is not None else chunk["slot_id"].rsplit(":", 1)[-1])


def eligible(chunk, role, target, mode):
    if chunk["source_role"] != role:
        return False
    if chunk["source_kind"] == "guidance":
        return True
    dates = chunk["asof_dates"]
    return target in dates if mode == "asof" else any(anchor <= target for anchor in dates)


def segmenter_default():
    from sudachipy import Dictionary, SplitMode
    tokenizer = Dictionary(dict="core").tokenizer()
    return lambda text: [token.normalized_form() for token in tokenizer.tokenize(text, SplitMode.A)]


def normalized_terms(text, segmenter):
    return set(token for token in segmenter(text)
               if any(not character.isspace() and not unicodedata.category(character).startswith("P") for character in token))


def dense_order(ids, scores, limit):
    # Independent full ordering for the explicitly sampled queries only.
    return sorted(zip(ids, scores), key=lambda item: (-float(item[1]), item[0]))[:limit]


def readback(index_dir, queries_path, query_embeddings_dir, discovery_dir, prepared_dir,
             source_dir, splits=("train", "valid"), *, segmenter=None, rank_sample_size=20,
             protocol_path=Path("configs/retrieval-protocol.json")):
    started = time.perf_counter()
    index, query_dir, discovery = Path(index_dir), Path(query_embeddings_dir), Path(discovery_dir)
    prepared, source = Path(prepared_dir), Path(source_dir)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    feature_names = sorted(protocol["rank_features"])
    limit, rrf_k = protocol["discovery_top_k"], protocol["rrf_k"]
    manifest = json.loads((discovery / "discovery.json").read_text(encoding="utf-8"))
    corpus_encoding = json.loads((index / "embeddings/encoding.json").read_text(encoding="utf-8"))
    query_encoding = json.loads((query_dir / "encoding.json").read_text(encoding="utf-8"))
    queries_all = load_jsonl(queries_path)
    selected = [query for query in queries_all if query["split"] in splits]
    queries = {query["query_id"]: query for query in selected}
    rows = load_jsonl(discovery / "candidates.jsonl")
    issues = []

    def check(condition, description):
        if not condition:
            issues.append(description)

    check(set(splits) <= {"train", "valid"}, "readback requested a split outside TRAIN/VALID")
    check(manifest["complete"] and corpus_encoding["complete"] and query_encoding["complete"], "discovery/encoder completion disagreement")
    check(manifest["splits"] == list(splits), "declared discovery splits changed")
    check(manifest["features"] == feature_names, "discovery feature-name declaration changed")
    check(manifest["discovery_top_k"] == limit and manifest["rrf_k"] == rrf_k, "discovery budget or RRF constant changed")
    for key in ("model_id", "revision"):
        check(manifest["model_pin"][key] == corpus_encoding["model_pin"][key] == query_encoding["model_pin"][key], f"encoder {key} disagreement")
    check(len(queries) == len(selected), "original selected query IDs are duplicated")
    check(Counter(row["query_id"] for row in rows) == Counter(query["query_id"] for query in selected), "discovery query IDs are missing, extra or duplicated")
    check(manifest["queries"] == len(rows), "discovery row count disagrees with manifest")

    chunks_list = load_jsonl(index / "chunks.jsonl")
    chunks = {chunk["chunk_id"]: chunk for chunk in chunks_list}
    tokens = {row["chunk_id"]: set(row["tokens"]) for row in load_jsonl(index / "lexical-tokens.jsonl")}
    chunk_ids = json.loads((index / "embeddings/chunk-ids.json").read_text(encoding="utf-8"))
    corpus_aliases = np.load(index / "embeddings/vector-row-indices.npy", mmap_mode="r", allow_pickle=False)
    vectors = np.load(index / "embeddings/vectors.npy", mmap_mode="r", allow_pickle=False)
    encoded_ids = json.loads((query_dir / "query-ids.json").read_text(encoding="utf-8"))
    encoded_texts = json.loads((query_dir / "query-texts.json").read_text(encoding="utf-8"))
    query_aliases = np.load(query_dir / "vector-row-indices.npy", mmap_mode="r", allow_pickle=False)
    query_vectors = np.load(query_dir / "vectors.npy", mmap_mode="r", allow_pickle=False)
    check(chunk_ids == [chunk["chunk_id"] for chunk in chunks_list] and len(chunk_ids) == len(corpus_aliases), "corpus vector identities disagree")
    check(len(chunks) == len(chunks_list), "index chunk IDs are duplicated")
    check(vectors.ndim == query_vectors.ndim == 2 and vectors.shape[1] == query_vectors.shape[1], "vector dimensions disagree")
    variant = query_encoding.get("variant", "all")
    encoded_queries = [query for query in queries_all if variant == "all" or query["query_variant"] == variant]
    check(encoded_ids == [query["query_id"] for query in encoded_queries], "encoded query ID order differs from original input")
    check(len(encoded_ids) == len(query_aliases), "query alias row count disagrees")
    check(len(encoded_texts) == query_vectors.shape[0], "query-text/vector row count disagrees")
    check(len(encoded_texts) == len(set(encoded_texts)), "unique query texts are duplicated")
    check(np.issubdtype(corpus_aliases.dtype, np.integer) and np.issubdtype(query_aliases.dtype, np.integer), "vector aliases are not integer rows")
    check(not len(corpus_aliases) or (int(corpus_aliases.min()) >= 0 and int(corpus_aliases.max()) < len(vectors)), "corpus vector alias out of range")
    check(not len(query_aliases) or (int(query_aliases.min()) >= 0 and int(query_aliases.max()) < len(query_vectors)), "query vector alias out of range")
    document_text_rows = {}
    for chunk, alias in zip(chunks_list, corpus_aliases):
        if chunk["text"] not in document_text_rows:
            document_text_rows[chunk["text"]] = len(document_text_rows)
        check(int(alias) == document_text_rows[chunk["text"]], f"document-text alias disagreement: {chunk['chunk_id']}")
    check(len(document_text_rows) == len(vectors), "unique document text/vector count disagrees")
    expected_texts = list(dict.fromkeys(query["query_text"] for query in encoded_queries))
    check(encoded_texts == expected_texts, "ordered query texts differ from first-occurrence input order")
    for query, alias in zip(encoded_queries, query_aliases):
        check(encoded_texts[int(alias)] == query["query_text"], f"query-text alias disagreement: {query['query_id']}")
    vector_checks = {}
    for name, matrix in (("corpus", vectors), ("query", query_vectors)):
        max_unit_error, nonfinite_rows = 0., 0
        for offset in range(0, len(matrix), 2048):
            batch = np.asarray(matrix[offset:offset + 2048], dtype=np.float64)
            finite = np.isfinite(batch).all(axis=1)
            nonfinite_rows += int((~finite).sum())
            if finite.any():
                max_unit_error = max(max_unit_error, float(np.abs(np.linalg.norm(batch[finite], axis=1) - 1).max()))
        check(nonfinite_rows == 0 and max_unit_error <= UNIT_TOLERANCE, f"{name} vectors are nonfinite or not unit normalized")
        vector_checks[name] = {"rows": len(matrix), "nonfinite_rows": nonfinite_rows, "max_unit_norm_error": max_unit_error}
    vector_rows = dict(zip(chunk_ids, map(int, corpus_aliases)))
    query_rows = dict(zip(encoded_ids, map(int, query_aliases)))
    target_map = {tuple(row["address_key"]): row for row in load_jsonl(prepared / "target-addresses.jsonl")}
    bindings = {(row["law_id"], row["requested_asof"]): row for row in load_jsonl(source / "asof-bindings.jsonl")}
    parent_chunks = defaultdict(list)
    for chunk in chunks_list:
        parent_chunks[chunk["parent_passage_id"]].append(chunk["chunk_id"])
        if chunk["source_kind"] == "guidance":
            check(not any(field in chunk for field in ("law_id", "requested_asof", "asof_dates")), f"guidance assigned legal dates: {chunk['chunk_id']}")
    pools = {}
    for query in selected:
        key = (query["source_role"], query["target_asof"])
        if key not in pools:
            pools[key] = {mode: sorted(chunk["chunk_id"] for chunk in chunks_list if eligible(chunk, *key, mode)) for mode in ("asof", "history")}
    sample_ids = set()
    per_split = rank_sample_size // len(splits)
    for split in splits:
        ids = sorted(query["query_id"] for query in selected if query["split"] == split)
        for position in np.linspace(0, len(ids) - 1, min(per_split, len(ids)), dtype=int) if ids else []:
            sample_ids.add(ids[int(position)])
    segmenter = segmenter if segmenter is not None else segmenter_default()
    term_cache = {}
    candidate_lists = candidate_hits = address_rows = feature_rows = current_scored = 0
    dense_checks, seen_cache = [], set()
    metadata_fields = ("query_id", "query_variant", "target_asof", "source_role", "split")
    for row in rows:
        query_id = row["query_id"]
        if query_id not in queries:
            continue
        query = queries[query_id]
        check(all(row[field] == query[field] for field in metadata_fields), f"query metadata changed: {query_id}")
        role, target_date = query["source_role"], query["target_asof"]
        pool = pools[(role, target_date)]
        check(row["bank_counts"] == {mode: len(ids) for mode, ids in pool.items()}, f"bank census mismatch: {query_id}")
        good_ids = True
        for mode in ("asof", "history"):
            for method in ("bm25", "dense", "hybrid"):
                hits = row[mode][method]
                ids = [hit["chunk_id"] for hit in hits]
                candidate_lists += 1
                candidate_hits += len(ids)
                check(len(ids) <= limit and len(ids) == len(set(ids)), f"duplicate/over-budget {mode}/{method}: {query_id}")
                check(all(chunk_id in chunks for chunk_id in ids), f"unknown {mode}/{method} chunk: {query_id}")
                if any(chunk_id not in chunks for chunk_id in ids):
                    good_ids = False
                    continue
                check(all(eligible(chunks[chunk_id], role, target_date, mode) for chunk_id in ids), f"ineligible {mode}/{method} chunk: {query_id}")
                check(all(math.isfinite(hit["score"]) for hit in hits), f"nonfinite hit scores: {query_id}")
                check(hits == sorted(hits, key=lambda hit: (-hit["score"], hit["chunk_id"])), f"saved score ordering changed: {query_id}/{mode}/{method}")
            scores = defaultdict(float)
            for method in ("bm25", "dense"):
                for rank, hit in enumerate(row[mode][method], 1):
                    scores[hit["chunk_id"]] += 1 / (rrf_k + rank)
            fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
            check([hit["chunk_id"] for hit in row[mode]["hybrid"]] == [item[0] for item in fused], f"RRF ID ordering mismatch: {query_id}/{mode}")
            check(all(math.isclose(hit["score"], expected[1], rel_tol=RTOL, abs_tol=ATOL)
                      for hit, expected in zip(row[mode]["hybrid"], fused)), f"RRF score mismatch: {query_id}/{mode}")
        if not good_ids:
            continue
        qvector = query_vectors[query_rows[query_id]]
        if query_id in sample_ids:
            unique_scores = vectors @ qvector
            for mode in ("asof", "history"):
                ids = pool[mode]
                expected = dense_order(ids, unique_scores[[vector_rows[chunk_id] for chunk_id in ids]], limit)
                actual = row[mode]["dense"]
                check([hit["chunk_id"] for hit in actual] == [item[0] for item in expected], f"sampled full-bank dense ordering mismatch: {query_id}/{mode}")
                check(all(math.isclose(hit["score"], float(item[1]), rel_tol=RTOL, abs_tol=ATOL)
                          for hit, item in zip(actual, expected)), f"sampled dense score mismatch: {query_id}/{mode}")
            dense_checks.append({"query_id": query_id, "split": query["split"], "query_variant": query["query_variant"]})
        groups = defaultdict(list)
        for hit in row["history"]["hybrid"]:
            groups[address(chunks[hit["chunk_id"]])].append(hit)
        check(Counter(tuple(candidate["address"]) for candidate in row["addresses"]) == Counter({key: 1 for key in groups}), f"address aggregation membership mismatch: {query_id}")
        ranks = {}
        for method, hits in row["history"].items():
            ranks[method] = {}
            for rank, hit in enumerate(hits, 1):
                ranks[method].setdefault(address(chunks[hit["chunk_id"]]), rank)
        if query["query_text"] not in term_cache:
            term_cache[query["query_text"]] = normalized_terms(query["query_text"], segmenter)
        terms = term_cache[query["query_text"]]
        asof_ids = set(pool["asof"])
        row_current = 0
        for candidate in row["addresses"]:
            address_rows += 1
            key = tuple(candidate["address"])
            if key not in groups:
                continue
            hits = groups[key]
            first = chunks[hits[0]["chunk_id"]]
            guidance = first["source_kind"] == "guidance"
            check(candidate["source_kind"] == first["source_kind"], f"address kind mismatch: {query_id}/{key}")
            revisions, best_ids, parents, expected_history = {}, {}, set(), []
            for hit in hits:
                chunk = chunks[hit["chunk_id"]]
                revision = chunk["source_id"] if guidance else chunk["revision_id"]
                if revision not in revisions or hit["score"] > revisions[revision]:
                    revisions[revision], best_ids[revision] = hit["score"], hit["chunk_id"]
                parents.add(chunk["parent_passage_id"])
                expected_history.append({**hit, "parent_passage_id": chunk["parent_passage_id"],
                    "revision_id": chunk.get("revision_id"), "asof_dates": [anchor for anchor in chunk.get("asof_dates", []) if anchor <= target_date]})
            check(candidate["history_chunk_hits"] == expected_history, f"history provenance mismatch: {query_id}/{key}")
            check(candidate["history_parent_passage_ids"] == sorted(parents), f"history parent mismatch: {query_id}/{key}")
            check(candidate["historical_best_chunk_ids"] == best_ids, f"history best-chunk mismatch: {query_id}/{key}")
            check(candidate["revision_scores"] == ({} if guidance else revisions), f"revision vote mismatch: {query_id}/{key}")
            check(candidate["support_count"] == len(revisions)
                  and math.isclose(candidate["max_score"], max(revisions.values()), rel_tol=RTOL, abs_tol=ATOL)
                  and math.isclose(candidate["mean_score"], sum(revisions.values()) / len(revisions), rel_tol=RTOL, abs_tol=ATOL),
                  f"raw history aggregate mismatch: {query_id}/{key}")
            if guidance:
                status, target_ids, usable = "available", sorted(parents), sorted(parents)
                check(candidate["source_id"] == first["source_id"] and candidate["url"] == first["url"], f"guidance source metadata mismatch: {query_id}/{key}")
                check("target_asof" not in candidate and "address_key" not in candidate, f"guidance received a dated law binding: {query_id}/{key}")
            else:
                dated_key = (key[0], target_date, key[1], key[2])
                binding, target = bindings.get(dated_key[:2]), target_map.get(dated_key)
                if binding is None or binding["status"] != "collected":
                    status, target_ids = "collection_failure", []
                else:
                    status, target_ids = (target["status"], target["passage_ids"]) if target else ("absent", [])
                usable = target_ids if status == "available" else []
                check(candidate["law_id"] == first["law_id"] and candidate["scope"] == first["scope"]
                      and candidate["article_num"] == first["article_num"], f"scope/branch metadata mismatch: {query_id}/{key}")
                check(tuple(candidate["address_key"]) == dated_key and candidate["target_asof"] == target_date, f"dated binding key mismatch: {query_id}/{key}")
                check(candidate["revision_id"] == (binding.get("revision_id") if binding else None), f"actual target revision mismatch: {query_id}/{key}")
            check(candidate["status"] == status and candidate["target_passage_ids"] == target_ids
                  and candidate["parent_passage_ids"] == usable, f"target status/hydration mismatch: {query_id}/{key}")
            current_ids = sorted(chunk_id for parent in usable for chunk_id in parent_chunks[parent] if chunk_id in asof_ids)
            check(all(not chunks[chunk_id].get("deleted", False) and not chunks[chunk_id].get("hidden", False) for chunk_id in current_ids), f"hidden/deleted current hydration: {query_id}/{key}")
            row_current += len(current_ids)
            check(candidate["current_feature_chunk_count"] == len(current_ids), f"current-chunk work count mismatch: {query_id}/{key}")
            current_scores = {chunk_id: float(vectors[vector_rows[chunk_id]] @ qvector) for chunk_id in current_ids}
            maximum = max(current_scores.values(), default=-1.)
            best = candidate["current_best_chunk_id"]
            check((best is None if not current_ids else best in current_scores and math.isclose(current_scores[best], maximum, rel_tol=RTOL, abs_tol=ATOL)), f"current best-chunk mismatch: {query_id}/{key}")
            overlap = max((len(terms.intersection(tokens[chunk_id])) / len(terms) for chunk_id in current_ids), default=0.) if terms else 0.
            anchors = [anchor for hit in hits for anchor in chunks[hit["chunk_id"]].get("asof_dates", []) if anchor <= target_date]
            gap = max(0., (date.fromisoformat(target_date) - date.fromisoformat(max(anchors))).days / 365.25) if anchors else 0.
            expected_features = {"max_history_rrf": max(revisions.values()), "mean_history_rrf": sum(revisions.values()) / len(revisions),
                "distinct_revision_count": 0 if guidance else len(revisions),
                "history_best_rank_rrf": ranks["hybrid"].get(key, limit + 1), "history_best_rank_dense": ranks["dense"].get(key, limit + 1),
                "history_best_rank_bm25": ranks["bm25"].get(key, limit + 1), "current_best_cosine": maximum,
                "max_lexical_overlap_current": overlap, "available_target": float(status == "available"), "history_latest_anchor_gap_years": gap}
            features = candidate["features"]
            check(sorted(features) == feature_names, f"feature schema mismatch: {query_id}/{key}")
            check(all(math.isfinite(value) for value in features.values()), f"nonfinite address feature: {query_id}/{key}")
            check(all(math.isclose(features[name], expected_features[name], rel_tol=RTOL, abs_tol=ATOL) for name in feature_names), f"independent feature mismatch: {query_id}/{key}")
            feature_rows += 1
        check(row["current_feature_chunks"] == row_current, f"query current-chunk work count mismatch: {query_id}")
        cache_key = (query["query_text"], target_date, role)
        if cache_key not in seen_cache:
            current_scored += row_current
            seen_cache.add(cache_key)
    check(manifest["retrieval_runs"] == len(seen_cache) and manifest["cache_hits"] == len(rows) - len(seen_cache), "retrieval/cache work receipts disagree")
    check(manifest["current_chunk_scoring_count"] == current_scored, "current-chunk scoring receipt disagrees")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - started,
        "discovery_rows_checked": len(rows), "splits": list(splits), "query_metadata_fields": list(metadata_fields),
        "query_alias_rows_checked": len(encoded_ids), "query_alias_scope": "all encoded input IDs and texts; metadata only, no TEST candidate evaluation",
        "vector_checks": vector_checks, "unit_norm_tolerance": UNIT_TOLERANCE,
        "candidate_lists_checked": candidate_lists, "candidate_hits_checked": candidate_hits,
        "address_rows_checked": address_rows, "feature_rows_checked": feature_rows,
        "numeric_tolerance": {"relative": RTOL, "absolute": ATOL}, "current_chunk_scoring_count": current_scored,
        "dense_rank_queries_checked": len(dense_checks), "dense_rank_lists_checked": 2 * len(dense_checks),
        "dense_rank_queries": dense_checks,
        "dense_rank_sample_rule": "up to10 evenly spaced sorted query IDs per TRAIN/VALID split; at most20 total",
        "ordering_scope": "all saved IDs/eligibility/budgets/duplicates/score order and RRF checked; complete dense top50 ordering only sampled queries; no complete BM25 bank-order claim",
        "training_runs": 0, "model_inference_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--queries", type=Path, default=Path("data/processed/hidate/queries-exact.jsonl"))
    parser.add_argument("--query-embeddings-dir", type=Path, default=Path("data/index/hidate/queries-exact"))
    parser.add_argument("--discovery-dir", type=Path, default=Path("runs/hidate/discovery/exact-train-valid"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"), default=["train", "valid"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = readback(args.index_dir, args.queries, args.query_embeddings_dir, args.discovery_dir,
        args.prepared_dir, args.source_dir, splits=tuple(args.splits), protocol_path=args.protocol)
    write_json(args.output or args.discovery_dir / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
