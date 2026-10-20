"""CPU HiDate discovery and fixed address features from supplied embeddings."""

from collections import defaultdict
from datetime import date

import numpy as np

from .retrieval import BM25Index, aggregate_history, bind_addresses, eligible_chunks, rrf


FEATURE_NAMES = tuple(sorted((
    "max_history_rrf", "mean_history_rrf", "distinct_revision_count",
    "history_best_rank_rrf", "history_best_rank_dense", "history_best_rank_bm25",
    "current_best_cosine", "max_lexical_overlap_current", "available_target",
    "history_latest_anchor_gap_years")))


def chunk_address(chunk):
    if chunk["source_kind"] == "guidance":
        return ("guidance", chunk["source_id"])
    number = chunk["article_num"]
    if number is None:
        number = chunk["slot_id"].rsplit(":", 1)[-1]
    return (chunk["law_id"], chunk["scope"], number)


def _dense_hits(chunk_ids, scores, top_k):
    limit = min(top_k, len(chunk_ids))
    if not limit:
        return []
    cutoff = np.partition(scores, len(scores) - limit)[len(scores) - limit]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:limit - len(above)]
    chosen = np.concatenate((above, tied))
    chosen = chosen[np.lexsort((chosen, -scores[chosen]))]
    return [{"chunk_id": chunk_ids[row], "score": float(scores[row])} for row in chosen]


def _address_ranks(hits, chunks):
    ranks = {}
    for rank, hit in enumerate(hits, 1):
        ranks.setdefault(chunk_address(chunks[hit["chunk_id"]]), rank)
    return ranks


class DiscoveryEngine:
    """Reuse one sparse index per eligible bank and one score vector per query.

    ``vectors`` contains unique document vectors. ``vector_row_indices`` maps
    the supplied chunk-ID order onto those rows. All selected source-role families
    remain in the bank; query split and gold fields never affect discovery.
    """

    def __init__(self, chunks, document_tokens, vectors, vector_row_indices, chunk_ids,
                 target_index, bindings, top_k=50, rrf_k=60):
        self.chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.document_tokens = document_tokens
        self.vectors = vectors
        self.vector_rows = {chunk_id: int(row) for chunk_id, row in zip(chunk_ids, vector_row_indices)}
        self.target_index, self.bindings = target_index, bindings
        self.top_k, self.rrf_k = top_k, rrf_k
        self.parent_chunks = defaultdict(list)
        for chunk in chunks:
            self.parent_chunks[chunk["parent_passage_id"]].append(chunk["chunk_id"])
        self._pool, self._banks, self._cache = None, {}, {}
        self.retrieval_runs = self.cache_hits = 0
        self.current_chunk_scoring_count = 0

    def _bank(self, source_role, target_asof, mode):
        pool = (source_role, target_asof)
        if pool != self._pool:
            self._pool, self._banks = pool, {}
        if mode not in self._banks:
            corpus = [chunk for chunk in self.chunks.values() if chunk["source_role"] == source_role]
            eligible = eligible_chunks(corpus, target_asof, mode)
            ids = sorted(chunk["chunk_id"] for chunk in eligible)
            self._banks[mode] = {"ids": ids, "id_set": set(ids),
                "vector_rows": np.array([self.vector_rows[chunk_id] for chunk_id in ids], dtype=np.int64),
                "lexical": BM25Index({chunk_id: self.document_tokens[chunk_id] for chunk_id in ids})}
        return self._banks[mode]

    def _rank_bank(self, bank, query_tokens, unique_scores):
        lexical = bank["lexical"].rank(query_tokens, top_k=self.top_k)
        dense = _dense_hits(bank["ids"], unique_scores[bank["vector_rows"]], self.top_k)
        return {"bm25": lexical, "dense": dense, "hybrid": rrf([lexical, dense], k=self.rrf_k, top_k=self.top_k)}

    def _addresses(self, history, asof_bank, query_tokens, unique_scores, target_asof):
        aggregated = aggregate_history(history["hybrid"], self.chunks)
        bound = bind_addresses(aggregated, target_asof, self.target_index, self.bindings)
        ranks = {method: _address_ranks(hits, self.chunks) for method, hits in history.items()}
        supporting = defaultdict(list)
        for hit in history["hybrid"]:
            supporting[chunk_address(self.chunks[hit["chunk_id"]])].append(hit)
        query_terms = set(query_tokens)
        rows = []
        for original, target in zip(aggregated, bound):
            address = original["address"]
            hits = supporting[address]
            current_ids = sorted(chunk_id for parent in target["parent_passage_ids"]
                for chunk_id in self.parent_chunks[parent] if chunk_id in asof_bank["id_set"])
            current_best = min(current_ids, key=lambda chunk_id: (-unique_scores[self.vector_rows[chunk_id]], chunk_id)) if current_ids else None
            cosine = float(unique_scores[self.vector_rows[current_best]]) if current_best else -1.
            overlap = max((len(query_terms.intersection(self.document_tokens[chunk_id])) / len(query_terms)
                           for chunk_id in current_ids), default=0.) if query_terms else 0.
            anchors = [anchor for hit in hits for anchor in self.chunks[hit["chunk_id"]].get("asof_dates", [])
                       if anchor <= target_asof]
            gap = max(0., (date.fromisoformat(target_asof) - date.fromisoformat(max(anchors))).days / 365.25) if anchors else 0.
            values = {"max_history_rrf": original["max_score"], "mean_history_rrf": original["mean_score"],
                "distinct_revision_count": len(original["revision_scores"]),
                "history_best_rank_rrf": ranks["hybrid"].get(address, self.top_k + 1),
                "history_best_rank_dense": ranks["dense"].get(address, self.top_k + 1),
                "history_best_rank_bm25": ranks["bm25"].get(address, self.top_k + 1),
                "current_best_cosine": cosine, "max_lexical_overlap_current": overlap,
                "available_target": float(target["status"] == "available"), "history_latest_anchor_gap_years": gap}
            history_hits = [{**hit, "parent_passage_id": self.chunks[hit["chunk_id"]]["parent_passage_id"],
                "revision_id": self.chunks[hit["chunk_id"]].get("revision_id"),
                "asof_dates": [anchor for anchor in self.chunks[hit["chunk_id"]].get("asof_dates", []) if anchor <= target_asof]}
                for hit in hits]
            rows.append({**target, "address": list(address),
                "features": {name: float(values[name]) for name in FEATURE_NAMES},
                "target_passage_ids": target.get("target_passage_ids", target["parent_passage_ids"]),
                "current_best_chunk_id": current_best, "historical_best_chunk_ids": original["best_chunk_ids"],
                "current_feature_chunk_count": len(current_ids),
                "history_parent_passage_ids": original["parent_passage_ids"], "history_chunk_hits": history_hits})
            self.current_chunk_scoring_count += len(current_ids)
        return rows

    def discover(self, query, query_tokens, query_vector):
        """Return candidates; only text/date/role enter retrieval and features.

        Query ID, variant and split are attached after exact result-cache lookup.
        Gold, source oracle context and originating addresses are never read.
        """
        text, target_asof, role = query["query_text"], query["target_asof"], query["source_role"]
        key = (text, target_asof, role)
        if key in self._cache:
            self.cache_hits += 1
            payload = self._cache[key]
        else:
            asof_bank = self._bank(role, target_asof, "asof")
            history_bank = self._bank(role, target_asof, "history")
            unique_scores = self.vectors @ np.asarray(query_vector)
            asof = self._rank_bank(asof_bank, query_tokens, unique_scores)
            history = self._rank_bank(history_bank, query_tokens, unique_scores)
            payload = {"asof": asof, "history": history,
                "addresses": self._addresses(history, asof_bank, query_tokens, unique_scores, target_asof),
                "bank_counts": {"asof": len(asof_bank["ids"]), "history": len(history_bank["ids"])} }
            payload["current_feature_chunks"] = sum(row["current_feature_chunk_count"] for row in payload["addresses"])
            self._cache[key] = payload
            self.retrieval_runs += 1
        return {"query_id": query["query_id"], "query_variant": query["query_variant"], "split": query["split"],
                "target_asof": target_asof, "source_role": role, **payload}
