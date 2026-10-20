"""Common HiDate reranking budgets from saved discovery, with exact pair reuse."""

import math

from .ranking import address_probabilities


BASELINE_CONDITIONS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound", "history_bound", "history_current")
ALL_CONDITIONS = BASELINE_CONDITIONS + ("history_learned",)
PAIR_BUDGET = 10
FINAL_PASSAGES = 4


def _hit_candidates(hits, chunks, limit):
    candidates, seen = [], set()
    for rank, hit in enumerate(hits, 1):
        chunk = chunks[hit["chunk_id"]]
        parent = chunk["parent_passage_id"]
        if parent in seen:
            continue
        seen.add(parent)
        candidates.append({**chunk, "original_score": float(hit["score"]), "discovery_rank": rank,
                           "selection_rank": len(candidates) + 1})
        if len(candidates) == limit:
            break
    return candidates


def select_candidates(discovery, chunks, *, conditions=None, ranker=None, pair_budget=PAIR_BUDGET):
    conditions = tuple(conditions) if conditions is not None else (ALL_CONDITIONS if ranker is not None else BASELINE_CONDITIONS)
    if "history_learned" in conditions and ranker is None:
        raise ValueError("history_learned requires the frozen TRAIN ranker")
    probabilities = address_probabilities(ranker, [row["features"] for row in discovery["addresses"]]) if ranker is not None else [None] * len(discovery["addresses"])
    addresses = [{**row, "learned_probability": None if probability is None else float(probability)}
                 for row, probability in zip(discovery["addresses"], probabilities)]

    def order(condition):
        if condition == "history_current":
            return sorted(addresses, key=lambda row: (-row["features"]["current_best_cosine"],
                -row["features"]["max_lexical_overlap_current"], -row["features"]["max_history_rrf"], tuple(row["address"])))
        if condition == "history_learned":
            return sorted(addresses, key=lambda row: (-row["learned_probability"], tuple(row["address"])))
        return sorted(addresses, key=lambda row: (-row["features"]["max_history_rrf"], tuple(row["address"])))

    selected = {}
    for condition in conditions:
        if condition not in ALL_CONDITIONS:
            raise ValueError(f"Unknown HiDate reranking condition: {condition}")
        diagnostics = []
        if condition.startswith("asof_"):
            candidates = _hit_candidates(discovery["asof"][condition[5:]], chunks, pair_budget)
        elif condition == "history_unbound":
            candidates = _hit_candidates(discovery["history"]["hybrid"], chunks, pair_budget)
            diagnostics = order(condition)
        else:
            diagnostics, candidates, seen = order(condition), [], set()
            for rank, row in enumerate(diagnostics, 1):
                chunk_id = row["current_best_chunk_id"]
                if row["status"] != "available" or chunk_id is None:
                    continue
                chunk = chunks[chunk_id]
                if chunk["parent_passage_id"] in seen:
                    continue
                seen.add(chunk["parent_passage_id"])
                original_score = (row["learned_probability"] if condition == "history_learned"
                    else row["features"]["current_best_cosine"] if condition == "history_current"
                    else row["features"]["max_history_rrf"])
                candidates.append({**chunk, "address": row["address"], "features": row["features"],
                    "learned_probability": row["learned_probability"], "binding_status": row["status"],
                    "original_score": float(original_score), "address_rank": rank,
                    "selection_rank": len(candidates) + 1})
                if len(candidates) == pair_budget:
                    break
        selected[condition] = {"candidates": candidates, "address_diagnostics": diagnostics}
    return selected


class PairScoreCache:
    """One execution's exact raw-query/raw-document Ruri score cache."""

    def __init__(self, reranker, receipt_callback=None, batch_size=32):
        self.reranker, self.receipt_callback, self.batch_size = reranker, receipt_callback, batch_size
        self.cache = {}
        self.logical_pair_requests = self.physical_pair_count = self.model_call_count = 0
        self.encoded_input_tokens = self.padded_token_slots = 0
        self.model_wall_seconds_with_cuda_sync = 0.0
        self.sdpa_backend = None

    def score_pairs(self, pairs):
        self.logical_pair_requests += len(pairs)
        pending = list(dict.fromkeys(tuple(pair) for pair in pairs if tuple(pair) not in self.cache))
        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset:offset + self.batch_size]
            result = self.reranker.score(batch, include_inputs=True)
            if (len(result["scores"]) != len(batch) or len(result["logits"]) != len(batch)
                    or not all(math.isfinite(value) for value in result["scores"] + result["logits"])):
                raise ValueError("Ruri reranker returned invalid pair scores or logits")
            if len(result["token_counts"]) != len(batch) or sum(result["token_counts"]) != result["total_tokens"]:
                raise ValueError("Ruri reranker token receipt does not match the actual pair batch")
            for pair, score, logit, token_count in zip(batch, result["scores"], result["logits"], result["token_counts"]):
                self.cache[pair] = {"reranker_score": float(score), "reranker_logit": float(logit),
                                    "pair_token_count": int(token_count)}
            self.physical_pair_count += len(batch)
            self.model_call_count += 1
            self.encoded_input_tokens += result["total_tokens"]
            self.padded_token_slots += int(result["input_shape"][0] * result["input_shape"][1])
            self.model_wall_seconds_with_cuda_sync += result["seconds"]
            self.sdpa_backend = result["sdpa_backend"]
            if self.receipt_callback is not None:
                self.receipt_callback({"batch_number": self.model_call_count,
                    "pairs": [{"query_text": query, "document_text": document} for query, document in batch],
                    "result": result})
        return [self.cache[tuple(pair)] for pair in pairs]


def rerank_query(query, discovery, chunks, score_cache, *, conditions=None, ranker=None):
    selected = select_candidates(discovery, chunks, conditions=conditions, ranker=ranker)
    requested = [(query["query_text"], candidate["text"]) for row in selected.values() for candidate in row["candidates"]]
    scores = iter(score_cache.score_pairs(requested))
    output = {}
    for condition, selection in selected.items():
        candidates = [{**candidate, **next(scores)} for candidate in selection["candidates"]]
        ranked = sorted(candidates, key=lambda row: (-row["reranker_score"], row["selection_rank"], row["chunk_id"]))
        evidence, seen = [], set()
        for row in ranked:
            if row["parent_passage_id"] in seen:
                continue
            seen.add(row["parent_passage_id"])
            evidence.append(row)
            if len(evidence) == FINAL_PASSAGES:
                break
        output[condition] = {"requested_pair_count": len(candidates), "pair_budget": PAIR_BUDGET,
            "requested_input_tokens": sum(row["pair_token_count"] for row in candidates),
            "reranked_candidates": ranked, "final_evidence": evidence,
            "final_passage_ids": [row["parent_passage_id"] for row in evidence],
            "address_diagnostics": selection["address_diagnostics"]}
    return {"query_id": query["query_id"], "query_variant": query["query_variant"], "split": query["split"],
        "target_asof": query["target_asof"], "source_role": query["source_role"], "conditions": output}
