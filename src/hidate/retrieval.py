"""HiDate retrieval primitives; tokenization and model vectors are supplied by callers."""

import math
from array import array
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from .sources import _flag


def _guidance(record):
    return record.get("source_kind") == "guidance" or ("source_id" in record and "law_id" not in record)


def chunk_passages(records, tokenizer, max_body_tokens=448, overlap_tokens=64):
    """Return literal body spans using a supplied fast tokenizer's character offsets.

    Tokenize bodies without special tokens. ``header_text`` contains natural source
    labels, and ``text`` joins that header and the body for retrieval. Whole source
    bodies remain in the parent corpus rather than being copied into every chunk.
    """
    if max_body_tokens <= 0 or not 0 <= overlap_tokens < max_body_tokens:
        raise ValueError("overlap_tokens must be nonnegative and smaller than max_body_tokens")
    chunks = []
    for record in records:
        if _flag(record.get("deleted")) or _flag(record.get("hidden")):
            continue
        guidance = _guidance(record)
        body = record["text"] if guidance else record["visible_text"]
        if not body.strip():
            continue
        parent = record["source_id"] if guidance else record["passage_id"]
        fields = ("title",) if guidance else ("law_title", "title", "caption")
        header = " ".join(record.get(field, "") for field in fields if record.get(field))
        encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = encoded["offset_mapping"]
        metadata = {key: value for key, value in record.items()
                    if key not in {"text", "visible_text", "paragraphs", "visible_paragraphs"}}
        start, number = 0, 0
        while start < len(offsets):
            end = min(start + max_body_tokens, len(offsets))
            char_start, char_end = int(offsets[start][0]), int(offsets[end - 1][1])
            text = body[char_start:char_end]
            chunks.append({**metadata, "chunk_id": parent + f"|chunk{number}",
                "parent_passage_id": parent, "source_kind": "guidance" if guidance else "statute",
                "header_text": header, "body_text": text, "text": header + "\n" + text if header else text,
                "token_start": start, "token_end": end, "body_token_count": end - start,
                "char_start": char_start, "char_end": char_end})
            if end == len(offsets):
                break
            start, number = end - overlap_tokens, number + 1
    return chunks


def _rank(scores, top_k):
    return [{"chunk_id": chunk_id, "score": float(score)}
            for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max(0, top_k)]]


def bm25_rank(query_tokens, document_tokens, top_k=50, k1=1.2, b=0.75):
    """BM25 over supplied Japanese token lists, keyed by chunk ID.

    Use positive Robertson IDF ``log(1 + (N-df+.5)/(df+.5))`` and deterministic
    chunk-ID tie breaking. Zero-score documents can fill the remaining budget.
    """
    if not document_tokens:
        return []
    frequencies = {chunk_id: Counter(tokens) for chunk_id, tokens in document_tokens.items()}
    df = Counter()
    for frequency in frequencies.values():
        df.update(frequency.keys())
    n = len(frequencies)
    average_length = sum(map(len, document_tokens.values())) / n or 1
    query = Counter(query_tokens)
    scores = {}
    for chunk_id, frequency in frequencies.items():
        length = len(document_tokens[chunk_id])
        normalization = k1 * (1 - b + b * length / average_length)
        scores[chunk_id] = sum(query_count * math.log(1 + (n - df[term] + .5) / (df[term] + .5))
            * frequency[term] * (k1 + 1) / (frequency[term] + normalization)
            for term, query_count in query.items() if frequency[term])
    return _rank(scores, top_k)


class BM25Index:
    """Frozen BM25 document weights stored as sparse term postings.

    Build once per eligible bank. Float64 weights retain the reference score
    precision; the final CSC matrix stores one entry per document/term. Original
    token lists and Python document Counters are not retained by the index.
    """

    def __init__(self, document_tokens, k1=1.2, b=0.75):
        self.chunk_ids = sorted(document_tokens)
        self.vocabulary = {}
        self.k1, self.b = k1, b
        self.lengths = np.array([len(document_tokens[chunk_id]) for chunk_id in self.chunk_ids], dtype=np.float64)
        n = len(self.chunk_ids)
        self.average_length = (self.lengths.sum() / n if n else 1) or 1
        columns, frequencies, pointers = array("i"), array("d"), array("q", [0])
        df = []
        for chunk_id in self.chunk_ids:
            for term, frequency in Counter(document_tokens[chunk_id]).items():
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                    df.append(0)
                column = self.vocabulary[term]
                columns.append(column)
                frequencies.append(frequency)
                df[column] += 1
            pointers.append(len(columns))
        document_frequency = np.asarray(df, dtype=np.float64)
        self.idf = np.log1p((n - document_frequency + .5) / (document_frequency + .5))
        weights = csr_matrix((np.frombuffer(frequencies, dtype=np.float64),
            np.frombuffer(columns, dtype=np.int32), np.frombuffer(pointers, dtype=np.int64)),
            shape=(n, len(self.vocabulary)))
        for row, length in enumerate(self.lengths):
            start, end = weights.indptr[row:row + 2]
            tf = weights.data[start:end]
            normalization = k1 * (1 - b + b * length / self.average_length)
            weights.data[start:end] = self.idf[weights.indices[start:end]] * tf * (k1 + 1) / (tf + normalization)
        self.weights = weights.tocsc()

    def rank(self, query_tokens, top_k=50):
        limit = min(max(0, top_k), len(self.chunk_ids))
        if not limit:
            return []
        query = Counter(query_tokens)
        columns = [self.vocabulary[term] for term in query if term in self.vocabulary]
        counts = [query[term] for term in query if term in self.vocabulary]
        if not columns:
            return [{"chunk_id": chunk_id, "score": 0.} for chunk_id in self.chunk_ids[:limit]]
        scores = np.asarray(self.weights[:, columns] @ np.asarray(counts)).ravel()
        cutoff = np.partition(scores, len(scores) - limit)[len(scores) - limit]
        above = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)[:limit - len(above)]
        chosen = np.concatenate((above, tied))
        chosen = chosen[np.lexsort((chosen, -scores[chosen]))]
        return [{"chunk_id": self.chunk_ids[row], "score": float(scores[row])} for row in chosen]


def cosine_topk(query_vector, document_vectors, chunk_ids, top_k=50):
    """Rank supplied L2-normalized vectors by their dot-product cosine score."""
    if not len(chunk_ids):
        return []
    scores = np.asarray(document_vectors) @ np.asarray(query_vector)
    return _rank(dict(zip(chunk_ids, scores)), top_k)


def rrf(rankings, k=60, top_k=50):
    """Fuse hit lists using one reciprocal-rank contribution per ID per list."""
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for hit in ranking:
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] += 1 / (k + len(seen))
    return _rank(scores, top_k)


def eligible_chunks(corpus, target_asof, mode="asof"):
    """Filter statutes by captured bindings; static guidance stays available.

    The caller selects full or external roles before this function. As-of uses
    an exact captured target date; history permits captured anchors <= target.
    """
    if mode not in {"asof", "history"}:
        raise ValueError("mode must be asof or history")
    selected, seen = [], set()
    for record in corpus:
        dates = record.get("asof_dates", [])
        allowed = _guidance(record) or (target_asof in dates if mode == "asof" else any(date <= target_asof for date in dates))
        if allowed and record["chunk_id"] not in seen:
            selected.append(record)
            seen.add(record["chunk_id"])
    return selected


def aggregate_history(hits, chunk_index):
    """Use each scoped address/revision's maximum chunk score as one history vote.

    Static guidance is grouped separately by source_id. Its one captured source
    contributes one source vote; no legal revision or date is manufactured.
    """
    groups = {}
    for hit in hits:
        chunk = chunk_index[hit["chunk_id"]]
        guidance = _guidance(chunk)
        number = chunk.get("article_num")
        if guidance:
            address = ("guidance", chunk["source_id"])
            revision = chunk["source_id"]
        else:
            if number is None:
                number = chunk["slot_id"].rsplit(":", 1)[-1]
            address = (chunk["law_id"], chunk["scope"], number)
            revision = chunk["revision_id"]
        if address not in groups:
            metadata = ({"source_id": chunk["source_id"], "url": chunk.get("url")} if guidance
                        else {"law_id": chunk["law_id"], "scope": chunk["scope"], "article_num": chunk.get("article_num")})
            groups[address] = {**metadata, "address": address,
                "source_kind": "guidance" if guidance else "statute", "scores": {},
                "best_chunk_ids": {}, "parents": set()}
        group = groups[address]
        score = float(hit["score"])
        previous = group["scores"].get(revision)
        if previous is None or score > previous:
            group["scores"][revision] = score
            group["best_chunk_ids"][revision] = hit["chunk_id"]
        group["parents"].add(chunk["parent_passage_id"])
    rows = []
    for group in groups.values():
        scores = group.pop("scores")
        parents = group.pop("parents")
        group.update(max_score=max(scores.values()), mean_score=sum(scores.values()) / len(scores),
            support_count=len(scores), parent_passage_ids=sorted(parents),
            revision_scores={} if group["source_kind"] == "guidance" else scores)
        rows.append(group)
    return sorted(rows, key=lambda row: (-row["max_score"], row["address"]))


def bind_addresses(addresses, target_asof, target_index, bindings):
    """Resolve aggregate addresses against actual dated maps and complete bindings.

    ``target_index`` maps four-part dated keys to saved target-address rows;
    ``bindings`` maps (law_id, date) to collection receipts. Only available targets
    populate ``parent_passage_ids`` for evidence. Other known parent IDs remain
    in ``target_passage_ids`` as provenance. Guidance binds to its static source.
    """
    bound = []
    for address in addresses:
        if _guidance(address):
            bound.append({**address, "status": "available"})
            continue
        law_id, scope, number = address["address"]
        key = (law_id, target_asof, scope, number)
        binding = bindings.get((law_id, target_asof))
        target = target_index.get(key)
        if binding is None or binding["status"] != "collected":
            status, target_ids = "collection_failure", []
        elif target is None:
            status, target_ids = "absent", []
        else:
            status, target_ids = target["status"], target["passage_ids"]
        bound.append({**address, "address_key": key, "target_asof": target_asof,
            "status": status, "revision_id": binding.get("revision_id") if binding else None,
            "target_passage_ids": target_ids,
            "parent_passage_ids": target_ids if status == "available" else []})
    return bound
