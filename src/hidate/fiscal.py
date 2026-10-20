"""Frozen, zero-shot FiscalQA R3 adaptation of the HiDate address mechanism.

FiscalQA data and the scorer are CC-BY-4.0; the official harness is MIT.
Date selection and query windowing follow rosecymbler/fiscal-fr-bench.
No French labels enter discovery, features, candidate ranking, or prompts.
"""

from collections import Counter, defaultdict
from datetime import date
import re
import unicodedata

import numpy as np

from .discovery import FEATURE_NAMES
from .ranking import address_probabilities
from .retrieval import BM25Index, rrf


CGI = "LEGITEXT000006069577"
CONDITIONS = ("closed_book", "static_bm25", "static_dense", "asof_bm25", "asof_dense",
              "asof_hybrid", "history_unbound", "history_bound", "history_current",
              "history_learned", "oracle_asof")
SYSTEM = ("Tu es un assistant juridique fiscal. Réponds de façon précise et "
          "concise à la question, en citant l'article applicable et la valeur "
          "chiffrée exacte. Si un extrait de texte légal est fourni, fonde ta "
          "réponse dessus.")


def date_text(value):
    return None if value is None else str(value)[:10]


def usable(row):
    return row["etat"].upper() != "MODIFIE_MORT_NE" and row.get("date_debut") is not None


def select_versions(rows, anchor, *, current=False):
    """Official inclusive interval, latest start/end, NULL end sorts first.

    Static B first prefers VIGUEUR but never VIGUEUR_DIFF, exactly as the
    public harness. Its date fallback uses the declared frozen snapshot date.
    """
    grouped = defaultdict(list)
    for row in rows:
        if usable(row):
            grouped[row["cid"]].append(row)
    chosen = {}
    for cid, versions in grouped.items():
        flagged = [r for r in versions if r["etat"].upper().startswith("VIGUEUR")
                   and not r["etat"].upper().startswith("VIGUEUR_DIFF")] if current else []
        candidates = flagged or [r for r in versions if r["date_debut"] <= anchor
                                and (r["date_fin"] is None or anchor <= r["date_fin"])]
        if candidates:
            chosen[cid] = max(candidates, key=lambda r: (r["date_debut"], r["date_fin"] or "9999-12-31", r["id"]))
    return chosen


def french_tokens(text):
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def query_window(text, query, budget=8000, size=1500, overlap=250):
    """The official head-plus-query-relevant-chunks window, without gold input."""
    text = (text or "").strip()
    if len(text) <= size:
        return text[:budget]
    chunks = [(i, text[i:i + size]) for i in range(0, len(text), size - overlap)
              if text[i:i + size].strip()]
    query_terms = {t for t in french_tokens(query) if len(t) > 2}
    rest = sorted(chunks[1:], key=lambda c: -sum(t in query_terms for t in set(french_tokens(c[1])) if len(t) > 2))
    picked, used = [chunks[0]], len(chunks[0][1])
    for offset, chunk in rest:
        if used + len(chunk) > budget:
            if used >= budget * .9:
                break
            continue
        picked.append((offset, chunk))
        used += len(chunk)
    return "".join(chunk for _, chunk in sorted(picked))[:budget]


def article_chunks(rows, size=1500, overlap=250):
    chunks = []
    for row in rows:
        if not usable(row):
            continue
        body = (row["texte_clean"] or "").strip()
        for offset in range(0, len(body), size - overlap):
            text = body[offset:offset + size]
            if not text.strip():
                continue
            chunks.append({"chunk_id": f"{row['id']}|{offset}", "version_id": row["id"],
                "article_cid": row["cid"], "article_num": row["num"], "code_id": row["code_id"],
                "date_debut": row["date_debut"], "date_fin": row["date_fin"],
                "char_start": offset, "char_end": offset + len(text),
                "text": f"Article {row['num']} CGI\n{text}", "body_text": text})
            if offset + size >= len(body):
                break
    return chunks


class FiscalDiscovery:
    """One fixed full-CGI BM25 index; date masks precede ranking for every bank.

    Corpus-wide IDF stays fixed across dates. Dense uses max chunk evidence per
    article-version; all history starts at or before the question date. A CID
    discovered in history is hydrated only with its date-applicable version.
    """

    def __init__(self, rows, chunks, vectors, *, top_k=50, rrf_k=60, static_date="2026-10-18"):
        self.rows, self.chunks = rows, {c["chunk_id"]: c for c in chunks}
        self.ids, self.vectors = [c["chunk_id"] for c in chunks], np.asarray(vectors)
        self.top_k, self.rrf_k, self.static_date = top_k, rrf_k, static_date
        self.tokens = {c["chunk_id"]: french_tokens(c["text"]) for c in chunks}
        self.lexical = BM25Index(self.tokens)
        lexical_rows = {identity: i for i, identity in enumerate(self.lexical.chunk_ids)}
        self.lexical_order = np.array([lexical_rows[identity] for identity in self.ids])
        self.row_index = {identity: i for i, identity in enumerate(self.ids)}
        self.by_version = defaultdict(list)
        for chunk in chunks:
            self.by_version[chunk["version_id"]].append(chunk["chunk_id"])
        self.static = select_versions(rows, static_date, current=True)

    def lexical_scores(self, tokens):
        query = Counter(tokens)
        terms = [term for term in query if term in self.lexical.vocabulary]
        if not terms:
            return np.zeros(len(self.ids))
        columns = [self.lexical.vocabulary[term] for term in terms]
        scores = np.asarray(self.lexical.weights[:, columns] @ np.array([query[t] for t in terms])).ravel()
        return scores[self.lexical_order]

    def rank(self, eligible, scores):
        indices = np.array([i for i, identity in enumerate(self.ids) if self.chunks[identity]["version_id"] in eligible])
        if not len(indices):
            return []
        order = np.lexsort((np.array(self.ids)[indices], -scores[indices]))[:self.top_k]
        return [{"chunk_id": self.ids[i], "score": float(scores[i]),
                 "version_id": self.chunks[self.ids[i]]["version_id"],
                 "article_cid": self.chunks[self.ids[i]]["article_cid"]} for i in indices[order]]

    def discover(self, text, anchor, query_vector):
        tokens = french_tokens(text)
        lexical, cosine = self.lexical_scores(tokens), self.vectors @ np.asarray(query_vector)
        asof = select_versions(self.rows, anchor)
        banks = {"asof": {r["id"] for r in asof.values()},
                 "history": {r["id"] for r in self.rows if usable(r) and r["date_debut"] <= anchor},
                 "static": {r["id"] for r in self.static.values()}}
        result = {"anchor": anchor, "bank_version_counts": {k: len(v) for k, v in banks.items()}}
        for bank, eligible in banks.items():
            lexical_hits, dense_hits = self.rank(eligible, lexical), self.rank(eligible, cosine)
            hybrid = rrf([lexical_hits, dense_hits], k=self.rrf_k, top_k=self.top_k)
            hybrid = [{**h, "article_cid": self.chunks[h["chunk_id"]]["article_cid"],
                       "version_id": self.chunks[h["chunk_id"]]["version_id"]} for h in hybrid]
            result[bank] = {"bm25": lexical_hits, "dense": dense_hits, "hybrid": hybrid}
        addresses, support = [], defaultdict(list)
        for hit in result["history"]["hybrid"]:
            support[hit["article_cid"]].append(hit)
        ranks = {}
        for method, hits in result["history"].items():
            ranks[method] = {}
            for rank, hit in enumerate(hits, 1):
                ranks[method].setdefault(hit["article_cid"], rank)
        for cid, hits in support.items():
            bound = asof.get(cid)
            target_chunks = self.by_version[bound["id"]] if bound else []
            best = min(target_chunks, key=lambda k: (-cosine[self.row_index[k]], k)) if target_chunks else None
            revision_scores = {}
            for hit in hits:
                revision_scores[hit["version_id"]] = max(revision_scores.get(hit["version_id"], 0.), hit["score"])
            terms = set(tokens)
            overlap = max((len(terms.intersection(self.tokens[k])) / len(terms) for k in target_chunks), default=0.) if terms else 0.
            latest = max(self.chunks[h["chunk_id"]]["date_debut"] for h in hits)
            features = {"max_history_rrf": max(revision_scores.values()),
                "mean_history_rrf": sum(revision_scores.values()) / len(revision_scores),
                "distinct_revision_count": len(revision_scores),
                "history_best_rank_rrf": ranks["hybrid"].get(cid, self.top_k + 1),
                "history_best_rank_dense": ranks["dense"].get(cid, self.top_k + 1),
                "history_best_rank_bm25": ranks["bm25"].get(cid, self.top_k + 1),
                "current_best_cosine": float(cosine[self.row_index[best]]) if best else -1.,
                "max_lexical_overlap_current": overlap, "available_target": float(bound is not None),
                "history_latest_anchor_gap_years": max(0., (date.fromisoformat(anchor) - date.fromisoformat(latest)).days / 365.25)}
            addresses.append({"article_cid": cid, "bound_version_id": bound["id"] if bound else None,
                              "current_best_chunk_id": best, "features": features,
                              "history_chunk_ids": [h["chunk_id"] for h in hits]})
        result["addresses"] = sorted(addresses, key=lambda a: (-a["features"]["max_history_rrf"], a["article_cid"]))
        return result


def select_conditions(discovery, chunks, rows, *, ranker=None, pair_budget=10):
    """Original seven HiDate candidates plus static controls; no oracle input."""
    selected = {}

    def distinct(hits):
        seen, output = set(), []
        for hit in hits:
            chunk = chunks[hit["chunk_id"]]
            if chunk["version_id"] not in seen:
                seen.add(chunk["version_id"])
                output.append({**chunk, "discovery_score": hit["score"]})
            if len(output) == pair_budget:
                break
        return output

    for bank in ("static", "asof"):
        for method in (("bm25", "dense") if bank == "static" else ("bm25", "dense", "hybrid")):
            selected[f"{bank}_{method}"] = distinct(discovery[bank][method])
    selected["history_unbound"] = distinct(discovery["history"]["hybrid"])
    addresses = discovery["addresses"]
    orders = {"history_bound": addresses,
              "history_current": sorted(addresses, key=lambda a: (-a["features"]["current_best_cosine"],
                  -a["features"]["max_lexical_overlap_current"], -a["features"]["max_history_rrf"], a["article_cid"]))}
    if ranker is not None:
        probabilities = address_probabilities(ranker, [a["features"] for a in addresses])
        scored = [{**a, "learned_probability": float(p)} for a, p in zip(addresses, probabilities)]
        orders["history_learned"] = sorted(scored, key=lambda a: (-a["learned_probability"], a["article_cid"]))
    for condition, ordered in orders.items():
        selected[condition] = [{**chunks[a["current_best_chunk_id"]],
                               "learned_probability": a.get("learned_probability"),
                               "features": a["features"]} for a in ordered
                              if a["current_best_chunk_id"] is not None][:pair_budget]
    return selected


def reader_messages(prompt, context):
    user = prompt if not context else (f'Extrait de l\'article applicable :\n"""\n{context}\n"""\n\nQuestion : {prompt}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def pack_reader(prompt, candidates, versions, tokenizer, *, prompt_budget=4096, max_passages=4):
    """Official query windows followed by exact fixed-reader prompt budgeting."""
    def context(items):
        return "\n\n".join(item["text"] for item in items)

    def tokens(items):
        return len(tokenizer.apply_chat_template(reader_messages(prompt, context(items)), tokenize=True, add_generation_prompt=True))

    if tokens([]) > prompt_budget:
        raise ValueError("The complete FiscalQA question exceeds the fixed reader prompt budget")
    picked = []
    for candidate in candidates[:max_passages]:
        row = versions[candidate["version_id"]]
        body = query_window(row["texte_clean"], prompt)
        header = f"=== Article {row['num']} (en vigueur {row['date_debut']}..{row['date_fin']}) ===\n"
        item = {"article_cid": row["cid"], "version_id": row["id"], "text": header + body,
                "truncated": False, "window_characters": len(body)}
        if tokens(picked + [item]) <= prompt_budget:
            picked.append(item)
            continue
        low, high = 0, len(body)
        while low < high:
            middle = (low + high + 1) // 2
            trial = {**item, "text": header + body[:middle]}
            if tokens(picked + [trial]) <= prompt_budget:
                low = middle
            else:
                high = middle - 1
        if low:
            picked.append({**item, "text": header + body[:low], "truncated": True})
        break
    return {"messages": reader_messages(prompt, context(picked)), "evidence": picked,
            "prompt_tokens": tokens(picked), "prompt_budget": prompt_budget}


def score_answer(question, text, nuggets, evidence, scorer):
    """Official answer scoring plus explicitly separate retrieval/citation metrics."""
    normalized = scorer.normalize(text)
    numbers = scorer.parse_numbers(normalized)
    nuggets = [n for n in nuggets if n["kind"] != "provenance"]
    hits = [scorer.match_nugget(n, normalized, numbers) for n in nuggets]
    required = [hit for hit, nugget in zip(hits, nuggets) if nugget.get("required", True)]
    gold = question["gold"]
    version = gold["version_id"]
    cid = gold["article_cid"]
    article_nuggets = [(h, n) for h, n in zip(hits, nuggets) if n.get("nugget_id", "").endswith("-art")]
    return {"coverage": sum(hits) / len(hits) if hits else None,
            "strict": int(all(required) if required else all(hits)) if hits else None,
            "gold_article_recall_at_four": int(any(e["article_cid"] == cid for e in evidence)),
            "gold_version_recall_at_four": int(any(e["version_id"] == version for e in evidence)),
            "top_article_correct": int(bool(evidence) and evidence[0]["article_cid"] == cid),
            "top_version_correct": int(bool(evidence) and evidence[0]["version_id"] == version),
            "retrieved_version_id": evidence[0]["version_id"] if evidence else None,
            "article_citation_nugget": int(all(h for h, _ in article_nuggets)) if article_nuggets else None,
            "version_id_cited": int(version in text),
            "per_nugget_hits": {n.get("nugget_id", str(i)): bool(h) for i, (n, h) in enumerate(zip(nuggets, hits))}}


def summarize(scores, questions):
    scopes = {q["qid"]: q.get("scope", "in") for q in questions}
    metrics = ("strict", "coverage", "gold_article_recall_at_four", "gold_version_recall_at_four",
               "top_article_correct", "top_version_correct", "article_citation_nugget", "version_id_cited")
    strata = {"scored209": {qid for qid, scope in scopes.items() if scope == "in"},
              "published221": set(scopes),
              "excluded_review": {qid for qid, scope in scopes.items() if scope == "excluded_review"},
              "excluded_arrete": {qid for qid, scope in scopes.items() if scope == "excluded_arrete"}}
    output = {}
    for stratum, qids in strata.items():
        grouped = defaultdict(list)
        for row in scores:
            if row["qid"] in qids:
                grouped[row["condition"]].append(row)
        output[stratum] = {}
        for condition, rows in grouped.items():
            summary = {"n": len(rows), "expected_n": len(qids)}
            for metric in metrics:
                values = [r[metric] for r in rows if r.get(metric) is not None]
                summary[metric] = sum(values) / len(values) if values else None
            accepted = [r for r in rows if r.get("accepted", True)]
            summary["accepted_n"] = len(accepted)
            summary["acceptance_coverage"] = len(accepted) / len(rows)
            strict = [r["strict"] for r in accepted if r.get("strict") is not None]
            summary["accepted_strict"] = sum(strict) / len(strict) if strict else None
            output[stratum][condition] = summary
    return output
