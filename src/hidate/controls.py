"""Build source-removal banks from held-out labels before model evaluation."""

from collections import defaultdict
import unicodedata


def _wording(text):
    return "".join(unicodedata.normalize("NFC", text).split())


def _address(parent):
    number = parent["article_num"]
    if number is None:
        number = parent["slot_id"].rsplit(":", 1)[-1]
    return parent["law_id"], parent["scope"], number


def controlled_date_bank(chunks, corpus_parents, target_index, queries, target_date):
    """Remove all TEST gold targets at one date from one shared evaluation bank.

    Exact old text copies of the same scoped address are removed with the target.
    Different older wording remains available for discovery. The retriever receives
    only the resulting bank and target map, never the benchmark query gold.
    """
    parents = {row["passage_id"]: row for row in corpus_parents}
    by_address = defaultdict(list)
    for parent in corpus_parents:
        by_address[_address(parent)].append(parent)
    keys = {tuple(query["target_key"]) for query in queries
            if query["split"] == "test" and query["target_asof"] == target_date}
    target_copy = dict(target_index)
    removed, target_parents, blocked = set(), set(), []
    for key in sorted(keys):
        target = target_index.get(key)
        if target is None or target["status"] != "available":
            continue
        current = target["passage_ids"]
        texts = {_wording(parents[identity]["visible_text"]) for identity in current}
        address = (key[0], key[2], key[3])
        target_parents.update(current)
        removed.update(row["passage_id"] for row in by_address[address]
                       if _wording(row["visible_text"]) in texts)
        removed.update(current)
        target_copy[key] = {**target, "status": "corpus_support_unavailable"}
        blocked.append(list(key))
    kept = [chunk for chunk in chunks if chunk["parent_passage_id"] not in removed]
    return {"chunks": kept, "target_index": target_copy,
        "audit": {"target_date": target_date, "blocked_addresses": blocked,
            "removed_parent_ids": sorted(removed), "target_parent_ids": sorted(target_parents),
            "extra_exact_text_aliases": sorted(removed - target_parents),
            "removed_chunk_ids": [chunk["chunk_id"] for chunk in chunks
                                  if chunk["parent_passage_id"] in removed],
            "operation": "shared date-wise removal of all available TEST targets and exact wording aliases"}}
