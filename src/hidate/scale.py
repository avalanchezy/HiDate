"""Nested distractor banks for the unchanged single-split HiDate TEST panel."""

from collections import defaultdict

import numpy as np


def nested_law_banks(selected, original, sizes=(120, 500, 1000)):
    ids = [row["law_info"]["law_id"] for row in selected]
    original_ids = [row["law_info"]["law_id"] for row in original]
    if ids[:len(original_ids)] != original_ids or len(set(ids)) != len(ids):
        raise ValueError("Nested banks must keep the original law order and unique law identities")
    if max(sizes) > len(ids):
        raise ValueError("The complete selected law list is required for every declared scale")
    return {str(size): ids[:size] for size in sizes}


def snapshot_plan(selected, original, anchors):
    original_ids = {row["law_info"]["law_id"] for row in original}
    return [(row["law_info"]["law_id"], anchor) for row in selected
            if row["law_info"]["law_id"] not in original_ids for anchor in anchors]


def add_passage(corpus, record):
    identity = record["passage_id"]
    if identity not in corpus:
        passage = {key: value for key, value in record.items() if key not in {"requested_asof", "raw_path"}}
        corpus[identity] = {**passage, "asof_dates": [], "raw_paths": []}
    passage = corpus[identity]
    if record["requested_asof"] not in passage["asof_dates"]:
        passage["asof_dates"].append(record["requested_asof"])
    if record["raw_path"] not in passage["raw_paths"]:
        passage["raw_paths"].append(record["raw_path"])


class SharedCosines:
    """Reuse one actual vector product when a query visits all three scales."""

    def __init__(self, vectors):
        self.vectors, self.query, self.scores = vectors, None, None
        self.products = 0

    def __matmul__(self, query):
        query = np.asarray(query)
        if self.query is None or not np.array_equal(query, self.query):
            self.query = query.copy()
            self.scores = self.vectors @ query
            self.products += 1
        return self.scores


def metric_panels(rows):
    cells = defaultdict(list)
    for row in rows:
        subsets = ["all", "stratum=" + row["stratum"]]
        if row["quote_absent"] is True:
            subsets.append("quote_absent")
        elif row["quote_absent"] is False:
            subsets.append("quote_present")
        for subset in subsets:
            cells[row["query_variant"] + "|" + subset].append(row)
    output = {}
    for name, values in cells.items():
        accepted = [row for row in values if row["accepted"]]
        output[name] = {"n": len(values), **{metric: sum(row[metric] for row in values) / len(values)
            for metric in ("top_parent_correct", "top_revision_correct", "target_mrr", "target_recall_at_four")},
            "accepted_n": len(accepted), "coverage": len(accepted) / len(values),
            "accepted_parent_precision": sum(row["top_parent_correct"] for row in accepted) / len(accepted) if accepted else None}
    return output
