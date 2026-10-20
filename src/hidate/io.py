"""Small JSONL helpers and explicit checks for experimental leakage."""

import json
import random
import unicodedata
from pathlib import Path


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def load_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _texts(row):
    return {normalize(t) for t in [row["source"], *row.get("references", [])] if normalize(t)}


def audit_splits(splits):
    issues, seen_ids, seen_groups, seen_texts = [], {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            key = row["id"]
            if key in seen_ids:
                issues.append({"kind": "duplicate_id", "id": key, "splits": [seen_ids[key], split]})
            seen_ids[key] = split
            group = row.get("group_id")
            if not group:
                issues.append({"kind": "missing_group", "id": key, "split": split})
            elif group in seen_groups and seen_groups[group] != split:
                issues.append({"kind": "group_overlap", "group_id": group, "splits": [seen_groups[group], split]})
            else:
                seen_groups[group] = split
            for text in _texts(row):
                if text in seen_texts and seen_texts[text] != split:
                    issues.append({"kind": "text_overlap", "id": key, "splits": [seen_texts[text], split]})
                else:
                    seen_texts[text] = split
    return issues


def split_rows(rows, seed=42, train_fraction=0.8, dev_fraction=0.1):
    """Group split, connecting exact duplicate source/reference texts as well.

    group_id must identify the actual grouping unit. Sentence groups are not
    document groups; the dataset manifest must state this limitation.
    """
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 - train_fraction:
        raise ValueError("Fractions must leave nonempty train/dev/test proportions")
    rows = sorted(rows, key=lambda r: r["id"])
    if any(not r.get("group_id") for r in rows):
        raise ValueError("Every row needs an explicit group_id")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ids")
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for i, row in enumerate(rows):
        keys = [("group", row["group_id"])] + [("text", t) for t in _texts(row)]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    n_train = round(len(groups) * train_fraction)
    n_dev = round(len(groups) * dev_fraction)
    if min(n_train, n_dev, len(groups) - n_train - n_dev) < 1:
        raise ValueError("Not enough independent groups for three splits")
    bounds = {"train": groups[:n_train], "dev": groups[n_train:n_train + n_dev], "test": groups[n_train + n_dev:]}
    return {split: [row for group in selected for row in group] for split, selected in bounds.items()}
