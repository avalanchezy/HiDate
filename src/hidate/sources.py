"""Scoped e-Gov provisions and the frozen HiDate source catalogue."""

import json
import random
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


LEGAL_CONTAINERS = {"Part", "Chapter", "Section", "Subsection", "Division"}
TEXT_BLOCKS = {"Paragraph", "Item", "Table", "TableStruct", "TableRow", "TableHeaderRow"} | {
    f"Subitem{i}" for i in range(1, 11)}


def _text(node, skip=()):
    if isinstance(node, str):
        return node
    if node["tag"] in skip:
        return ""
    children = node.get("children", [])
    if node["tag"] in {"TableRow", "TableHeaderRow"}:
        return "\t".join(_text(child, skip) for child in children)
    parts, previous_block = [], False
    for child in children:
        text = _text(child, skip)
        if not text:
            continue
        block = isinstance(child, dict) and child["tag"] in TEXT_BLOCKS
        if parts and (block or previous_block):
            parts.append("\n")
        parts.append(text)
        previous_block = block
    return "".join(parts)


def _flag(value):
    return value is True or value == "true"


def _children(node, path):
    for index, child in enumerate(node.get("children", [])):
        if isinstance(child, dict):
            yield child, path + [index]


def _direct_text(node, tags):
    return "".join(_text(child) for child, _ in _children(node, []) if child["tag"] in tags)


def extract_provisions(snapshot):
    """Extract source addresses without interpreting quoted NewProvision trees.

    ``tree_path`` contains exact child indices from ``law_full_text``. A scoped
    slot is a source address only; it does not assert lineage after renumbering.
    """
    tree = snapshot["law_full_text"]
    law_id = snapshot["law_info"]["law_id"]
    revision = snapshot["revision_info"]
    records = []
    occurrences = Counter()

    def add(node, path, scope, slot, deleted=False, hidden=False):
        tag = node["tag"]
        paragraphs = [(child, child_path) for child, child_path in _children(node, path)
                      if child["tag"] == "Paragraph"]
        if tag == "Paragraph":
            paragraphs = [(node, path)]
        paragraph_rows = []
        for child, child_path in paragraphs:
            paragraph_attrs = child.get("attr", {})
            paragraph_text = _text(child, {"ParagraphNum"})
            paragraph_rows.append({"num": paragraph_attrs.get("Num"),
                                   "text": paragraph_text, "tree_path": child_path,
                                   "deleted": _flag(paragraph_attrs.get("Delete")) or paragraph_text.strip() == "削除",
                                   "hidden": _flag(paragraph_attrs.get("Hide"))})
        titles = {"ArticleTitle"} if tag == "Article" else {tag + "Title"}
        caption = _direct_text(node, {"ArticleCaption"})
        title = _direct_text(node, titles)
        ignored = titles | {"ArticleCaption", "ParagraphNum", "SupplProvisionLabel"}
        if tag == "Article":
            text = "\n".join(_text(child, {"ParagraphNum"}) for child in node.get("children", [])
                             if not isinstance(child, dict) or child["tag"] not in ignored)
        else:
            text = _text(node, ignored)
        attrs = node.get("attr", {})
        records.append({"law_id": law_id, "revision_id": revision["law_revision_id"],
                        "law_title": revision["law_title"], "scope": scope,
                        "slot_id": f"{law_id}:{scope}:{slot}",
                        "article_num": attrs.get("Num") if tag == "Article" else None,
                        "title": title, "caption": caption, "text": text,
                        "deleted": deleted or _flag(attrs.get("Delete")) or text.strip() == "削除",
                        "hidden": hidden or _flag(attrs.get("Hide")),
                        "paragraphs": paragraph_rows, "tree_path": path})

    def scoped(node, path, scope, deleted=False, hidden=False):
        deleted = deleted or _flag(node.get("attr", {}).get("Delete"))
        hidden = hidden or _flag(node.get("attr", {}).get("Hide"))
        for child, child_path in _children(node, path):
            tag = child["tag"]
            if tag == "Article":
                add(child, child_path, scope, child.get("attr", {}).get("Num"), deleted, hidden)
            elif tag in LEGAL_CONTAINERS:
                scoped(child, child_path, scope, deleted, hidden)
            elif tag == "Paragraph":
                add(child, child_path, scope, "Paragraph_" + str(child.get("attr", {}).get("Num")), deleted, hidden)
            elif tag.startswith(("Appdx", "SupplProvisionAppdx")):
                appendix(child, child_path, scope, deleted, hidden)

    def appendix(node, path, parent_scope="", deleted=False, hidden=False):
        tag = node["tag"]
        base = (parent_scope + ":" if parent_scope else "") + tag
        occurrences[base] += 1
        num = node.get("attr", {}).get("Num") or str(occurrences[base])
        scope = base + ":" + num
        # Num can be absent, and repeated blocks must still be distinct.
        occurrences[scope] += 1
        if occurrences[scope] > 1:
            scope += "#" + str(occurrences[scope])
        add(node, path, scope, "block", deleted, hidden)

    def envelope(node, path):
        tag = node["tag"]
        if tag == "MainProvision":
            scoped(node, path, "MainProvision")
        elif tag == "SupplProvision":
            attrs = node.get("attr", {})
            amendment = attrs.get("AmendLawNum", attrs.get("AmendmentLawNum")) or "original"
            scope = "SupplProvision:" + amendment
            occurrences[scope] += 1
            if occurrences[scope] > 1:
                scope += "#" + str(occurrences[scope])
            scoped(node, path, scope)
        elif tag.startswith(("Appdx", "SupplProvisionAppdx")):
            appendix(node, path)
        elif tag in {"Law", "LawBody"}:
            for child, child_path in _children(node, path):
                envelope(child, child_path)

    envelope(tree, [])
    return records


def source_exclusions(law_list):
    """Keep the complete LawQA law and guidance source reservation."""
    ids, titles, guidance = [], [], []
    for source in law_list:
        url = urlparse(source["url"])
        parts = url.path.rstrip("/").split("/")
        if url.hostname == "laws.e-gov.go.jp" and len(parts) >= 3 and parts[-2] == "law":
            ids.append(parts[-1])
            titles.append(source["title"])
        else:
            guidance.append(dict(source))
    return {"law_ids": sorted(set(ids)), "guidance": guidance,
            "family_titles": sorted(set(titles))}


def catalogue_selection(laws, excluded_ids, excluded_titles, count_per_type=40, seed=42, protocol=None):
    """Select metadata-eligible law IDs using one seed42 shuffle per type.

    Returned official catalogue entries retain all raw metadata and add only
    ``family_id``. Only ``revision_info`` describes the requested as-of state.
    """
    defaults = json.loads((Path(__file__).resolve().parents[2] / "configs/source-protocol.json").read_text(encoding="utf-8"))
    settings = defaults | (protocol or {})
    acts = sorted([(row["revision_info"]["law_title"], row["law_info"]["law_id"])
                   for row in laws if row["law_info"]["law_type"] == "Act"],
                  key=lambda pair: (-len(pair[0]), pair[1]))

    def family(title, law_id, law_type):
        if law_type != "Act":
            for act_title, act_id in acts:
                if title.startswith(act_title):
                    return act_id
        return law_id

    reserved = set(excluded_ids)
    reserved_titles = set(excluded_titles)
    reserved_families = set()
    for row in laws:
        info, revision = row["law_info"], row["revision_info"]
        if info["law_id"] in reserved or revision["law_title"] in reserved_titles:
            reserved_families.add(family(revision["law_title"], info["law_id"], info["law_type"]))
    for title in reserved_titles:
        for act_title, act_id in acts:
            if title.startswith(act_title):
                reserved_families.add(act_id)
                break
    selected = []
    for law_type in settings["law_types"]:
        eligible = {}
        for row in laws:
            info, revision = row["law_info"], row["revision_info"]
            law_id, title = info["law_id"], revision["law_title"]
            family_id = family(title, law_id, info["law_type"])
            promulgation = info.get("promulgation_date")
            enforcement = revision.get("amendment_enforcement_date")
            reserved_title = title in reserved_titles or (
                info["law_type"] != "Act" and family_id == law_id
                and any(title.startswith(prefix) for prefix in reserved_titles))
            if (info["law_type"] != law_type or law_id in reserved or family_id in reserved_families
                    or reserved_title
                    or not promulgation or promulgation > settings["promulgation_at_most"]
                    or not enforcement or not (settings["enforcement_after"] < enforcement <= settings["enforcement_at_most"])
                    or revision.get("current_revision_status") not in settings["catalogue_statuses"]
                    or revision.get("amendment_type") != settings["catalogue_amendment_type"]
                    or revision.get("mission") != settings["catalogue_mission"]
                    or revision.get("repeal_status") != settings["catalogue_repeal_status"]):
                continue
            eligible[law_id] = dict(row, family_id=family_id)
        ids = sorted(eligible)
        random.Random(seed).shuffle(ids)
        selected.extend(eligible[law_id] for law_id in ids[:count_per_type])
    return selected


def lawqa_records(payload):
    """Preserve released question units even when generic question stems repeat."""
    return [dict(sample, group_id=sample["ファイル名"]) for sample in payload["samples"]]
