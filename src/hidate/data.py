"""Pure preparation of dated HiDate address pairs and source-derived queries."""

import random
import unicodedata
from collections import defaultdict

from .sources import _flag, _text


SAMPLED_STRATA = {"same_date", "changed_cross_date", "unavailable_target"}


def family_split(selected_laws, seed=42):
    families = sorted({row["family_id"] for row in selected_laws})
    random.Random(seed).shuffle(families)
    train_end = int(0.70 * len(families))
    valid_end = train_end + int(0.15 * len(families))
    return {family: "train" if i < train_end else "valid" if i < valid_end else "test"
            for i, family in enumerate(families)}


def _at_path(tree, path):
    for index in path:
        tree = tree["children"][index]
    return tree


def _visible_tree(node):
    if isinstance(node, str):
        return node
    attrs = node.get("attr", {})
    if _flag(attrs.get("Delete")) or _flag(attrs.get("Hide")):
        return dict(node, children=[]) if node["tag"] in {"TableColumn", "TableHeaderColumn"} else None
    if node["tag"] == "Paragraph" and _text(node, {"ParagraphNum"}).strip() == "削除":
        return None
    children = [_visible_tree(child) for child in node.get("children", [])]
    return dict(node, children=[child for child in children if child is not None])


def visible_provision(record, snapshot):
    """Retain raw text while deriving text eligible for visible source evidence."""
    passage_id = record["revision_id"] + "|" + record["slot_id"] + "|" + ".".join(map(str, record["tree_path"]))
    result = dict(record, passage_id=passage_id, visible_text="", visible_paragraphs=[])
    if _flag(record.get("deleted")) or _flag(record.get("hidden")):
        return result
    tree = snapshot["law_full_text"]
    node = _visible_tree(_at_path(tree, record["tree_path"]))
    if node is None:
        return result
    if node["tag"] == "Article":
        result["visible_text"] = "\n".join(_text(child, {"ParagraphNum"}) for child in node["children"]
            if not isinstance(child, dict) or child["tag"] not in {"ArticleTitle", "ArticleCaption"})
    else:
        result["visible_text"] = _text(node, {node["tag"] + "Title", "ParagraphNum", "SupplProvisionLabel"})
    for paragraph in record["paragraphs"]:
        if _flag(paragraph.get("deleted")) or _flag(paragraph.get("hidden")):
            continue
        visible = _visible_tree(_at_path(tree, paragraph["tree_path"]))
        if visible is not None:
            result["visible_paragraphs"].append({"num": paragraph["num"],
                "text": _text(visible, {"ParagraphNum"}), "tree_path": paragraph["tree_path"]})
    return result


def address_index(provisions, bindings):
    """Keep each occurrence at its exact dated address, including duplicate paths."""
    addresses = defaultdict(list)
    for record in provisions:
        address = record["article_num"] if record["article_num"] is not None else record["slot_id"].rsplit(":", 1)[-1]
        addresses[(record["law_id"], record["requested_asof"], record["scope"], address)].append(record)
    return {"addresses": dict(addresses),
            "bindings": {(row["law_id"], row["requested_asof"]): row for row in bindings}}


def _normal_text(text):
    return "".join(unicodedata.normalize("NFC", text).split())


def _address_status(index, key):
    binding = index["bindings"].get(key[:2])
    records = index["addresses"].get(key, [])
    if binding is None or binding["status"] != "collected":
        return "collection_failure", None
    if len(records) > 1:
        return "ambiguous", None
    if not records:
        return "absent", None
    record = records[0]
    if _flag(record.get("deleted")):
        return "deleted", record
    if _flag(record.get("hidden")):
        return "hidden", record
    text = _normal_text(record["visible_text"])
    if text == "削除":
        return "deleted", record
    return ("available" if text else "no_substantive_text"), record


def address_pairs(index, anchor_dates, allocation):
    """Census past-to-target MainProvision addresses without copying article text."""
    pairs = []
    anchors = sorted(set(anchor_dates))
    for source_key, records in sorted(index["addresses"].items()):
        law_id, source_asof, scope, article_num = source_key
        origin = records[0]
        if (scope != "MainProvision" or origin["article_num"] is None
                or source_asof not in anchors or origin["source_role"] != "full"):
            continue
        family_id = origin["family_id"]
        if family_id not in allocation:
            continue
        source_status, source = _address_status(index, source_key)
        if source_status not in {"available", "ambiguous", "collection_failure"}:
            continue
        for target_asof in anchors:
            if target_asof < source_asof:
                continue
            target_key = (law_id, target_asof, scope, article_num)
            target_status, target = _address_status(index, target_key)
            if "collection_failure" in {source_status, target_status}:
                stratum = "collection_failure"
            elif "ambiguous" in {source_status, target_status}:
                stratum = "ambiguous"
            elif target_status != "available":
                stratum = "unavailable_target"
            elif source_asof == target_asof:
                stratum = "same_date"
            elif _normal_text(source["visible_text"]) == _normal_text(target["visible_text"]):
                stratum = "unchanged_cross_date"
            else:
                stratum = "changed_cross_date"
            target_binding = index["bindings"].get((law_id, target_asof), {})
            pairs.append({"source_key": source_key, "target_key": target_key,
                "law_id": law_id, "family_id": family_id, "source_asof": source_asof,
                "target_asof": target_asof, "stratum": stratum, "target_status": target_status,
                "source_status": source_status, "source_role": "full", "split": allocation[family_id],
                "source_revision_id": source["revision_id"] if source else index["bindings"].get(source_key[:2], {}).get("revision_id"),
                "target_revision_id": target["revision_id"] if target else target_binding.get("revision_id"),
                "source_passage_id": source["passage_id"] if source else None,
                "target_passage_id": target["passage_id"] if target else None})
    return pairs


def select_query_pairs(pairs, cap_per_family_target_stratum=25, seed=42):
    """Select one latest permitted origin, then cap unique addresses per family."""
    groups = defaultdict(list)
    for row in pairs:
        if row["stratum"] not in SAMPLED_STRATA or row["source_asof"] > row["target_asof"]:
            continue
        if row["stratum"] == "same_date" and row["source_asof"] != row["target_asof"]:
            continue
        groups[(row["family_id"], row["target_asof"], row["stratum"])].append(row)
    selected, counts = [], []
    rng = random.Random(seed)
    for (family_id, target_asof, stratum), rows in sorted(groups.items()):
        representatives = {}
        for row in rows:
            law_id, _, scope, article_num = row["target_key"]
            address = (law_id, scope, article_num)
            previous = representatives.get(address)
            if previous is None or row["source_asof"] > previous["source_asof"]:
                representatives[address] = row
        addresses = sorted(representatives)
        rng.shuffle(addresses)
        chosen = addresses[:cap_per_family_target_stratum]
        selected.extend(representatives[address] for address in chosen)
        counts.append({"family_id": family_id, "target_asof": target_asof, "stratum": stratum,
            "census_pairs": len(rows), "eligible_addresses": len(addresses),
            "selected_addresses": len(chosen), "cap": cap_per_family_target_stratum})
    return selected, counts


def exact_quote_queries(selected_pairs, index, quote_chars=96, min_chars=24):
    """Build literal visible-paragraph spans; provenance and gold stay separate."""
    queries = []
    for pair in selected_pairs:
        source = index["addresses"][pair["source_key"]][0]
        for paragraph in source["visible_paragraphs"]:
            text = paragraph["text"]
            start = 0
            quote = text[start:start + quote_chars]
            if len("".join(quote.split())) < min_chars:
                continue
            end = start + len(quote)
            query_id = (pair["source_passage_id"] + "|" + pair["source_asof"] + "->" + pair["target_asof"]
                        + "|exact_quote|" + str(paragraph["num"]) + f":{start}:{end}")
            quote_present = None
            if pair["target_status"] == "available":
                target = index["addresses"][pair["target_key"]][0]
                quote_present = _normal_text(quote) in _normal_text(target["visible_text"])
            queries.append({**pair, "query_id": query_id, "query_text": quote, "query_variant": "exact_quote",
                "quote_present_in_target": quote_present,
                "source_quote_absent_from_target": None if quote_present is None else not quote_present,
                "source_paragraph_num": paragraph["num"], "source_tree_path": paragraph["tree_path"],
                "source_span_start": start, "source_span_end": end, "source_quote_text": quote,
                "source_paragraph_text": text, "source_body_text": source["visible_text"],
                "source_raw_path": source.get("raw_path"),
                "labels": {"target_key": pair["target_key"], "target_status": pair["target_status"],
                    "revision_id": pair["target_revision_id"],
                    "passage_id": pair["target_passage_id"] if pair["target_status"] == "available" else None}})
            break
    return queries


def attach_paraphrases(exact_queries, generated_records):
    generated = {row["base_query_id"]: row for row in generated_records}
    return [{**query, "base_query_id": query["query_id"], "query_id": query["query_id"] + "|paraphrase",
             "query_variant": "paraphrase", "query_text": generated[query["query_id"]]["paraphrase"]}
            for query in exact_queries if query["query_id"] in generated]
