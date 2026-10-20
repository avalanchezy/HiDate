"""Fixed-CID Fiscal version realization with matched article-level packing."""

import json

from . import fiscal as fiscal


CONDITIONS = ("binding_bound", "binding_source")


def freeze_addresses(question, retrieval, versions, max_passages=4):
    """Freeze raw learned final addresses; select versions without reading gold.

    Saved history_chunk_ids preserve the original fused-hit order. Its first
    hit for this CID attains that revision's maximum RRF and wins the original
    deterministic chunk-ID tie rule. No oldest-version heuristic is applied.
    """
    selected = retrieval["conditions"]["history_learned"]["evidence"][:max_passages]
    addresses = {a["article_cid"]: a for a in retrieval["addresses"]}
    applicable = fiscal.select_versions(versions.values(), question["date_anchor"])
    cids = [e["article_cid"] for e in selected]
    if len(cids) != len(set(cids)):
        raise ValueError("Frozen learned final CID list contains duplicate addresses")
    slots = []
    for number, evidence in enumerate(selected, 1):
        cid = evidence["article_cid"]
        address = addresses[cid]
        source_chunk = address["history_chunk_ids"][0]
        source_id = source_chunk.split("|", 1)[0]
        source = versions[source_id]
        bound = applicable.get(cid)
        if (source["cid"] != cid or not fiscal.usable(source)
                or source["date_debut"] > question["date_anchor"]):
            raise ValueError("Highest fused source revision is not eligible at the question date")
        if bound is None or bound["id"] != evidence["version_id"]:
            raise ValueError("Saved learned final bound version differs from dated realization")
        slots.append({"slot": number, "article_cid": cid, "bound_version_id": bound["id"],
                      "source_version_id": source_id, "source_history_chunk_id": source_chunk,
                      "source_selection": "first preserved fused hit for this CID; original RRF/tie order",
                      "source_max_history_rrf": address["features"]["max_history_rrf"]})
    return slots


def literal_token_prefix(text, token_cap, tokenizer):
    """Return a literal prefix bounded by the shared body-token allowance."""
    if len(tokenizer.encode(text, add_special_tokens=False)) <= token_cap:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenizer.encode(text[:middle], add_special_tokens=False)) <= token_cap:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def pack_paired_versions(prompt, slots, versions, tokenizer, prompt_budget=4096):
    """Use whole articles/query_window and one uniform shared slot body cap.

    Every frozen CID slot is retained in both arms. Header-only cost determines
    the initial cap; actual complete rendered prompts enforce the joint limit.
    No reranking, address refill, extra chunking or per-arm slot dropping occurs.
    """
    arms = {}
    for condition, version_key in zip(CONDITIONS, ("bound_version_id", "source_version_id")):
        items = []
        for slot in slots:
            row = versions[slot[version_key]]
            body = fiscal.query_window(row["texte_clean"], prompt)
            header = f"=== Article {row['num']} (en vigueur {row['date_debut']}..{row['date_fin']}) ===\n"
            items.append({"slot": slot["slot"], "article_cid": slot["article_cid"], "version_id": row["id"],
                          "header": header, "body": body})
        arms[condition] = items

    def render(items, cap):
        evidence = []
        for item in items:
            included = literal_token_prefix(item["body"], cap, tokenizer)
            evidence.append({"slot": item["slot"], "article_cid": item["article_cid"], "version_id": item["version_id"],
                "text": item["header"] + included, "truncated": len(included) < len(item["body"]),
                "window_characters": len(item["body"]), "included_body_characters": len(included),
                "body_tokens": len(tokenizer.encode(included, add_special_tokens=False)), "shared_cap": cap})
        messages = fiscal.reader_messages(prompt, "\n\n".join(e["text"] for e in evidence))
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return {"messages": messages, "evidence": evidence, "input_ids": list(ids),
                "attention_mask": [1] * len(ids), "prompt_tokens": len(ids),
                "prompt_budget": prompt_budget, "slot_body_token_cap": cap}

    header_cost = max(render(items, 0)["prompt_tokens"] for items in arms.values())
    if header_cost > prompt_budget:
        raise ValueError("Frozen question and all fixed CID headers exceed the common prompt budget")
    cap = (prompt_budget - header_cost) // len(slots) if slots else 0
    while True:
        packed = {condition: render(items, cap) for condition, items in arms.items()}
        excess = max(p["prompt_tokens"] for p in packed.values()) - prompt_budget
        if excess <= 0:
            return packed
        cap = max(0, cap - max(1, (excess + len(slots) - 1) // len(slots)))


def exact_input_key(record):
    """Literal reader input identity; no digest or normalized-text substitute."""
    return json.dumps({k: record[k] for k in ("messages", "input_ids", "attention_mask", "decode")}
                      | {"reader_pin": {k: record["reader_pin"][k] for k in ("model_id", "revision")}},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def single_sequence(values):
    return values[0] if values and isinstance(values[0], list) else values


def response_from_result(record, result, provenance):
    evidence = record["evidence"]
    return {"qid": record["qid"], "condition": record["condition"], "model": record["reader_pin"]["model_id"],
            "reader_pin": record["reader_pin"], "decode": record["decode"],
            "response_text": result["text"], "gold_version_id": record["gold_version_id"],
            "retrieved_version_id": evidence[0]["version_id"] if evidence else None,
            "evidence": evidence, "frozen_article_cids": record["frozen_article_cids"],
            "accepted": record["accepted"], "support_confidence": record["support_confidence"],
            "support_threshold": record["support_threshold"], "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"], "end_status": result["end_status"],
            "provenance": provenance}
