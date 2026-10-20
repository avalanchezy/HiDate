"""Stream an independent audit of frozen HiDate final retrieval and model receipts.

No model forward is repeated. Frozen local tokenizers verify raw text against
recorded input IDs; logged scores, selections, labels and work are recomputed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import time
import unicodedata


METHODS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound",
           "history_bound", "history_current", "history_learned")
GROUPS = (("test-natural-exact", "natural", "exact_quote"),
          ("test-natural-paraphrase", "natural", "generated_paraphrase"),
          ("test-controlled-exact", "controlled", "exact_quote"),
          ("test-controlled-paraphrase", "controlled", "generated_paraphrase"),
          ("lawqa", "lawqa", "natural_question"))
SYSTEM = ("法令に関する四択問題に答えてください。参考資料がある場合は資料を用いて判断し、"
          "回答は a、b、c、d のいずれか一文字だけを出力してください。")
OMIT = {"text", "body_text", "visible_text", "visible_paragraphs", "paragraphs"}
ATOL, RTOL = 2e-6, 2e-5


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_encoding(tokenizer, pairs, pin):
    """Reconstruct model inputs from the declared stack and frozen raw pair text."""
    qwen3 = pin.get("runtime") == "qwen3" or pin["model_id"].startswith("Qwen/Qwen3-Reranker-")
    if not qwen3:
        return tokenizer([p["query_text"] for p in pairs], text_pair=[p["document_text"] for p in pairs],
                         padding=True, truncation=True, max_length=1024, return_token_type_ids=False)
    instruction = pin.get("conventions", {}).get("instruction",
        "Given a web search query, retrieve relevant passages that answer the query")
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prefix_ids, suffix_ids = [tokenizer.encode(text, add_special_tokens=False) for text in (prefix, suffix)]
    texts = [f"<Instruct>: {instruction}\n<Query>: {p['query_text']}\n<Document>: {p['document_text']}" for p in pairs]
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, padding=False, truncation="longest_first", return_attention_mask=False,
                        return_token_type_ids=False, max_length=1024-len(prefix_ids)-len(suffix_ids))
    encoded["input_ids"] = [prefix_ids + row + suffix_ids for row in encoded["input_ids"]]
    return tokenizer.pad(encoded, padding=True)


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def address(chunk):
    if chunk["source_kind"] == "guidance":
        return ("guidance", chunk["source_id"])
    number = chunk["article_num"]
    return (chunk["law_id"], chunk["scope"],
            number if number is not None else chunk["slot_id"].rsplit(":", 1)[-1])


def parent_address(parent):
    number = parent["article_num"]
    return (parent["law_id"], parent["scope"],
            number if number is not None else parent["slot_id"].rsplit(":", 1)[-1])


def eligible(chunk, role, date, mode):
    if chunk["source_role"] != role or chunk.get("deleted") or chunk.get("hidden"):
        return False
    if chunk["source_kind"] == "guidance":
        return True
    dates = chunk["asof_dates"]
    return date in dates if mode == "asof" else any(anchor <= date for anchor in dates)


def probability(model, features):
    value = model["intercept"] + sum((features[name] - mean) / scale * coefficient
        for name, mean, scale, coefficient in zip(model["feature_names"], model["mean"],
                                                model["scale"], model["coefficient"]))
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def selections(discovery, chunks, model):
    diagnostics = [{**row, "learned_probability": probability(model, row["features"])}
                   for row in discovery["addresses"]]
    result = {}
    for method in METHODS:
        if method.startswith("asof_") or method == "history_unbound":
            hits = (discovery["history"]["hybrid"] if method == "history_unbound"
                    else discovery["asof"][method[5:]])
            chosen, parents = [], set()
            for rank, hit in enumerate(hits, 1):
                chunk = chunks[hit["chunk_id"]]
                if chunk["parent_passage_id"] in parents:
                    continue
                parents.add(chunk["parent_passage_id"])
                chosen.append({**chunk, "original_score": hit["score"], "discovery_rank": rank,
                               "selection_rank": len(chosen) + 1})
                if len(chosen) == 10:
                    break
            ordered = sorted(diagnostics, key=lambda row: (-row["features"]["max_history_rrf"], tuple(row["address"]))) if method == "history_unbound" else []
        else:
            if method == "history_learned":
                ordering = lambda row: (-row["learned_probability"], tuple(row["address"]))
            elif method == "history_current":
                ordering = lambda row: (-row["features"]["current_best_cosine"],
                    -row["features"]["max_lexical_overlap_current"],
                    -row["features"]["max_history_rrf"], tuple(row["address"]))
            else:
                ordering = lambda row: (-row["features"]["max_history_rrf"], tuple(row["address"]))
            ordered, chosen, parents = sorted(diagnostics, key=ordering), [], set()
            for rank, item in enumerate(ordered, 1):
                identity = item["current_best_chunk_id"]
                if item["status"] != "available" or identity is None:
                    continue
                chunk = chunks[identity]
                if chunk["parent_passage_id"] in parents:
                    continue
                parents.add(chunk["parent_passage_id"])
                score = (item["learned_probability"] if method == "history_learned" else
                    item["features"]["current_best_cosine"] if method == "history_current" else item["features"]["max_history_rrf"])
                chosen.append({**chunk, "address": item["address"], "features": item["features"],
                    "learned_probability": item["learned_probability"], "binding_status": item["status"],
                    "original_score": score, "address_rank": rank, "selection_rank": len(chosen) + 1})
                if len(chosen) == 10:
                    break
        result[method] = (chosen, ordered)
    return result


def evidence_order(candidates, alpha):
    scored = [{**row, "final_score": alpha * row["learned_probability"] + (1 - alpha) * row["reranker_score"]
               if alpha else row["reranker_score"]} for row in candidates]
    scored.sort(key=lambda row: (-row["final_score"], row["selection_rank"], row["chunk_id"]))
    selected, parents = [], set()
    for row in scored:
        if row["parent_passage_id"] not in parents:
            selected.append(row)
            parents.add(row["parent_passage_id"])
            if len(selected) == 4:
                break
    return selected


def confidence(method, evidence, diagnostics):
    if not evidence:
        return 0.
    if method == "history_learned":
        value = evidence[0]["learned_probability"]
        other = max((row["learned_probability"] for row in diagnostics if row["status"] != "available"), default=-1.)
        return 0. if other >= value else value
    if method in {"history_bound", "history_current"} and diagnostics and diagnostics[0]["status"] != "available":
        return 0.
    return evidence[0]["reranker_score"]


def structural(query, evidence, scope):
    available = scope == "natural" and query["target_status"] == "available"
    key = query["target_key"]
    gold_address = (key[0], key[2], key[3])
    top = evidence[0] if evidence else None
    rank = next((i for i, row in enumerate(evidence, 1)
                 if available and row["parent_passage_id"] == query["target_passage_id"]), None)
    return {"target_available": available,
        "top_address_correct": bool(top and address(top) == gold_address),
        "top_revision_correct": bool(available and top and address(top) == gold_address
                                     and top.get("revision_id") == query["target_revision_id"]),
        "top_parent_correct": bool(available and top and top["parent_passage_id"] == query["target_passage_id"]),
        "target_parent_rank": rank, "target_mrr": 1 / rank if rank else 0., "target_recall_at_four": rank is not None,
        "task": "source-derived addressed retrieval and citation; no semantic QA gold"}


def compact(row):
    return {**{key: value for key, value in row.items() if key not in OMIT}, "address": list(address(row))}


def prompt(question, evidence):
    references = "\n\n".join(f"参考資料 {i + 1}\n{item['text']}" for i, item in enumerate(evidence))
    content = (references + "\n\n" if references else "") + "問題\n" + question["問題文"] + "\n\n選択肢\n" + question["選択肢"]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def packed_evidence(question, passages, tokenizer):
    def length(items):
        return len(tokenizer.apply_chat_template(prompt(question, items), tokenize=True, add_generation_prompt=True))
    chosen = []
    for passage in passages[:4]:
        text = passage["text"]
        item = {"chunk_id": passage["chunk_id"], "text": text, "truncated": False,
                "included_characters": len(text), "original_characters": len(text)}
        if length(chosen + [item]) <= 4096:
            chosen.append(item)
            continue
        lower, upper = 0, len(text)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if length(chosen + [{**item, "text": text[:middle]}]) <= 4096:
                lower = middle
            else:
                upper = middle - 1
        if lower:
            chosen.append({**item, "text": text[:lower], "truncated": True, "included_characters": lower})
        break
    return chosen


def readback(final_dir, discovery_dir, prepared_dir, index_dir, source_dir, controls_dir,
             policy_path, stack_path, *, ranker_path=None, protocol_path=Path("configs/retrieval-protocol.json"),
             preflight_path=None, tokenizers=None, expected_counts=None):
    started = time.perf_counter()
    final, discovery_root, prepared, index, source, controls = map(Path,
        (final_dir, discovery_dir, prepared_dir, index_dir, source_dir, controls_dir))
    manifest = read_json(final / "evaluation.json")
    policy, stack, protocol = read_json(policy_path), read_json(stack_path), read_json(protocol_path)
    reranker_pin = stack["models"]["reranker"]
    qwen3 = reranker_pin.get("runtime") == "qwen3" or reranker_pin["model_id"].startswith("Qwen/Qwen3-Reranker-")
    expected_backend = "default" if qwen3 else "math"
    ranker_path = Path(ranker_path or policy["ranker_path"])
    model = read_json(ranker_path)
    issues = []

    def check(condition, description):
        if not condition:
            issues.append(description)

    check(manifest["complete"] and manifest["phase"] == "complete", "final execution is not complete")
    check(policy["complete"] and model["complete"] and stack["complete"], "frozen inputs are incomplete")
    check(manifest["frozen_policy"] == policy and read_json(final / "ranker-used.json") == model,
          "frozen policy or numeric ranker/provenance changed")
    check(manifest["policy_path"] == str(Path(policy_path).resolve())
          and manifest["ranker_path"] == str(ranker_path.resolve()), "frozen policy/ranker path provenance changed")
    check(manifest["training_runs"] == manifest["dev_updates"] == 0, "final execution performed training/DEV updates")
    check(manifest["conditions"] == list(METHODS) and set(policy["method_thresholds"]) == set(METHODS), "method coverage changed")
    check(manifest["pair_budget"] == 10 and manifest["final_parent_budget"] == manifest["reader_max_passages"] == 4
          and manifest["reader_prompt_budget"] == 4096 and manifest["reranker_max_length"] == 1024, "frozen budgets changed")
    for role in ("reranker", "reader"):
        pin = {key: stack["models"][role][key] for key in ("model_id", "revision")}
        check(manifest["model_pins"][role] == pin and stack["models"][role]["download_complete"]
              and manifest["model_local_paths"][role] == stack["models"][role]["local_path"], f"{role} model pin/path changed")
    if preflight_path is not None:
        preflight = read_json(preflight_path)
        preflight_pins = {role: {key: preflight["model_pins"][role][key] for key in ("model_id", "revision")}
                          for role in ("reranker", "reader")}
        check(preflight["status"] == "PASS" and preflight_pins == manifest["model_pins"], "GPU preflight pins disagree")
    if expected_counts is not None:
        check(model.get("fit_runs") == 1 and model.get("random_state") == 42, "one seed42 TRAIN fit provenance changed")
        check(model["feature_names"] == protocol["rank_features"], "ranker feature order changed")
        check(manifest.get("reranker_released_before_reader") is True, "reranker release before reader not recorded")
        for role in ("reranker", "reader"):
            device = manifest.get("devices", {}).get(role, {})
            expected_dtype = preflight["stages"][role]["model"]["dtype"] if preflight_path is not None else "torch.bfloat16"
            check(bool(device.get("name")) and device.get("dtype") == expected_dtype, f"{role} actual GPU dtype differs from preflight")
        if not qwen3:
            check(manifest["devices"]["reranker"]["dtype"] == "torch.bfloat16", "frozen Ruri BF16 weights changed")
        check(manifest["devices"]["reader"].get("attention_implementation") == "sdpa", "reader attention backend changed")
        check(manifest["reranker_sdpa_backend"] == expected_backend, "declared reranker attention backend changed")
        check(manifest["stack_model_pins"] == {role: {key: item[key] for key in ("model_id", "revision")}
            for role, item in stack["models"].items()}, "complete frozen model-stack pins changed")
    if tokenizers is None:
        from transformers import AutoTokenizer
        tokenizers = {role: AutoTokenizer.from_pretrained(stack["models"][role]["local_path"], local_files_only=True)
                      for role in ("reranker", "reader")}

    chunks = {row["chunk_id"]: row for row in rows(index / "chunks.jsonl")}
    queries = {}
    for filename in ("queries-exact.jsonl", "queries-paraphrase.jsonl", "lawqa-retrieval-queries.jsonl"):
        for row in rows(prepared / filename):
            if row["split"] in {"test", "external"}:
                check(row["query_id"] not in queries, "duplicate frozen query ID")
                queries[row["query_id"]] = row
    questions = {row["ファイル名"]: row for row in rows(prepared / "lawqa-external.jsonl")}
    original_questions = read_json(source / "lawqa/selection.json")["samples"]
    check(list(questions.values()) == original_questions, "released original QA fields/references changed")
    targets = {tuple(row["address_key"]): row for row in rows(prepared / "target-addresses.jsonl")}
    bindings = {(row["law_id"], row["requested_asof"]): row for row in rows(source / "asof-bindings.jsonl")}
    control_dates = {q["target_asof"] for q in queries.values() if q["split"] == "test" and q["target_status"] == "available"}
    audits = {date: read_json(controls / (date + ".json")) for date in control_dates}
    blocked_keys = {tuple(q["target_key"]) for q in queries.values()
                    if q["split"] == "test" and q["target_status"] == "available"}
    blocked_addresses = {(key[0], key[2], key[3]) for key in blocked_keys}
    parent_wording, parents_by_address = {}, defaultdict(list)
    for parent in rows(prepared / "corpus-passages.jsonl"):
        key = parent_address(parent)
        if key in blocked_addresses:
            wording = "".join(unicodedata.normalize("NFC", parent["visible_text"]).split())
            parent_wording[parent["passage_id"]] = wording
            parents_by_address[key].append(parent["passage_id"])
    for date, audit in audits.items():
        keys = {key for key in blocked_keys if key[1] == date}
        target_parents = {identity for key in keys for identity in targets[key]["passage_ids"]}
        removed = set(target_parents)
        for key in keys:
            texts = {parent_wording[identity] for identity in targets[key]["passage_ids"]}
            removed.update(identity for identity in parents_by_address[(key[0], key[2], key[3])]
                           if parent_wording[identity] in texts)
        expected_overrides = {key: {**targets[key], "status": "corpus_support_unavailable"} for key in keys}
        check(set(audit["removed_parent_ids"]) == removed and set(audit["target_parent_ids"]) == target_parents
              and set(audit["removed_chunk_ids"]) == {identity for identity, chunk in chunks.items() if chunk["parent_passage_id"] in removed},
              "independent target/exact-wording-copy removal audit disagrees")
        check({tuple(row["address_key"]): row for row in audit["target_overrides"]} == expected_overrides
              and set(audit["eligible_base_query_ids"]) == {q["query_id"] for q in queries.values()
                if q["query_variant"] == "exact_quote" and q["target_asof"] == date and q["target_status"] == "available"},
              "shared controlled target overrides/base query coverage changed")
    expected = {(scope, identity) for _, scope, variant in GROUPS for identity, q in queries.items()
                if q["query_variant"] == variant and (scope != "controlled" or q["target_status"] == "available")}
    derived_counts = dict(Counter(scope for scope, _ in expected))
    check(manifest["expected_retrieval_rows_by_scope"] == derived_counts, "declared final query counts disagree with input maps")
    if expected_counts is not None:
        check(derived_counts == expected_counts and len(questions) == 140, "complete production counts differ from 6752/140")
    external_pairs = [(q["original_question_id"], q["target_asof"]) for q in queries.values() if q["split"] == "external"]
    check(len(external_pairs) == len(set(external_pairs)) and set(external_pairs) == {
        (identity, date) for identity in questions for date in protocol["LawQA_corpus_conditions"]}, "external mirrored corpus-date units changed")
    for q in queries.values():
        if q["split"] == "external":
            check(q["query_text"] == questions[q["original_question_id"]]["問題文"]
                  and q["source_role"] == "external", "external retrieval query contains non-question input")
        else:
            key = tuple(q["target_key"])
            binding, target = bindings.get(key[:2]), targets.get(key)
            status = "collection_failure" if binding is None or binding["status"] != "collected" else target["status"] if target else "absent"
            passage = target["passage_ids"][0] if target is not None and status not in {"ambiguous", "collection_failure"} else None
            revision = binding.get("revision_id") if binding else None
            check(key[1] == q["target_asof"] and q["target_status"] == status
                  and q["target_revision_id"] == revision and q["target_passage_id"] == passage,
                  "original structural dated gold disagrees with actual source binding")
            check(q["labels"] == {"target_key": q["target_key"], "target_status": status,
                "revision_id": revision, "passage_id": passage if status == "available" else None}, "source-derived structural gold labels changed")

    # Retain compact numeric pair receipts, never padded-ID batches or giant logs.
    query_text_ids = {text: i for i, text in enumerate(dict.fromkeys(q["query_text"] for q in queries.values()))}
    document_text_ids = {text: i for i, text in enumerate(dict.fromkeys(c["text"] for c in chunks.values()))}
    pair_scores, pair_tokens = {}, Counter()
    pair_work, pair_batches = Counter(), 0

    def receipt_inputs(result, description):
        ids, masks = result["input_ids"], result["attention_mask"]
        shape = result["input_shape"]
        check(shape == [len(ids), len(ids[0])] and len(masks) == len(ids)
              and all(len(row) == shape[1] for row in ids + masks), description + ": input shape mismatch")
        check(all(isinstance(value, int) and value >= 0 for row in ids for value in row)
              and all(value in (0, 1) for row in masks for value in row), description + ": invalid IDs/masks")
        counts = list(map(sum, masks))
        check(counts == result["token_counts"] and sum(counts) == result["total_tokens"], description + ": nonpadding work mismatch")
        check(math.isfinite(result["seconds"]) and result["seconds"] >= 0, description + ": invalid elapsed work")
        return counts

    for receipt in rows(final / "model-pair-receipts.jsonl"):
        pair_batches += 1
        result, pairs = receipt["result"], receipt["pairs"]
        counts = receipt_inputs(result, f"pair batch {pair_batches}")
        check(receipt["batch_number"] == pair_batches and 0 < len(pairs) <= 32 and len(pairs) == len(counts), "physical pair batch coverage changed")
        encoded = pair_encoding(tokenizers["reranker"], pairs, reranker_pin)
        check(encoded["input_ids"] == result["input_ids"] and encoded["attention_mask"] == result["attention_mask"], "pair raw text does not reproduce recorded IDs/masks")
        check(result["output_shape"] == [len(pairs), 1], "reranker output shape changed")
        check(len(result["scores"]) == len(result["logits"]) == len(pairs), "reranker score coverage changed")
        if expected_counts is not None:
            check(result["sdpa_backend"] == expected_backend, "physical pair receipt attention backend changed")
        if qwen3:
            check(result["score_format"] == "conditional_yes_no" and result["logit_format"] == "yes_minus_no",
                  "Qwen3 conditional score convention changed")
            check(result["yes_token_id"] == tokenizers["reranker"].convert_tokens_to_ids("yes")
                  and result["no_token_id"] == tokenizers["reranker"].convert_tokens_to_ids("no"),
                  "Qwen3 yes/no token IDs changed")
            check(same(result["logits"], [yes-no for no, yes in result["no_yes_logits"]]),
                  "Qwen3 saved scalar logits differ from yes-minus-no margins")
        for i, pair in enumerate(pairs):
            text, document = pair["query_text"], pair["document_text"]
            check(text in query_text_ids and document in document_text_ids, "physical pair input is not frozen query/chunk text")
            if text not in query_text_ids or document not in document_text_ids:
                continue
            key = (query_text_ids[text], document_text_ids[document])
            logit, score = result["logits"][i], result["scores"][i]
            check(math.isfinite(logit) and math.isfinite(score), "nonfinite physical pair logit/score")
            if not math.isfinite(logit) or not math.isfinite(score):
                continue
            sigmoid = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            check(key not in pair_scores and same(score, sigmoid), "physical pair repeated or logit/sigmoid inconsistent")
            pair_scores[key] = {"reranker_score": score, "reranker_logit": logit, "pair_token_count": counts[i]}
            pair_tokens[key] += 1
        pair_work.update(input_tokens=result["total_tokens"], padded_token_slots=result["input_shape"][0] * result["input_shape"][1])
        pair_work["wall_seconds"] += result["seconds"]

    def discoveries():
        for name, scope, variant in GROUPS:
            report = read_json(discovery_root / name / "discovery.json")
            check(report["complete"], f"incomplete CPU discovery: {name}")
            count = 0
            for item in rows(discovery_root / name / "candidates.jsonl"):
                count += 1
                yield scope, variant, item
            check(report["query_rows"] == report["completed_query_rows"] == count, "CPU discovery manifest row accounting changed")

    checked_chunk_bindings = set()

    def chunk_bindings(chunk):
        identity = chunk["chunk_id"]
        if identity in checked_chunk_bindings or chunk["source_kind"] == "guidance":
            return
        checked_chunk_bindings.add(identity)
        for date in chunk["asof_dates"]:
            binding = bindings.get((chunk["law_id"], date))
            key = address(chunk)
            target = targets.get((key[0], date, key[1], key[2]))
            check(binding is not None and binding["status"] == "collected"
                  and binding["revision_id"] == chunk["revision_id"] and target is not None
                  and chunk["parent_passage_id"] in target["passage_ids"], "chunk captured-anchor binding/revision disagrees with actual source")

    def validate_discovery(item, scope, query):
        date, role = query["target_asof"], query["source_role"]
        audit = audits[date] if scope == "controlled" else {}
        removed_chunks, removed_parents = set(audit.get("removed_chunk_ids", [])), set(audit.get("removed_parent_ids", []))
        overrides = {tuple(row["address_key"]): row for row in audit.get("target_overrides", [])}
        for mode in ("asof", "history"):
            check(set(item[mode]) == {"bm25", "dense", "hybrid"}, "CPU ranking method schema changed")
            for hits in item[mode].values():
                identities = [hit["chunk_id"] for hit in hits]
                check(len(hits) <= 50 and len(identities) == len(set(identities)), "CPU ranking budget/duplicate violation")
                for hit in hits:
                    chunk = chunks[hit["chunk_id"]]
                    chunk_bindings(chunk)
                    check(eligible(chunk, role, date, mode) and math.isfinite(hit["score"]), "CPU hit violates role/date/visibility")
                    check(hit["chunk_id"] not in removed_chunks and chunk["parent_passage_id"] not in removed_parents, "controlled removed support entered CPU rankings")
                check(hits == sorted(hits, key=lambda hit: (-hit["score"], hit["chunk_id"])), "CPU ranking score/ID ordering changed")
            fused = defaultdict(float)
            for method in ("bm25", "dense"):
                for rank, hit in enumerate(item[mode][method], 1):
                    fused[hit["chunk_id"]] += 1 / (protocol["rrf_k"] + rank)
            expected_fused = [{"chunk_id": identity, "score": score} for identity, score in
                sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:50]]
            check(same(item[mode]["hybrid"], expected_fused), "CPU independent RRF scores/order changed")
        groups = defaultdict(list)
        for hit in item["history"]["hybrid"]:
            groups[address(chunks[hit["chunk_id"]])].append(hit)
        check(Counter(tuple(row["address"]) for row in item["addresses"]) == Counter({key: 1 for key in groups}),
              "historical address groups missing, duplicated or invented")
        for row in item["addresses"]:
            check(all(math.isfinite(value) for value in row["features"].values()) and set(row["features"]) == set(protocol["rank_features"]), "nonfinite/changed discovery feature keys")
            group = groups[tuple(row["address"])]
            scores, best, parents = {}, {}, set()
            expected_hits = []
            for hit in group:
                chunk = chunks[hit["chunk_id"]]
                revision = chunk["source_id"] if chunk["source_kind"] == "guidance" else chunk["revision_id"]
                if revision not in scores or hit["score"] > scores[revision]:
                    scores[revision], best[revision] = hit["score"], hit["chunk_id"]
                parents.add(chunk["parent_passage_id"])
                expected_hits.append({**hit, "parent_passage_id": chunk["parent_passage_id"],
                    "revision_id": chunk.get("revision_id"), "asof_dates": [anchor for anchor in chunk.get("asof_dates", []) if anchor <= date]})
            check(row["history_parent_passage_ids"] == sorted(parents) and same(row["history_chunk_hits"], expected_hits)
                  and row["historical_best_chunk_ids"] == best and row["best_chunk_ids"] == best
                  and same(row["max_score"], max(scores.values())) and same(row["mean_score"], sum(scores.values()) / len(scores))
                  and row["support_count"] == len(scores), "historical aggregation/provenance changed")
            if row["source_kind"] == "guidance":
                check(row["address"] == ["guidance", row["source_id"]] and row["status"] == "available"
                      and row["parent_passage_ids"] == row["target_passage_ids"] == sorted(parents)
                      and row["revision_scores"] == {} and "law_id" not in row and "target_asof" not in row, "static guidance binding changed")
                current = row["current_best_chunk_id"]
                check(current is not None and chunks[current]["source_kind"] == "guidance"
                      and chunks[current]["source_id"] == row["source_id"] and eligible(chunks[current], role, date, "asof"), "guidance current evidence changed")
                continue
            check(row["revision_scores"] == scores and row["law_id"] == row["address"][0]
                  and row["scope"] == row["address"][1]
                  and row["article_num"] == chunks[group[0]["chunk_id"]]["article_num"], "scoped statute metadata/revision scores changed")
            key = (row["address"][0], date, row["address"][1], row["address"][2])
            binding, target = bindings.get((key[0], date)), overrides.get(key, targets.get(key))
            status = "collection_failure" if binding is None or binding["status"] != "collected" else target["status"] if target else "absent"
            target_ids = target["passage_ids"] if target is not None and status != "collection_failure" else []
            check(tuple(row["address_key"]) == key and row["status"] == status
                  and row["revision_id"] == (binding.get("revision_id") if binding else None)
                  and row["target_passage_ids"] == target_ids
                  and row["parent_passage_ids"] == (target_ids if status == "available" else []), "actual scoped target binding disagreement")
            current = row["current_best_chunk_id"]
            if status != "available":
                check(current is None and row["features"]["available_target"] == 0, "unavailable target hydrated into current evidence")
            elif current is not None:
                chunk = chunks[current]
                chunk_bindings(chunk)
                check(chunk["parent_passage_id"] in target_ids and address(chunk) == tuple(row["address"])
                      and eligible(chunk, role, date, "asof") and current not in removed_chunks
                      and chunk["parent_passage_id"] not in removed_parents, "bound current candidate invalid")

    seen, requested_pairs = Counter(), Counter()
    logical_by_method, tokens_by_method, final_parents = Counter(), Counter(), Counter()
    scoped_pairs, scoped_tokens, structural_cells = Counter(), Counter(), defaultdict(Counter)
    external_selections = {}
    for candidate, record in zip_longest(discoveries(), rows(final / "retrieval.jsonl")):
        check(candidate is not None and record is not None, "CPU/final retrieval row coverage differs")
        if candidate is None or record is None:
            continue
        scope, variant, discovery = candidate
        identity = record["query_id"]
        check(identity in queries and identity == discovery["query_id"], "final row differs from CPU query order/identity")
        if identity not in queries:
            continue
        query = queries[identity]
        seen[(scope, identity)] += 1
        check((scope, identity) in expected and record["evaluation_scope"] == scope
              and variant == query["query_variant"], "unexpected final scope/variant")
        for field in ("query_variant", "split", "target_asof", "source_role"):
            check(record[field] == discovery[field] == query[field], "query metadata changed: " + field)
        check(record["corpus_condition_asof"] == query["target_asof"], "corpus condition date changed")
        validate_discovery(discovery, scope, query)
        if scope == "lawqa":
            check(record["original_question_id"] == query["original_question_id"], "original LawQA identity changed")
        else:
            check(record["base_query_id"] == query.get("base_query_id", identity)
                  and record["target_key"] == query["target_key"] and record["stratum"] == query["stratum"], "structural query provenance changed")
            check(record["original_gold"] == {key: query[key] for key in
                ("target_status", "target_key", "target_revision_id", "target_passage_id", "labels")}, "original dated gold changed")
            check(record["target_status"] == ("corpus_support_unavailable" if scope == "controlled" else query["target_status"]), "reported target status changed")
        check(set(record["conditions"]) == set(METHODS), "final seven-method coverage changed")
        expected_selections = selections(discovery, chunks, model)
        for method, condition in record["conditions"].items():
            chosen, diagnostics = expected_selections[method]
            check(same(condition["address_diagnostics"], diagnostics), "independent address probability/order mismatch")
            scored = []
            for chunk in chosen:
                key = (query_text_ids[query["query_text"]], document_text_ids[chunk["text"]])
                requested_pairs[key] += 1
                check(key in pair_scores, "requested candidate has no physical pair receipt")
                if key in pair_scores:
                    scored.append({**chunk, **pair_scores[key]})
            scored.sort(key=lambda row: (-row["reranker_score"], row["selection_rank"], row["chunk_id"]))
            check(same(condition["reranked_candidates"], [compact(row) for row in scored]), "canonical top10 candidates/scores/metadata mismatch")
            alpha = policy["alpha"] if method == "history_learned" else 0.
            evidence = evidence_order(scored, alpha)
            check(same(condition["alpha"], alpha) and same(condition["evidence"], [compact(row) for row in evidence]), "canonical alpha/top4 selected evidence mismatch")
            check(condition["evidence_chunk_ids"] == [row["chunk_id"] for row in evidence]
                  and condition["evidence_parent_ids"] == [row["parent_passage_id"] for row in evidence], "evidence ID/order mismatch")
            score, threshold = confidence(method, evidence, diagnostics), policy["method_thresholds"][method]["threshold"]
            accepted = score >= threshold
            check(same(condition["support_confidence"], score) and same(condition["support_threshold"], threshold)
                  and condition["accepted"] is accepted, "support confidence/threshold/acceptance mismatch")
            token_count = sum(row["pair_token_count"] for row in scored)
            check(condition["requested_pair_count"] == len(scored) <= 10
                  and condition["requested_input_tokens"] == token_count, "logical pair/token budget mismatch")
            logical_by_method[method] += len(scored)
            tokens_by_method[method] += token_count
            scoped_key = "|".join((scope, query["query_variant"], method))
            scoped_pairs[scoped_key] += len(scored)
            scoped_tokens[scoped_key] += token_count
            final_parents[method] += len(evidence)
            if scope == "lawqa":
                external_selections[(identity, method)] = {"accepted": accepted, "support_confidence": score,
                    "support_threshold": threshold, "selected_evidence_chunk_ids": [row["chunk_id"] for row in evidence],
                    "selected_evidence_parent_ids": [row["parent_passage_id"] for row in evidence]}
            else:
                metrics = structural(query, evidence, scope)
                check(same(condition["structural"], metrics), "independent structural dated-gold metrics mismatch")
                cell = structural_cells["|".join((scope, variant, query["stratum"], method))]
                cell.update(queries=1, accepted=int(accepted), top_address_correct=int(metrics["top_address_correct"]),
                    top_revision_correct=int(metrics["top_revision_correct"]), top_parent_correct=int(metrics["top_parent_correct"]),
                    accepted_correct_support=int(accepted and metrics["top_parent_correct"]), recall_at_four=int(metrics["target_recall_at_four"]))
                cell["mrr_sum"] += metrics["target_mrr"]
    check(seen == Counter({key: 1 for key in expected}), "final row IDs missing, duplicated or extra")
    check(set(requested_pairs) == set(pair_scores), "physical pair receipts have missing or unused inputs")
    checks = {"retrieval_rows_by_scope": dict(Counter(scope for scope, _ in seen.elements())),
        "logical_pair_requests": sum(requested_pairs.values()), "physical_pair_count": len(pair_scores),
        "exact_pair_cache_reuses": sum(requested_pairs.values()) - len(pair_scores), "reranker_model_call_count": pair_batches,
        "reranker_encoded_input_tokens": pair_work["input_tokens"], "reranker_padded_token_slots": pair_work["padded_token_slots"],
        "reranker_wall_seconds_with_cuda_sync": pair_work["wall_seconds"], "condition_pair_requests": dict(logical_by_method),
        "condition_requested_input_tokens": dict(tokens_by_method), "scope_variant_condition_pair_requests": dict(scoped_pairs),
        "scope_variant_condition_requested_input_tokens": dict(scoped_tokens), "final_parent_counts": dict(final_parents),
        "structural_counts": {key: dict(value) for key, value in structural_cells.items()}}
    for key, value in checks.items():
        check(same(manifest[key], value), "independent manifest accounting mismatch: " + key)

    answer_seen, answer_cells, reader_work = Counter(), defaultdict(Counter), Counter()
    reader_calls = 0
    for answer, receipt in zip_longest(rows(final / "answers.jsonl"), rows(final / "model-reader-receipts.jsonl")):
        check(answer is not None and receipt is not None, "answer/reader receipt coverage differs")
        if answer is None or receipt is None:
            continue
        reader_calls += 1
        result = receipt["result"]
        receipt_inputs(result, f"reader call {reader_calls}")
        check(receipt["call_number"] == answer["reader_receipt_call_number"] == reader_calls, "reader call ID sequence changed")
        identity, method, scope = answer["original_question_id"], answer["method"], answer["evaluation_scope"]
        question = questions[identity]
        check(all(same(answer[key], value) for key, value in receipt.items() if key not in {"messages", "result", "call_number"}), "answer identity differs from reader receipt")
        if scope == "lawqa":
            query = queries[answer["query_id"]]
            unit = (scope, query["query_id"], method)
            expected_identity = external_selections.get((query["query_id"], method))
            check(expected_identity is not None and all(same(answer[key], value) for key, value in expected_identity.items()), "reader did not use frozen external evidence/support decision")
            check(query["original_question_id"] == identity and answer["target_asof"] == query["target_asof"]
                  and answer["query_variant"] == "natural_question" and answer["source_role"] == "external", "mirrored original external unit changed")
            full_passages = [chunks[key] for key in answer["selected_evidence_chunk_ids"]]
        else:
            unit = (scope, identity, method)
            check(scope == "lawqa_reference" and method in {"closed_book", "released_oracle"}
                  and answer["accepted"] is True and "target_asof" not in answer and "query_id" not in answer,
                  "reference was dated, duplicated per condition or withheld")
            full_passages = [] if method == "closed_book" else [{"chunk_id": identity + "|released_context", "text": question["コンテキスト"]}]
        answer_seen[unit] += 1
        evidence = answer["evidence"]
        check(same(evidence, packed_evidence(question, full_passages, tokenizers["reader"])), "reader canonical evidence prefix/budget packing changed")
        check(len(evidence) <= 4 and len(evidence) <= len(full_passages), "reader evidence budget changed")
        for i, item in enumerate(evidence):
            original = full_passages[i]
            check(item["chunk_id"] == original["chunk_id"] and item["text"] == original["text"][:item["included_characters"]]
                  and item["included_characters"] == len(item["text"]) and item["original_characters"] == len(original["text"])
                  and item["truncated"] is (item["included_characters"] < item["original_characters"]), "reader evidence literal prefix/provenance changed")
            check(not item["truncated"] or i == len(evidence) - 1, "reader continued after a truncated passage")
        messages = prompt(question, evidence)
        check(receipt["messages"] == messages, "reader prompt changed full original question/options/evidence")
        encoded = tokenizers["reader"].apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
        check([encoded["input_ids"]] == result["input_ids"] and [encoded["attention_mask"]] == result["attention_mask"], "reader raw messages do not reproduce IDs/mask")
        counts = result["token_counts"]
        check(len(counts) == 1 and answer["prompt_tokens"] == result["prompt_tokens"] == result["total_tokens"] <= 4096,
              "reader actual prompt budget changed")
        values, choice_ids = result["choice_logprobs"], result["choice_token_ids"]
        check(set(values) == set(choice_ids) == set("abcd") and all(math.isfinite(value) and value <= 0 for value in values.values()), "invalid reader log probabilities")
        check(len(set(choice_ids.values())) == 4 and sum(math.exp(value) for value in values.values()) <= 1 + ATOL,
              "reader choice IDs/probability mass invalid")
        check(all(tokenizers["reader"].encode(label, add_special_tokens=False) == [choice_ids[label]] for label in "abcd"), "reader choice token IDs changed")
        position = max(i for i, value in enumerate(result["attention_mask"][0]) if value)
        check(result["choice_position"] == position and result["output_shape"][:2] == result["input_shape"]
              and result["output_shape"][2] > max(choice_ids.values()), "reader choice position/output shape mismatch")
        prediction = max("abcd", key=lambda label: values[label])
        gold = question["output"].strip().lower()
        check(answer["prediction"] == result["prediction"] == prediction and same(answer["choice_logprobs"], values)
              and answer["released_answer"] == gold and answer["correct"] is (prediction == gold), "reader prediction/released-answer correctness changed")
        check(answer["evidence_is_corpus_condition"] is (scope == "lawqa"), "corpus-condition answer semantics changed")
        key = "|".join((scope, method, answer.get("target_asof", "once_per_question")))
        cell = answer_cells[key]
        cell.update(calls=1, correct=int(prediction == gold), accepted=int(answer["accepted"]),
            accepted_correct=int(answer["accepted"] and prediction == gold), input_tokens=result["total_tokens"],
            padded_token_slots=result["input_shape"][0] * result["input_shape"][1])
        cell["wall_seconds_with_cuda_sync"] += result["seconds"]
        reader_work.update(input_tokens=result["total_tokens"], padded_token_slots=result["input_shape"][0] * result["input_shape"][1])
        reader_work["wall_seconds_with_cuda_sync"] += result["seconds"]
    answer_expected = {("lawqa", identity, method) for identity, method in external_selections} | {
        ("lawqa_reference", identity, method) for identity in questions for method in ("closed_book", "released_oracle")}
    check(answer_seen == Counter({key: 1 for key in answer_expected}), "all forced-choice/reference IDs not exactly once")
    check(reader_calls == manifest["reader_calls"] == manifest["expected_reader_calls"]
          and (expected_counts is None or reader_calls == 3220), "complete reader call count changed")
    check(same(manifest["answer_counts"], {key: dict(value) for key, value in answer_cells.items()})
          and same(manifest["reader_work"], dict(reader_work)), "independent reader work/outcome accounting mismatch")
    return {"status": "FAIL" if issues else "PASS", "issues": issues, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("PBS_JOBID"), "retrieval_rows_checked": sum(seen.values()),
        "retrieval_rows_by_scope": checks["retrieval_rows_by_scope"], "method_outcomes_checked": sum(seen.values()) * len(METHODS),
        "reader_responses_checked": reader_calls, "original_questions_checked": len(questions),
        "physical_pairs_checked": len(pair_scores), "pair_batches_checked": pair_batches,
        "statute_chunk_bindings_checked": len(checked_chunk_bindings), "controlled_date_masks_checked": len(audits),
        "logical_pair_requests_checked": sum(requested_pairs.values()), "reader_work": dict(reader_work),
        "tokenizer_ids_checked": True, "production_counts_checked": expected_counts is not None,
        "model_pins": manifest["model_pins"], "devices": manifest.get("devices"),
        "reranker_sdpa_backend": manifest["reranker_sdpa_backend"],
        "frozen_policy_path": str(Path(policy_path).resolve()), "frozen_ranker_path": str(ranker_path.resolve()),
        "score_scope": "all saved physical logits/sigmoids, choice log probabilities/predictions and work receipts; no repeated model forward",
        "retrieval_scope": "all final rows/methods and all five CPU candidate files; prospective eligibility, controls, actual bindings, independent frozen-ranker selection/support/metrics",
        "answer_scope": "all original question/options and recorded prompt IDs/masks; mirrored dates are corpus conditions, not temporal answer gold",
        "training_runs": 0, "dev_updates": 0, "model_forward_calls": 0, "wall_seconds": time.perf_counter() - started,
        "numeric_tolerance": {"absolute": ATOL, "relative": RTOL}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=Path("runs/hidate/final"))
    parser.add_argument("--discovery-dir", type=Path, default=Path("runs/hidate/final-discovery"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index/hidate"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--controls-dir", type=Path, default=Path("runs/hidate/controlled-removal"))
    parser.add_argument("--policy", type=Path, default=Path("runs/hidate/policy/frozen.json"))
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = readback(args.final_dir, args.discovery_dir, args.prepared_dir, args.index_dir, args.source_dir,
        args.controls_dir, args.policy, args.stack, ranker_path=args.ranker, protocol_path=args.protocol,
        preflight_path=args.preflight, expected_counts={"natural": 3172, "controlled": 3160, "lawqa": 420})
    path = args.output or args.final_dir / "verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "issues"}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
