"""Extract four traceable HiDate illustrations from saved outputs; no model calls."""

import argparse
import json
from pathlib import Path
import re


JP_IDS = (
    "322AC0000000050_20180525_430AC0000000031|322AC0000000050:MainProvision:21|1.2.2.3.1|2018-09-30->2026-09-30|exact_quote|1:0:79",
    "金商法_第3章_選択式_根拠条文_問題番号39|corpus_asof=2026-09-30",
    "薬機法_第5章_選択式_根拠条文_問題番号39|corpus_asof=2026-09-30",
)
FRENCH_ID = "R3-WS-156-2009-01-01"
METHODS = ("asof_hybrid", "history_unbound", "history_learned")


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def sourced(path, line_number, row):
    return {"source_path": str(path).replace("\\", "/"), "source_line": line_number, "saved": row}


def excerpt(text, anchor="", length=360):
    start = text.lower().find(anchor.lower()) if anchor else 0
    start = max(0, start)
    return {"text": text[start:start+length], "start_character": start,
            "end_character": min(len(text), start+length), "full_text_characters": len(text)}


def extraction_methods(condition):
    return {key: condition[key] for key in ("alpha", "support_confidence", "support_threshold", "accepted",
        "evidence", "evidence_chunk_ids", "evidence_parent_ids", "reranked_candidates", "structural") if key in condition}


def french_bundle(root):
    root = Path(root)
    prepared = root / "data/processed/fiscal"
    outputs = root / "runs/hidate/fiscal-transfer/fiscal"
    retrieval = root / "runs/hidate/fiscal-transfer/fiscal-retrieval/retrieval.jsonl"
    n, query = next((n, q) for n, q in rows(prepared / "questions.jsonl") if q["qid"] == FRENCH_ID)
    result = {"query": sourced(prepared / "questions.jsonl", n, query), "records": {}}
    n, retrieved = next((n, r) for n, r in rows(retrieval) if r["qid"] == FRENCH_ID)
    address = next(a for a in retrieved["addresses"] if a["article_cid"] == query["gold"]["article_cid"])
    result["discovery_and_binding"] = sourced(retrieval, n, {"date_anchor": retrieved["date_anchor"],
        "address": address, "conditions": {m: retrieved["conditions"][m] for m in METHODS}})
    source_id = next(cid.split("|")[0] for cid in address["history_chunk_ids"]
                     if cid.split("|")[0] != address["bound_version_id"])
    needed = {source_id, address["bound_version_id"]}
    for n, article in rows(prepared / "articles.jsonl"):
        if article["id"] in needed:
            metadata = {k: v for k, v in article.items() if k != "texte_clean"}
            metadata["excerpt"] = excerpt(article["texte_clean"], "des déficits provenant d'exploitations agricoles", 350)
            result["historical_source" if article["id"] == source_id else "asof_target"] = sourced(
                prepared / "articles.jsonl", n, metadata)
    for filename in ("responses.jsonl", "scores.jsonl", "reader-receipts.jsonl"):
        selected = {}
        for n, row in rows(outputs / filename):
            if row["qid"] == FRENCH_ID and row["condition"] in METHODS:
                if filename == "reader-receipts.jsonl":
                    row = {**row, "result": {k: v for k, v in row["result"].items() if k != "input_ids"}}
                selected[row["condition"]] = sourced(outputs / filename, n, row)
        result["records"][filename] = selected
    assert query["scope"] == "in"
    return result


def japanese_cases(root):
    root = Path(root)
    prepared, final = root / "data/processed/hidate", root / "runs/hidate/final"
    queries = {}
    for filename in ("queries-exact.jsonl", "lawqa-retrieval-queries.jsonl"):
        for n, q in rows(prepared / filename):
            if q["query_id"] in JP_IDS:
                queries[q["query_id"]] = sourced(prepared / filename, n, q)
    selected = {}
    # Match the first identity field before parsing only the three requested
    # records in the existing 3GB file; no archive is downloaded or expanded.
    path = final / "retrieval.jsonl"
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            match = re.match(r'\{"query_id": "([^"]+)"', line)
            if not match or match.group(1) not in JP_IDS:
                continue
            r = json.loads(line)
            if r["evaluation_scope"] not in ("natural", "lawqa"):
                continue
            selected[r["query_id"]] = sourced(path, n, r)
    result, needed = [], set()
    for identity in JP_IDS:
        record, query = selected[identity], queries[identity]
        r, q = record["saved"], query["saved"]
        learned = r["conditions"]["history_learned"]
        top = learned["evidence"][0]
        address = ([q["law_id"], q["scope"], q["article_num"]] if r["evaluation_scope"] == "natural" else top["address"])
        bound = next(a for a in learned["address_diagnostics"] if a["address"] == address)
        current_id = q.get("target_passage_id", top["parent_passage_id"])
        source_id = q.get("source_passage_id") or next(p["parent_passage_id"] for p in
            r["conditions"]["history_unbound"]["evidence"] if p["address"] == address)
        needed.update((source_id, current_id))
        case = {"query": query, "evaluation_scope": r["evaluation_scope"], "address": address,
            "historical_parent_id": source_id, "asof_parent_id": current_id,
            "discovery_and_binding": sourced(record["source_path"], record["source_line"], bound),
            "retrieval": sourced(record["source_path"], record["source_line"],
                {m: extraction_methods(r["conditions"][m]) for m in METHODS}), "reader": {}}
        result.append(case)
    parents = {p["passage_id"]: sourced(prepared / "corpus-passages.jsonl", n, p)
               for n, p in rows(prepared / "corpus-passages.jsonl") if p["passage_id"] in needed}
    for case in result:
        for role, key in (("historical_source", "historical_parent_id"), ("asof_target", "asof_parent_id")):
            p = parents[case[key]]
            content = {k: v for k, v in p["saved"].items() if k not in
                       ("text", "visible_text", "paragraphs", "visible_paragraphs")}
            anchor = "四投資運用業" if case["address"][-1] == "15_7" else ""
            content["excerpt"] = excerpt(p["saved"]["visible_text"], anchor, 600)
            case[role] = {**p, "saved": content}
    questions = {q["ファイル名"]: sourced(prepared / "lawqa-external.jsonl", n, q)
                 for n, q in rows(prepared / "lawqa-external.jsonl")}
    for filename in ("answers.jsonl", "model-reader-receipts.jsonl"):
        for n, r in rows(final / filename):
            if r.get("query_id") not in JP_IDS or r.get("method") not in METHODS:
                continue
            case = result[JP_IDS.index(r["query_id"])]
            saved = {k: v for k, v in r.items() if k != "evidence"}
            if filename == "model-reader-receipts.jsonl":
                saved["result"] = {k: v for k, v in r["result"].items() if k != "input_ids"}
            case["reader"].setdefault(filename, {})[r["method"]] = sourced(final / filename, n, saved)
    for case in result:
        if case["evaluation_scope"] == "lawqa":
            identity = case["query"]["saved"]["original_question_id"]
            case["released_question"] = questions[identity]
            a, b = (case["reader"]["answers.jsonl"][m]["saved"] for m in ("asof_hybrid", "history_learned"))
            assert not a["correct"] and b["correct"]
        else:
            assert not case["reader"]
    return result


def annotated_report(root, japanese, french):
    notes = [
        {"case_id": "C1", "case_role": "main", "title": "Commuting-injury benefits: the same provision with an updated cross-reference",
         "query_english": "The insurance benefits for commuting injuries under Article 7(1)(ii) are medical-treatment benefits, absence-from-work benefits, disability benefits, survivors' benefits, funeral benefits, injury/disease pensions and nursing-care benefits.",
         "historical_english": "Article 21 lists the seven commuting-injury benefits and refers to Article 7(1)(ii).",
         "current_english": "The applicable Article 21 lists the same seven benefits and now refers to Article 7(1)(iii). The operative topic and benefit list persist; this is a cross-reference update, not an asserted change in benefit amounts.",
         "interpretation": "The historical quote leads to the same Workers' Accident Compensation Insurance Act Article 21. Binding supplies its 2026 revision with the updated item-(iii) cross-reference. As-of hybrid already includes the target parent at rank two; the learned method places it first. Unbound history ranks an older Article 21 first. This illustrates current-version priority within the evidence set, not a baseline recall-at-four failure. No semantic reader answer was run for this structural query.",
         "reader_scope": "not_run_structural_task", "is_semantic_answer_claim": False},
        {"case_id": "C2", "case_role": "main", "title": "Investment-management capital requirements and a new custody exception",
         "query_english": "Which Cabinet Order provision specifies the minimum capital or total contributions for investment management, excluding investment management for qualified investors?",
         "historical_english": "Item (iv) specifies JPY 50 million for the stated investment-management category, excluding the earlier listed categories.",
         "current_english": "Item (iv) retains JPY 50 million but adds JPY 10 million where the operator neither receives customer money/securities in custody nor causes them to be deposited with the person specified in Article 15-4-2.",
         "interpretation": "Unbound retrieval fills all four slots with older versions of Article 15-7. Binding supplies the 2026 version, including its added numerical/conditional exception. Both as-of hybrid and learned retrieval put the relevant current parent first; their remaining evidence differs. The actual forced-choice answer changes from b (wrong) to a (the released correct provision, Article 15-7(1)(iv)). This is not evidence that the reader generated the JPY 10 million amount, nor a new date-specific LawQA gold label.",
         "reader_scope": "real_forced_choice", "is_semantic_answer_claim": True},
        {"case_id": "C3", "case_role": "backup", "title": "An objection to medical-device certification: same wording, current provenance",
         "query_english": "Which provision allows an applicant for conformity certification to apply to the Minister when objecting to a registered certification body's result?",
         "historical_english": "An applicant may ask the Minister to order an examination or a new examination where the registered certification body does not examine the application or the applicant objects to its certification result.",
         "current_english": "The same application-to-the-Minister provision appears in the applicable current Article 23-14(1). The quoted operative wording is unchanged between these saved source and target passages.",
         "interpretation": "Unbound retrieval returns four historical versions of the same article. The learned method binds that address to the 2026 revision and uses four distinct current parents. Both hybrid and learned methods already rank the correct current parent first; the observed reader contrast is c (Article 23-13, wrong) versus d (Article 23-14(1), correct). The case illustrates provenance and evidence composition, not a substantive change in this operative sentence or an isolated causal attribution to one feature.",
         "reader_scope": "real_forced_choice", "is_semantic_answer_claim": True},
    ]
    cases = [{**note, **case} for note, case in zip(notes, japanese)]
    cases.append({"case_id": "C4", "case_role": "main", "title": "French transfer: historical francs versus the applicable euro threshold",
        "evaluation_scope": "primary_in_scope", "reader_scope": "real_free_generation",
        "is_semantic_answer_claim": True,
        "query_english": "Under the text in force on 1 January 2009, above what net income from other categories can agricultural losses no longer be offset against a household's aggregate income?",
        "historical_english": "Agricultural losses cannot be offset when net income from other sources exceeds FRF 100,000; they may instead be deducted from same-kind profits through the fifth following year.",
        "current_english": "The 2009 text sets EUR 101,300 and allows deduction from same-kind profits through the sixth following year.",
        "interpretation": "The historical hit belongs to stable CGI Article 156. Unbound evidence contains only older versions and the actual reader answers FRF 100,000. Binding supplies the version valid on 1 January 2009 and the learned reader answers EUR 101,300. As-of hybrid already retrieves the gold version first, but its reader incorrectly answers EUR 15,300 and reaches the output limit. Thus the hybrid comparison concerns evidence composition/reader output, while the unbound comparison directly exposes the date-binding failure. The question has scope=in; the excluded-arrete Article 199 undecies A candidate is not used.",
        **french})
    ranker_path = Path(root) / "runs/hidate/final/ranker-used.json"
    ranker = json.loads(ranker_path.read_text(encoding="utf-8"))
    policy_path = Path(root) / "runs/hidate/policy/frozen.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return {"complete": True, "illustration_only": True, "model_forward_calls": 0, "training_runs": 0,
        "selection_scope": "Three main saved-output mechanism illustrations (C1, C2, C4) and one backup (C3): a same-topic cross-reference update, a capital-requirement exception, and primary in-scope French transfer. They do not estimate effect sizes or change any evaluation denominator.",
        "source_roots": {"japanese_archive": ".", "french_targeted_extract": "runs/hidate/case-studies/french-evidence.json"},
        "date_scope": "Japanese LawQA dates are corpus conditions, not newly annotated date-specific semantic gold labels. Structural queries have no semantic reader gold. French C4 uses its released date anchor and scope=in.",
        "method_definition": {"historical_rrf": "RRF(c)=sum_m 1/(60+rank_m(c)); BM25 and dense historical top50",
            "address": "Japanese (law_id, scope, article_num); French stable article_cid",
            "binding": "Map each discovered stable address to the eligible version/passages at target_asof; unavailable addresses remain explicit.",
            "address_probability": "p(a,q)=sigmoid(w^T((x(a,q)-mean)/scale)+intercept)",
            "fit": "One TRAIN StandardScaler and class-balanced LogisticRegression, C=1, seed42; no case-based fit or tuning",
            "feature_names_in_saved_model_order": ranker["feature_names"],
            "learned_final_score": "alpha*p(address,query)+(1-alpha)*reranker_sigmoid(query,current_chunk)",
            "alpha": policy["alpha"], "reranked_pair_budget": 10, "distinct_parent_evidence_budget": 4,
            "acceptance": "Reported separately from forced reader correctness; saved outputs include forced answers even when the support policy withholds.",
            "source_paths": [str(ranker_path).replace("\\", "/"), str(policy_path).replace("\\", "/"),
                "src/hidate/discovery.py", "src/hidate/ranking.py", "src/hidate/selection.py",
                "src/hidate/fiscal.py", "scripts/fiscal_benchmark.py"]}, "cases": cases}


def markdown(report):
    out = ["# HiDate saved-output case studies", "", report["selection_scope"], "", report["date_scope"], "",
        "English passages below are author translations/paraphrases of the adjacent saved original text; they are not official translations or model-generated answers.", "",
        "## Exact method used by these saved traces", "",
        "Historical BM25 and dense ranks are fused as `RRF(c) = sum_m 1/(60 + rank_m(c))`. Historical chunks identify stable addresses; binding maps them to the target-date version before current-text reranking.", "",
        "The saved TRAIN ranker uses `p(a,q) = sigmoid(w^T ((x-mean)/scale) + intercept)`. Its feature fields, in saved-model order, are:", "",
        *[f"- `{name}`" for name in report["method_definition"]["feature_names_in_saved_model_order"]], "",
        "The final learned score is `0.75*p(a,q) + 0.25*reranker_sigmoid(q,current_chunk)`, with ten reranked candidates and at most four distinct parent passages. Acceptance/withholding is separate from the saved forced reader answer.", ""]
    for case in sorted(report["cases"], key=lambda c: (c["case_role"] != "main", c["case_id"])):
        fr = case["case_id"] == "C4"
        q = case["query"]["saved"]
        query = q["prompt"] if fr else q["query_text"]
        identity = q["qid"] if fr else q["query_id"]
        out.extend([f"## {case['case_id']}. {case['title']} ({case['case_role']})", "", f"Query ID: `{identity}`", "",
            f"Scope: `{case['evaluation_scope']}`. Reader evidence: `{case['reader_scope']}`.", "",
            f"Original query: {query}", "", f"English meaning: {case['query_english']}", ""])
        for role, label, translation in (("historical_source", "Historical source", "historical_english"),
                                         ("asof_target", "Applicable target", "current_english")):
            r = case[role]["saved"]
            text = r["excerpt"]["text"] if case["case_id"] == "C1" else r["excerpt"]["text"].split("\n")[0]
            if fr:
                text = text.split("inclusivement.")[0] + "inclusivement."
                meta = f"version `{r['id']}`, stable article `{r['cid']}`, Article {r['num']}, valid `{r['date_debut']}` to `{r['date_fin']}`"
            else:
                meta = f"{r['law_title']}, Article `{r['article_num']}`, revision `{r['revision_id']}`, saved anchors `{', '.join(r['asof_dates'])}`"
            out.extend([f"**{label}:** {meta}.", "", "> " + text.replace("\n", "\n> "), "", case[translation], "",
                f"Source: `{case[role]['source_path']}` line {case[role]['source_line']}; JSON includes exact excerpt offsets.", ""])
        if fr:
            bound = case["discovery_and_binding"]["saved"]["address"]
            out.extend([f"Historical discovery: stable CID `{bound['article_cid']}` has {bound['features']['distinct_revision_count']} historical revisions; first historical RRF address hit rank {bound['features']['history_best_rank_rrf']}. Bound target is `{bound['bound_version_id']}`.", "",
                "| Method | Top version correct | Gold version in four | Strict | Accepted | Actual reader output |",
                "|---|---:|---:|---:|---:|---|"])
            for m in METHODS:
                s = case["records"]["scores.jsonl"][m]["saved"]
                response = case["records"]["responses.jsonl"][m]["saved"]
                short = response["response_text"].split("\n")[0].replace("|", "\\|")
                out.append(f"| {m} | {s['top_version_correct']} | {s['gold_version_recall_at_four']} | {s['strict']} | {s['accepted']} | {short} |")
            out.extend(["", "The hybrid output ended at `max_tokens`; the learned output ended at `eos`. All complete original responses and reader receipts are preserved in the JSON.", ""])
        else:
            b = case["discovery_and_binding"]["saved"]
            out.extend([f"Stable address: `{case['address']}`. Historical RRF hit rank {b['features']['history_best_rank_rrf']}; {b['features']['distinct_revision_count']} supporting revisions. Bound revision `{b['revision_id']}`. Full historical hit IDs, features, parent IDs and binding record are retained in the JSON.", "",
                "| Method | First evidence address | First evidence revision | Target parent rank | Accepted |",
                "|---|---|---|---:|---:|"])
            for m in METHODS:
                c = case["retrieval"]["saved"][m]
                top = c["evidence"][0] if c["evidence"] else {}
                rank = c.get("structural", {}).get("target_parent_rank", "not labelled for LawQA")
                out.append(f"| {m} | `{top.get('address')}` | `{top.get('revision_id')}` | {rank} | {c['accepted']} |")
            out.append("")
            if case["reader"]:
                out.extend(["Actual forced-choice reader results:", ""])
                for m in METHODS:
                    r = case["reader"]["answers.jsonl"][m]
                    a = r["saved"]
                    out.append(f"- `{m}`: prediction `{a['prediction']}`, released answer `{a['released_answer']}`, correct `{a['correct']}`, accepted `{a['accepted']}`; `{r['source_path']}` line {r['source_line']}, reader call {a['reader_receipt_call_number']}.")
                out.extend(["", "Released choices:", "", "```text", case["released_question"]["saved"]["選択肢"], "```", ""])
            else:
                out.extend(["No reader response exists for this structural query. Its success criterion is inclusion of the target parent in the four-passage evidence set.", ""])
        out.extend([case["interpretation"], "", "Trace source: `" + case["discovery_and_binding"]["source_path"] + "` line " + str(case["discovery_and_binding"]["source_line"]) + ".", ""])
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fiscal-only", action="store_true")
    parser.add_argument("--fiscal-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fiscal_only:
        result = french_bundle(args.root)
    else:
        result = annotated_report(args.root, japanese_cases(args.root),
            json.loads(args.fiscal_evidence.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.fiscal_only:
        args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
