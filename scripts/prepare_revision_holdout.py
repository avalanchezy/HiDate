"""Freeze new HiDate test families from identity/date metadata; optional PBS text census."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import random


ANCHORS = ("2018-09-30", "2021-09-30", "2024-09-30", "2026-09-30")
ALLOCATION = {"Act": 34, "CabinetOrder": 33, "MinisterialOrdinance": 33}
SEED = 42


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def family_resolver(catalogue):
    """The exact longest Act-title-prefix rule of hidate.sources."""
    acts = sorted(((r["revision_info"]["law_title"], r["law_info"]["law_id"])
                   for r in catalogue if r["law_info"]["law_type"] == "Act"),
                  key=lambda pair: (-len(pair[0]), pair[1]))

    def resolve(row):
        info, revision = row["law_info"], row["revision_info"]
        if info["law_type"] != "Act":
            for title, law_id in acts:
                if revision["law_title"].startswith(title):
                    return law_id
        return info["law_id"]

    return resolve, acts


def freeze(args):
    output, source, original = args.output, args.sources_root, args.original_sources
    selected, old = read(source / "selected-laws.json"), read(original / "selected-laws.json")
    split, reserved = read(args.original_split), read(original / "source-exclusions.json")
    catalogue = read(original / "catalogue.json")
    resolve, acts = family_resolver(catalogue)
    for row in selected:
        if resolve(row) != row["family_id"]:
            raise ValueError("Saved family differs from the original longest-prefix rule: " + row["law_info"]["law_id"])
    old_ids = {r["law_info"]["law_id"] for r in old}
    old_families = {r["family_id"] for r in old}
    if set(split) != old_families:
        raise ValueError("Original selected families and saved TRAIN/VALID/TEST allocation differ")
    lawqa_ids, lawqa_titles = set(reserved["law_ids"]), set(reserved["family_titles"])
    lawqa_families = set(lawqa_ids)
    for row in catalogue:
        if row["law_info"]["law_id"] in lawqa_ids or row["revision_info"]["law_title"] in lawqa_titles:
            lawqa_families.add(resolve(row))
    for title in lawqa_titles:
        for act_title, act_id in acts:
            if title.startswith(act_title):
                lawqa_families.add(act_id)
                break
    # Also compare the actual frozen LawQA law list to the collection reservation.
    from hidate.sources import source_exclusions
    lawqa_release = source_exclusions(read(original / "lawqa/law_list.json"))
    if set(lawqa_release["law_ids"]) != lawqa_ids or set(lawqa_release["family_titles"]) != lawqa_titles:
        raise ValueError("LawQA release sources and saved source-exclusions disagree")
    bindings = {(r["law_id"], r["requested_asof"]): r for r in rows(source / "asof-bindings.jsonl")}
    extras = [r for r in selected if r["law_info"]["law_id"] not in old_ids]
    if len(extras) != 880 or len({r["family_id"] for r in extras}) != 880:
        raise ValueError("Expected exactly 880 distinct new distractor families")

    def binding_view(law_id, day):
        b = bindings[law_id, day]
        f = source / b["raw_path"]
        rev = b["revision_info"]
        return {"requested_asof": day, "revision_id": b["revision_id"],
                "amendment_enforcement_date": rev.get("amendment_enforcement_date"),
                "amendment_promulgate_date": rev.get("amendment_promulgate_date"),
                "amendment_scheduled_enforcement_date": rev.get("amendment_scheduled_enforcement_date"),
                "current_revision_status": rev.get("current_revision_status"),
                "repeal_status": rev.get("repeal_status"), "status": b["status"],
                "raw_path": b["raw_path"], "source_file_exists": f.is_file(),
                "source_file_bytes": f.stat().st_size if f.is_file() else None,
                "official_url": f"https://laws.e-gov.go.jp/api/2/law_data/{law_id}?asof={day}&law_full_text_format=json&json_format=full&response_format=json"}

    candidates, excluded = [], []
    for law in extras:
        info, revision, family = law["law_info"], law["revision_info"], law["family_id"]
        law_id, title = info["law_id"], revision["law_title"]
        why = []
        if family in set(split) | old_families or law_id in old_ids:
            why.append("original_supervised_or_test_family")
        if law_id in lawqa_ids or family in lawqa_families or title in lawqa_titles:
            why.append("reserved_lawqa_family")
        if info["law_type"] != "Act" and family == law_id and any(title.startswith(t) for t in lawqa_titles):
            why.append("reserved_lawqa_title_prefix")
        anchor_bindings = [binding_view(law_id, day) for day in ANCHORS]
        if any(b["status"] != "collected" or not b["source_file_exists"] or not b["source_file_bytes"] for b in anchor_bindings):
            why.append("incomplete_official_anchor_sources")
        revision_count = len({b["revision_id"] for b in anchor_bindings})
        if revision_count < 2:
            why.append("no_observed_revision_change_across_anchors")
        item = {"family_id": family, "law_id": law_id, "law_type": info["law_type"],
                "law_title": title, "law_num": info["law_num"], "category": revision.get("category"),
                "promulgation_date": info.get("promulgation_date"),
                "prior_role": "distractor_only_in_completed_scale_corpus",
                "anchor_bindings": anchor_bindings, "distinct_anchor_revisions": revision_count,
                "text_census_status": "pending_PBS_visible_MainProvision_census"}
        if why:
            excluded.append({**item, "reasons": why})
        else:
            candidates.append(item)
    candidate_map = {r["family_id"]: r for r in candidates}
    expert_laws, expert_boundaries = read(source / "expert-selected-laws.json"), read(source / "expert-boundaries.json")
    boundaries = {r["family_id"]: r for r in expert_boundaries}
    expert_ids = sorted(r["family_id"] for r in expert_laws)
    if len(expert_ids) != 40 or len(set(expert_ids)) != 40 or not set(expert_ids) <= set(candidate_map):
        raise ValueError("Expert reservation requires 40 eligible, original/LawQA-disjoint families")
    experts = []
    for family in expert_ids:
        item, boundary = candidate_map[family], boundaries[family]
        before = binding_view(item["law_id"], boundary["before"])
        after = binding_view(item["law_id"], boundary["after"])
        if ((date.fromisoformat(boundary["after"]) - date.fromisoformat(boundary["before"])).days != 1
                or before["revision_id"] == after["revision_id"]
                or after["revision_id"] != boundary["revision_id"]
                or any(b["status"] != "collected" or not b["source_file_exists"] for b in (before, after))):
            raise ValueError("Official before/after boundary does not match its saved sources: " + family)
        experts.append({**item, "boundary": {**boundary, "before_binding": before, "after_binding": after},
                        "planned_base_question_slots": 6, "planned_paired_subset_slots": 2})
    # One sorted family-ID population per law type; one seed-42 RNG, consumed in this order.
    rng, confirmatory = random.Random(SEED), []
    for kind, count in ALLOCATION.items():
        pool = sorted((r for r in candidates if r["law_type"] == kind and r["family_id"] not in expert_ids),
                      key=lambda r: r["family_id"])
        rng.shuffle(pool)
        if len(pool) < count:
            raise ValueError("Insufficient eligible families for " + kind)
        confirmatory.extend(pool[:count])
    confirm_ids = {r["family_id"] for r in confirmatory}
    assert not confirm_ids.intersection(expert_ids)
    role_ids = {"confirmatory_test": [r["family_id"] for r in confirmatory], "expert_scenarios": expert_ids}
    if (output / "family-manifest.json").exists():
        previous = read(output / "family-manifest.json")
        previous_ids = {k: [r["family_id"] for r in previous["roles"][k]] for k in role_ids}
        if previous_ids != role_ids:
            raise ValueError("Existing frozen family identities differ; keep the existing manifest")
    output.mkdir(parents=True, exist_ok=True)
    exclusions = {"original_sources": original.as_posix(), "original_split": args.original_split.as_posix(),
                  "original_law_ids": sorted(old_ids), "original_family_allocation": split,
                  "lawqa_law_ids": sorted(lawqa_ids), "lawqa_family_ids_including_roots": sorted(lawqa_families),
                  "lawqa_family_titles": sorted(lawqa_titles), "lawqa_guidance_sources": reserved["guidance"],
                  "fiscal": {"jurisdiction": "France", "corpus_id": "LEGITEXT000006069577",
                             "identity_namespace": "French LEGITEXT/LEGIARTI IDs, disjoint from Japanese e-Gov IDs",
                             "new_fiscal_questions_or_labels_used": False}}
    protocol = {"seed": SEED, "sampling_unit": "family", "confirmatory_allocation": ALLOCATION,
                "algorithm": "Sort family IDs within Act/CabinetOrder/MinisterialOrdinance; shuffle each list with one Random(42) RNG consumed in that order; take 34/33/33.",
                "expert_reservation": "All 40 existing metadata-selected Act families with complete real enforcement-boundary pairs; excluded from confirmatory sampling.",
                "eligibility": ["new law and family outside all original TRAIN/VALID/TEST families",
                                "outside original LawQA IDs and exact longest-Act-title-prefix families",
                                "four official collected anchor JSONs present; at least two returned revision IDs"],
                "selection_inputs": ["catalogue identity/family metadata", "official as-of bindings", "file existence and byte length", "existing expert boundary metadata"],
                "confirmatory_role": "Automatic source-derived retrieval test only; never TRAIN/VALID or method selection.",
                "expert_role": "Independent human-authored scenarios, gold evidence and dated answers; metadata is not an automatic answer key.",
                "query_preparation": {"source_protocol": "configs/data-protocol.json", "primary_scope": "MainProvision",
                                      "anchor_dates": list(ANCHORS), "seed": SEED,
                                      "original_cap_per_family_target_stratum": 25,
                                      "eligibility": "existing visible/unambiguous address and first-prefix text rules; enumerate under PBS",
                                      "empty_family_policy": "retain frozen identity and report zero eligible rows; no performance-driven replacements"},
                "evaluation": {"discovery_top_k": 50, "reranker_top_k": 10, "evidence_parents": 4,
                               "family_cluster_bootstrap_seed": SEED, "original_split_modified": False,
                               "recipe_freeze": "Fit and select on original TRAIN/VALID; evaluate the frozen recipe once on confirmatory queries.",
                               "exposure": "These texts previously served as scale distractors, without supervised training/selection targets."},
                "text_preparation": "Bulk visible-text parsing and query/label generation use PBS; this freeze reads only identity/date/file metadata."}
    summary = {"new_laws": len(extras), "new_families": len({r["family_id"] for r in extras}),
               "eligible_families": len(candidates), "excluded_new_families": len(excluded),
               "candidate_types": dict(Counter(r["law_type"] for r in candidates)),
               "original_laws": len(old), "original_families": len(split),
               "original_split_counts": dict(Counter(split.values())),
               "lawqa_reserved_laws": len(lawqa_ids), "lawqa_reserved_family_ids_including_roots": len(lawqa_families),
               "confirmatory_families": len(confirmatory), "confirmatory_types": dict(Counter(r["law_type"] for r in confirmatory)),
               "expert_families": len(experts), "expert_types": dict(Counter(r["law_type"] for r in experts)),
               "confirmatory_expert_overlap": 0, "original_family_overlap": len((confirm_ids | set(expert_ids)) & old_families),
               "lawqa_family_overlap": len((confirm_ids | set(expert_ids)) & lawqa_families),
               "four_anchor_bindings": len(extras) * len(ANCHORS), "all_saved_bindings": len(bindings),
               "source_collection_complete": read(source / "manifest.json")["complete"],
               "source_provision_count_is_unparsed_placeholder": all(b.get("provision_count") == 0 for b in bindings.values()),
               "expert_real_before_after_pairs": len(experts), "distinct_anchor_revision_counts": dict(Counter(r["distinct_anchor_revisions"] for r in candidates)),
               "human_authored_questions": 0, "automatically_generated_test_queries": 0,
               "text_census_status": "pending_PBS", "training_runs": 0, "inference_runs": 0}
    manifest = {"schema_version": 1, "frozen_on": "2026-10-19", "status": "family_identities_frozen",
                "sources_root": source.as_posix(), "original_sources": original.as_posix(),
                "protocol": protocol, "summary": summary,
                "roles": {"confirmatory_test": confirmatory, "expert_scenarios": experts,
                          "remaining_distractor_only": sorted(set(candidate_map) - confirm_ids - set(expert_ids))}}
    write(output / "family-manifest.json", manifest)
    write(output / "frozen-original-exclusions.json", exclusions)
    write(output / "sampling-protocol.json", protocol)
    write(output / "freeze-summary.json", summary)
    with (output / "candidate-families.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(candidates, key=lambda r: r["family_id"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(output / "excluded-families.json", excluded)
    write(output / "expert-material-plan.json", {
        "family_role": "expert_scenarios", "families": len(experts), "base_questions": 240,
        "base_questions_per_family": 6, "paired_question_subset": 80,
        "paired_questions_per_family": 2, "total_question_date_instances": 320,
        "scenario_types": {"old_terminology_at_explicit_date": 60, "independent_real_scenario": 60,
                           "changed_condition_amount_or_subject": 60, "insufficient_target_date_evidence": 60},
        "paired_subset": {"old_terminology_at_explicit_date": 30, "independent_real_scenario": 30,
                          "changed_condition_amount_or_subject": 20, "insufficient_target_date_evidence": 0},
        "paired_answer_change_plan": {"changed": 40, "unchanged": 40},
        "authorship": "Humans write questions without model ranks/results and without paraphrasing original generated source prefixes.",
        "gold_fields": ["question_ja", "asof_date", "answerability", "gold_answer_ja", "acceptable_law_ids_and_scoped_addresses", "revision_ids", "literal_evidence_spans", "source_urls", "author_id", "authored_on"],
        "annotation": "Accept multiple valid evidence addresses; date-specific changed/unchanged labels are verified by experts. A changed or negated legal rule can be answerable; missing evidence is a separate label.",
        "independent_ratings": ["answer_correct", "citation_support", "dated_version"],
        "aggregation": "Independent expert scores and agreement summaries; no fabricated ratings or automatic expert adjudication.",
        "pilot": "First 40 base questions across all families, then remaining 200; pilot quality checks use no model outcome.",
        "human_authored_questions": 0, "gold_answers_completed": 0})
    return summary


def census(args):
    from hidate.sources import extract_provisions
    from hidate.data import visible_provision
    manifest = read(args.output / "family-manifest.json")
    tasks = []
    for role in ("confirmatory_test", "expert_scenarios"):
        for family in manifest["roles"][role]:
            bindings = family["anchor_bindings"] if role == "confirmatory_test" else [family["boundary"][k] for k in ("before_binding", "after_binding")]
            tasks.extend((role, family, b) for b in bindings)
    result, totals = [], Counter()
    for role, family, binding in tasks:
        snapshot = read(args.sources_root / binding["raw_path"])
        if snapshot["revision_info"]["law_revision_id"] != binding["revision_id"]:
            raise ValueError("Returned revision differs from frozen binding")
        groups = defaultdict(list)
        for raw in extract_provisions(snapshot):
            if raw["scope"] == "MainProvision" and raw["article_num"] is not None:
                groups[raw["article_num"]].append(visible_provision(raw, snapshot))
        visible = [values[0] for values in groups.values() if len(values) == 1 and values[0]["visible_text"].strip()]
        eligible = [r for r in visible if any(len("".join(p["text"][:96].split())) >= 24
                                            for p in r["visible_paragraphs"])]
        row = {"role": role, "family_id": family["family_id"], "law_id": family["law_id"],
               "requested_asof": binding["requested_asof"], "revision_id": binding["revision_id"],
               "main_article_addresses": len(groups), "ambiguous_addresses": sum(len(v) > 1 for v in groups.values()),
               "unambiguous_visible_articles": len(visible), "eligible_first_prefix_articles": len(eligible),
               "visible_characters": sum(len(r["visible_text"]) for r in visible)}
        result.append(row);totals[role + "_snapshots"] += 1
    with (args.output / "visible-text-census.jsonl").open("w", encoding="utf-8") as stream:
        for row in result:stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    receipt = {"complete": True, "job_id": os.environ.get("PBS_JOBID"), "seed": SEED,
               "at_utc": datetime.now(timezone.utc).isoformat(), "snapshots": len(result),
               "counts": dict(totals), "family_selection_modified": False,
               "query_rows_generated": 0, "gold_answers_generated": 0}
    write(args.output / "visible-text-census.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "census"), nargs="?", default="freeze")
    parser.add_argument("--sources-root", type=Path, default=Path("data/raw/hidate/extended-benchmark/extension"))
    parser.add_argument("--original-sources", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--original-split", type=Path, default=Path("data/processed/hidate/family-split.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hidate-holdout"))
    args = parser.parse_args()
    print(json.dumps((freeze if args.phase == "freeze" else census)(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
