"""Build HiDate automatic-result tables and two editable figures from saved summaries."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_ablations import configure, export_figure, vector_qa, write_csv


METHODS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound", "history_bound", "history_current", "history_learned")
VARIANTS = (("exact_quote", "Exact quote"), ("generated_paraphrase", "Generated paraphrase"))
CORAL, MINT, INK, MUTED, GRID = "#E48173", "#74B5A2", "#27323A", "#65717A", "#E7EBEE"
COLORS = {"history_learned": CORAL, "asof_hybrid": MINT}
LABELS = {"history_learned": "Full method", "asof_hybrid": "As-of hybrid"}


def load(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result["complete"]:
        raise ValueError(f"Completed saved summaries are required: {path}")
    return result


def overall(source, scope, variant, absent=None):
    return next(p for p in source[scope] if p["query_variant"] == variant and p["stratum"] == "all"
                and p["target_asof"] == "all" and p["source_quote_absent_from_target"] is absent)


def plot_data(scale, ruri, qwen, paths):
    scale_rows, stack_rows = [], []
    for variant, title in VARIANTS:
        for count in (120, 500, 1000):
            for method in ("history_learned", "asof_hybrid"):
                panel = scale["panels"][f"{count}|{method}"][variant + "|quote_absent"]
                if panel["n"] != 263:
                    raise ValueError("The scale figure requires the frozen263-query quote-absent panel")
                value = panel["target_recall_at_four"]
                scale_rows.append(dict(query_variant=variant, variant_label=title, laws=count, method=method,
                    n_queries=263, n_families=13, recalled_queries=round(value * 263), recall_at_four=value,
                    recall_percent=value * 100, source_file=paths["scale"].as_posix()))
        for stack, source in (("Ruri", ruri), ("Qwen3", qwen)):
            panel = overall(source, "natural", variant, True)
            if (panel["n_pairs"], panel["n_groups"]) != (263, 13):
                raise ValueError("The matched-stack figure requires263 queries and13 source families")
            for method in ("history_learned", "asof_hybrid"):
                estimate = panel["estimates"][method]["recall_at4"]
                stack_rows.append(dict(query_variant=variant, variant_label=title, stack=stack, method=method,
                    n_queries=263, n_families=13, recalled_queries=int(estimate["numerator"]),
                    recall_at_four=estimate["mean"], recall_percent=estimate["mean"] * 100,
                    ci_low_percent=estimate["ci95"][0] * 100, ci_high_percent=estimate["ci95"][1] * 100,
                    ci_method="saved family-cluster percentile interval", bootstrap_seed=42, bootstrap_draws=2000,
                    frozen_alpha=source["frozen_policy"]["alpha"], source_file=paths[stack].as_posix()))
    return scale_rows, stack_rows


def axes_style(ax, *, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if horizontal:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", length=3, width=.6, pad=4)


def title(figure, heading, subtitle):
    figure.text(.055, .967, heading, ha="left", va="top", size=11.5, weight="bold")
    figure.text(.055, .891, subtitle, ha="left", va="top", size=8.5, color=MUTED)


def legend(figure):
    handles = [Line2D([0], [0], color=COLORS[m], marker="o", markersize=4.7, linewidth=1.6,
                      label=LABELS[m]) for m in ("history_learned", "asof_hybrid")]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(.976, .838), ncol=2,
                  frameon=False, fontsize=8.5, handlelength=1.7, columnspacing=1.7)


def scale_figure(rows):
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.65), sharey=True)
    figure.subplots_adjust(left=.085, right=.976, bottom=.205, top=.705, wspace=.18)
    title(figure, "Scaling the distractor corpus", "Source-quote absent  |  263 queries, 13 families per variant")
    legend(figure)
    for index, ((variant, name), ax) in enumerate(zip(VARIANTS, axes)):
        for method in ("history_learned", "asof_hybrid"):
            selected = [r for r in rows if r["query_variant"] == variant and r["method"] == method]
            x, y = [r["laws"] for r in selected], [r["recall_percent"] for r in selected]
            ax.plot(x, y, marker="o", markersize=5, color=COLORS[method], linewidth=1.75,
                    markeredgecolor="white", markeredgewidth=.6, zorder=3)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 9 if method == "history_learned" else -14),
                    textcoords="offset points", ha="center", va="center", size=8.5,
                    color=INK if method == "history_learned" else "#467F70")
        ax.set_xlim(30, 1090)
        ax.set_ylim(54, 102)
        ax.set_xticks([120, 500, 1000])
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_title(f"({'ab'[index]})  {name}", loc="left", weight="bold", pad=9)
        ax.set_xlabel("Laws in retrieval corpus", labelpad=7)
        axes_style(ax)
    axes[0].set_ylabel("Target-parent Recall@4 (%)", labelpad=7)
    figure.text(.055, .035, "Point estimates; same TEST queries, original Ruri ranker and VALID policy frozen.",
                ha="left", va="bottom", size=8, color=MUTED)
    return figure


def stack_figure(rows):
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.65), sharey=True)
    figure.subplots_adjust(left=.12, right=.976, bottom=.235, top=.705, wspace=.18)
    title(figure, "Matched methods across retrieval stacks", "Source-quote absent  |  263 queries, 13 families per variant")
    legend(figure)
    for index, ((variant, name), ax) in enumerate(zip(VARIANTS, axes)):
        for stack, base_y in (("Ruri", 1), ("Qwen3", 0)):
            for method, offset in (("history_learned", .17), ("asof_hybrid", -.17)):
                row = next(r for r in rows if (r["query_variant"], r["stack"], r["method"]) == (variant, stack, method))
                mean, low, high = row["recall_percent"], row["ci_low_percent"], row["ci_high_percent"]
                y = base_y + offset
                ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", markersize=5.5,
                    markeredgecolor="white", markeredgewidth=.6, color=COLORS[method], elinewidth=1.4,
                    capsize=2.8, capthick=.8, zorder=4)
                ax.text(mean, y + (.13 if method == "history_learned" else -.13), f"{mean:.2f}",
                        ha="center", va="center", size=8.5, color=INK if method == "history_learned" else "#467F70")
        ax.set_xlim(38, 105)
        ax.set_ylim(-.56, 1.56)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_yticks([1, 0], ["Ruri", "Qwen3"])
        ax.set_title(f"({'ab'[index]})  {name}", loc="left", weight="bold", pad=9)
        axes_style(ax, horizontal=True)
    figure.text(.55, .143, "Target-parent Recall@4 (%)", ha="center", va="center", size=9)
    figure.text(.055, .06, "Bars: saved 95% family-cluster intervals. Each stack uses its own frozen VALID policy.",
                ha="left", va="bottom", size=8, color=MUTED)
    return figure


def qwen_csv(source, source_path):
    rows = []
    for scope in ("natural", "controlled", "lawqa"):
        for panel in source[scope]:
            for method, metrics in panel["estimates"].items():
                for metric, value in metrics.items():
                    ci = value["ci95"] or [None, None]
                    rows.append(dict(scope=scope, query_variant=panel.get("query_variant", "lawqa"),
                        stratum=panel.get("stratum", "all"), target_asof=panel["target_asof"],
                        source_quote_absent=panel.get("source_quote_absent_from_target"),
                        n_pairs=panel["n_pairs"], n_groups=panel["n_groups"], method=method, metric=metric,
                        mean=value["mean"], numerator=value["numerator"], denominator=value["denominator"],
                        denominator_groups=value["denominator_groups"], ci95_low=ci[0], ci95_high=ci[1],
                        valid_bootstrap_draws=value["n_valid_draws"], source_file=source_path.as_posix()))
    return rows


def complete_csvs(scale, qwen, fiscal, paths, output):
    output.mkdir(parents=True, exist_ok=True)
    qrows = qwen_csv(qwen, paths["Qwen3"])
    srows, frows = [], []
    for key, facets in scale["panels"].items():
        laws, method = key.split("|")
        for facet, panel in facets.items():
            variant, subset = facet.split("|")
            for metric, value in panel.items():
                if metric in ("n", "accepted_n"):
                    continue
                srows.append(dict(laws=int(laws), method=method, query_variant=variant, subset=subset,
                    n=panel["n"], accepted_n=panel["accepted_n"], metric=metric, value=value, source_file=paths["scale"].as_posix()))
    for scope, methods in fiscal["strata"].items():
        for method, panel in methods.items():
            for metric, value in panel.items():
                if metric in ("n", "expected_n", "accepted_n"):
                    continue
                frows.append(dict(scope=scope, method=method, n=panel["n"], expected_n=panel["expected_n"],
                    accepted_n=panel["accepted_n"], metric=metric, value=value,
                    contract="unprompted diagnostic; not official citation metric" if metric == "version_id_cited" else
                             "retrieval provenance" if metric in ("top_version_correct", "gold_version_recall_at_four") else "saved metric",
                    source_file=paths["fiscal"].as_posix()))
    for name, rows in (("qwen3-all-metrics.csv", qrows), ("scale-all-metrics.csv", srows), ("fiscal-all-metrics.csv", frows)):
        write_csv(output / name, rows)
    return {"qwen3_all_metric_rows": len(qrows), "scale_all_metric_rows": len(srows), "fiscal_all_metric_rows": len(frows)}


def percent(value):
    return "undefined" if value is None else f"{100 * value:.2f}"


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def metric_table(lines, estimates, names, *, methods=METHODS):
    def display(metric, value):
        if metric in ("mrr_at4", "raw_mrr_at50"):
            return "undefined" if value is None else f"{value:.4f}"
        return percent(value)
    table(lines, ["Method", *names], [[f"`{method}`", *[display(metric, estimates[method][metric]["mean"])
        for metric in names]] for method in methods])


def report(scale, ruri, qwen, fiscal, paths, output, counts):
    lines = ["# HiDate completed automatic results", "", "Evidence read back on 2026-10-18. All results below are saved model outputs; no model inference or new statistical resampling was run to produce this report.", "",
        "## Execution and scope", "",
        "One fixed seed 42 and the original TRAIN/VALID/TEST allocation are retained. The Japanese ranker uses 14,030 TRAIN queries and 2,490 VALID queries. The natural TEST contains 3,172 queries (1,586 per variant); 3,160 controlled inputs and 420 LawQA retrieval inputs are separate panels. Human expert assessment remains pending and is not represented by automatic scores.", ""]
    table(lines, ["Execution", "Job", "Verified output", "Completion evidence"], [
        ["Ruri nested scale", "5100278", "66612 = 3172 × 3 scales × 7 methods", "evaluation.complete=true;66612 saved rows"],
        ["Qwen3 full pipeline", "5100219", "6752 retrieval rows;3220 reader answers", "exit0;pipeline_complete=true;independent verification PASS;analysis.complete=true"],
        ["Fiscal reader", "5100298", "2431 = 221 × 11 responses", "reader.complete=true,2026-10-18 07:30:49 UTC"],
        ["Fiscal final score", "5100299", "2431 scored responses", "score-completion.complete=true,2026-10-18 07:30:58 UTC"]])
    lines += ["Sources below are paths relative to the repository root:", "",
        *[f"- {label}: `{path.as_posix()}`" for label, path in paths.items()],
        "- Qwen3 independent readback: `runs/hidate/extended-benchmark/qwen3/final/verification.json`.",
        "- Fiscal final score receipt: `runs/hidate/fiscal-transfer/fiscal/score-completion.json`.", "",
        "Rate/accuracy/coverage cells below are percentages; MRR cells retain their 0–1 scale. `undefined` means a zero denominator, including selective precision/accuracy when no response was accepted. No undefined value is replaced by zero. Full saved-facet values, numerators, denominators and existing confidence intervals are retained in the CSV tables linked below.", "",
        "## Main observations", "",
        "- In the 1,000-law bank, full-method Recall@4 on the 263-query source-quote-absent subset is 90.11% exact/91.63% generated; matched as-of hybrid obtains 65.78%/62.36%. Expansion reduces full-method overall Recall@4, especially for generated queries.",
        "- On the same 263-query/13-family subset, the original Ruri full method obtains 95.44%/95.44% and Qwen3 obtains 96.20%/95.82%. This is a robustness comparison between complete retrieval stacks with their own frozen VALID policies, not an isolated encoder intervention.",
        "- Qwen3 full-method coverage and rejection behavior vary by query variant. Controlled false-support is 6.14% exact/0.38% generated, versus 0.06%/0.13% for as-of hybrid. The stronger hard-subset retrieval result does not establish uniformly better abstention.",
        "- On Fiscal scored209, strict accuracy is 78.95% for the frozen Japanese learned method and 79.43% for history-bound. As-of/bound gold-version Recall@4 reaches 100%, versus 96.65% for learned. French transfer therefore does not dominate the simpler dated baselines.", "",
        "## Qwen3: all seven frozen methods", "",
        "Source: `tables/qwen3-all-metrics.csv`, exported from the completed Qwen3 analysis without re-scoring. Natural overall panels have 1,586 inputs/15 source families per variant, but target-parent/revision/MRR/Recall metrics use 1,580 available targets. Coverage uses all 1,586 inputs; support precision/risk uses accepted inputs. Quote-absent panels have 263 available targets/13 families. Existing 95% intervals use 2,000 source-family bootstrap draws with seed 42; these draws do not train models or define extra splits.", "",
        "Both retrieval models are Qwen3-0.6B. The fixed reader remains Qwen2.5-7B-Instruct. Qwen3 VALID selects alpha 1.0; the original Ruri policy uses alpha 0.75. Neither policy is selected on TEST or French data.", ""]
    lines += ["Raw@50 follows the saved analysis definition: first target-parent chunk rank for as-of/unbound methods, and full address ordering for bound/current/learned methods. These rank units should not be treated as identical retrieval stages.", ""]
    for variant, title in VARIANTS:
        for absent, subset in ((None, "All natural TEST"), (True, "Source-quote absent")):
            p = overall(qwen, "natural", variant, absent)
            lines += [f"### {title}: {subset}", "", f"Inputs={p['n_pairs']}; source families={p['n_groups']}. Rates are percentages; MRR is on the 0–1 scale.", ""]
            metric_table(lines, p["estimates"], ("top_address_accuracy", "target_revision_accuracy", "target_parent_accuracy", "mrr_at4", "recall_at4", "raw_mrr_at50", "raw_recall_at50"))
            metric_table(lines, p["estimates"], ("coverage", "support_precision", "support_risk", "unavailable_rejection_rate", "unavailable_false_support_rate"))
    lines += ["### Controlled target removal", "", "All targets in these panels were removed from the support corpus. False-support and rejection rates use all controlled queries; evidence availability and address discovery are reported separately. A lower false-support rate can accompany very low natural coverage.", ""]
    for variant, title in VARIANTS:
        for absent, subset in ((None, "all"), (True, "quote-absent")):
            p = overall(qwen, "controlled", variant, absent)
            lines += [f"#### {title}, {subset}: n={p['n_pairs']}, families={p['n_groups']}", ""]
            metric_table(lines, p["estimates"], ("rejection_rate", "false_support_rate", "nonempty_evidence_rate", "address_discovery_rate", "support_precision"))
    lines += ["### LawQA: all seven methods and two reader references", "", "Each table uses the same 140 released original choice questions. The three dates are corpus conditions, not three independently gold-labeled temporal QA sets; 420 retrieval inputs do not create 420 independent questions. Closed-book/oracle answers are computed once per question and displayed as shared references. Choice accuracy and source-derived retrieval support are separate endpoints.", ""]
    for p in qwen["lawqa"]:
        lines += [f"#### Corpus date {p['target_asof']}; n=140 original questions", ""]
        metric_table(lines, p["estimates"], ("accuracy", "coverage", "selective_accuracy", "selective_risk"), methods=("closed_book", "released_oracle", *METHODS))
    lines += ["## Ruri nested-law evaluation: all seven methods", "",
        "Source: `tables/scale-all-metrics.csv`. The three banks reuse identical natural TEST queries, original chunk IDs/boundaries, original Ruri vectors where available, and the original ranker/alpha 0.75/support thresholds. The 880 new laws enter only as distractors at four fixed anchors; expert before/after dates are excluded. There is no new ranker fitting or DEV policy selection. All three scales were evaluated uniformly in job 5100278.", "",
        "The scale summary uses all 1,586 inputs per variant for its overall means, including 6 unavailable targets assigned zero structural hits. It is not the available-target denominator used by the Qwen3 natural structural tables. The 263-query quote-absent subset contains available targets only and matches the figure denominator. Within each scale table, coverage uses all inputs; accepted precision uses accepted inputs.", ""]
    table(lines, ["Laws", "Eligible full-corpus chunks"], [[n, scale["scale_full_chunk_counts"][str(n)]] for n in (120, 500, 1000)])
    for n in (120, 500, 1000):
        for variant, title in VARIANTS:
            for subset, desc in (("all", "all natural TEST"), ("quote_absent", "source-quote absent")):
                panels = {m: scale["panels"][f"{n}|{m}"][variant + "|" + subset] for m in METHODS}
                lines += [f"### {n} laws, {title}, {desc}; n={panels[METHODS[0]]['n']}", ""]
                table(lines, ["Method", "Top parent", "Top revision", "MRR@4", "Recall@4", "Accepted n", "Coverage", "Accepted parent precision"],
                    [[f"`{m}`", percent(panels[m]["top_parent_correct"]), percent(panels[m]["top_revision_correct"]),
                      f"{panels[m]['target_mrr']:.4f}", percent(panels[m]["target_recall_at_four"]), panels[m]["accepted_n"],
                      percent(panels[m]["coverage"]), percent(panels[m]["accepted_parent_precision"])] for m in METHODS])
    lines += ["## FiscalQA R3 external evaluation: all eleven conditions", "",
        "Source: `tables/fiscal-all-metrics.csv`. Models, Japanese TRAIN ranker and Japanese VALID policy are frozen; French fit/update runs=0. The reader is fixed and greedy. The 221 published R3 questions comprise 209 primary scored questions, 8 review exclusions and 4 arrêté exclusions. The excluded strata are retained as secondary diagnostics and are not silently added to the official scored209 endpoint.", "",
        "Strict and nugget coverage call the released deterministic matching functions. `Nugget coverage` means answer-rubric coverage, not acceptance coverage. Article-nugget accuracy matches the official article-number regex in the answer. Retrieval top1 article/version identity and gold recall among up to four actually packed evidence passages are distinct evidence metrics. All conditions are forced to answer even if the frozen support policy withholds them; accepted-only statistics remain separate. The proprietary C-prod system is not reproduced.", ""]
    for scope, methods in fiscal["strata"].items():
        n = next(iter(methods.values()))["expected_n"]
        lines += [f"### {scope}: n={n} questions per condition", ""]
        table(lines, ["Condition", "Strict", "Nugget coverage", "Article nugget", "Accepted n", "Acceptance coverage", "Accepted strict"],
            [[f"`{m}`", percent(v["strict"]), percent(v["coverage"]), percent(v["article_citation_nugget"]),
              v["accepted_n"], percent(v["acceptance_coverage"]), percent(v["accepted_strict"])] for m, v in methods.items()])
        table(lines, ["Condition", "Gold article Recall@4", "Gold version Recall@4", "Top1 article", "Top1 version"],
            [[f"`{m}`", *[percent(v[k]) for k in ("gold_article_recall_at_four", "gold_version_recall_at_four", "top_article_correct", "top_version_correct")]] for m, v in methods.items()])
    lines += ["## Figure sources and complete metric tables", "",
        "- [Scale figure](figures/scale_quote_absent.pdf): point estimates from the frozen Ruri 120/500/1000-law bank, exact and generated variants. Underlying data: [12 rows](figures/scale_quote_absent_plotdata.csv). No new confidence intervals were computed for the scale curves.",
        "- [Matched-stack figure](figures/stack_quote_absent.pdf): Ruri/Qwen3 full method and matched as-of hybrid on the same 263-query/13-family panel. Bars are existing 95% family-cluster intervals, not a newly fitted cross-stack contrast. Underlying data: [8 rows](figures/stack_quote_absent_plotdata.csv).",
        f"- [All Qwen3 metrics](tables/qwen3-all-metrics.csv): {counts['qwen3_all_metric_rows']} saved-facet rows, including dates, strata, denominators and existing CIs.",
        f"- [All scale metrics](tables/scale-all-metrics.csv): {counts['scale_all_metric_rows']} rows across all three banks, seven methods and saved subsets.",
        f"- [All Fiscal metrics](tables/fiscal-all-metrics.csv): {counts['fiscal_all_metric_rows']} rows across all eleven conditions and all four published/scored/excluded strata; diagnostic fields remain labeled.",
        "- [Citation-contract research record](research-record-fiscal-citation.md): official retrieval provenance and the unprompted literal-version-ID diagnostic are explained separately. That diagnostic is not treated as official version-citation ability.",
        "- The existing [feature-group ablation report](../../../reports/feature-ablations.md) remains a separate experiment. Its without-history selector is not relabeled as the matched as-of-hybrid method.", "",
        "This package contains completed automatic results only. No human expert scores, new training seeds, new folds, TEST-selected thresholds or additional model runs are introduced.", ""]
    (output / "results-automatic.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruri", type=Path, default=Path("runs/hidate/analysis/analysis.json"))
    parser.add_argument("--qwen3", type=Path, default=Path("runs/hidate/extended-benchmark/qwen3/analysis/analysis.json"))
    parser.add_argument("--scale", type=Path, default=Path("runs/hidate/extended-benchmark/scale/evaluation.json"))
    parser.add_argument("--fiscal", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/hidate/extended-benchmark"))
    args = parser.parse_args()
    paths = {"Ruri": args.ruri, "Qwen3": args.qwen3, "scale": args.scale, "fiscal": args.fiscal}
    sources = {key: load(path) for key, path in paths.items()}
    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    configure()
    scale_rows, stack_rows = plot_data(sources["scale"], sources["Ruri"], sources["Qwen3"], paths)
    write_csv(figures / "scale_quote_absent_plotdata.csv", scale_rows)
    write_csv(figures / "stack_quote_absent_plotdata.csv", stack_rows)
    counts = complete_csvs(sources["scale"], sources["Qwen3"], sources["fiscal"], paths, output / "tables")
    report(sources["scale"], sources["Ruri"], sources["Qwen3"], sources["fiscal"], paths, output, counts)
    quality = {"sources": {k: str(v) for k, v in paths.items()}, "scale_plot_rows": len(scale_rows),
        "stack_plot_rows": len(stack_rows), **counts, "figures": {}}
    for name, figure in (("scale_quote_absent", scale_figure(scale_rows)), ("stack_quote_absent", stack_figure(stack_rows))):
        path = figures / name
        clipped = export_figure(figure, path)
        quality["figures"][name] = {**vector_qa(path), "text_outside_canvas": clipped}
    quality["complete"] = all(q["native_vector"] and q["editable_text"] and q["pdf_min_font_points"] >= 7.99
                              and not q["text_outside_canvas"] for q in quality["figures"].values())
    (figures / "automatic-figure-qa.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    if not quality["complete"]:
        raise ValueError("Native vector, readable font and canvas checks are required")


if __name__ == "__main__":
    main()
