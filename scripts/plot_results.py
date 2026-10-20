"""Render manuscript figures from completed HiDate evaluation summaries."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
import numpy as np

from plot_ablations import configure, export_figure, vector_qa, write_csv, plot_rows, recall_figure, effect_figure
from plot_automatic_results import load, overall, plot_data, scale_figure, stack_figure


METHODS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound", "history_bound", "history_current", "history_learned")
LABELS = {"asof_bm25": "As-of BM25", "asof_dense": "As-of dense", "asof_hybrid": "As-of hybrid",
          "history_unbound": "History unbound", "history_bound": "History bound", "history_current": "History current",
          "history_learned": "HiDate", "closed_book": "Closed book", "static_bm25": "Static BM25",
          "static_dense": "Static dense", "oracle_asof": "As-of oracle"}
VARIANTS = (("exact_quote", "Exact quote"), ("generated_paraphrase", "Generated paraphrase"))
CORAL, MINT, GRAY, INK, MUTED, GRID = "#E48173", "#74B5A2", "#A2ACB5", "#27323A", "#65717A", "#E7EBEE"
AMBER, LIGHT, BLUE = "#D8AE65", "#DEE4E8", "#92AABB"
CMAP = LinearSegmentedColormap.from_list("hidate_profile", ["#F3F6F6", "#C0DCD4", "#4D8B79"])
CMAP.set_bad("#F0F2F3")


def color(method):
    return CORAL if method == "history_learned" else MINT if method == "asof_hybrid" else GRAY


def header(fig, name, subtitle):
    fig.text(.045, .975, name, ha="left", va="top", size=12, weight="bold")
    fig.text(.045, .922, subtitle, ha="left", va="top", size=8.5, color=MUTED)


def clean(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, width=.6, pad=4)


def polish(fig):
    fig.canvas.draw()
    def rename(text):
        return text.replace("Full method", "HiDate").replace("Current text", "History current").replace("Gold oracle", "As-of oracle")
    for ax in fig.axes:
        labels = [item.get_text() for item in ax.get_yticklabels()]
        if any(rename(text) != text for text in labels):
            ax.set_yticks(ax.get_yticks(), [rename(text) for text in labels])
    for item in fig.findobj(Text):
        item.set_text(rename(item.get_text()))
        if item.get_fontsize() < 9.5:
            item.set_fontsize(9.5)
    return fig


def heatmap(ax, matrix, column_labels, *, highlight=None):
    matrix = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(matrix)
    ax.pcolormesh(np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1), masked,
                  cmap=CMAP, vmin=0, vmax=100, edgecolors="white", linewidth=.8, rasterized=False)
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(matrix.shape[1]) + .5, column_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=5, labelsize=8.5)
    ax.set_yticks([])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            label = "NA" if np.isnan(value) else f"{value:.1f}"
            ax.text(j + .5, i + .5, label, ha="center", va="center", size=8.5,
                    color="white" if value >= 78 else INK)
    if highlight is not None:
        ax.add_patch(Rectangle((0, highlight), matrix.shape[1], 1, fill=False, edgecolor=CORAL, linewidth=1.2))
    for spine in ax.spines.values():
        spine.set_visible(False)


def primary_data(ruri, source):
    rows = []
    for variant, _ in VARIANTS:
        for subset, absent in (("all", None), ("quote_absent", True)):
            panel = overall(ruri, "natural", variant, absent)
            for method in METHODS:
                for metric in ("target_revision_accuracy", "mrr_at4", "recall_at4", "coverage", "support_precision"):
                    v = panel["estimates"][method][metric]
                    rows.append(dict(variant=variant, subset=subset, n_inputs=panel["n_pairs"], n_families=panel["n_groups"],
                        method=method, metric=metric, numerator=v["numerator"], denominator=v["denominator"],
                        value=v["mean"], percent=None if v["mean"] is None else v["mean"] * 100, source_file=source.as_posix()))
    return rows


def primary_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.6), gridspec_kw={"width_ratios": [1, 1.08]})
    fig.subplots_adjust(left=.205, right=.98, top=.745, bottom=.14, wspace=.34, hspace=.67)
    header(fig, "Seven-method retrieval profiles", "Original Ruri stack  |  fixed natural TEST and source-quote-absent queries")
    fig.legend(handles=[Line2D([0], [0], marker="D", color=GRAY, markerfacecolor="white", linestyle="", markersize=4.8,
                              label="All natural: 1,580 available targets"),
                        Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=5, label="Quote absent: 263 targets")],
               loc="upper left", bbox_to_anchor=(.045, .877), frameon=False, ncol=2, fontsize=8.5,
               columnspacing=1.4, handletextpad=.45)
    for row_index, (variant, name) in enumerate(VARIANTS):
        left, right = axes[row_index]
        values = {(r["subset"], r["method"], r["metric"]): r for r in rows if r["variant"] == variant}
        matrix = []
        for i, method in enumerate(METHODS):
            y = 6 - i
            all_value = values["all", method, "recall_at4"]["percent"]
            absent = values["quote_absent", method, "recall_at4"]["percent"]
            c = color(method)
            left.plot([absent, all_value], [y, y], color=c, linewidth=1.4, alpha=.8)
            left.scatter(all_value, y, marker="D", s=25, facecolors="white", edgecolors=c, linewidths=1, zorder=4)
            left.scatter(absent, y, s=29, color=c, edgecolors="white", linewidths=.4, zorder=5)
            left.text(absent - 1.6, y, f"{absent:.1f}", ha="right", va="center", size=8.3,
                      color=INK if method == "history_learned" else MUTED)
            matrix.append([values["quote_absent", method, metric]["percent"]
                           if values["quote_absent", method, metric]["percent"] is not None else np.nan
                           for metric in ("target_revision_accuracy", "mrr_at4", "recall_at4", "coverage", "support_precision")])
        left.set_xlim(44, 103)
        left.set_ylim(-.6, 6.6)
        left.set_xticks([50, 75, 100])
        left.set_yticks(range(6, -1, -1), [LABELS[m] for m in METHODS], fontsize=8.7)
        for tick, method in zip(left.get_yticklabels(), METHODS):
            if method in ("history_learned", "asof_hybrid"):
                tick.set_color(color(method)); tick.set_fontweight("bold")
        left.set_xlabel("Target-parent Recall@4 (%)", labelpad=5)
        left.set_title(f"({'ac'[row_index]})  {name}", loc="left", size=9.5, weight="bold", pad=30)
        clean(left)
        heatmap(right, matrix, ["Top1", "MRR", "R@4", "Cov.", "Prec."], highlight=6)
        right.set_title(f"({'bd'[row_index]})  Quote-absent profile", loc="left", size=9.5, weight="bold", pad=30)
    fig.text(.045, .055, "Profile: top1 revision, MRR@4 × 100, Recall@4, coverage and accepted precision (%).",
             ha="left", va="bottom", size=8, color=MUTED)
    fig.text(.045, .025, "Natural: 1,580 available; quote absent: 263 / 13 families. NA: no accepted predictions.",
             ha="left", va="bottom", size=8, color=MUTED)
    return fig


def binding_data(ruri, qwen, sources):
    rows = []
    for stack, source in (("Ruri", ruri), ("Qwen3", qwen)):
        for variant, _ in VARIANTS:
            panel = overall(source, "natural", variant, True)
            assert (panel["n_pairs"], panel["n_groups"]) == (263, 13)
            for method in ("history_unbound", "history_bound", "history_current", "history_learned"):
                estimates = panel["estimates"][method]
                addresses = int(estimates["top_address_accuracy"]["numerator"])
                revisions = int(estimates["target_revision_accuracy"]["numerator"])
                assert 0 <= revisions <= addresses <= 263
                for outcome, count in (("correct_revision", revisions), ("correct_address_other_revision", addresses - revisions),
                                       ("other_address", 263 - addresses)):
                    rows.append(dict(stack=stack, variant=variant, method=method, outcome=outcome, n_queries=263,
                        n_families=13, count=count, percent=count / 263 * 100, source_file=sources[stack].as_posix()))
    return rows


def binding_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.6), sharey=True)
    fig.subplots_adjust(left=.238, right=.978, top=.755, bottom=.19, wspace=.16)
    header(fig, "From article addresses to dated revisions", "Source-quote absent  |  263 queries and 13 families per query variant")
    outcomes = (("correct_revision", "Target revision", MINT),
                ("correct_address_other_revision", "Correct address, other revision", AMBER),
                ("other_address", "Other address", LIGHT))
    fig.legend(handles=[Patch(facecolor=c, label=label) for _, label, c in outcomes], ncol=3,
               loc="upper left", bbox_to_anchor=(.045, .873), frameon=False, fontsize=8.3,
               handlelength=1.1, columnspacing=1.2)
    methods = ("history_unbound", "history_bound", "history_current", "history_learned")
    labels = []
    keys = []
    short = {"history_unbound": "Unbound", "history_bound": "Bound", "history_current": "History current", "history_learned": "HiDate"}
    for stack in ("Ruri", "Qwen3"):
        for method in methods:
            labels.append(stack + " · " + short[method]); keys.append((stack, method))
    for panel_index, ((variant, name), ax) in enumerate(zip(VARIANTS, axes)):
        for i, (stack, method) in enumerate(keys):
            y, start = 7 - i, 0
            for outcome, _, c in outcomes:
                r = next(r for r in rows if (r["variant"], r["stack"], r["method"], r["outcome"]) == (variant, stack, method, outcome))
                ax.barh(y, r["percent"], left=start, height=.63, color=c, edgecolor="white", linewidth=.5)
                if r["percent"] >= 10:
                    ax.text(start + r["percent"] / 2, y, f"{r['percent']:.1f}", ha="center", va="center", size=8.5, color=INK)
                start += r["percent"]
        ax.axhline(3.5, color=GRID, linewidth=.9)
        ax.set_xlim(0, 100)
        ax.set_ylim(-.65, 7.65)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(range(7, -1, -1), labels, fontsize=8.6)
        ax.set_title(f"({'ab'[panel_index]})  {name}", loc="left", weight="bold", pad=9)
        ax.set_xlabel("Top-ranked evidence (%)", labelpad=6)
        clean(ax)
    for label, (_, method) in zip(axes[0].get_yticklabels(), keys):
        if method == "history_learned":label.set_color(CORAL); label.set_fontweight("bold")
    fig.text(.045, .035, "Each row partitions 263 targets; an address match can still select another revision.",
             ha="left", va="bottom", size=8, color=MUTED)
    return fig


def fiscal_data(fiscal, source):
    rows = []
    for method, summary in fiscal["strata"]["scored209"].items():
        assert summary["n"] == summary["expected_n"] == 209
        for metric in ("strict", "gold_article_recall_at_four", "gold_version_recall_at_four", "top_version_correct"):
            value = summary[metric]
            rows.append(dict(method=method, metric=metric, n_queries=209, value=value, percent=100 * value,
                numerator=round(209 * value), source_file=source.as_posix(), scope="scored209"))
    return rows


def fiscal_figure(rows):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 5.15), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.204, right=.978, top=.752, bottom=.15, wspace=.34)
    header(fig, "French tax-law transfer", "FiscalQA R3  |  209 officially scored questions  |  frozen Japanese ranker")
    left, right = axes
    matrix = []
    for i, method in enumerate(methods):
        y = len(methods) - 1 - i
        strict = next(r["percent"] for r in rows if r["method"] == method and r["metric"] == "strict")
        left.barh(y, strict, height=.64, color=color(method), edgecolor="white", linewidth=.4)
        left.text(strict + 1.4, y, f"{strict:.2f}", ha="left", va="center", size=8.4,
                  weight="bold" if method == "history_learned" else "normal", color=INK)
        matrix.append([next(r["percent"] for r in rows if r["method"] == method and r["metric"] == metric)
                       for metric in ("gold_article_recall_at_four", "gold_version_recall_at_four", "top_version_correct")])
    left.set_xlim(0, 100)
    left.set_ylim(-.6, len(methods) - .4)
    left.set_xticks([0, 25, 50, 75, 100])
    left.set_yticks(range(len(methods) - 1, -1, -1), [LABELS[m] for m in methods], fontsize=8.6)
    for tick, method in zip(left.get_yticklabels(), methods):
        if method in ("history_learned", "asof_hybrid"):
            tick.set_color(color(method)); tick.set_fontweight("bold")
    left.set_title("(a)  Official strict accuracy", loc="left", weight="bold", pad=8)
    left.set_xlabel("All required answer nuggets (%)", labelpad=7)
    clean(left)
    heatmap(right, matrix, ["Article\nRecall@4", "Version\nRecall@4", "Top1\nversion"], highlight=methods.index("history_learned"))
    right.set_title("(b)  Evidence provenance (%)", loc="left", weight="bold", pad=31)
    fig.text(.045, .038, "Released strict answer scoring; evidence profiles use the actually packed passages.",
             ha="left", va="bottom", size=8, color=MUTED)
    return fig


def temporal_data(ruri, source):
    rows = []
    dates = sorted({p["target_asof"] for p in ruri["natural"] if p["target_asof"] != "all"})
    def count(variant, date, stratum="all", absent=None):
        return next((p["n_pairs"] for p in ruri["natural"] if p["query_variant"] == variant and p["target_asof"] == date
                     and p["stratum"] == stratum and p["source_quote_absent_from_target"] is absent), 0)
    for date in dates:
        values = {key: count("exact_quote", date, key) for key in ("same_date", "changed_cross_date", "unavailable_target")}
        total, absent = count("exact_quote", date), count("exact_quote", date, absent=True)
        assert sum(values.values()) == total and absent <= values["changed_cross_date"]
        assert all(count("exact_quote", date, key) == count("generated_paraphrase", date, key)
                   for key in ("all", "same_date", "changed_cross_date", "unavailable_target"))
        rows.append(dict(target_date=date, n_per_variant=total, **values, source_quote_absent=absent,
            changed_quote_present=values["changed_cross_date"] - absent,
            quote_absent_fraction_of_changed=absent / values["changed_cross_date"] if values["changed_cross_date"] else None,
            source_file=source.as_posix()))
    assert sum(r["n_per_variant"] for r in rows) == 1586 and sum(r["source_quote_absent"] for r in rows) == 263
    return rows


def temporal_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.05), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.08, right=.978, top=.735, bottom=.22, wspace=.33)
    header(fig, "Temporal coverage of the held-out queries", "Natural TEST  |  1,586 queries per variant; 3,172 exact and generated inputs in total")
    left, right = axes
    x, bottom = np.arange(len(rows)), np.zeros(len(rows))
    status = (("same_date", "Same date", BLUE), ("changed_cross_date", "Changed cross-date", MINT),
              ("unavailable_target", "Unavailable", CORAL))
    for key, label, c in status:
        vals = [r[key] for r in rows]
        left.bar(x, vals, bottom=bottom, width=.65, label=label, color=c, edgecolor="white", linewidth=.5)
        for i, value in enumerate(vals):
            if value >= 30:left.text(i, bottom[i] + value / 2, str(value), ha="center", va="center", size=8.5, color=INK)
        bottom += vals
    for i, r in enumerate(rows):
        left.text(i, r["n_per_variant"] + 11, str(r["n_per_variant"]), ha="center", size=8.5, weight="bold")
    left.set_xticks(x, [r["target_date"][:4] for r in rows])
    left.set_ylim(0, 505)
    left.set_yticks([0, 100, 200, 300, 400, 500])
    left.set_ylabel("Queries per variant", labelpad=6)
    left.set_title("(a)  Targets by corpus date", loc="left", weight="bold", pad=9)
    left.set_xlabel("Target corpus date (30 September)", labelpad=7)
    clean(left, "y")
    fig.legend(handles=[Patch(facecolor=c, label=label) for _, label, c in status], loc="upper left",
                bbox_to_anchor=(.045, .864), ncol=3, frameon=False, fontsize=8,
                handlelength=.85, handletextpad=.35, columnspacing=.8)
    changed = [r for r in rows if r["changed_cross_date"]]
    for i, r in enumerate(changed):
        y = 2 - i
        absent = r["quote_absent_fraction_of_changed"] * 100
        right.barh(y, absent, color=CORAL, height=.57, edgecolor="white", linewidth=.5)
        right.barh(y, 100 - absent, left=absent, color=LIGHT, height=.57, edgecolor="white", linewidth=.5)
        right.text(absent / 2, y, f"{r['source_quote_absent']} ({absent:.1f}%)", ha="center", va="center", size=8.7, color=INK)
        right.text(absent + (100 - absent) / 2, y, str(r["changed_quote_present"]), ha="center", va="center", size=8.7, color=INK)
    right.set_yticks([2, 1, 0], [f"{r['target_date'][:4]} · {r['changed_cross_date']}" for r in changed], fontsize=8.4)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_xlim(0, 100)
    right.set_ylim(-.65, 2.65)
    right.set_xlabel("Changed cross-date queries (%)", labelpad=7)
    right.set_title("(b)  Source-quote absence", loc="left", weight="bold", pad=9)
    fig.legend(handles=[Patch(facecolor=CORAL, label="Absent"), Patch(facecolor=LIGHT, label="Present")],
                 loc="upper left", bbox_to_anchor=(.63, .864), ncol=2, frameon=False, fontsize=8.3, handlelength=1)
    clean(right)
    fig.text(.045, .035, "Shared counts per variant. Unavailable: 0 / 2 / 2 / 2. Quote absent: 263 / 13 families.",
             ha="left", va="bottom", size=8, color=MUTED)
    return fig


def copy_existing(source, output, ruri, qwen, scale, sources):
    copied = []
    ablation = load(Path("runs/hidate/extended-benchmark/ablations/paired-analysis.json"))
    recall, effects = plot_rows(ablation)
    scale_rows, stack_rows = plot_data(scale, ruri, qwen, {"scale": sources["Scale"], "Ruri": sources["Ruri"], "Qwen3": sources["Qwen3"]})
    figures = {"ablation_recall": recall_figure(recall), "history_paired_effect": effect_figure(effects),
               "scale_quote_absent": scale_figure(scale_rows), "stack_quote_absent": stack_figure(stack_rows)}
    for stem in ("ablation_recall", "history_paired_effect", "scale_quote_absent", "stack_quote_absent"):
        clipped = export_figure(polish(figures[stem]), output / stem)
        if clipped:
            raise ValueError(f"Figure text is clipped: {stem}: {clipped}")
        shutil.copyfile(source / (stem + "_plotdata.csv"), output / (stem + "_plotdata.csv"))
        csv_path = output / (stem + "_plotdata.csv")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            points = list(csv.DictReader(stream))
        if "method_label" in points[0]:
            for point in points:
                point["method_label"] = point["method_label"].replace("Full method", "HiDate")
            write_csv(csv_path, points)
        copied.append(stem)
    return copied


def notes(output, sources, row_counts):
    content = """# HiDate evaluation figures

All figures use completed saved HiDate outputs. No model inference, new fit, new seed, new split, threshold selection or additional statistical resampling was run. The sources are the original Ruri analysis, the completed Qwen3 analysis, and the completed FiscalQA evaluation.

## New figures

1. **`method_profile` — Seven-method retrieval profiles.** The two left panels display all-natural versus source-quote-absent target-parent Recall@4 for all seven methods, for exact and generated variants. Open diamonds are all-natural values, filled circles are quote-absent values, and the line only connects the same method's two cohorts. Natural structural Recall uses 1,580 available targets, not all 1,586 inputs. Quote-absent has 263 available targets and 13 source families. The two right panels show quote-absent top1 revision accuracy, MRR@4, Recall@4, coverage and accepted support precision. Every profile value is displayed on a 0–100 scale, including MRR × 100. `NA` means no predictions accepted. This is a point-estimate figure, not a new significance analysis. HiDate is coral; matched as-of hybrid is mint; the remaining method marks are gray.

2. **`binding_mechanism` — From article addresses to dated revisions.** Stacked top1 outcomes are shown for unbound/bound/current/HiDate methods in the original Ruri and Qwen3 stacks. Each row partitions the same 263 quote-absent queries: target revision; correct address with another revision; other address. Counts are obtained exactly from saved numerator counts: `revision_hits`, `address_hits - revision_hits`, and `263 - address_hits`. This partition is licensed by `scripts/evaluate_final.py:59-61`, where top-revision correctness requires top-address correctness. No independence assumption, model output reconstruction or new scoring is involved. “Other revision” means the selected revision differs from the target revision; the chart does not turn that identity comparison into a separate legal-validity judgment. For example, Ruri unbound exact has 253 address hits but only 9 revision hits: 244 of 263 select the correct address with another revision. HiDate is retained even where its top1 revision hit rate is not the highest.

3. **`fiscal_transfer` — French tax-law transfer.** All eleven conditions are shown on the official 209-question R3 scoring cohort. Panel(a) is official strict answer-nugget accuracy. Panel(b) reports gold article Recall@4, gold version Recall@4 and top1 version identity among actual packed evidence. These retrieval identities are distinct from generated answer citation correctness. The Japanese TRAIN ranker and Japanese VALID policy remain frozen; no French fitting or tuning occurred. HiDate does not dominate the dated baselines: history-bound strict is 79.43% versus 78.95% for HiDate; HiDate gold-version Recall@4 is 96.65% versus 100% for the dated baselines. The unprompted literal-version-ID field is not plotted or interpreted as citation ability. Its preserved diagnostic status is documented in `../extended-benchmark/research-record-fiscal-citation.md`.

4. **`test_composition` — Temporal coverage of the held-out queries.** The original saved date-by-stratum facets close exactly: 293/401/443/449 queries at the four target dates, totaling 1,586 per variant. They comprise 1,187 same-date, 393 changed-cross-date and 6 unavailable targets. The absent-source-quote counts are 0/66/96/101, totaling 263. The right panel uses the changed-cross-date denominators 102/142/149; the absent fractions are 66/102, 96/142 and 101/149. The exact and generated variants have identical stratum counts, so the chart displays one set of counts and states the combined total of 3,172 inputs explicitly. This is a target-date composition figure, not a fabricated revision-gap histogram.

## Reused figures

`ablation_recall`, `history_paired_effect`, `scale_quote_absent` and `stack_quote_absent` use the plotted values and confidence intervals from `../extended-benchmark/figures/`. Vector exports use DejaVu text at a minimum of 9.5pt and a coral/mint palette. The matched-stack figure reports the as-of-hybrid method (Ruri 68.82%/66.54% on the quote-absent subset); the without-history ablation is a separate condition (68.44%/66.16%). Each stack uses its frozen validation alpha (Ruri 0.75; Qwen3 1.0).

## Method names

Visible labels use HiDate (`history_learned`, or the original full-frozen ablation reference), History current (`history_current`) and As-of oracle (`oracle_asof`). Saved method IDs, values and confidence intervals remain unchanged.

## Sources and denominators

"""
    content += "\n".join(f"- {name}: `{path.as_posix()}`" for name, path in sources.items())
    content += "\n\nThe full numerical tables remain in `../extended-benchmark/results-automatic.md` and its `tables/` directory. The new figure CSV row counts are " + ", ".join(f"`{name}`={n}" for name, n in row_counts.items()) + ". All CSVs retain the raw fractional values or integer counts and exact source paths.\n\nEach new figure is saved as native SVG/PDF and PNG. SVG text remains text; PDF text is embedded TrueType/CID, with minimum 9.5pt type and zero raster image objects. The new PNGs and Poppler renders are inspected for clipping, overlap and legibility. Figures contain no human expert ratings and no proposed framework graphic.\n"
    (output.parent / "figure-notes.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruri", type=Path, default=Path("runs/hidate/analysis/analysis.json"))
    parser.add_argument("--qwen3", type=Path, default=Path("runs/hidate/extended-benchmark/qwen3/analysis/analysis.json"))
    parser.add_argument("--fiscal", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal/evaluation.json"))
    parser.add_argument("--scale", type=Path, default=Path("runs/hidate/extended-benchmark/scale/evaluation.json"))
    parser.add_argument("--existing", type=Path, default=Path("paper/hidate/extended-benchmark/figures"))
    parser.add_argument("--output", type=Path, default=Path("paper/hidate/manuscript/figures"))
    args = parser.parse_args()
    sources = {"Ruri": args.ruri, "Qwen3": args.qwen3, "FiscalQA": args.fiscal, "Scale": args.scale}
    ruri, qwen, fiscal, scale = load(args.ruri), load(args.qwen3), load(args.fiscal), load(args.scale)
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    data = {"method_profile": primary_data(ruri, args.ruri),
            "binding_mechanism": binding_data(ruri, qwen, sources),
            "fiscal_transfer": fiscal_data(fiscal, args.fiscal),
            "test_composition": temporal_data(ruri, args.ruri)}
    makers = {"method_profile": primary_figure, "binding_mechanism": binding_figure,
              "fiscal_transfer": fiscal_figure, "test_composition": temporal_figure}
    qa = {"sources": {k: v.as_posix() for k, v in sources.items()}, "new_figures": {}, "reused_figures": {}}
    for name, rows in data.items():
        path = args.output / name
        write_csv(args.output / (name + "_plotdata.csv"), rows)
        clipped = export_figure(polish(makers[name](rows)), path)
        qa["new_figures"][name] = {**vector_qa(path), "csv_rows": len(rows), "text_outside_canvas": clipped}
    for name in copy_existing(args.existing, args.output, ruri, qwen, scale, sources):
        qa["reused_figures"][name] = {**vector_qa(args.output / name), "source_directory": args.existing.as_posix()}
    notes(args.output, sources, {k: len(v) for k, v in data.items()})
    qa["complete"] = all(q["native_vector"] and q["editable_text"] and q["pdf_min_font_points"] >= 9.49
                         and not q.get("text_outside_canvas", []) for q in [*qa["new_figures"].values(), *qa["reused_figures"].values()])
    (args.output / "figure-qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if not qa["complete"]:
        raise ValueError("Figure QA requires native vectors, editable text, minimum9.5pt and no clipped text")


if __name__ == "__main__":
    main()
