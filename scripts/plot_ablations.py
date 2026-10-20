"""Plot the completed HiDate paired ablations as editable native scientific figures."""

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


METHODS = (
    ("full_frozen", "Full method"),
    ("without_history", "Without history"),
    ("without_current_text", "Without current text"),
    ("without_revision_gap", "Without revision gap"),
    ("without_availability", "Without availability"),
)
VARIANTS = (("exact_quote", "Exact quote"), ("generated_paraphrase", "Generated paraphrase"))
COLORS = {"full_frozen": "#E48173", "without_history": "#74B5A2"}
GRAY, INK, MUTED, GRID = "#A2ACB5", "#27323A", "#65717A", "#E7EBEE"


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": "#B9C2C9", "axes.linewidth": .6,
        "axes.unicode_minus": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "savefig.dpi": 300,
    })


def plot_rows(source):
    recall, effects = [], []
    if not source["complete"]:
        raise ValueError("Plot only the completed paired analysis")
    for variant, title in VARIANTS:
        panel = source["panels"][variant + "|quote_absent"]
        if panel["n"] != 263 or panel["families"] != 13:
            raise ValueError("These figures describe the frozen263-query/13-family quote-absent panel")
        for method, label in METHODS:
            value = panel["methods"][method]["target_recall_at_four"]
            recall.append({"query_variant": variant, "variant_label": title, "method": method,
                "method_label": label, "n_queries": panel["n"], "n_families": panel["families"],
                "recalled_queries": round(value * panel["n"]), "recall_at_four": value,
                "recall_percent": value * 100})
        full = panel["methods"]["full_frozen"]["target_recall_at_four"]
        ablated = panel["methods"]["without_history"]["target_recall_at_four"]
        effect = source["full_minus_ablation"][variant + "|quote_absent|without_history"]["target_recall_at_four"]
        if not np.isclose(full - ablated, effect["delta_b_minus_a"], rtol=0, atol=1e-12):
            raise ValueError("The paired effect direction must be full minus without_history")
        effects.append({"query_variant": variant, "variant_label": title,
            "contrast": "full_frozen_minus_without_history", "metric": "target_recall_at_four",
            "full_recall": full, "without_history_recall": ablated,
            "delta_fraction": effect["delta_b_minus_a"], "delta_percentage_points": 100 * effect["delta_b_minus_a"],
            "ci_low_percentage_points": 100 * effect["ci95"][0],
            "ci_high_percentage_points": 100 * effect["ci95"][1],
            "n_pairs": effect["n_pairs"], "n_families": effect["n_groups"],
            "n_resamples": effect["n_resamples"], "interval_method": effect["method"], "seed": effect["seed"]})
    return recall, effects


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.65)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3, width=.6, pad=4)


def recall_figure(rows):
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharey=True)
    figure.subplots_adjust(left=.248, right=.982, bottom=.23, top=.745, wspace=.18)
    figure.text(.035, .966, "Feature-group ablations", ha="left", va="top", size=11.5, weight="bold")
    figure.text(.035, .886, "Source-quote absent  |  263 paired queries, 13 families per variant",
                ha="left", va="top", size=8.5, color=MUTED)
    for panel_index, ((variant, title), ax) in enumerate(zip(VARIANTS, axes)):
        selected = {r["method"]: r for r in rows if r["query_variant"] == variant}
        for index, (method, _) in enumerate(METHODS):
            y = 4 - index
            value = selected[method]["recall_percent"]
            color = COLORS.get(method, GRAY)
            ax.hlines(y, 60, value, color=color, linewidth=1.5 if method in COLORS else 1.0, alpha=.42)
            ax.scatter(value, y, s=45 if method == "full_frozen" else 39 if method == "without_history" else 27,
                       color=color, edgecolors="white", linewidths=.65, zorder=4)
            ax.text(value + 1.3, y, f"{value:.2f}", ha="left", va="center", size=8.5,
                    color=INK if method in COLORS else MUTED, weight="bold" if method == "full_frozen" else "normal")
        ax.set_xlim(60, 110)
        ax.set_ylim(-.6, 4.6)
        ax.set_xticks([60, 70, 80, 90, 100])
        ax.set_yticks(range(4, -1, -1), [label for _, label in METHODS])
        ax.set_title(f"({'ab'[panel_index]})  {title}", loc="left", pad=12, weight="bold")
        style_axis(ax)
    for tick, (method, _) in zip(axes[0].get_yticklabels(), METHODS):
        if method == "full_frozen":
            tick.set_color(COLORS[method])
            tick.set_fontweight("bold")
        elif method == "without_history":
            tick.set_color("#4F8D7D")
    figure.text(.615, .070, "Target-parent Recall@4 (%)", ha="center", va="center", size=9)
    return figure


def effect_figure(rows):
    figure, ax = plt.subplots(figsize=(6.25, 2.7))
    figure.subplots_adjust(left=.282, right=.971, bottom=.267, top=.705)
    figure.text(.035, .967, "Historical matching contribution", ha="left", va="top", size=11.5, weight="bold")
    figure.text(.035, .873, "Full method minus without history", ha="left", va="top", size=8.5, color=MUTED)
    color = COLORS["full_frozen"]
    for index, row in enumerate(rows):
        y = 1 - index
        mean = row["delta_percentage_points"]
        lower, upper = row["ci_low_percentage_points"], row["ci_high_percentage_points"]
        ax.errorbar(mean, y, xerr=[[mean - lower], [upper - mean]], color=color,
                    fmt="o", markersize=6, markeredgecolor="white", markeredgewidth=.7,
                    elinewidth=2.0, capsize=3.3, capthick=1.1, zorder=4)
        ax.text(mean, y + .23, f"+{mean:.2f}  [{lower:.2f}, {upper:.2f}]",
                ha="center", va="bottom", size=9, color=INK)
    ax.axvline(0, color="#A1ABB3", linestyle=(0, (3, 3)), linewidth=.85, zorder=2)
    ax.set_xlim(-4, 55)
    ax.set_ylim(-.5, 1.7)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([1, 0], [row["variant_label"] for row in rows])
    style_axis(ax)
    ax.set_xlabel("Recall@4 difference (percentage points)", labelpad=8)
    figure.text(.035, .035, "95% family-cluster paired intervals  |  263 paired queries, 13 families",
                ha="left", va="bottom", size=8, color=MUTED)
    return figure


def export_figure(figure, path):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    clipped = []
    for item in figure.findobj(Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            clipped.append(item.get_text())
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(path.with_suffix("." + suffix), metadata={"Creator": "Matplotlib; HiDate completed paired analysis"}
                       if suffix in {"svg", "pdf"} else None)
    plt.close(figure)
    return clipped


def vector_qa(path):
    import pymupdf
    root = ET.parse(path.with_suffix(".svg")).getroot()
    text_elements = list(root.iter("{http://www.w3.org/2000/svg}text"))
    svg_sizes = [float(value) for node in text_elements for value in
                 re.findall(r"(?:font-size:\s*|font:\s*)([\d.]+)(?:px|pt)", node.attrib.get("style", ""))]
    doc = pymupdf.open(path.with_suffix(".pdf"))
    spans = [span for page in doc for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"] if span["text"].strip()]
    fonts = sorted({font[3] for page in doc for font in page.get_fonts()})
    result = {"svg_image_elements": len(list(root.iter("{http://www.w3.org/2000/svg}image"))),
              "svg_text_elements": len(text_elements), "svg_min_font_points": min(svg_sizes) if svg_sizes else None,
              "pdf_image_objects": sum(len(page.get_images()) for page in doc), "pdf_pages": len(doc),
              "pdf_text_spans": len(spans), "pdf_fonts": fonts,
              "pdf_min_font_points": min(span["size"] for span in spans),
              "pdf_font_types": sorted({font[2] for page in doc for font in page.get_fonts()}),
              "editable_text": bool(text_elements and spans),
              "native_vector": not list(root.iter("{http://www.w3.org/2000/svg}image"))
                               and not any(page.get_images() for page in doc)}
    doc.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/hidate/extended-benchmark/ablations/paired-analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/hidate/extended-benchmark/figures"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    recall, effects = plot_rows(source)
    write_csv(args.output / "ablation_recall_plotdata.csv", recall)
    write_csv(args.output / "history_paired_effect_plotdata.csv", effects)
    stems = (("ablation_recall", recall_figure(recall)), ("history_paired_effect", effect_figure(effects)))
    quality = {"source": str(args.source), "evaluation_job": source["evaluation_job"],
               "cohort": {"stratum": "source_quote_absent", "queries_per_variant": 263, "families_per_variant": 13},
               "recall_plot_rows": len(recall), "paired_effect_rows": len(effects), "figures": {}}
    for name, figure in stems:
        path = args.output / name
        clipped = export_figure(figure, path)
        quality["figures"][name] = {**vector_qa(path), "text_outside_canvas": clipped}
    quality["complete"] = all(row["native_vector"] and row["editable_text"] and row["pdf_min_font_points"] >= 7.99
                              and not row["text_outside_canvas"] for row in quality["figures"].values())
    (args.output / "figure-qa.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    if not quality["complete"]:
        raise ValueError("Figure QA requires editable native vectors, fonts >=8pt and no clipped canvas text")


if __name__ == "__main__":
    main()
