"""Editable temporal case cards from the saved Japanese and French traces."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/hidate/manuscript/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
INK = '#263238'
MUTED = '#64747b'
COLORS = [('#fbf2de', '#cba661'), ('#e8f4f0', '#72b6a4'), ('#fceae7', '#df7c6d')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.3), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.025, .96, 'From historical wording to dated legal evidence',
            fontsize=14, fontweight='bold', color=INK, va='top')
    ax.text(.025, .905, 'Historical source', color=MUTED, fontsize=10)
    ax.text(.36, .905, 'Requested-date binding', color=MUTED, fontsize=10)
    ax.text(.695, .905, 'Saved reader result', color=MUTED, fontsize=10)
    rows = [
        {'y': .51, 'title': '(a) Investment-management capital · Article 15-7',
         'cards': [
             ('2018 revision', 'JPY 50 million\nfor the stated investment-\nmanagement category'),
             ('2026 corpus', 'JPY 50 million;\nJPY 10 million under\nthe added custody condition'),
             ('HiDate', 'a: Article 15-7(1)(iv)\nCorrect released choice\nHybrid: b')],
         'foot': 'Five historical revisions identify one address; binding supplies its updated provision.'},
        {'y': .105, 'title': '(b) Agricultural-loss offset · CGI Article 156',
         'cards': [
             ('1990–1991 version', 'Net income threshold:\nFRF 100,000\nUnbound reader: FRF 100,000'),
             ('1 January 2009', 'Net income threshold:\nEUR 101,300\nVersion valid at query date'),
             ('HiDate', 'EUR 101,300\nOfficial strict: correct\nHybrid: EUR 15,300')],
         'foot': 'Stable article identity connects the archive hit to the version valid on 1 January 2009.'}
    ]
    for row in rows:
        y = row['y']
        ax.text(.025, y + .335, row['title'], va='top', fontsize=11, fontweight='bold', color=INK)
        for i, ((head, body), (fill, border)) in enumerate(zip(row['cards'], COLORS)):
            x = .025 + i * .335
            ax.add_patch(FancyBboxPatch((x, y + .055), .285, .23,
                         boxstyle='round,pad=0.012,rounding_size=0.012',
                         facecolor=fill, edgecolor=border, linewidth=1.1))
            ax.text(x + .012, y + .257, head, va='top', color=INK, fontsize=10, fontweight='bold')
            ax.text(x + .012, y + .195, body, va='top', color=INK, fontsize=9.5, linespacing=1.4)
            if i < 2:
                ax.add_patch(FancyArrowPatch((x + .302, y + .17), (x + .327, y + .17),
                             arrowstyle='-|>', mutation_scale=12, linewidth=1.4, color=MUTED))
        ax.text(.025, y + .015, row['foot'], va='top', color=MUTED, fontsize=9.5)
    ax.text(.025, .035, 'English summaries preserve source amounts and conditions; reader outputs are recorded results.',
            fontsize=9.5, color=MUTED, va='bottom')
    for suffix in ['svg', 'pdf', 'png']:
        fig.savefig(OUT / f'temporal_case_studies.{suffix}', dpi=300, facecolor='white')
    (OUT / 'temporal_case_studies_plotdata.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()
