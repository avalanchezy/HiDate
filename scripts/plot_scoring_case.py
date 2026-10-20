"""Editable illustration of a saved same-candidate historical/current CE trace."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


def main():
    source = Path('runs/hidate/revision-aware/transfer/ruri-scale-final/case-records.json')
    case = json.loads(source.read_text(encoding='utf-8'))['cases'][-1]
    assert case['case_id'] == 'SCALE-MAIN-MECHANISM-3'
    out = Path('paper/hidate/development/figures')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    jp = FontProperties(family=['Noto Sans CJK JP', 'Yu Gothic', 'Meiryo', 'sans-serif'], size=9)
    fig = plt.figure(figsize=(8.2, 4.2), facecolor='white')
    left = fig.add_axes([.015, .055, .57, .89])
    left.set(xlim=(-.025, 1.025), ylim=(0, 1)); left.axis('off')
    dark, historical, current = '#27364b', '#b44e49', '#4b83a6'
    left.text(0, 1.02, '(a) One legal address, two textual realizations',
              fontsize=10.5, weight='bold', color=dark)
    left.add_patch(FancyBboxPatch((0, .76), .98, .225, boxstyle='round,pad=0.012',
                    facecolor='#f1f4f8', edgecolor='#d6dce5', linewidth=.8))
    left.text(.025, .947, 'QUERY  ·  requested date 2021-09-30', fontsize=8,
              weight='bold', color=dark)
    left.text(.025, .90, 'Terminology for electronic administrative procedures\nbased on Act No. 151 of 2002.',
              color=dark, va='top', linespacing=1.5)
    left.text(.025, .788, 'English translation of the saved Japanese query',
              fontsize=7.7, color='#68768a')
    cards = [(.41, 'HISTORICAL VIEW  ·  2018-09-30', '#fbefed', historical,
              'Full statutory reference', '…（平成十四年法律第百五十一号。）…',
              'Terms follow those in the named Act (2002, No. 151).'),
             (.08, 'TARGET-DATE VIEW  ·  2021-09-30', '#edf4f8', current,
              'Shortened statutory reference', '法で使用する用語の例による。',
              'Terms follow those used in “the Act”.')]
    for y, title, fill, color, heading, quote, gloss in cards:
        left.add_patch(FancyBboxPatch((0, y), .98, .285, boxstyle='round,pad=.012',
                        facecolor=fill, edgecolor=color, linewidth=1))
        left.text(.025, y + .244, title, fontsize=8.2, color=color, weight='bold')
        left.text(.025, y + .187, heading, color=dark, weight='bold', fontsize=9)
        left.text(.025, y + .12, quote, fontproperties=jp, color=dark)
        left.text(.025, y + .05, gloss, fontsize=8.2, color=dark)
    left.text(.49, .035, 'Article 2 · Definitions  |  same scoped address',
              ha='center', fontsize=8.2, weight='bold', color=dark)

    ax = fig.add_axes([.69, .16, .27, .69])
    ax.set(xlim=(-.13, 1.13), ylim=(10.6, .35))
    ax.axhspan(.5, 4.5, facecolor='#e9f2e9', zorder=0)
    ax.axhline(4.5, color='#8eac90', linewidth=.8, linestyle=(0, (3, 3)))
    ax.set_xticks([0, 1], ['Historical CE', 'Current CE'])
    ax.xaxis.tick_top(); ax.tick_params(axis='both', length=0, pad=7)
    ax.set_yticks(range(1, 11)); ax.set_ylabel('Rank among the same ten candidates', labelpad=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    for c in case['same_common_ten']:
        if c['is_target_parent']: continue
        ax.plot([0, 1], [c['historical_CE_rank'], c['current_CE_rank']],
                color='#c8ced7', linewidth=.85, alpha=.8, zorder=1)
        ax.scatter([0, 1], [c['historical_CE_rank'], c['current_CE_rank']],
                   s=16, color='#c2c9d2', zorder=2)
    t = case['target_scores_and_ranks']
    ax.plot([0, 1], [t['historical_CE_rank'], t['current_CE_rank']],
            color=historical, linewidth=2.5, zorder=4)
    ax.scatter([0, 1], [t['historical_CE_rank'], t['current_CE_rank']],
               s=58, color=historical, edgecolor='white', linewidth=.8, zorder=5)
    ax.text(.08, 1.05, f"Target · {t['historical_ce']['reranker_score']:.3f}",
            color=historical, fontsize=8, weight='bold')
    ax.text(.95, 8.65, f"Target · {t['current_ce']['reranker_score']:.3f}",
            color=historical, fontsize=8, weight='bold', ha='right')
    ax.text(.05, 4.25, 'Top four', color='#4b6f4f', fontsize=8,
            bbox={'facecolor':'#e9f2e9', 'edgecolor':'none', 'pad':1})
    fig.text(.64, .963, '(b) Change the scoring view', fontsize=10.5,
             weight='bold', color=dark)
    fig.text(.805, .055, 'Historical scoring returns the\nbound 2021 article at rank 1.',
             ha='center', fontsize=9, color=dark, linespacing=1.4)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('scoring_case.' + ext), dpi=240, bbox_inches='tight', pad_inches=.09)
    with (out / 'scoring_case.csv').open('w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['law_id', 'scope', 'article', 'target',
            'historical_rank', 'current_rank', 'historical_score', 'current_score'])
        w.writeheader()
        for c in case['same_common_ten']:
            w.writerow(dict(zip(w.fieldnames, [*c['address'], c['is_target_parent'],
                c['historical_CE_rank'], c['current_CE_rank'], c['historical_ce']['reranker_score'],
                c['current_ce']['reranker_score']])))
    (out / 'scoring_case_source.json').write_text(json.dumps({'source': str(source),
        'case_id': case['case_id'], 'query_id': case['query']['query_id'],
        'text_note': 'English translations and explicit shortened source quotations; ranks and scores are saved outputs.'},
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
