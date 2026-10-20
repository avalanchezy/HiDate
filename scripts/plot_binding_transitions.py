"""Show all paired FiscalQA strict-answer transitions as native vector flows."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from plot_ablations import configure, export_figure, vector_qa, write_csv


CORAL, MINT, GRAY, GOLD = '#E48173', '#74B5A2', '#A2ACB5', '#D8AE65'
INK, MUTED = '#27323A', '#65717A'
FLOWS = (
    ('both_strict_correct', 'Correct', 'Correct', 'Remain correct', MINT),
    ('bound_only_strict_correct', 'Incorrect', 'Correct', 'Become correct', CORAL),
    ('source_only_strict_correct', 'Correct', 'Incorrect', 'Become incorrect', GOLD),
    ('both_strict_incorrect', 'Incorrect', 'Incorrect', 'Remain incorrect', GRAY),
)


def source_rows(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data['complete'] or not data['cid_order_invariant']:
        raise ValueError('Use the complete fixed-address matched intervention')
    counts = data['strict_transitions']
    n = data['paired_questions']
    if n != 209 or sum(counts[key] for key, *_ in FLOWS) != n:
        raise ValueError('The four strict transitions must partition all209 paired questions')
    source_correct = counts['both_strict_correct'] + counts['source_only_strict_correct']
    target_correct = counts['both_strict_correct'] + counts['bound_only_strict_correct']
    for arm, correct in (('binding_source', source_correct), ('binding_bound', target_correct)):
        if not math.isclose(data['conditions'][arm]['strict'], correct/n, abs_tol=1e-12):
            raise ValueError('Transition marginals differ from saved strict correctness')
    rows = [{'transition_key': key, 'source_status': source, 'target_status': target,
        'transition_label': label, 'count': counts[key], 'fraction_of_paired_questions': counts[key]/n,
        'source_node_count': source_correct if source == 'Correct' else n-source_correct,
        'target_node_count': target_correct if target == 'Correct' else n-target_correct,
        'paired_questions': n, 'color': color, 'source': path.as_posix()}
        for key, source, target, label, color in FLOWS]
    return data, rows


def ribbon(x0, x1, left_low, left_high, right_low, right_high, color, *, zorder=2):
    control = .47*(x1-x0)
    vertices = [(x0, left_low), (x0+control, left_low), (x1-control, right_low), (x1, right_low),
                (x1, right_high), (x1-control, right_high), (x0+control, left_high),
                (x0, left_high), (x0, left_low)]
    codes = [MplPath.MOVETO, *([MplPath.CURVE4]*3), MplPath.LINETO,
             *([MplPath.CURVE4]*3), MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), facecolor=color,
                     edgecolor='white' if zorder > 2 else 'none',
                     linewidth=.55 if zorder > 2 else 0, zorder=zorder)


def transition_figure(data, rows):
    counts = {row['transition_key']: row['count'] for row in rows}
    both, gain, loss, neither = [counts[key] for key, *_ in FLOWS]
    source_correct, target_correct = both+loss, both+gain
    n = data['paired_questions']
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    # Every vertical flow thickness and node height uses this one count scale.
    top, bottom, gap = .80, .12, .065
    unit = (top-bottom-gap)/n
    x0, x1, node_width = .215, .785, .015
    source = {'Correct': (top-source_correct*unit, top),
              'Incorrect': (bottom, top-source_correct*unit-gap)}
    target = {'Correct': (top-target_correct*unit, top),
              'Incorrect': (bottom, top-target_correct*unit-gap)}
    bands = {
        'both_strict_correct': (top-both*unit, top, top-both*unit, top),
        'bound_only_strict_correct': (source['Incorrect'][1]-gain*unit, source['Incorrect'][1],
                                    target['Correct'][0], top-both*unit),
        'source_only_strict_correct': (source['Correct'][0], top-both*unit,
                                     target['Incorrect'][1]-loss*unit, target['Incorrect'][1]),
        'both_strict_incorrect': (bottom, bottom+neither*unit, bottom, bottom+neither*unit),
    }
    for row in rows:
        key = row['transition_key']
        low, high, right_low, right_high = bands[key]
        assert math.isclose(high-low, row['count']*unit, abs_tol=1e-12)
        assert math.isclose(right_high-right_low, high-low, abs_tol=1e-12)
        row['band_height_figure_fraction'] = high-low
        row['points_per_question'] = unit*fig.get_figheight()*72
        ax.add_patch(ribbon(x0, x1, *bands[key], row['color'],
                            zorder=4 if key == 'source_only_strict_correct' else 2))

    for nodes, x, label_x, align in ((source, x0-node_width, .184, 'right'),
                                    (target, x1, .816, 'left')):
        for label, (low, high) in nodes.items():
            count = round((high-low)/unit)
            ax.add_patch(Rectangle((x, low), node_width, high-low,
                facecolor='#4D8B79' if label == 'Correct' else '#788590', linewidth=0, zorder=5))
            ax.text(label_x, (low+high)/2, f'{label}\n{count}', ha=align, va='center',
                    fontsize=10, linespacing=1.4, color=INK)
    ax.text(.208, .934, 'Source-version text', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(.792, .934, 'Requested-date text', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(.50, top-both*unit/2, f'{both} remain correct', ha='center', va='center',
            fontsize=10.2, color=INK)
    ax.text(.615, .550, str(gain), ha='center', va='center', fontsize=19, weight='bold', color=INK)
    ax.text(.615, .475, 'become correct', ha='center', va='center', fontsize=10.2, color=INK)
    ax.text(.50, bottom+neither*unit/2, f'{neither} remain incorrect', ha='center', va='center',
            fontsize=10.2, color=INK)
    # The two-question reverse transition keeps its true narrow width; the
    # callout identifies it rather than enlarging it or changing the scale.
    t = .09
    bezier_x = ((1-t)**3*x0 + 3*(1-t)**2*t*(x0+.47*(x1-x0))
                + 3*(1-t)*t*t*(x1-.47*(x1-x0)) + t**3*x1)
    yleft = (bands['source_only_strict_correct'][0]+bands['source_only_strict_correct'][1])/2
    yright = (bands['source_only_strict_correct'][2]+bands['source_only_strict_correct'][3])/2
    bezier_y = yleft + (3*t*t-2*t**3)*(yright-yleft)
    ax.annotate(f'{loss} become incorrect', xy=(bezier_x, bezier_y), xytext=(.49, .851),
        ha='center', va='center', fontsize=9.5, color='#926721',
        arrowprops={'arrowstyle': '->', 'color': MUTED, 'linewidth': .7, 'linestyle': (0, (2.5, 2.5)),
                    'connectionstyle': 'arc3,rad=.12', 'shrinkA': 4, 'shrinkB': 1}, zorder=7)
    ax.text(.5, .035, f'{n} paired questions', ha='center', va='center', fontsize=9.5, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
        default=Path('runs/hidate/mechanism-analysis/binding-fiscal/evaluation.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('paper/hidate/manuscript/figures'))
    args = parser.parse_args()
    data, rows = source_rows(args.input)
    configure()
    plt.rcParams.update({'font.size': 9.5, 'axes.labelsize': 9.5})
    fig = transition_figure(data, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'binding_answer_transitions'
    clipped = export_figure(fig, path)
    write_csv(args.output_dir/'binding_answer_transitions_plotdata.csv', rows)
    qa = {**vector_qa(path), 'text_outside_canvas': clipped,
        'source': args.input.as_posix(), 'paired_questions': data['paired_questions'],
        'strict_transitions': data['strict_transitions'],
        'size_inches': [7.2, 3.3], 'flow_width': 'one common scale; count proportional at both nodes'}
    print(json.dumps(qa, indent=2))
    if clipped or not qa['native_vector'] or not qa['editable_text'] or qa['pdf_min_font_points'] < 9.5:
        raise ValueError('Require native vectors, editable text, fonts>=9.5pt and no clipped text')


if __name__ == '__main__':
    main()
