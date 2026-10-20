"""Produce manuscript tables directly from completed HiDate saved aggregates."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/hidate/manuscript'
METHODS = ['asof_bm25', 'asof_dense', 'asof_hybrid', 'history_unbound',
           'history_bound', 'history_current', 'history_learned']
NAMES = dict(zip(METHODS, ['As-of BM25', 'As-of dense', 'As-of hybrid',
                         'History unbound', 'History bound', 'History current', 'HiDate']))


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def fmt(value, percent=True, signed=False):
    if value is None:
        return '--'
    return f'{value * (100 if percent else 1):+.{2 if percent else 3}f}' if signed else f'{value * (100 if percent else 1):.{2 if percent else 3}f}'


def panel(data, scope, variant, absent=None):
    return next(x for x in data[scope] if x.get('query_variant') == variant
                and x.get('stratum') == 'all' and x.get('target_asof') == 'all'
                and x.get('source_quote_absent_from_target') is absent)['estimates']


def table(name, caption, label, header, rows, spec, star=True):
    env = 'table*' if star else 'table'
    body = '\n'.join(' & '.join(str(v) for v in row) + r' \\' for row in rows)
    text = (f'\\begin{{{env}}}[!tp]\n\\centering\\small\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
            '\\setlength{\\tabcolsep}{4pt}\n'
            f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n'
            + ' & '.join(header) + r' \\' + '\n\\midrule\n' + body
            + f'\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n')
    (OUT / name).write_text(text, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ruri = load('runs/hidate/analysis/analysis.json')
    qwen = load('runs/hidate/extended-benchmark/qwen3/analysis/analysis.json')
    ab = load('runs/hidate/extended-benchmark/ablations/paired-analysis.json')
    variants = ['exact_quote', 'generated_paraphrase']
    for stack, data in [('ruri', ruri), ('qwen3', qwen)]:
        rows = []
        for method in METHODS:
            values = []
            for absent in [None, True]:
                for variant in variants:
                    p = panel(data, 'natural', variant, absent)[method]
                    values += [fmt(p['recall_at4']['mean']), fmt(p['mrr_at4']['mean'], False),
                               fmt(p['target_parent_accuracy']['mean'])]
            rows.append([NAMES[method]] + values)
        table(f'table-natural-{stack}.tex',
              f'{"Ruri" if stack == "ruri" else "Qwen3"} natural retrieval. Each block gives Recall@4 (R4, \\%), MRR@4 (MRR), and Parent@1 (P1, \\%). All-available panels contain 1,580 queries per variant; quote-absent panels contain 263 queries from 13 families. Both stacks use the same test identities.',
              f'tab:natural-{stack}',
              ['Method', '\\multicolumn{3}{c}{All / literal}', '\\multicolumn{3}{c}{All / generated}',
               '\\multicolumn{3}{c}{Absent / literal}', '\\multicolumn{3}{c}{Absent / generated}'],
              [['', 'R4', 'MRR', 'P1', 'R4', 'MRR', 'P1', 'R4', 'MRR', 'P1', 'R4', 'MRR', 'P1']] + rows,
              'l' + 'r' * 12)

    # All methods and both stacks at their fixed validation-selected operating points.
    rows = []
    for stack, data in [('Ruri', ruri), ('Qwen3', qwen)]:
        for method in METHODS:
            vals = []
            for variant in variants:
                natural = panel(data, 'natural', variant)[method]
                control = panel(data, 'controlled', variant)[method]
                vals += [fmt(natural['coverage']['mean']), fmt(natural['support_precision']['mean']),
                         fmt(control['false_support_rate']['mean'])]
            rows.append([stack, NAMES[method]] + vals)
    table('table-support.tex', 'Frozen support-selection policies. NC is natural coverage, SP is accepted top-parent precision, and FS is controlled false support (all \\%). Natural coverage uses 1,586 inputs; controlled false support uses 1,580 controlled query inputs per variant. A dash denotes an empty accepted set.', 'tab:support',
          ['Stack', 'Method', '\\multicolumn{3}{c}{Literal}', '\\multicolumn{3}{c}{Generated}'],
          [['', '', 'NC', 'SP', 'FS', 'NC', 'SP', 'FS']] + rows, 'llrrrrrr')

    rows = []
    ab_names = {'full_frozen': 'HiDate (full)', 'without_history': 'Without historical features',
                'without_current_text': 'Without current-text features', 'without_revision_gap': 'Without temporal-distance feature',
                'without_availability': 'Without availability feature'}
    for method, name in ab_names.items():
        vals = []
        for variant in variants:
            p = ab['panels'][variant + '|quote_absent']['methods'][method]
            vals += [fmt(p['target_recall_at_four']), fmt(p['target_mrr'], False)]
        alpha = .75 if method == 'full_frozen' else ab['coverage_readback'][method]['alpha']
        rows.append([name, f'{alpha:g}'] + vals)
    table('table-ablation.tex', 'Feature-group ablation with the original Ruri stack. Each reduced model is fitted once on the original training data and selects its mixture on validation. The 263 quote-absent queries and full-model policy remain fixed. R4 is Recall@4 (\\%).', 'tab:ablation',
          ['Model', '$\\alpha$', '\\multicolumn{2}{c}{Literal}', '\\multicolumn{2}{c}{Generated}'],
          [['', '', 'R4', 'MRR', 'R4', 'MRR']] + rows, 'lrrrrr')

    scale = list(csv.DictReader((ROOT / 'paper/hidate/extended-benchmark/tables/scale-all-metrics.csv').open(encoding='utf-8')))
    def sv(laws, method, variant, subset, metric):
        return float(next(x['value'] for x in scale if x['laws'] == str(laws) and x['method'] == method
                          and x['query_variant'] == variant and x['subset'] == subset and x['metric'] == metric))
    rows = []
    for laws in [120, 500, 1000]:
        for method in METHODS:
            vals = [fmt(sv(laws, method, variant, subset, 'target_recall_at_four'))
                    for subset in ['all', 'quote_absent'] for variant in variants]
            rows.append([laws, NAMES[method]] + vals)
    table('table-scale.tex', 'Nested-corpus Recall@4 (\\%) for every method. The same 3,172 query inputs, ranker, mixture and cutoffs are used at each scale. All-query means use 1,586 inputs per variant, including six unavailable targets; quote-absent means use the same 263 available queries as the main analysis.', 'tab:scale',
          ['Laws', 'Method', '\\multicolumn{2}{c}{All queries}', '\\multicolumn{2}{c}{Quote absent}'],
          [['', '', 'Literal', 'Generated', 'Literal', 'Generated']] + rows, 'rlrrrr')

    fiscal = list(csv.DictReader((ROOT / 'paper/hidate/extended-benchmark/tables/fiscal-all-metrics.csv').open(encoding='utf-8')))
    def fv(method, metric, scope='scored209'):
        return float(next(x['value'] for x in fiscal if x['method'] == method and x['metric'] == metric and x['scope'] == scope))
    fiscal_methods = ['closed_book', 'static_bm25', 'static_dense'] + METHODS + ['oracle_asof']
    fn = dict(NAMES, closed_book='Closed book', static_bm25='Static BM25', static_dense='Static dense', oracle_asof='As-of oracle')
    print('Fiscal metrics:', sorted({x['metric'] for x in fiscal}))
    rows = []
    for m in fiscal_methods:
        rows.append([fn[m]] + [fmt(fv(m, key)) for key in ['strict', 'coverage', 'article_citation_nugget', 'gold_version_recall_at_four', 'top_version_correct']] + [fmt(fv(m, 'strict', 'published221'))])
    table('table-fiscal.tex', 'French FiscalQA R3 transfer with the frozen Japanese Qwen3 model and policy. Primary results use 209 questions. Strict, nugget coverage (Nuggets), article-nugget correctness (Article), version Recall@4 (VR4) and top-version accuracy (V1) are percentages. Published gives strict correctness across all 221 released questions.', 'tab:fiscal',
          ['Condition', 'Strict', 'Nuggets', 'Article', 'VR4', 'V1', 'Published'], rows, 'lrrrrrr')

    rows = []
    dates = ['2024-09-30', '2025-11-04', '2026-09-30']
    for m in ['closed_book', 'released_oracle'] + METHODS:
        vals = []
        for data in [ruri, qwen]:
            for date in dates:
                est = data['lawqa_references']['estimates'][m] if m in ['closed_book', 'released_oracle'] else next(x for x in data['lawqa'] if x['target_asof'] == date)['estimates'][m]
                vals.append(fmt(est['accuracy']['mean']))
        rows.append([dict(NAMES, closed_book='Closed book', released_oracle='Released-context oracle')[m]] + vals)
    table('table-lawqa.tex', 'LawQA original-choice accuracy (\\%) on all 140 released questions. Columns are corpus dates rather than new question sets. The fixed reader, original labels and options are shared; closed-book and oracle answers are evaluated once per question.', 'tab:lawqa',
          ['Condition', '\\multicolumn{3}{c}{Ruri evidence}', '\\multicolumn{3}{c}{Qwen3 evidence}'],
          [['', '2024', 'Release', '2026', '2024', 'Release', '2026']] + rows, 'lrrrrrr')

    # Paired effects are read from existing intervals, not resampled here.
    effects = load('runs/hidate/analysis/paired-intervals.json')['contrasts']
    rows = []
    for comp in ['asof_hybrid', 'history_unbound', 'history_bound', 'history_current']:
        vals = []
        for variant in variants:
            x = next(x for x in effects if x['evaluation_scope'] == 'natural' and x['query_variant'] == variant and x['stratum'] == 'all' and x['target_asof'] == 'all' and x['source_quote_absent_from_target'] is True and x['left'] == comp and x['right'] == 'history_learned' and x['metric'] == 'recall_at4')
            vals.append(f"{fmt(x['delta_right_minus_left'], signed=True)} [{fmt(x['ci95'][0], signed=True)}, {fmt(x['ci95'][1], signed=True)}]")
        rows.append([NAMES[comp]] + vals)
    table('table-paired.tex', 'Paired HiDate-minus-comparator Recall@4 effects in percentage points on the 263 quote-absent queries under Ruri. Brackets give saved 95\\% source-family bootstrap intervals (13 families, 2,000 draws).', 'tab:paired',
          ['Comparator', 'Literal', 'Generated'], rows, 'lrr')

    # Bibliography entries are maintained separately from result tables.
    print('Wrote manuscript tables to', OUT)


if __name__ == '__main__':
    main()
