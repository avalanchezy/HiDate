"""Render manuscript tables from completed aggregate receipts; no new statistics."""
import csv
import json
from pathlib import Path

ROOT = Path('runs/hidate/revision-aware')
PAPER = Path('paper/hidate/relevance-analysis')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(name, caption, label, headers, records, formats, *, ours=None, best=True):
    maxima = [max(float(row[j + 1]) for row in records) for j in range(len(formats))]
    lines = [r'\begin{table*}[t]', r'\centering\small',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\begin{tabular}{l' + 'r' * len(formats) + '}', r'\toprule',
             ' & '.join(headers) + r'\\', r'\midrule']
    for row in records:
        cells = [r'\textbf{' + row[0] + ' (Ours)}' if row[0] == ours else row[0]]
        if row[0] == ours:
            lines.append(r'\rowcolor{gray!10}')
        for j, fmt in enumerate(formats):
            value = float(row[j + 1])
            text = format(value, fmt)
            cells.append(r'\textbf{' + text + '}' if best and abs(value - maxima[j]) < 1e-12 else text)
        lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    (PAPER / name).write_text('\n'.join(lines), encoding='utf-8')


def main():
    freeze = read(ROOT / 'frozen-recipe.json')
    assert freeze['complete']
    dev = {r['method']: r for r in freeze['candidate_results']}
    names = [
        ('RRF native', 'RRF_native'),
        ('Historical CE', 'common_Historical_CE'),
        ('Current CE', 'common_Current_CE'),
        ('Rank fusion, equal weights', 'common_HC_rank_a0.5'),
        ('Min--max fusion, equal weights', 'common_HC_minmax_a0.5'),
        ('CE-conditioned LambdaMART (13)', 'ce_lambdamart_base13'),
        ('CE-conditioned LambdaMART (20)', 'ce_lambdamart_revision20'),
        ('CE-conditioned gate (13)', 'ce_gate_base13'),
        ('CE-conditioned gate (20)', 'ce_gate_revision20')]
    # Freeze keys retain all declared comparison recipes.
    rows = [(label, dev[key]['mrr4'], dev[key]['recall4'] * 100) for label, key in names]
    table('table-scoring-development.tex',
          'Scoring-view development with Ruri on the same RRF ten candidates. Values are macro means over four original validation cells. Historical and current scoring retain the same bound target-date text for each address. Higher is better; bold marks the best value in each column.',
          'tab:scoring-development', ['Scoring rule', r'MRR@4 $\uparrow$', r'Recall@4 (\%) $\uparrow$'],
          rows, ['.4f', '.2f'])
    head_names = [
        ('RRF', 'RRF_historical_CE'),
        ('Original LambdaMART (10)', 'Original_LM10_historical_CE'),
        ('Dual-relevance LambdaMART (11)', 'Dual_relevance_LM11_historical_CE'),
        ('Without focused alignment (16)', 'Lean_noalignment_LM16_historical_CE'),
        ('HiDate', 'Full_LM18_historical_CE')]
    rows = [(label, dev[key]['mrr4'], dev[key]['recall4'] * 100) for label, key in head_names]
    table('table-address-development.tex',
          'Ruri address selection with one shared historical-CE final stage. Each head selects its own ten candidates from the same discovered pool. The original four-cell validation MRR@4 determines the final recipe.',
          'tab:address-development', ['Candidate selector', r'MRR@4 $\uparrow$', r'Recall@4 (\%) $\uparrow$'],
          rows, ['.4f', '.2f'], ours='HiDate')

    ruri_components = read(ROOT / 'native-development/valid-components/evaluation.json')
    qwen_components = read(ROOT / 'qwen-native-development/valid-components/evaluation.json')
    component_names = [('Base retrieval features (9)', 'lambdamart_base'),
                       ('Dual-view query coverage (11)', 'lambdamart_dual_relevance'),
                       ('Without term/number change (14)', 'lambdamart_no_query_change'),
                       ('Without focused alignment (16)', 'lambdamart_no_alignment'),
                       ('Full revision features (18)', 'lambdamart_revision')]
    rows = []
    for label, key in component_names:
        r = ruri_components['conditions'][key + '|own_pool']
        q = qwen_components['conditions'][key + '|own_pool']
        rows.append((label, r['mrr4'], 100 * r['recall4'], q['mrr4'], 100 * q['recall4']))
    table('table-feature-controls.tex',
          'Fixed-policy feature controls on the original validation set. Each query-grouped LambdaMART uses the same fitting settings and its native score for final ordering, with no CE or fusion-weight change. Values are four-cell macro metrics. These address-head comparisons are distinct from the complete historical-CE pipeline.',
          'tab:feature-controls', ['Address features', r'Ruri MRR@4 $\uparrow$', r'Ruri R4 (\%) $\uparrow$',
                                   r'Qwen MRR@4 $\uparrow$', r'Qwen R4 (\%) $\uparrow$'],
          rows, ['.4f', '.2f', '.4f', '.2f'])

    scale = read(ROOT / 'transfer/ruri-scale-final/evaluation.json')
    assert scale['complete'] and scale['selected_method'] == freeze['selected_method']
    names = [
        ('Target-date hybrid', 'AsOf_hybrid_native'),
        ('Target-date hybrid + current CE', 'AsOf_hybrid_CE'),
        ('Historical RRF', 'RRF_native'),
        ('Historical RRF + current CE', 'common_Current_CE'),
        ('Original logistic/current-CE mixture', 'Original_HiDate_LR_CE'),
        ('Original LambdaMART native', 'Original_LM10_native'),
        ('Dual-relevance native', 'Dual_relevance_LM11_native'),
        ('Full-address native', 'Full_LM18_native'),
        ('Without focused alignment native', 'Lean_noalignment_LM16_native'),
        ('Historical RRF + historical CE', 'RRF_historical_CE'),
        ('Original LambdaMART + historical CE', 'Original_LM10_historical_CE'),
        ('Dual relevance + historical CE', 'Dual_relevance_LM11_historical_CE'),
        ('Without focused alignment + historical CE', 'Lean_noalignment_LM16_historical_CE'),
        ('CE-conditioned LambdaMART (13)', 'ce_lambdamart_base13'),
        ('CE-conditioned LambdaMART (20)', 'ce_lambdamart_revision20'),
        ('CE-conditioned gate (13)', 'ce_gate_base13'),
        ('CE-conditioned gate (20)', 'ce_gate_revision20'),
        ('HiDate', freeze['selected_method'])]
    shared_names = names
    rows = []
    for label, key in names:
        r = scale['available_four_cell_macro'][key]
        rows.append((label, r['target_mrr'], 100 * r['target_recall_at_four'], 100 * r['top_parent_correct']))
    table('table-scale-revised.tex',
          'Frozen ranking recipes at 1,000 laws, evaluated on the original regression queries with the Ruri stack. The table reports four-cell macro metrics over the 3,160 available-target observations. All history-based conditions deliver the same target-date realization for a given address. Bold marks column maxima.',
          'tab:scale-revised', ['Method', r'MRR@4 $\uparrow$', r'Recall@4 (\%) $\uparrow$', r'Parent@1 (\%) $\uparrow$'],
          rows, ['.4f', '.2f', '.2f'], ours='HiDate')
    confirmation_path = ROOT / 'confirmation/final/evaluation.json'
    if confirmation_path.exists():
        confirmation = read(confirmation_path)
        assert confirmation['complete'] and confirmation['selected_method'] == freeze['selected_method']
        rows = []
        for label, key in shared_names:
            r = confirmation['available_four_cell_macro'][key]
            rows.append((label, r['target_mrr'], 100 * r['target_recall_at_four'], 100 * r['top_parent_correct']))
        available = confirmation['panels'][freeze['selected_method']]['available']
        table('table-confirmation.tex',
              'Additional-family confirmation in the unchanged 1,000-law bank. The ranking recipe was fixed on the original validation set before these outcomes were examined. Values are four-cell macro metrics over '
              + f"{available['n']:,} available-target observations from {confirmation['family_count']} families. "
              + 'Bold marks column maxima.',
              'tab:confirmation', ['Method', r'MRR@4 $\uparrow$', r'Recall@4 (\%) $\uparrow$', r'Parent@1 (\%) $\uparrow$'],
              rows, ['.4f', '.2f', '.2f'], ours='HiDate')
    ruri = read(ROOT / 'transfer/ruri-original-final/evaluation.json')
    qwen = read(ROOT / 'transfer/qwen-test/evaluation.json')
    qwen_ce = read(ROOT / 'transfer/qwen-test/ce-controls/evaluation.json')
    assert ruri['complete'] and qwen['complete'] and qwen_ce['complete']
    qwen_metrics = {**qwen['available_four_cell_macro'], **qwen_ce['available_four_cell_macro']}
    names = [
        ('Target-date hybrid', 'AsOf_hybrid_native', 'AsOf_hybrid_native'),
        ('Target-date hybrid + current CE', 'AsOf_hybrid_CE', 'AsOf_hybrid_current_CE'),
        ('Historical RRF', 'RRF_native', 'RRF_native__native'),
        ('Historical RRF + current CE', 'common_Current_CE', 'RRF_native__current_ce'),
        ('Original LambdaMART native', 'Original_LM10_native', 'Original_LM10__native'),
        ('Dual-relevance native', 'Dual_relevance_LM11_native', 'Dual_relevance_LM11__native'),
        ('Full-address native', 'Full_LM18_native', 'Full_LM18__native'),
        ('Without focused alignment native', 'Lean_noalignment_LM16_native', 'Lean_noalignment_LM16__native'),
        ('Historical RRF + historical CE', 'RRF_historical_CE', 'RRF_native__historical_ce'),
        ('Original LambdaMART + historical CE', 'Original_LM10_historical_CE', 'Original_LM10__historical_ce'),
        ('Dual relevance + historical CE', 'Dual_relevance_LM11_historical_CE', 'Dual_relevance_LM11__historical_ce'),
        ('Without focused alignment + historical CE', 'Lean_noalignment_LM16_historical_CE', 'Lean_noalignment_LM16__historical_ce'),
        ('CE-conditioned LambdaMART (13)', 'ce_lambdamart_base13', 'CE_lambdamart_base13'),
        ('CE-conditioned LambdaMART (20)', 'ce_lambdamart_revision20', 'CE_lambdamart_revision20'),
        ('CE-conditioned gate (13)', 'ce_gate_base13', 'CE_gate_base13'),
        ('CE-conditioned gate (20)', 'ce_gate_revision20', 'CE_gate_revision20'),
        ('HiDate', 'Full_LM18_historical_CE', 'Full_LM18__historical_ce')]
    rows = []
    for label, rk, qk in names:
        r, q = ruri['available_four_cell_macro'][rk], qwen_metrics[qk]
        rows.append((label, r['target_mrr'], r['target_recall_at_four'] * 100,
                     q['target_mrr'], q['target_recall_at_four'] * 100))
    table('table-backbone-revised.tex',
          'Complete-stack regression evaluation in the original 120-law corpus. Each stack uses its own training discovery, embeddings and CE scores; the selected recipe remains fixed. Metrics are four-cell macro values over 3,160 available-target observations. Bold marks column maxima.',
          'tab:backbone-revised', ['Method', r'Ruri MRR@4 $\uparrow$', r'Ruri R4 (\%) $\uparrow$',
                                  r'Qwen MRR@4 $\uparrow$', r'Qwen R4 (\%) $\uparrow$'],
          rows, ['.4f', '.2f', '.4f', '.2f'], ours='HiDate')
    with (ROOT / 'cost-benchmark/latency-summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        costs = {r['method']: r for r in csv.DictReader(stream)}
    names = [('Historical RRF', 'RRF_native'), ('Target-date hybrid', 'AsOf_hybrid_native'),
             ('Target-date hybrid + current CE', 'AsOf_hybrid_current_CE'),
             ('Historical RRF + historical CE', 'Historical_CE'),
             ('Original LambdaMART + historical CE', 'Original_LM10_historical_CE'),
             ('Dual-CE LambdaMART (20)', 'Development_lambdamart_revision20_dual_CE'),
             ('HiDate', 'Full_LM18_historical_CE')]
    rows = [(label, costs[key]['p50_ms'], costs[key]['p95_ms'], costs[key]['CE_actual_unique_pairs'],
             costs[key]['CE_calls'], costs[key]['CE_input_tokens']) for label, key in names]
    table('table-online-cost.tex',
          'Actual independent-query latency on an RTX A6000 with four CPU threads and the 120-law index. Each method processes the same 100 validation queries with fresh query encoding and no cross-request score reuse. CE counts and nonpadding input tokens are totals over 100 requests. Model/index setup and warm-up are excluded.',
          'tab:online-cost', ['Method', r'p50 ms $\downarrow$', r'p95 ms $\downarrow$', r'CE pairs $\downarrow$', r'CE calls $\downarrow$', r'CE tokens $\downarrow$'],
          rows, ['.1f', '.1f', ',.0f', ',.0f', ',.0f'], ours='HiDate', best=False)

    fiscal_path = ROOT / 'transfer/fiscal-ce/analysis-expanded/analysis.json'
    if not fiscal_path.exists():
        fiscal_path = ROOT / 'transfer/fiscal-ce/analysis/analysis.json'
    if fiscal_path.exists():
        fiscal = read(fiscal_path)
        assert fiscal['complete'] and len(fiscal['methods']) in (21, 26)
        names = [
            ('Closed-book reader', 'old.closed_book'),
            ('Static BM25', 'old.static_bm25'),
            ('Static dense', 'old.static_dense'),
            ('Target-date BM25 (original)', 'old.asof_bm25'),
            ('Target-date dense (original)', 'old.asof_dense'),
            ('Target-date hybrid (original)', 'old.asof_hybrid'),
            ('Historical search, unbound', 'old.history_unbound'),
            ('Historical search + binding (original)', 'old.history_bound'),
            ('Current-view address ordering (original)', 'old.history_current'),
            ('Original logistic-address policy', 'old.history_learned'),
            ('Target-date hybrid native', 'AsOf_hybrid_native'),
            ('Target-date hybrid + current CE', 'AsOf_hybrid_current_CE'),
            ('Historical RRF native', 'RRF_native'),
            ('Historical RRF + historical CE', 'RRF_historical_CE'),
            ('Original LambdaMART native', 'Original_LM10_native'),
            ('Dual-relevance native', 'Dual_relevance_LM11_native'),
            ('Full-address native', 'Full_LM18_native'),
            ('Without focused alignment native', 'Lean_noalignment_LM16_native'),
            ('Original LambdaMART + historical CE', 'Original_LM10_historical_CE'),
            ('Dual relevance + historical CE', 'Dual_relevance_LM11_historical_CE'),
            ('CE-conditioned LambdaMART (13)', 'Qwen_lambdamart_base13'),
            ('CE-conditioned LambdaMART (20)', 'Qwen_lambdamart_revision20'),
            ('TimelyRAG interval, native versions', 'TimelyRAG_interval_native'),
            ('TimelyRAG interval + CID binding', 'TimelyRAG_interval_bound'),
            ('HiDate', 'Full_LM18_historical_CE'),
            ('Gold-article reference', 'old.oracle_asof')]
        names = [(label, key) for label, key in names if key in fiscal['results']]
        assert len(names) == len(fiscal['methods'])
        metrics = ['strict', 'coverage', 'gold_article_recall_at_four', 'gold_version_recall_at_four']
        rows = [(label, *(100 * fiscal['results'][key]['scored209'][m] for m in metrics)) for label, key in names]
        table('table-fiscal-revised.tex',
              'French FiscalQA transfer on all 209 primary questions with the same frozen reader and official scorer. All values are percentages. Article and version recall use the evidence actually retained in the reader prompt. Japanese-trained heads transfer without French fitting. The gold-article condition supplies a reference; bold marks column maxima.',
              'tab:fiscal-revised', ['Method', r'Strict $\uparrow$', r'Nuggets $\uparrow$', r'Article R4 $\uparrow$', r'Version R4 $\uparrow$'],
              rows, ['.2f'] * 4, ours='HiDate')

    stage_rows = []
    for part in ('regression', 'confirmation'):
        source = ROOT / f'diagnostics/stage-reachability/{part}/stage-metrics.csv'
        if source.exists():
            with source.open(encoding='utf-8-sig', newline='') as stream:
                stage_rows.extend(csv.DictReader(stream))
    if stage_rows:
        cohorts = [('ruri120', 'Ruri, 120 laws'), ('ruri1000', 'Ruri, 1,000 laws'),
                   ('qwen120', 'Qwen, 120 laws'), ('confirmation', 'Additional families, Ruri')]
        lines = [r'\begin{table*}[t]', r'\centering\small',
            r'\caption{Stage-wise target retention on available queries. Each block uses four-cell macro percentages. Discovery and bindable-parent recall coincide in these cohorts; candidate selection and final ranking account for the subsequent losses. The three selectors share historical discovery and the historical-CE final stage.}',
            r'\label{tab:retention-stages}', r'\begin{tabular}{lrrrr}', r'\toprule',
            r'Selector & Discovered $\uparrow$ & Bindable $\uparrow$ & Own-ten $\uparrow$ & Final-four $\uparrow$\\', r'\midrule']
        for cohort, label in cohorts:
            data = {r['method']: r for r in stage_rows if r['cohort'] == cohort and r['scope'] == 'available_four_cell_macro'}
            if not data:
                continue
            keys = [('RRF', 'RRF_native__historical_ce' if cohort == 'qwen120' else 'RRF_historical_CE'),
                    ('Original LambdaMART', 'Original_LM10__historical_ce' if cohort == 'qwen120' else 'Original_LM10_historical_CE'),
                    (r'\textbf{HiDate (Ours)}', 'Full_LM18__historical_ce' if cohort == 'qwen120' else 'Full_LM18_historical_CE')]
            lines.append(r'\multicolumn{5}{l}{\textit{' + label + r'}}\\')
            for name, key in keys:
                r = data[key]
                lines.append(name + ' & ' + ' & '.join(f'{100 * float(r[k]):.2f}' for k in
                    ('discovery_address_hit', 'bindable_parent_hit', 'candidate_parent_at10', 'final_recall_at4')) + r'\\')
            lines.append(r'\addlinespace')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
        (PAPER / 'table-retention-stages.tex').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
