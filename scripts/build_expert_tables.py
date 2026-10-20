"""Render manuscript tables from completed, frozen expert retrieval analysis."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "outputs" / "tables"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'runs/hidate/expert-benchmark/analysis/analysis.json'
A = json.loads(SOURCE.read_text(encoding='utf-8'))
assert A['complete'] and A['selected_method'] == 'Full_LM18_historical_CE'
P = A['panels']['answerable']['methods']
D = A['date_pair_panels']['all_verified_pairs']['methods']
COMPLETE = 'revision_correct_set_complete_at_k'
COVERAGE = 'revision_correct_set_coverage_at_k'
RR = 'first_complete_set_reciprocal_rank_at_k'
PAIR = 'both_dates_retrieval_complete_at4'
NAMES = {
 'AsOf_hybrid_native': 'Target-date hybrid, native',
 'AsOf_hybrid_CE': 'Target-date hybrid + current CE',
 'RRF_native': 'Historical RRF, native',
 'RRF_historical_CE': 'Historical RRF + historical CE',
 'common_Current_CE': 'Common RRF ten + current CE',
 'Original_HiDate_LR_CE': 'Original LR/current-CE mixture',
 'Original_LR_native': 'Original LR, native',
 'Original_LR_historical_CE': 'Original LR + historical CE',
 'Original_LM10_native': 'Original LM10, native',
 'Original_LM10_historical_CE': 'Original LM10 + historical CE',
 'Dual_relevance_LM11_native': 'Dual-relevance LM11, native',
 'Dual_relevance_LM11_historical_CE': 'Dual-relevance LM11 + historical CE',
 'Lean_noalignment_LM16_native': 'Reduced-alignment LM16, native',
 'Lean_noalignment_LM16_historical_CE': 'Reduced-alignment LM16 + historical CE',
 'Full_LM18_native': 'Full LM18, native',
 'Full_LM18_historical_CE': r'\textbf{HiDate (Ours)}',
 'ce_lambdamart_base13': 'Common RRF ten + CE-LM13',
 'ce_lambdamart_revision20': 'Common RRF ten + CE-LM20',
 'ce_gate_base13': 'Common RRF ten + CE-gate13',
 'ce_gate_revision20': 'Common RRF ten + CE-gate20',
 'RRF_identity_view': 'RRF + identity-dependent CE view',
 'common_Historical_CE': 'Common RRF ten + historical CE',
 'common_RRF_native': 'Common RRF ten, native',
}
for mode in ('rank', 'minmax'):
    label = 'rank' if mode == 'rank' else 'min--max'
    for weight in (0, .25, .5, .75, 1):
        NAMES[f'common_HC_{mode}_a{weight:g}'] = rf'Common RRF ten, {label} H/C, $\alpha={weight:g}$'

MAIN = ['AsOf_hybrid_native', 'AsOf_hybrid_CE', 'RRF_native',
        'common_Current_CE', 'RRF_historical_CE', 'Original_HiDate_LR_CE',
        'Original_LM10_native', 'Original_LM10_historical_CE',
        'Dual_relevance_LM11_native', 'Dual_relevance_LM11_historical_CE',
        'Lean_noalignment_LM16_native', 'Lean_noalignment_LM16_historical_CE',
        'Full_LM18_native', 'Full_LM18_historical_CE',
        'ce_lambdamart_base13', 'ce_lambdamart_revision20',
        'ce_gate_base13', 'ce_gate_revision20']


def value(method, metric):
    return P[method][metric]['mean']


def write(name, lines):
    destination = HERE / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')


lines = [r'\begin{table*}[t]', r'\centering\small',
 r'\caption{Expert-scenario retrieval with the frozen Ruri stack and a shared 1,000-law bank. Set coverage and Complete@4 use 259 provision-eligible instances; Set RR@4 is the reciprocal rank of the first prefix completing a sufficient dated evidence set. Both dates requires complete retrieval at each date in 79 eligible pairs. Instance measures are query-weighted; both-date completeness is pair-weighted. All methods return at most four parents.}',
 r'\label{tab:expert-main}', r'\setlength{\tabcolsep}{6pt}',
 r'\begin{tabular}{lrrrr}', r'\toprule',
 r'Method & Set coverage (\%) $\uparrow$ & Complete@4 (\%) $\uparrow$ & Set RR@4 $\uparrow$ & Both dates (\%) $\uparrow$\\', r'\midrule']
for i, method in enumerate(MAIN):
    if i in (2, 6, 14):
        lines.append(r'\midrule')
    if method == A['selected_method']:
        lines.append(r'\rowcolor{gray!10}')
    lines.append(f"{NAMES[method]} & {100*value(method,COVERAGE):.2f} & {100*value(method,COMPLETE):.2f} & {value(method,RR):.4f} & {100*D[method][PAIR]['mean']:.2f}" + r'\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
write('table-expert-main.tex', lines)

COMPARE = ['AsOf_hybrid_native', 'AsOf_hybrid_CE', 'RRF_historical_CE',
           'Original_LM10_historical_CE', 'Full_LM18_historical_CE']
lines = [r'\begin{table*}[t]', r'\centering\small',
 r'\caption{Expert-scenario categories and real-date pairs. Entries are successful instances or pairs divided by the panel size ($\uparrow$: higher is better). Category rows require a complete dated evidence set within four parents. Pair rows require this at both dates, retaining the same facts and question except for the requested date.}',
 r'\label{tab:expert-panels}', r'\begin{tabular}{lrrrrr}', r'\toprule',
 r'Panel & Target-date native & Target-date CE & RRF + hist. CE & LM10 + hist. CE & \textbf{HiDate (Ours)}\\', r'\midrule']
for name, label in [('category=old_terms', 'Historical terminology'),
                    ('category=real_world', 'Practical fact patterns'),
                    ('category=changed_conditions', 'Amended conditions')]:
    panel = A['panels'][name]
    cells = [f"{round(panel['methods'][m][COMPLETE]['mean']*panel['n'])}/{panel['n']}" for m in COMPARE]
    lines.append(label + ' & ' + ' & '.join(cells) + r'\\')
lines.append(r'\midrule')
for name, label in [('verified_changed', 'Answer changes across dates'),
                    ('verified_unchanged', 'Answer remains unchanged'),
                    ('all_verified_pairs', 'All eligible date pairs')]:
    panel = A['date_pair_panels'][name]
    cells = [f"{round(panel['methods'][m][PAIR]['mean']*panel['n'])}/{panel['n']}" for m in COMPARE]
    lines.append(label + ' & ' + ' & '.join(cells) + r'\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
write('table-expert-panels.tex', lines)

ORDER = MAIN + [m for m in P if m not in MAIN]
assert set(ORDER) == set(P) == set(NAMES) and len(ORDER) == 33
lines = [r'\begin{table*}[p]', r'\centering\small',
 r'\caption{All 33 frozen expert-retrieval conditions. Coverage and Complete@4 are percentages over 259 provision-eligible instances; both-date completeness uses 79 pairs. The final column reports \method\ minus the row method in Complete@4 percentage points, with a paired 95\% family-bootstrap interval. H/C mixtures assign weight $\alpha$ to historical CE and $1-\alpha$ to current CE. Shared-pool endpoint rows are retained as separate predefined conditions.}',
 r'\label{tab:expert-all}', r'\setlength{\tabcolsep}{4pt}',
 r'\begin{tabular}{lrrrrr}', r'\toprule',
 r'Method & Coverage $\uparrow$ & Complete@4 $\uparrow$ & Set RR@4 $\uparrow$ & Both dates $\uparrow$ & $\Delta$ Complete [95\% CI]\\', r'\midrule']
for i, method in enumerate(ORDER):
    if i in (2, 6, 14, 18, 23):
        lines.append(r'\midrule')
    q = P[method][COMPLETE]
    delta = 100*q['selected_minus_reference']
    lo, hi = (100*x for x in q['selected_minus_reference_ci95'])
    if method == A['selected_method']:
        lines.append(r'\rowcolor{gray!10}')
    interval = f'${delta:.2f}$ [${lo:.2f}$, ${hi:.2f}$]'
    lines.append(f"{NAMES[method]} & {100*value(method,COVERAGE):.2f} & {100*value(method,COMPLETE):.2f} & {value(method,RR):.4f} & {100*D[method][PAIR]['mean']:.2f} & {interval}" + r'\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
write('table-expert-all.tex', lines)
write('qa/expert-table-source.json', [json.dumps({'source':SOURCE.relative_to(ROOT).as_posix(),
    'analysis_complete':A['complete'],'selected_method':A['selected_method'],
    'main_conditions':MAIN,'appendix_conditions':ORDER,'provision_instances':259,
    'eligible_pairs':79,'training_runs':0,'selection_updates':0},indent=2)])
print('Generated three tables from completed analysis: 18 main and 33 appendix conditions.')
