"""Render already-computed complete structural risk/coverage curves."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SOURCES={
    'Ruri':Path('runs/hidate/analysis/risk-coverage.json'),
    'Qwen3':Path('runs/hidate/extended-benchmark/qwen3/analysis/risk-coverage.json')}
METHODS={
    'asof_bm25':('As-of BM25','#aab1b8','-'),
    'asof_dense':('As-of dense','#8d79b5','-'),
    'asof_hybrid':('As-of hybrid','#3579ab','-'),
    'history_unbound':('Unbound history','#b7a56e','--'),
    'history_bound':('History + binding','#4f9a88','-'),
    'history_current':('Current-text ordering','#c58368','--'),
    'history_learned':('Original LR support policy','#25354e','-')}


def main():
    out=Path('paper/hidate/development/figures');out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),sharex=True,sharey=True)
    rows=[];points=[]
    for row,(stack,path) in enumerate(SOURCES.items()):
        source=json.loads(path.read_text(encoding='utf-8'))
        for col,variant in enumerate(('exact_quote','generated_paraphrase')):
            ax=axes[row,col]
            selected=[c for c in source['curves'] if c['evaluation_scope']=='natural'
                and c['stratum']=='all' and c['target_asof']=='all'
                and c['source_quote_absent_from_target'] is None and c['query_variant']==variant]
            assert len(selected)==7
            for c in selected:
                label,color,style=METHODS[c['method']]
                valid=[p for p in c['points'] if p['risk'] is not None]
                ax.step([p['coverage'] for p in valid],[p['risk'] for p in valid],where='post',
                    color=color,ls=style,lw=1.8 if c['method']=='history_learned' else 1.1,label=label)
                p=c['frozen_point']
                if p['risk'] is not None:ax.scatter(p['coverage'],p['risk'],s=20,color=color,zorder=5,edgecolor='white',linewidth=.4)
                points.append({'stack':stack,'variant':variant,'method':c['method'],**p})
                rows.extend({'stack':stack,'variant':variant,'method':c['method'],**p} for p in c['points'])
            ax.set_title(stack+' · '+('Literal' if col==0 else 'Generated'),loc='left',weight='bold',pad=8)
            ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.16,linewidth=.6);ax.set_axisbelow(True)
            if row==1:ax.set_xlabel('Coverage ↑')
            if col==0:ax.set_ylabel('Structural error rate ↓')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=8.2)
    fig.subplots_adjust(left=.095,right=.98,top=.95,bottom=.19,hspace=.26,wspace=.16)
    for ext in ('pdf','svg','png'):fig.savefig(out/('structural_risk_coverage.'+ext),dpi=220,bbox_inches='tight',pad_inches=.08)
    for name,data in [('structural_risk_coverage.csv',rows),('structural_risk_operating_points.csv',points)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (out/name).open('w',encoding='utf-8',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(data)
    (out/'structural_risk_coverage_caption.txt').write_text(
        'Structural risk–coverage curves for the original frozen support policies on the regression set. '
        'Each curve sweeps all positive-score thresholds, retaining tied scores together; markers denote the '
        'original validation-selected cutoffs. Curves stop at each policy’s attainable coverage. '
        'Correctness here means target-parent support, distinct from reader-answer correctness. '
        'The original LR policy is not the newly developed two-view ranker.\n',encoding='utf-8')
    print(json.dumps({'curves':28,'saved_threshold_points':len(rows),'output':str(out)}))


if __name__=='__main__':main()
