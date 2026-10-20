"""Saved-data scoring-view diagnostic with fixed per-address date bindings."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    source = Path('runs/hidate/revision-aware/diagnostics/revision-dualview/evaluation.json')
    report = json.loads(source.read_text(encoding='utf-8'))
    out = Path('paper/hidate/development/figures')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    methods = [('Historical_CE_before_binding','Historical-view CE','#CA574B','o'),
               ('Current_CE','Current-view CE','#337EA1','s'),
               ('RRF_native','Historical RRF','#818B96','D')]
    variants = [('exact_quote','Literal query'),('generated_paraphrase','Generated query')]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),sharey=True)
    records=[]
    for ax,(variant,label) in zip(axes,variants):
        for method,name,color,marker in methods:
            values=[]
            for stratum in ('same_date','changed_cross_date'):
                cell=report['heads'][method]['cells'][variant+'|'+stratum]
                values.append(cell['mrr_at4']*100)
                records.append({'variant':variant,'stratum':stratum,'method':method,
                    'n':cell['n'],'mrr4':cell['mrr_at4'],'recall4':cell['recall_at4']})
            ax.plot([0,1],values,color=color,marker=marker,lw=2,ms=5,label=name,zorder=3)
            offset=4 if method=='Historical_CE_before_binding' else -12 if method=='Current_CE' else 4
            ax.annotate(f'{values[1]:.1f}',(1,values[1]),xytext=(6,offset),
                        textcoords='offset points',color=color,fontsize=9)
        ax.set_title(label,fontweight='bold',pad=12)
        ax.set_xticks([0,1],['Same-date','Changed cross-date'])
        ax.set_xlim(-.15,1.28);ax.set_ylim(62,103);ax.set_yticks([65,75,85,95,100])
        ax.grid(axis='y',color='#e5e9ed',lw=.7,zorder=0)
        ax.spines['left'].set_color('#b0b8c1');ax.spines['bottom'].set_color('#b0b8c1')
        ax.tick_params(length=0,pad=6)
    axes[0].set_ylabel('MRR@4 (%) ↑')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,
               bbox_to_anchor=(.5,.005),handlelength=2,columnspacing=1.6,fontsize=9)
    fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.23,wspace=.17)
    for extension in ('pdf','svg','png'):
        fig.savefig(out/('scoring_view_diagnostic.'+extension),dpi=300,facecolor='white')
    with (out/'scoring_view_diagnostic.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'scoring_view_diagnostic_caption.txt').write_text(
        'Original VALID diagnostic with identical historical-RRF ten-address candidates and fixed '
        'requested-date text for each address. The cross-encoder scores either the retrieved historical representative '
        'or the canonical requested-date chunk; all returned evidence remains date-bound. '
        'The same-date and changed cross-date cells contain884 and345 available questions per query '
        'variant, respectively. Data: completed job5100468. This figure is a development diagnostic.\n',encoding='utf-8')
    print(out/'scoring_view_diagnostic.pdf')


if __name__=='__main__':main()
