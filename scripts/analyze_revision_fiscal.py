"""Read back complete French reader runs and paired gold-CID intervals."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

from hidate.io import write_json
from hidate.statistics import paired_bootstrap
from run_revision_fiscal import rows,read,receipt


METRICS=('strict','coverage','gold_article_recall_at_four','gold_version_recall_at_four',
         'top_article_correct','top_version_correct','article_citation_nugget')
MAIN='Full_LM18_historical_CE'
REFERENCES=('RRF_native','RRF_historical_CE','Original_LM10_native',
    'Original_LM10_historical_CE','Dual_relevance_LM11_historical_CE',
    'AsOf_hybrid_native','AsOf_hybrid_current_CE',
    'TimelyRAG_interval_native','TimelyRAG_interval_bound','old.history_learned')
STRONG_REFERENCES=('Qwen_lambdamart_base13','Qwen_lambdamart_revision20',
    'Full_LM18_native','Lean_noalignment_LM16_native','Dual_relevance_LM11_native')


def load_complete(directory,prefix,questions):
    meta=read(directory/'evaluation.json');reader=read(directory/'reader.json')
    if not meta['complete'] or not reader['complete']:
        raise ValueError('Readback requires complete reader and official scores: '+str(directory))
    conditions=meta['conditions'];expected={(qid,c) for qid in questions for c in conditions}
    grouped=defaultdict(dict);seen=set()
    for row in rows(directory/'scores.jsonl'):
        key=(row['qid'],row['condition'])
        if key in seen or key not in expected:raise ValueError('Duplicate/unexpected official score identity')
        seen.add(key);grouped[prefix+row['condition']][row['qid']]=row
    if seen!=expected:raise ValueError('Official score rows do not cover all221 questions/conditions')
    return grouped,meta,reader


def merge_new(target,new):
    duplicate=0
    for condition,items in new.items():
        if condition in target:
            if set(target[condition])!=set(items):raise ValueError('Repeated condition query sets differ')
            for qid,row in items.items():
                if any(row[k]!=target[condition][qid][k] for k in METRICS):
                    raise ValueError('Incrementally reused condition has different official scores')
            duplicate+=len(items)
        else:target[condition]=items
    return duplicate


def summarize_items(items,ids):
    values={k:[items[q][k] for q in ids if items[q][k] is not None] for k in METRICS}
    return {'n':len(ids),'metric_n':{k:len(v) for k,v in values.items()},
        **{k:float(np.mean(v)) if v else None for k,v in values.items()}}


def analyze(args):
    questions={q['qid']:q for q in rows(args.prepared/'questions.jsonl')}
    primary={qid:q for qid,q in questions.items() if q['scope']=='in'}
    if len(questions)!=221 or len(primary)!=209:raise ValueError('Keep original published221/scored209 roles')
    groups={qid:q['gold']['article_cid'] for qid,q in primary.items()}
    old,old_meta,old_reader=load_complete(args.original,'old.',questions)
    base,base_meta,base_reader=load_complete(args.output/'reader','',questions)
    extra,extra_meta,extra_reader=load_complete(args.output/'reader-frozen-main','',questions)
    merged={**old,**base};duplicates=merge_new(merged,extra)
    if extra_meta.get('main_method')!=MAIN:raise ValueError('Main method label must match actual VALID freeze')
    sources={method:(old_meta if method.startswith('old.') else extra_meta if method in extra else base_meta)
        for method in merged}
    extra_directories=getattr(args,'extra_reader',[]) or []
    expanded_readers={}
    for directory in extra_directories:
        added,added_meta,added_reader=load_complete(directory,'',questions)
        if added_meta.get('main_method')!=MAIN:raise ValueError('Additional controls require the same VALID freeze')
        duplicates+=merge_new(merged,added)
        sources.update({method:added_meta for method in added})
        expanded_readers[str(directory)]=added_reader
    results={}
    for method,items in merged.items():
        results[method]={}
        for name,ids in [('scored209',set(primary)),('published221',set(questions))]:
            results[method][name]=summarize_items(items,ids)
        source=sources[method]
        label=method.removeprefix('old.')
        saved=source['strata']['scored209'][label]
        actual=results[method]['scored209']
        if any((actual[k]!=saved[k] if actual[k] is None or saved[k] is None
                else abs(actual[k]-saved[k])>1e-12) for k in METRICS):
            raise ValueError('Readback disagrees with the saved official summary')
    comparisons=[]
    references=list(REFERENCES)+[method for method in STRONG_REFERENCES if method in merged]
    for reference in references:
        for metric in ('strict','coverage','gold_article_recall_at_four','gold_version_recall_at_four'):
            a=[{'id':qid,'group_id':groups[qid],metric:merged[reference][qid][metric]} for qid in primary]
            b=[{'id':qid,'group_id':groups[qid],metric:merged[MAIN][qid][metric]} for qid in primary]
            comparisons.append({'method':MAIN,'reference':reference,
                **paired_bootstrap(a,b,metric,seed=42,n_resamples=2000)})
    args.analysis.mkdir(parents=True,exist_ok=True)
    meta=receipt(complete=True,questions=221,primary_questions=209,gold_article_clusters=len(set(groups.values())),
        main_method=MAIN,methods=list(results),results=results,paired_comparisons=comparisons,
        repeated_condition_rows_verified=duplicates,source_directories=[str(args.original),str(args.output/'reader'),
            str(args.output/'reader-frozen-main')]+[str(p) for p in extra_directories],
        CE=read(args.output/'scoring.json'),readers={'first8':base_reader,'frozen_main3':extra_reader,
            **expanded_readers},
        scoring='original official strict and mean per-question nugget coverage; actual packed evidence',
        intervals='2000 paired sample-weighted gold-article-CID cluster draws, fixed statistical seed42')
    write_json(args.analysis/'analysis.json',meta)
    with (args.analysis/'results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','cohort','n',*METRICS])
        writer.writeheader()
        for method,cohorts in results.items():
            for cohort,result in cohorts.items():
                writer.writerow({'method':method,'cohort':cohort,
                    **{k:result[k] for k in ('n',*METRICS)}})
    lines=['# HiDate French transfer: completed reader readback','',
        f'All {3+len(extra_directories)} source evaluations are complete. Primary cohort:209 questions in {len(set(groups.values()))} gold-article CID clusters; published cohort:221.',
        f'Frozen main method:`{MAIN}`. Duplicate OriginalLM10 records agree on all official metrics ({duplicates} query rows).','',
        'Strict is official all-required-nuggets correctness. Nugget coverage is the mean per-question fraction, excluding provenance. Article/version recall uses actual packed reader evidence. All values below are percentages.','',
        '| Method | n | Strict ↑ | Nugget coverage ↑ | Article R@4 ↑ | Version R@4 ↑ | Article@1 ↑ | Version@1 ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    order=[MAIN]+[x for x in results if x!=MAIN and not x.startswith('old.')]+[x for x in results if x.startswith('old.')]
    for method in order:
        r=results[method]['scored209']
        lines.append('| '+method+' | '+str(r['n'])+' | '+' | '.join(
            f'{100*r[k]:.2f}' if r[k] is not None else '—' for k in METRICS[:6])+' |')
    lines+=['','Paired differences below are main minus comparator, in percentage points, over the same209 questions.','',
        '| Comparator | Strict Δ [95% CI] | Nugget coverage Δ [95% CI] |','|---|---:|---:|']
    for reference in references:
        values=[]
        for metric in ('strict','coverage'):
            c=next(c for c in comparisons if c['reference']==reference and c['metric']==metric)
            values.append(f"{100*c['delta_b_minus_a']:+.2f} [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}]")
        lines.append('| '+reference+' | '+' | '.join(values)+' |')
    lines+=['','Actual execution and cache accounting:','',
        f"- New CE: {meta['CE']['new_physical_pairs_total']} exact pairs; {meta['CE']['physical_work_total']['input_tokens']} nonpadding tokens; {meta['CE']['physical_work_total']['model_seconds']:.3f}s model time.",
        f"- First8 reader: {base_reader['logical_outputs']} logical outputs, {base_reader['new_physical_calls_total']} new physical prompts.",
        f"- Frozen-main3 reader: {extra_reader['logical_outputs']} logical outputs, {extra_reader['new_physical_calls_total']} new physical prompts; exact earlier prompts reused.",
        '- Reader: unchanged official query windows and4096-token packing; greedy, at most192 new tokens; pinned original Qwen2.5-7B-Instruct.',
        '- Original11 conditions are retained above with the `old.` prefix. Their legacy support decisions are not applied to the new native/CE conditions.',
        '- Full published221 results, article/version paired intervals, source paths and complete cost receipts are in `analysis.json`.','']
    for directory,reader in expanded_readers.items():
        lines.append(f"- Additional fixed controls `{directory}`: {reader['logical_outputs']} logical outputs, "
            f"{reader['new_physical_calls_total']} new physical prompts; exact earlier prompts reused. "
            'Japanese-Qwen TRAIN heads transfer without French fitting; existing French CE scores are reused.')
    (args.analysis/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'complete':True,'methods':len(results),'primary_questions':209,
        'gold_article_clusters':len(set(groups.values())),'paired_comparisons':len(comparisons)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('data/processed/fiscal'))
    p.add_argument('--original',type=Path,default=Path('runs/hidate/fiscal-transfer/fiscal'))
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/transfer/fiscal-ce'))
    p.add_argument('--analysis',type=Path,default=Path('runs/hidate/revision-aware/transfer/fiscal-ce/analysis'))
    p.add_argument('--extra-reader',type=Path,nargs='*',default=[])
    args=p.parse_args()
    analyze(args)


if __name__=='__main__':main()
