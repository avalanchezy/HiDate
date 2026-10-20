"""Extract three illustrative saved scale cases; no model execution."""
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path

from hidate.io import write_json
from run_core_mechanisms import read,rows


HEADS=('RRF','Original_LM10','Full_LM18')


def historical_order(candidates,pool,view='historical_ce'):
    return sorted(range(len(pool)),key=lambda rank:(
        -candidates[pool[rank]][view]['reranker_score'],rank,
        candidates[pool[rank]]['current_chunk_id']))


def excerpt(text,offset=0,limit=1600):
    start=max(0,offset-300);end=min(len(text),start+limit)
    return {'text':text[start:end],'start_character':start,'end_character':end,
            'full_characters':len(text),'truncated':start>0 or end<len(text)}


def text_pair(candidate,chunks):
    h=chunks[candidate['historical_chunk_id']];c=chunks[candidate['current_chunk_id']]
    edits=[x for x in SequenceMatcher(None,h['text'],c['text'],autojunk=False).get_opcodes() if x[0]!='equal']
    hi,ci=(edits[0][1],edits[0][3]) if edits else (0,0)
    def view(chunk,at):
        return {**{k:chunk[k] for k in ('chunk_id','parent_passage_id','law_title','title','caption',
            'law_id','revision_id','scope','article_num','asof_dates','url') if k in chunk},
            'excerpt':excerpt(chunk['text'],at)}
    return {'historical':view(h,hi),'requested_date':view(c,ci),
        'same_parent':h['parent_passage_id']==c['parent_passage_id'],
        'same_revision':h.get('revision_id')==c.get('revision_id'),
        'same_text':h['text']==c['text']}


def extract(args):
    if not read(args.input/'evaluation.json')['complete']:raise ValueError('Use completed saved scale results')
    improvements={}
    for row in rows(args.input/'predictions.jsonl'):
        if row['query_variant']!='generated_paraphrase' or row['quote_absent'] is not True or row['target_status']!='available':continue
        full=row['conditions']['Full_LM18_historical_CE'];old=row['conditions']['Original_LM10_historical_CE']
        delta=full['structural']['target_mrr']-old['structural']['target_mrr']
        if delta>0:
            improvements[row['query_id']]={'prediction':row,'delta':delta,
                'recall_gain':int(full['structural']['target_recall_at_four'])-int(old['structural']['target_recall_at_four']),
                'pool_gain':full['candidate_parent_at10']-old['candidate_parent_at10']}
    queries={q['query_id']:q for p in args.queries for q in rows(p) if q['query_id'] in improvements}
    if set(queries)!=set(improvements):raise ValueError('Original query metadata missing for eligible cases')
    ordered=sorted(improvements,key=lambda qid:(-improvements[qid]['pool_gain'],
        -improvements[qid]['recall_gain'],-improvements[qid]['delta'],qid))
    selected=[];families=set();addresses=set()
    for diverse in (True,False):
        for qid in ordered:
            q=queries[qid];address=tuple(q['target_key'])
            if qid in selected or address in addresses or (diverse and q['family_id'] in families):continue
            selected.append(qid);families.add(q['family_id']);addresses.add(address)
            if len(selected)==3:break
        if len(selected)==3:break
    scored={r['query_id']:r for r in rows(args.input/'scored.jsonl') if r['query_id'] in selected}
    needed={c[v] for r in scored.values() for c in r['candidates'] for v in ('historical_chunk_id','current_chunk_id')}
    chunks={c['chunk_id']:c for c in rows(args.chunks) if c['chunk_id'] in needed}
    cases=[]
    for qid in selected:
        q=queries[qid];saved=scored[qid];cc=saved['candidates'];pred=improvements[qid]['prediction'];heads={}
        for name in HEADS:
            pool=saved['pools'][name];order=historical_order(cc,pool)
            expected=pred['conditions'][name+'_historical_CE']['evidence_chunk_ids']
            actual=[cc[pool[i]]['current_chunk_id'] for i in order[:4]]
            if actual!=expected:raise ValueError('Historical CE ordering does not reproduce saved evidence')
            candidates=[]
            for rank,i in enumerate(pool,1):
                c=cc[i];candidates.append({'address':c['address'],'preselection_rank':rank,
                    'historical_CE_rank_within_ten':order.index(rank-1)+1,
                    'historical_chunk_id':c['historical_chunk_id'],'current_chunk_id':c['current_chunk_id'],
                    'is_target_parent':chunks[c['current_chunk_id']]['parent_passage_id']==q['target_passage_id'],
                    'historical_ce':c['historical_ce'],'current_ce_if_cached':c.get('current_ce')})
            target=next((c for c in candidates if c['is_target_parent']),None)
            heads[name]={'target_preselection_rank':target['preselection_rank'] if target else None,
                'target_historical_CE_rank':target['historical_CE_rank_within_ten'] if target else None,
                'target_historical_CE_score':target['historical_ce'] if target else None,
                'saved_structural':pred['conditions'][name+'_historical_CE']['structural'],
                'candidates':candidates,'final4_current_chunk_ids':actual}
        gold=next(cc[i] for i in saved['pools']['Full_LM18']
            if chunks[cc[i]['current_chunk_id']]['parent_passage_id']==q['target_passage_id'])
        old_top=cc[saved['pools']['Original_LM10'][historical_order(cc,saved['pools']['Original_LM10'])[0]]]
        rrf_top=cc[saved['pools']['RRF'][historical_order(cc,saved['pools']['RRF'])[0]]]
        old=next((c for c in heads['Original_LM10']['candidates'] if c['is_target_parent']),None)
        same_pair=old is not None and all(old[k]==gold[k] for k in ('current_chunk_id','historical_chunk_id'))
        explanation=('Target parent is absent from Original_LM10 ten but present in Full_LM18 ten: the observable difference is address preselection.' if old is None else
            'Both heads retain the identical target current/historical pair; its CE score is unchanged. Different competing ten-candidate sets or CE tie order change its final rank.' if same_pair else
            'Both pools retain the target parent but use different representative chunk pairs; inspect those IDs before attributing the final-rank change.')
        cases.append({'case_id':'SCALE-ILLUSTRATIVE-'+str(len(cases)+1),
            'query':{k:q[k] for k in ('query_id','query_text','source_quote_text','query_variant','family_id','stratum',
                'source_asof','target_asof','target_status','target_key','target_passage_id','target_revision_id') if k in q},
            'quote_absent':True,'delta_full_minus_original_LM_historical_CE':improvements[qid]['delta'],
            'heads':heads,'target_text_views':text_pair(gold,chunks),
            'target_features':{k:v for k,v in gold['features'].items() if k!='available_target'},
            'original_LM_and_full_target_pair_identical':same_pair,
            'original_LM_top1_text_views':text_pair(old_top,chunks),'RRF_top1_text_views':text_pair(rrf_top,chunks),
            'observed_stage_difference':explanation,
            'interpretation_boundary':'Illustrative post-hoc retrieval case, not a causal attribution to one feature or an answer/human evaluation.'})
    result={'complete':True,'job_id':os.environ.get('PBS_JOBID'),'seed':42,'cases':cases,
        'eligible_generated_quote_absent_improvements':len(improvements),'selected_cases':len(cases),
        'selection':'positive Full18-history-CE versus originalLM10-history-CE MRR difference; prefer pool/recall recovery and different families/addresses; stable query-ID tie break',
        'source_role':'original TEST regression at1000-law scale; qualitative illustration only',
        'sources':{'predictions':str(args.input/'predictions.jsonl'),'scores':str(args.input/'scored.jsonl'),'chunks':str(args.chunks)},
        'training_runs':0,'model_forward_calls':0,'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(args.output,result);return result


def main_mechanism(args):
    """Add one same-RRF-ten text-view intervention to the two LM cases."""
    candidates={}
    for r in rows(args.input/'predictions.jsonl'):
        if r['query_variant']!='generated_paraphrase' or r['quote_absent'] is not True or r['target_status']!='available':continue
        h=r['conditions']['common_Historical_CE']['structural'];c=r['conditions']['common_Current_CE']['structural']
        if h['target_mrr']>c['target_mrr']:
            candidates[r['query_id']]={'prediction':r,'delta':h['target_mrr']-c['target_mrr'],
                'recall_gain':int(h['target_recall_at_four'])-int(c['target_recall_at_four'])}
    queries={q['query_id']:q for p in args.queries for q in rows(p) if q['query_id'] in candidates}
    scored={};priority=[]
    for r in rows(args.input/'scored.jsonl'):
        qid=r['query_id']
        if qid not in candidates:continue
        q=queries[qid];address=(q['target_key'][0],q['target_key'][2],q['target_key'][3])
        target=next((r['candidates'][i] for i in r['pools']['RRF'] if tuple(r['candidates'][i]['address'])==address),None)
        if target is None or target['historical_chunk_id']==target['current_chunk_id']:continue
        f=target['features'];priority.append(((-candidates[qid]['recall_gain'],-candidates[qid]['delta'],
            -int(f['query_numeric_changed']>0),-f['query_terms_lost'],qid),qid))
        scored[qid]=r
    shortlist=[qid for key,qid in sorted(priority)[:20]]
    needed={scored[qid]['candidates'][i][view] for qid in shortlist for i in scored[qid]['pools']['RRF']
            for view in ('historical_chunk_id','current_chunk_id')}
    chunks={c['chunk_id']:c for c in rows(args.chunks) if c['chunk_id'] in needed}
    chosen=None
    for qid in shortlist:
        q=queries[qid];r=scored[qid]
        target=next((r['candidates'][i] for i in r['pools']['RRF']
            if chunks[r['candidates'][i]['current_chunk_id']]['parent_passage_id']==q['target_passage_id']),None)
        if target is None:continue
        pair=text_pair(target,chunks)
        if args.same_caption:
            historical=chunks[target['historical_chunk_id']];current=chunks[target['current_chunk_id']]
            if not historical.get('caption') or historical.get('caption')!=current.get('caption'):continue
        if not pair['same_revision'] and not pair['same_text']:
            chosen=qid;break
    if chosen is None:raise ValueError('No substantive different-revision target in the bounded mechanism shortlist')
    q=queries[chosen];r=scored[chosen];cc=r['candidates'];pool=r['pools']['RRF'];pred=candidates[chosen]['prediction']
    orders={view:historical_order(cc,pool,view+'_ce') for view in ('historical','current')}
    for view,method in [('historical','common_Historical_CE'),('current','common_Current_CE')]:
        actual=[cc[pool[i]]['current_chunk_id'] for i in orders[view][:4]]
        if actual!=pred['conditions'][method]['evidence_chunk_ids']:raise ValueError('Same-ten view ordering changed')
    target=next(cc[i] for i in pool if chunks[cc[i]['current_chunk_id']]['parent_passage_id']==q['target_passage_id'])
    common=[]
    for rank,i in enumerate(pool,1):
        c=cc[i];common.append({'address':c['address'],'preselection_rank':rank,
            'historical_chunk_id':c['historical_chunk_id'],'current_chunk_id':c['current_chunk_id'],
            'is_target_parent':chunks[c['current_chunk_id']]['parent_passage_id']==q['target_passage_id'],
            'historical_ce':c['historical_ce'],'current_ce':c['current_ce'],
            'historical_CE_rank':orders['historical'].index(rank-1)+1,'current_CE_rank':orders['current'].index(rank-1)+1})
    case={'case_id':'SCALE-MAIN-MECHANISM-3','query':{k:q[k] for k in ('query_id','query_text','source_quote_text',
        'query_variant','family_id','stratum','source_asof','target_asof','target_status','target_key','target_passage_id','target_revision_id') if k in q},
        'quote_absent':True,'contrast':'historical CE versus current CE inside identical RRF ten',
        'target_text_views':text_pair(target,chunks),'target_scores_and_ranks':next(c for c in common if c['is_target_parent']),
        'same_common_ten':common,'saved_conditions':{name:pred['conditions'][name] for name in ('common_Historical_CE','common_Current_CE','Full_LM18_historical_CE')},
        'current_CE_top1_text_views':text_pair(cc[pool[orders['current'][0]]],chunks),
        'observed_stage_difference':'Only the CE input view changes inside the identical RRF ten. Both orders deliver their selected fixed current chunks; no address preselection or date-binding change is involved.',
        'interpretation_boundary':'Post-hoc illustrative retrieval case, not an answer/human evaluation or a causal attribution to one amendment token.'}
    base=read(args.base_cases or args.output)
    if base['selected_cases']!=2:raise ValueError('Retain the two previously extracted LM comparison cases')
    write_json(args.output.with_name('case-records-lm-comparison.json'),base)
    base['cases'].append(case);base.update(selected_cases=3,
        main_mechanism_eligible_improvements=len(candidates),main_mechanism_shortlist=len(shortlist),
        main_mechanism_same_caption_required=args.same_caption,
        main_mechanism_job=os.environ.get('PBS_JOBID'),
        third_case_selection='same RRF ten; generated quote-absent; prioritize recall/MRR gain then changed numerical query terms and lost query terms; require real differing revision/text')
    write_json(args.output,base);return base


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path('runs/hidate/revision-aware/transfer/ruri-scale-final')
    p.add_argument('--input',type=Path,default=root);p.add_argument('--output',type=Path,default=root/'case-records.json')
    p.add_argument('--chunks',type=Path,default=Path('data/index/hidate/extended-benchmark/scale/chunks.jsonl'))
    p.add_argument('--queries',nargs='+',type=Path,default=[Path('data/processed/hidate/queries-exact.jsonl'),Path('data/processed/hidate/queries-paraphrase.jsonl')])
    p.add_argument('--mode',choices=('lm','main-mechanism'),default='lm')
    p.add_argument('--same-caption',action='store_true')
    p.add_argument('--base-cases',type=Path)
    args=p.parse_args()
    d=(extract if args.mode=='lm' else main_mechanism)(args)
    print(json.dumps({'complete':d['complete'],'eligible':d['eligible_generated_quote_absent_improvements'],'cases':d['selected_cases']}))


if __name__=='__main__':main()
