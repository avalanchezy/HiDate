"""Frozen Japanese-Qwen address heads on the existing French FiscalQA cache.

The interval-adapted TimelyRAG uses paper Eq.3 distances and the official
repository's auto-alpha/gate/margin recipe. QIT/DIT remain unknown; no auxiliary
time term is enabled. Native source-version and shared-binding outputs stay
separate. No French labels enter feature construction or candidate selection.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate import fiscal as fiscal
from hidate.revision import normalize_scores, revision_features
from hidate.retrieval import rrf
from hidate.timelyrag import ALPHA_GRID, DAY_GAP_CAP, DAY_GRANULARITY_SIGNAL


BASE_FEATURES = ('max_history_rrf','mean_history_rrf','distinct_revision_count',
    'history_best_rank_rrf','history_best_rank_dense','history_best_rank_bm25',
    'current_best_cosine','max_lexical_overlap_current','available_target',
    'history_latest_anchor_gap_years')
TREE_HEADS = ('Original_LM10','Base_LM9','Dual_relevance_LM11',
              'Full_LM18','Lean_noalignment_LM16')
TIMELY_SOURCES = [
    'https://arxiv.org/html/2609.11572v1#S3.SS3',
    'https://github.com/kaist-dmlab/TimelyRAG/blob/main/time_scores.py',
    'https://github.com/kaist-dmlab/TimelyRAG/blob/main/alpha.py',
    'https://github.com/kaist-dmlab/TimelyRAG/blob/main/retrieval.py',
    'https://github.com/kaist-dmlab/TimelyRAG/blob/main/time_utils.py',
    'https://github.com/kaist-dmlab/TimelyRAG/blob/main/config.py']


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_row(stream, row):
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def receipt(**kw):
    return {'seed':42,'job_id':os.environ.get('PBS_JOBID'),
        'at_utc':datetime.now(timezone.utc).isoformat(),'training_runs':0,
        'french_fit_runs':0,'french_dev_updates':0,'neural_model_calls':0,
        'encoder_calls':0,'reranker_calls':0,'reader_calls':0,**kw}


def enrich_fiscal_addresses(question, saved, chunks, document_tokens=None):
    """Use all saved addresses, including missing current versions, for ranks."""
    addresses=saved['addresses']
    historical=normalize_scores([a['features']['max_history_rrf'] for a in addresses])
    current=normalize_scores([a['features']['current_best_cosine'] for a in addresses])
    terms=fiscal.french_tokens(question['prompt'])
    tokens=document_tokens if document_tokens is not None else {
        cid:fiscal.french_tokens(chunk['text']) for cid,chunk in chunks.items()}
    result=[]
    for i,a in enumerate(addresses):
        hid=a['history_chunk_ids'][0] if a['history_chunk_ids'] else None
        cid=a['current_best_chunk_id']
        if cid is not None:
            c=chunks[cid]
            if c['article_cid']!=a['article_cid'] or c['version_id']!=a['bound_version_id']:
                raise ValueError('Saved current chunk differs from its CID/date-bound version')
            if not (c['date_debut']<=question['date_anchor'] and
                    (c['date_fin'] is None or question['date_anchor']<=c['date_fin'])):
                raise ValueError('Saved current representative is outside its effective interval')
        for historical_id in a['history_chunk_ids']:
            c=chunks[historical_id]
            if c['article_cid']!=a['article_cid'] or c['date_debut']>question['date_anchor']:
                raise ValueError('Saved historical hit violates CID or history eligibility')
        f=revision_features(terms,tokens.get(hid,[]),tokens[cid] if cid is not None else None)
        current_normalized=float(current[i]) if cid is not None else float(historical[i])
        f['query_score_disagreement']=float(historical[i])-current_normalized
        result.append({**a,'address_key':[fiscal.CGI,a['article_cid']],
            'historical_chunk_id':hid,'current_chunk_id':cid,
            'binding_available':a['bound_version_id'] is not None,
            'deliverable':cid is not None,'historical_normalized':float(historical[i]),
            'current_normalized':current_normalized,'features':{**a['features'],**f}})
    return result


def candidate(chunk, score, rank, historical_id=None, address_index=None):
    return {'article_cid':chunk['article_cid'],'version_id':chunk['version_id'],
        'chunk_id':chunk['chunk_id'],'current_chunk_id':chunk['chunk_id'],
        'historical_chunk_id':historical_id,'native_score':float(score),
        'selection_rank':rank,'address_index':address_index}


def delivered(candidates, pool=None):
    return {'candidates':candidates,'evidence':candidates[:4],
        'pool_chunk_ids':pool if pool is not None else [c['chunk_id'] for c in candidates],
        'support_policy':'none; native structural ranking',
        'candidate_budget':10,'evidence_budget':4}


def rank_fiscal_heads(addresses, tree_scores, chunks):
    """Raw-margin own ten and reordered common RRF ten, without fusion."""
    score_map={'RRF_native':[a['features']['max_history_rrf'] for a in addresses],**tree_scores}
    rrf_order=sorted(range(len(addresses)),key=lambda i:
        (-float(score_map['RRF_native'][i]),addresses[i]['article_cid']))

    def select(order,scores):
        selected,seen=[],set()
        for i in order:
            a=addresses[i]; cid=a['current_chunk_id']
            if cid is None: continue
            chunk=chunks[cid]
            if chunk['version_id'] in seen: continue
            seen.add(chunk['version_id'])
            selected.append(candidate(chunk,scores[i],len(selected)+1,a['historical_chunk_id'],i))
            if len(selected)==10: break
        return selected

    common=select(rrf_order,score_map['RRF_native'])
    common_indices=[c['address_index'] for c in common]
    output={}
    for name,scores in score_map.items():
        if len(scores)!=len(addresses): raise ValueError('One native score per full address is required')
        order=sorted(range(len(addresses)),key=lambda i:(-float(scores[i]),addresses[i]['article_cid']))
        fixed_order=sorted(common_indices,key=lambda i:(-float(scores[i]),addresses[i]['article_cid']))
        output[name]={'score_semantics':'raw historical RRF' if name=='RRF_native' else 'raw tree margin',
            'full_address_scores':[float(v) for v in scores],
            'own_pool':delivered(select(order,scores)),
            'common_rrf10':delivered(select(fixed_order,scores),[c['chunk_id'] for c in common])}
    return output


def history_dense_hits(scores, ordered_ids, chunks, anchor):
    eligible=[i for i,cid in enumerate(ordered_ids) if chunks[cid]['date_debut']<=anchor]
    eligible.sort(key=lambda i:(-float(scores[i]),ordered_ids[i]))
    return [{'chunk_id':ordered_ids[i],'score':float(scores[i])} for i in eligible[:50]]


def direct_asof_condition(hits,chunks):
    candidates,seen=[],set()
    for hit in hits:
        chunk=chunks[hit['chunk_id']]
        if chunk['version_id'] in seen: continue
        seen.add(chunk['version_id'])
        candidates.append(candidate(chunk,hit['score'],len(candidates)+1,chunk['chunk_id']))
        if len(candidates)==10: break
    return {'score_semantics':'raw as-of hybrid RRF','own_pool':delivered(candidates)}


def temporal_bound_addresses(hits,chunks,selected_versions,by_version,cosine,row_index):
    """Bind every discovered dense CID, independently of the RRF address pool."""
    output=[]
    for cid in sorted({chunks[h['chunk_id']]['article_cid'] for h in hits}):
        version=selected_versions.get(cid)
        candidates=by_version.get(version['id'],[]) if version else []
        best=min(candidates,key=lambda k:(-float(cosine[row_index[k]]),k)) if candidates else None
        output.append({'article_cid':cid,'current_chunk_id':best})
    return output


def score_interval_pool(base_scores,intervals,anchor):
    """Paper interval distance; official auto-alpha defaults, unknown QIT/DIT.

    Official implementation has point-only DET. The only temporal-interface
    extension is interval distance, with gap_avg computed from interval starts.
    Day signal=2/3, explicit day GAP_MAX=60 (corrects upstream ID mapping).
    """
    target=date.fromisoformat(anchor)
    starts=[date.fromisoformat(s) if s is not None else None for s,e in intervals]
    ends=[date.fromisoformat(e) if e is not None else None for s,e in intervals]
    delta=np.array([max(0,(s-target).days,(target-e).days if e is not None else 0)
        if s is not None else np.nan for s,e in zip(starts,ends)],dtype=float)
    raw=np.asarray(base_scores,dtype=float)
    if len(raw)!=len(delta): raise ValueError('Interval and dense scores must align')
    base=(raw-raw.min())/(np.ptp(raw)+1e-9) if len(raw) else raw.copy()
    finite=np.isfinite(delta); normalized=delta.copy()
    if finite.any():
        spread=np.ptp(delta[finite]); normalized[finite]=(delta[finite]-delta[finite].min())/spread if spread else 0.
    affinity=np.full(len(raw),.5); affinity[finite]=np.exp(-2.*normalized[finite])
    penalty=np.full(len(raw),1.05)
    if finite.any():
        order=np.argsort(normalized[finite],kind='stable'); ranks=np.empty(finite.sum())
        ranks[order]=np.arange(finite.sum())
        penalty[finite]=ranks/(finite.sum()-1) if finite.sum()>1 else 0.
        penalty=np.clip(penalty+1e-6*((np.arange(len(raw))%997)/997.),0.,1.05)
    known=float(finite.mean()) if len(raw) else 0.
    ordered_starts=sorted(s for s in starts if s is not None)
    gap=float(np.mean([(b-a).days for a,b in zip(ordered_starts[:-1],ordered_starts[1:])])) if len(ordered_starts)>1 else 0.
    text_std=float(np.clip(np.std(base)/.35,0.,1.)) if len(base) else 0.
    time_std=float(np.clip(np.std(penalty)/.35,0.,1.)) if len(penalty) else 0.
    valid_mean=known/4.  # only QET_DET is observed, not QET_DIT.
    importance=float(np.clip(.32*np.clip(gap/DAY_GAP_CAP,0.,1.)+.18+
        .08*DAY_GRANULARITY_SIGNAL+.22*time_std+.18*valid_mean+.02*known-.20*text_std,0.,1.))
    alpha=min(ALPHA_GRID,key=lambda v:abs(v-(1.-importance)))
    gate=.35+.10*DAY_GRANULARITY_SIGNAL+.35*valid_mean+.20*time_std
    if time_std<.10: gate*=.25
    gate=float(np.clip(gate,0.,1.))
    margin=np.clip(1.-(base.max()-base)/max(1e-6,float(np.ptp(base))),0.,1.) if len(base) else base.copy()
    weights=(1.-alpha)*gate*margin
    final=(1.-weights)*base+weights*affinity
    return {'method':'TimelyRAG-interval (adapted)','qet':anchor,'qit':None,'dit':None,
        'date_source':'released_legal_effective_interval','intervals':[list(v) for v in intervals],
        'distance_days':[float(v) if np.isfinite(v) else None for v in delta],
        'time_affinity':affinity.tolist(),'semantic_scores':base.tolist(),'final_scores':final.tolist(),
        'alpha_semantic':float(alpha),'gate':gate,'gap_avg_days':gap,
        'valid_ratios':{'QET_DET':known,'QIT_DIT':0.,'QIT_DET':0.,'QET_DIT':0.},
        'signals':{'text_std_normalized':text_std,'time_std_normalized':time_std,
                   'valid_mean':valid_mean,'time_importance':importance},
        'granularity':'day','day_gap_cap':DAY_GAP_CAP,'auxiliary_terms_used':False}


def timely_interval_conditions(hits,chunks,anchor,addresses):
    hits=hits[:50]
    scored=score_interval_pool([h['score'] for h in hits],
        [(chunks[h['chunk_id']]['date_debut'],chunks[h['chunk_id']]['date_fin']) for h in hits],anchor)
    source=[{**h,'timelyrag_score':s,'original_candidate_rank':i+1}
        for i,(h,s) in enumerate(zip(hits,scored['final_scores']))]
    source.sort(key=lambda h:(-h['timelyrag_score'],h['original_candidate_rank'],h['chunk_id']))
    native,seen=[],set()
    for hit in source:
        chunk=chunks[hit['chunk_id']]
        if chunk['version_id'] in seen: continue
        seen.add(chunk['version_id']);native.append(candidate(chunk,hit['timelyrag_score'],len(native)+1,chunk['chunk_id']))
        if len(native)==10: break
    by_cid={a['article_cid']:a for a in addresses}
    bound,seen=[],set()
    for hit in source:
        original=chunks[hit['chunk_id']]; cid=original['article_cid']
        if cid in seen: continue
        a=by_cid.get(cid)
        if a is None or a['current_chunk_id'] is None: continue
        seen.add(cid)
        bound.append(candidate(chunks[a['current_chunk_id']],hit['timelyrag_score'],len(bound)+1,hit['chunk_id']))
        if len(bound)==10: break
    return {'source_hits':source,'scoring':scored,'conditions':{
        'TimelyRAG_interval_native':{'score_semantics':'official auto-alpha interval-adapted source ranking',
            'own_pool':delivered(native)},
        'TimelyRAG_interval_bound':{'score_semantics':'same temporal source ranking plus shared date binding',
            'own_pool':delivered(bound)}}}


def prepare_ce_pairs(question,conditions,chunks,cache):
    """Deduplicate exact raw query/document pairs across heads, scopes and views."""
    pending={}
    for name,condition in conditions.items():
        for scope in ('own_pool','common_rrf10'):
            if scope not in condition: continue
            for c in condition[scope]['candidates']:
                delivered_view='source' if name=='TimelyRAG_interval_native' else 'current'
                for view,chunk_id in ((delivered_view,c['chunk_id']),('historical',c.get('historical_chunk_id'))):
                    if chunk_id is None: continue
                    key=(question['prompt'],chunks[chunk_id]['text'])
                    if key not in pending:
                        pending[key]={'query_text':key[0],'document_text':key[1],
                            'views':[],'uses':[],'cached_score':cache.get(key)}
                    row=pending[key]
                    if view not in row['views']: row['views'].append(view)
                    row['uses'].append({'method':name,'candidate_scope':scope,'view':view,
                        'article_cid':c['article_cid'],'chunk_id':chunk_id})
    return list(pending.values())


def load_pair_cache(path):
    cache={}
    for receipt_row in rows(path):
        pairs,result=receipt_row['pairs'],receipt_row['result']
        if (result['sdpa_backend'],result['score_format'],result['logit_format']) != (
                'default','conditional_yes_no','yes_minus_no'):
            raise ValueError('Reuse original Qwen French CE template/backend and score semantics')
        if not (len(pairs)==len(result['scores'])==len(result['logits'])==len(result['token_counts'])):
            raise ValueError('French CE receipt pairs and actual scores differ')
        for pair,score,logit,tokens in zip(pairs,result['scores'],result['logits'],result['token_counts']):
            cache[tuple(pair)]={'reranker_score':score,'reranker_logit':logit,'pair_tokens':tokens}
    return cache


def load_trees(args):
    from hidate.lambdamart import lightgbm_module
    original=read(args.original_tree.parent/'fit.json')
    fit=read(args.development/'models/fit.json')
    parts=read(args.development/'models/component-controls.json')
    if not all(r['complete'] and r['seed']==42 for r in (original,fit,parts)):
        raise ValueError('Use completed seed42 Japanese Qwen TRAIN fits')
    if original['validation_used'] or fit['validation_used_for_fitting'] or parts['validation_used_for_fitting']:
        raise ValueError('No VALID or French fitting')
    if original['feature_names']!=list(BASE_FEATURES):
        raise ValueError('Original Qwen ten-feature order differs')
    models=args.development/'models'
    specs={'Original_LM10':(args.original_tree,original['feature_names'])}
    for name,key,source in [('Base_LM9','lambdamart_base',fit),
        ('Dual_relevance_LM11','lambdamart_dual_relevance',parts),
        ('Full_LM18','lambdamart_revision',fit),('Lean_noalignment_LM16','lambdamart_no_alignment',parts)]:
        specs[name]=(models/(key+'.txt'),source['fits'][key]['feature_names'])
    if [len(names) for path,names in specs.values()]!=[10,9,11,18,16]:
        raise ValueError('Frozen Japanese transfer head counts differ')
    lgb=lightgbm_module();trees={}
    for name,(path,names) in specs.items():
        tree=lgb.Booster(model_file=str(path))
        if tree.feature_name()!=names: raise ValueError('Model feature order differs from fit receipt')
        trees[name]=tree
    return trees,{k:{'path':str(p),'feature_names':names} for k,(p,names) in specs.items()}


def prepare(args):
    started=time.perf_counter();out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=read(args.protocol);old=read(args.retrieved/'retrieval.json');encoding=read(args.encoded/'encoding.json')
    if not old['complete'] or not encoding['complete']: raise ValueError('Complete original Fiscal inputs required')
    articles=list(rows(args.prepared/'articles.jsonl'));chunk_list=list(rows(args.prepared/'chunks.jsonl'))
    chunks={c['chunk_id']:c for c in chunk_list};questions=list(rows(args.prepared/'questions.jsonl'))
    if len(questions)!=221 or sum(q['scope']=='in' for q in questions)!=209:
        raise ValueError('Keep all published221 questions and original scored209 scope')
    docs=np.load(args.encoded/'documents.npy',mmap_mode='r');qvecs=np.load(args.encoded/'queries.npy',mmap_mode='r')
    if len(docs)!=len(chunk_list) or len(qvecs)!=len(questions): raise ValueError('Saved row-order vector contract differs')
    if old['embedding_pin']!=encoding['embedding_pin']: raise ValueError('Original French discovery encoder context differs')
    stack=read(args.stack);pin=deepcopy(stack['models']['reranker'])
    pin.setdefault('conventions',{})['instruction']=protocol['retrieval_instruction']
    if old['reranker_pin']!=pin: raise ValueError('French exact CE cache requires the original full effective pin')
    trees,definitions=load_trees(args)
    source={r['qid']:r for r in rows(args.retrieved/'retrieval.jsonl')}
    if set(source)!={q['qid'] for q in questions}: raise ValueError('Original full-address cache must cover221 questions')
    cache=load_pair_cache(args.retrieved/'reranker-receipts.jsonl')
    engine=fiscal.FiscalDiscovery(articles,chunk_list,docs,top_k=50,rrf_k=60,static_date=protocol['static_snapshot_date'])
    metadata=receipt(complete=False,questions=221,scored_questions=209,completed_questions=0,
        models=definitions,original_retrieval=str(args.retrieved/'retrieval.json'),
        source_chunks=str(args.prepared/'chunks.jsonl'),source_questions=str(args.prepared/'questions.jsonl'),
        original_conditions_preserved=old['conditions'],reranker_pin=pin,model_max_length=protocol['model_max_length'],
        pair_cache_source=str(args.retrieved/'reranker-receipts.jsonl'),
        tokenizer='existing French NFKC/NFKD accent removal + lowercase ASCII word/number tokens',
        historical_representative='first saved history_chunk_ids entry, original highest-RRF order',
        normalization='all saved discovered addresses before filtering missing versions or taking ten',
        score_policy='raw native margin/RRF; no CE, fusion, support tuning or French selection',
        timelyrag={'name':'TimelyRAG-interval (adapted)','official_sources':TIMELY_SOURCES,
            'verified_date':'2026-10-19','paper_distance':'Eq3 zero inside inclusive effective interval, otherwise nearest boundary',
            'normalization':'candidate-pool min-max','gamma':2.,'auxiliary_time_terms':'not used: QIT and DIT unknown',
            'alpha':'official repository auto-alpha grid 0,.1,...,1; semantic coefficient, not paper Eq7 sigmoid',
            'auto_alpha_coefficients':{'gap_normalized':.32,'qet_exists':.18,'granularity_signal':.08,
                'penalty_std_normalized':.22,'valid_mean':.18,'finite_ratio':.02,'text_std_normalized':-.20},
            'gate_and_margin':'official repository constants .35,.10,.35,.20; std reference .35; margin power1',
            'other_defaults':{'missing_margin':.05,'base_epsilon':1e-9,'distance_tie_perturbation':1e-6,
                'day_granularity_signal':DAY_GRANULARITY_SIGNAL,'day_gap_cap':DAY_GAP_CAP,
                'gate_low_variance_cutoff':.10,'gate_low_variance_factor':.25},
            'interval_gap':'mean adjacent effective-start gaps; no insertion-date proxy',
            'day_mapping':'explicit day cap60; upstream reversed granularity-ID mapping corrected',
            'candidate_pool':'history dense top50 from existing Qwen French vectors',
            'source_and_binding':'native source versions; separate shared-binding current dense-best output'},
        timing_definition={'feature_seconds':'full-pool French revision features and normalization',
            'tree_seconds':'five frozen tree predictions',
            'retrieval_seconds':'existing-vector dense product and asof BM25/RRF; no neural calls',
            'timely_seconds':'interval reranking and native/shared-binding materialization'},
        outputs={'features':'addresses.jsonl','predictions':'predictions.jsonl','ce_pairs':'ce-pairs.jsonl'})
    write_json(out/'preparation.json',metadata)
    counts,timing=Counter(),Counter()
    with (out/'addresses.jsonl').open('w',encoding='utf-8') as features_out, \
         (out/'predictions.jsonl').open('w',encoding='utf-8') as predictions_out, \
         (out/'ce-pairs.jsonl').open('w',encoding='utf-8') as pairs_out:
        for index,q in enumerate(questions):
            saved=source[q['qid']]
            if saved['date_anchor']!=q['date_anchor'] or saved['scope']!=q['scope']:
                raise ValueError('Saved French question/date scope differs')
            t=time.perf_counter();addresses=enrich_fiscal_addresses(q,saved,chunks,engine.tokens)
            timing['feature_seconds']+=time.perf_counter()-t
            t=time.perf_counter();scores={}
            for name,tree in trees.items():
                matrix=np.asarray([[a['features'][f] for f in definitions[name]['feature_names']] for a in addresses])
                scores[name]=tree.predict(matrix,raw_score=True,num_threads=4) if len(matrix) else []
            timing['tree_seconds']+=time.perf_counter()-t
            conditions=rank_fiscal_heads(addresses,scores,chunks)
            t=time.perf_counter();cosine=docs@np.asarray(qvecs[index])
            dense=history_dense_hits(cosine,engine.ids,chunks,q['date_anchor'])
            selected=fiscal.select_versions(articles,q['date_anchor']);valid={a['id'] for a in selected.values()}
            lexical=engine.lexical_scores(fiscal.french_tokens(q['prompt']))
            asof=rrf([engine.rank(valid,lexical),engine.rank(valid,cosine)],k=60,top_k=50)
            conditions['AsOf_hybrid_native']=direct_asof_condition(asof,chunks)
            timing['retrieval_seconds']+=time.perf_counter()-t
            t=time.perf_counter()
            temporal_addresses=temporal_bound_addresses(dense,chunks,selected,
                engine.by_version,cosine,engine.row_index)
            temporal=timely_interval_conditions(dense,chunks,q['date_anchor'],temporal_addresses)
            conditions.update(temporal['conditions']);timing['timely_seconds']+=time.perf_counter()-t
            identity={k:q[k] for k in ('qid','date_anchor','scope')}
            save_row(features_out,{**identity,'query_text':q['prompt'],'addresses':addresses})
            save_row(predictions_out,{**identity,'conditions':conditions,'timelyrag':temporal,
                'original_conditions':saved['conditions'],'original_conditions_unchanged':True})
            pairs=prepare_ce_pairs(q,conditions,chunks,cache)
            for pair_index,pair in enumerate(pairs,1):
                save_row(pairs_out,{'qid':q['qid'],'pair_id':f"{q['qid']}|P{pair_index:04d}",**pair})
                counts['exact_query_document_pairs']+=1
                counts['cached_exact_pairs']+=pair['cached_score'] is not None
                counts['missing_exact_pairs']+=pair['cached_score'] is None
                counts['logical_view_uses']+=len(pair['uses'])
            counts['questions']+=1;counts['addresses']+=len(addresses)
            counts['condition_records']+=len(conditions)
            counts['missing_current_addresses']+=sum(a['current_chunk_id'] is None for a in addresses)
            if counts['questions']%20==0:
                print(json.dumps({'phase':'prepare','questions':counts['questions'],'expected':221}),flush=True)
    metadata.update(complete=counts['questions']==221,completed_questions=counts['questions'],counts=dict(counts),
        timing=dict(timing),wall_seconds=time.perf_counter()-started)
    write_json(out/'preparation.json',metadata)
    return metadata


def native(args):
    """Write structural gold evaluation after candidate freezing; no reader scores."""
    metadata=read(args.output/'preparation.json')
    if not metadata['complete']: raise ValueError('Complete frozen French predictions first')
    questions={q['qid']:q for q in rows(args.prepared/'questions.jsonl')}
    groups=defaultdict(Counter);seen=set();count=0
    with (args.output/'structural.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows(args.output/'predictions.jsonl'):
            q=questions[row['qid']]
            if q['qid'] in seen: raise ValueError('Duplicate Fiscal prediction')
            seen.add(q['qid']);gold=q['gold']
            for name,condition in row['conditions'].items():
                for scope in ('own_pool','common_rrf10'):
                    if scope not in condition: continue
                    e=condition[scope]['evidence'];top=e[0] if e else None
                    ar=next((i for i,c in enumerate(e,1) if c['article_cid']==gold['article_cid']),None)
                    vr=next((i for i,c in enumerate(e,1) if c['version_id']==gold['version_id']),None)
                    metrics={'article_recall_at4':int(ar is not None),'version_recall_at4':int(vr is not None),
                        'article_mrr_at4':1./ar if ar else 0.,'version_mrr_at4':1./vr if vr else 0.,
                        'top_article_correct':int(bool(top) and top['article_cid']==gold['article_cid']),
                        'top_version_correct':int(bool(top) and top['version_id']==gold['version_id'])}
                    save_row(stream,{'qid':q['qid'],'scope':q['scope'],'condition':name,
                        'candidate_scope':scope,'evidence_chunk_ids':[c['chunk_id'] for c in e],**metrics})
                    for group in ('published221','scored209' if q['scope']=='in' else q['scope']):
                        g=groups[(group,name,scope)];g['n']+=1;g.update(metrics)
                    count+=1
    summary={}
    for (group,name,scope),values in groups.items():
        summary.setdefault(group,{})[name+'|'+scope]={k:(v if k=='n' else v/values['n']) for k,v in values.items()}
    result=receipt(complete=seen==set(questions),questions=len(seen),structural_records=count,
        preparation_path=str(args.output/'preparation.json'),panels=summary,
        evidence_stage='raw selected top4 before reader packing',
        answer_strict=None,answer_nuggets=None,answer_status='not run; original reader required',
        french_selection=False,original_conditions_preserved=metadata['original_conditions_preserved'])
    write_json(args.output/'evaluation.json',result)
    return {'complete':result['complete'],'questions':len(seen),'structural_records':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','native','all'),default='all')
    p.add_argument('--prepared',type=Path,default=Path('data/processed/fiscal'))
    p.add_argument('--encoded',type=Path,default=Path('data/index/fiscal'))
    p.add_argument('--retrieved',type=Path,default=Path('runs/hidate/fiscal-transfer/fiscal-retrieval'))
    p.add_argument('--protocol',type=Path,default=Path('configs/fiscal_protocol.json'))
    p.add_argument('--stack',type=Path,default=Path('models/qwen3/model-stack.json'))
    p.add_argument('--original-tree',type=Path,default=Path('runs/hidate/revision-aware/qwen-original-lambdamart/ranker/model.txt'))
    p.add_argument('--development',type=Path,default=Path('runs/hidate/revision-aware/qwen-native-development'))
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/transfer/fiscal-native'))
    args=p.parse_args()
    if args.phase in ('prepare','all'):
        result=prepare(args)
        print(json.dumps({'phase':'prepare','complete':result['complete'],'counts':result['counts']}),flush=True)
    if args.phase in ('native','all'): print(json.dumps({'phase':'native',**native(args)}),flush=True)


if __name__=='__main__': main()
