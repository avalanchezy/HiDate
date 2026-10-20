"""Actual French dual-view CE, fixed native/CE recipes and exact-prompt reader.

The Japanese-Qwen 13/20-feature CE heads are optional explicit frozen inputs.
No model is fitted or selected here. Reader packing and answer scoring call the
original FiscalQA helpers, with one compact record per unique physical prompt.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json
from hidate import fiscal as fiscal
from hidate.revision import gate_scores
from run_revision_fiscal import rows,read,save_row,receipt
from run_revision_ce_training import add_ce_features


READER_CONDITIONS=('RRF_native','RRF_historical_CE','Original_LM10_native',
    'Original_LM10_historical_CE','AsOf_hybrid_native','AsOf_hybrid_current_CE',
    'TimelyRAG_interval_native','TimelyRAG_interval_bound')


def check_context(meta,pin):
    saved=meta.get('reranker_pin') or meta.get('model_pin')
    if saved!=pin: raise ValueError('French CE requires the same complete pin/instruction context')
    if meta.get('max_length',meta.get('model_max_length',1024))!=1024:
        raise ValueError('Reuse the original1024-token CE contract')
    if meta.get('sdpa_backend','default')!='default':
        raise ValueError('Qwen CE requires the original default backend')


def validate_result(result,n):
    if (result['sdpa_backend'],result['score_format'],result['logit_format'])!=(
            'default','conditional_yes_no','yes_minus_no'):
        raise ValueError('Actual Qwen CE backend/score semantics changed')
    if not all(len(result[k])==n for k in ('scores','logits','token_counts')):
        raise ValueError('CE actual result arrays must match exact pairs')
    if not all(math.isfinite(v) for v in result['scores']+result['logits']):
        raise ValueError('Actual CE scores must be finite')
    if sum(result['token_counts'])!=result['total_tokens']:
        raise ValueError('Actual CE token accounting differs')


def pair_inputs(path):
    pairs=list(rows(path));refs={(r['qid'],r['pair_id']):r for r in pairs}
    if len(refs)!=len(pairs): raise ValueError('Exact pair references must be unique')
    return pairs,refs


def pair_cache(pairs,refs,output):
    cache={(r['query_text'],r['document_text']):r['cached_score'] for r in pairs if r['cached_score'] is not None}
    path=output/'new-ce-pair-scores.jsonl';work=Counter()
    if path.exists():
        for row in rows(path):
            result=row['result'];validate_result(result,len(row['pair_refs']))
            for i,ref in enumerate(row['pair_refs']):
                p=refs[(ref['qid'],ref['pair_id'])]
                cache[(p['query_text'],p['document_text'])]={'reranker_score':result['scores'][i],
                    'reranker_logit':result['logits'][i],'pair_tokens':result['token_counts'][i]}
            work.update(calls=1,pairs=len(row['pair_refs']),input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1],model_seconds=result['seconds'])
    return cache,work


def score_pair_file(source,output,pin,*,factory=None,batch_size=16):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'scoring.json').exists(): check_context(read(output/'scoring.json'),pin)
    pairs,refs=pair_inputs(source);cache,total=pair_cache(pairs,refs,output)
    pending={}
    for p in pairs:
        key=(p['query_text'],p['document_text'])
        if key not in cache: pending.setdefault(key,{'qid':p['qid'],'pair_id':p['pair_id']})
    todo=list(pending.items());model=None;this_job=Counter()
    meta=receipt(complete=False,reranker_pin=pin,max_length=1024,sdpa_backend='default',
        logical_pair_rows=len(pairs),unique_requested_pairs=len({(p['query_text'],p['document_text']) for p in pairs}),
        unique_missing_at_start=len(todo),input_pair_file=str(source),input_ID_arrays_saved=False,
        score_format='conditional_yes_no',logit_format='yes_minus_no')
    write_json(output/'scoring.json',meta)
    with (output/'new-ce-pair-scores.jsonl').open('a',encoding='utf-8') as stream:
        for offset in range(0,len(todo),batch_size):
            chosen=todo[offset:offset+batch_size]
            if model is None:
                if factory is None:
                    import torch
                    from hidate.models import create_reranker
                    torch.manual_seed(42);model=create_reranker(pin,device='cuda',max_length=1024)
                else: model=factory(pin)
            raw=model.score([key for key,ref in chosen],include_inputs=False)
            validate_result(raw,len(chosen))
            result={k:raw[k] for k in ('scores','logits','token_counts','total_tokens','input_shape',
                'seconds','sdpa_backend','score_format','logit_format')}
            save_row(stream,{'pair_refs':[ref for key,ref in chosen],'result':result});stream.flush()
            for i,(key,ref) in enumerate(chosen):
                cache[key]={'reranker_score':result['scores'][i],'reranker_logit':result['logits'][i],
                            'pair_tokens':result['token_counts'][i]}
            used=dict(calls=1,pairs=len(chosen),input_tokens=result['total_tokens'],
                padded_tokens=result['input_shape'][0]*result['input_shape'][1],model_seconds=result['seconds'])
            total.update(used);this_job.update(used)
            if this_job['calls']%10==0:
                print(json.dumps({'phase':'CE','physical_pairs':this_job['pairs'],'pending':len(todo)}),flush=True)
    with (output/'scored-pairs.jsonl').open('w',encoding='utf-8') as stream:
        for p in pairs:
            save_row(stream,{k:v for k,v in p.items() if k not in ('query_text','document_text','cached_score')} |
                {'score':cache[(p['query_text'],p['document_text'])]})
    meta.update(complete=all((p['query_text'],p['document_text']) in cache for p in pairs),
        new_physical_pairs_this_job=this_job['pairs'],new_physical_pairs_total=total['pairs'],
        work_this_job=dict(this_job),physical_work_total=dict(total),
        reranker_calls=total['calls'],neural_model_calls=total['calls'])
    write_json(output/'scoring.json',meta);return meta


def context(args):
    base=read(args.native/'preparation.json');protocol=read(args.protocol);stack=read(args.stack)
    if not base['complete']: raise ValueError('Complete frozen native French preparation first')
    pin=deepcopy(stack['models']['reranker'])
    pin.setdefault('conventions',{})['instruction']=protocol['retrieval_instruction']
    check_context(base,pin)
    return base,protocol,stack,pin


def score(args):
    base,protocol,stack,pin=context(args)
    return score_pair_file(args.native/'ce-pairs.jsonl',args.output,pin,batch_size=args.batch_size)


def ce_rerank(candidates,query,chunks,cache,view):
    ranked=[]
    for c in candidates:
        cid=c['historical_chunk_id'] if view=='historical' else c['chunk_id']
        actual=cache[(query,chunks[cid]['text'])]
        ranked.append({**c,'ce_view':view,'ce_document_chunk_id':cid,**actual})
    ranked.sort(key=lambda c:(-c['reranker_score'],c['selection_rank'],c['chunk_id']))
    return {'candidates':ranked,'evidence':ranked[:4],'candidate_budget':10,'evidence_budget':4,
        'score_semantics':'actual Qwen P(yes|yes,no); historical score delivers same bound current chunk'}


def optional_ce_models(path,pin):
    if path is None:return {},{}
    from hidate.lambdamart import lightgbm_module
    meta=read(path/'fit.json')
    if not meta['complete'] or meta['seed']!=42 or meta['validation_used_for_fitting']:
        raise ValueError('Use complete Japanese Qwen TRAIN-only CE fits')
    # The producing Qwen worker records actual stack context in its parent.
    scoring=read(path.parent/'scoring.json')
    training_pin=scoring.get('model_pin') or scoring.get('reranker_pin')
    if (not scoring['complete'] or any(training_pin[k]!=pin[k] for k in ('model_id','revision'))
            or scoring['sdpa_backend']!='default'):
        raise ValueError('French CE transfer requires Qwen-trained CE heads, not Ruri fits')
    trees={};models=meta['models']
    for name,model in models.items():
        if len(model['feature_names']) not in (13,20):raise ValueError('Use frozen13/20 CE feature definitions')
        if name.startswith('lambdamart'):
            trees[name]=lightgbm_module().Booster(model_file=str(path/(name+'.txt')))
            if trees[name].feature_name()!=model['feature_names']:raise ValueError('CE tree feature order differs')
    return models,trees


def rank(args):
    base,protocol,stack,pin=context(args)
    ce_source=getattr(args,'ce_source',None) or args.output
    scoring=read(ce_source/'scoring.json')
    check_context(scoring,pin)
    if not scoring['complete']:raise ValueError('All actual CE pairs are required before ranking')
    pairs,refs=pair_inputs(args.native/'ce-pairs.jsonl');cache,total=pair_cache(pairs,refs,ce_source)
    chunks={c['chunk_id']:c for c in rows(args.prepared/'chunks.jsonl')}
    questions={q['qid']:q for q in rows(args.prepared/'questions.jsonl')}
    addresses={r['qid']:r['addresses'] for r in rows(args.native/'addresses.jsonl')}
    models,trees=optional_ce_models(args.ce_models,pin);counts=Counter()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'ranked.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows(args.native/'predictions.jsonl'):
            q=questions[row['qid']];conditions={};all_scopes={}
            for name,record in row['conditions'].items():
                all_scopes[name]={}
                for scope in ('own_pool','common_rrf10'):
                    if scope not in record:continue
                    cc=record[scope]['candidates']
                    # Source-version methods have only their native output.
                    views=('current',) if name=='AsOf_hybrid_native' else ('historical','current')
                    results={'native':record[scope]}
                    if name=='TimelyRAG_interval_native':views=()
                    for view in views:results[view+'_CE']=ce_rerank(cc,q['prompt'],chunks,cache,view)
                    all_scopes[name][scope]=results
                own=all_scopes[name]['own_pool']
                if name=='RRF_native':
                    conditions['RRF_native']=own['native'];conditions['RRF_historical_CE']=own['historical_CE']
                elif name=='Original_LM10':
                    conditions['Original_LM10_native']=own['native']
                    conditions['Original_LM10_historical_CE']=own['historical_CE']
                elif name=='AsOf_hybrid_native':
                    conditions[name]=own['native'];conditions['AsOf_hybrid_current_CE']=own['current_CE']
                elif name.startswith('TimelyRAG_interval'):conditions[name]=own['native']
            if models:
                common=row['conditions']['RRF_native']['common_rrf10']['candidates']
                by_cid={a['article_cid']:a for a in addresses[q['qid']]}
                cc=[{**c,'features':by_cid[c['article_cid']]['features'],
                    'historical_ce':cache[(q['prompt'],chunks[c['historical_chunk_id']]['text'])],
                    'current_ce':cache[(q['prompt'],chunks[c['chunk_id']]['text'])]} for c in common]
                features,h,c=add_ce_features(cc)
                for name,model in models.items():
                    X=np.asarray([[f[k] for k in model['feature_names']] for f in features])
                    values=(np.asarray([]) if not cc else
                        trees[name].predict(X,raw_score=True,num_threads=4) if name in trees else gate_scores(model,X,h,c)[0])
                    order=sorted(range(len(cc)),key=lambda i:(-float(values[i]),cc[i]['selection_rank'],cc[i]['chunk_id']))
                    ranked=[{**cc[i],'frozen_ce_head_score':float(values[i])} for i in order]
                    conditions['Qwen_'+name]={'candidates':ranked,'evidence':ranked[:4],
                        'candidate_pool':'common_rrf10','score_semantics':'frozen Japanese Qwen TRAIN CE-conditioned ranking'}
            save_row(stream,{'qid':q['qid'],'date_anchor':q['date_anchor'],'scope':q['scope'],
                'all_head_scope_comparisons':all_scopes,'reader_conditions':conditions})
            counts['questions']+=1;counts['reader_condition_records']+=len(conditions)
    meta=receipt(complete=counts['questions']==base['questions'],counts=dict(counts),
        source_native=str(args.native),ce_scoring=str(ce_source/'scoring.json'),
        optional_ce_models=str(args.ce_models) if args.ce_models else None,
        reader_conditions=list(READER_CONDITIONS),method_selection=False,
        test_or_french_selection=False,reader_calls=0)
    write_json(args.output/'ranking.json',meta);return meta


def prompt_key(record):
    return json.dumps({k:record[k] for k in ('messages','reader_pin','decode','prompt_tokens')},
        ensure_ascii=False,sort_keys=True,separators=(',',':'))


def minimal_pin(pin):return {k:pin[k] for k in ('model_id','revision')}


def reader_dir(args):
    return getattr(args,'reader_output',None) or args.output/'reader'


def reader_evidence(row,name):
    if name in row['reader_conditions']:return row['reader_conditions'][name]['evidence']
    for suffix,stage in (('_historical_CE','historical_CE'),('_current_CE','current_CE'),('_native','native')):
        if name.endswith(suffix):
            return row['all_head_scope_comparisons'][name[:-len(suffix)]]['own_pool'][stage]['evidence']
    raise ValueError('Requested condition was not scored in the frozen own-ten comparison')


def compact_prompt_cache(directories,desired,pin,decode):
    """Reuse completed records, including the durable prefix of another run."""
    cached={}
    for directory in directories:
        directory=Path(directory);meta=read(directory/'preparation.json')
        if not meta['complete'] or meta['reader_pin']!=pin or meta['decode']!=decode:
            raise ValueError('Incremental compact reader cache has a different pin/decode')
        source_prompts={r['prompt_id']:r for r in rows(directory/'prompts.jsonl')}
        # A running producer can append a new final line after this snapshot.
        # Only newline-terminated, already durable records are cache entries.
        for raw in (directory/'physical-reader-results.jsonl').read_bytes().split(b'\n')[:-1]:
            if not raw:continue
            saved=json.loads(raw);source=source_prompts[saved['prompt_id']];key=prompt_key(source)
            if key not in desired:continue
            target=desired[key]
            if saved['result']['prompt_tokens']!=target['prompt_tokens']:
                raise ValueError('Compact cached actual prompt length differs from exact prepared prompt')
            cached.setdefault(target['prompt_id'],{'prompt_id':target['prompt_id'],
                'result':compact_reader_result(saved['result']),
                'provenance':{'kind':'compact_exact_prompt_reuse','path':str(directory),
                    'source_prompt_id':saved['prompt_id']}})
    return cached


def prepare_reader(args,*,tokenizer=None):
    ranking=read(args.output/'ranking.json');protocol=read(args.protocol);stack=read(args.stack)
    if not ranking['complete']:raise ValueError('Complete fixed CE/native rankings required')
    out=reader_dir(args);out.mkdir(parents=True,exist_ok=True)
    frozen=getattr(args,'frozen_recipe',None)
    if frozen is not None:
        recipe=read(frozen)
        if not recipe['complete'] or recipe['selection_split']!='original_valid':
            raise ValueError('Additional main-method readers require completed original-VALID freeze')
    if (out/'preparation.json').exists() and read(out/'preparation.json')['complete']:
        prior=read(out/'preparation.json')
        if prior['conditions']!=args.reader_conditions or prior['reader_pin']!=minimal_pin(stack['models']['reader']):
            raise ValueError('Retain completed reader inputs; use a new output for changed frozen conditions')
        return prior
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(stack['models']['reader']['local_path'],local_files_only=True)
    questions={q['qid']:q for q in rows(args.prepared/'questions.jsonl')}
    versions={a['id']:a for a in rows(args.prepared/'articles.jsonl')}
    pin=minimal_pin(stack['models']['reader']);decode={'do_sample':False,'max_new_tokens':192}
    if (protocol['reader_prompt_tokens'],protocol['reader_max_new_tokens'],protocol['reader_max_passages'])!=(4096,192,4):
        raise ValueError('Keep original4096/192/4 Fiscal reader protocol')
    prompts={};logical=[]
    for row in rows(args.output/'ranked.jsonl'):
        q=questions[row['qid']]
        for name in args.reader_conditions:
            chosen=reader_evidence(row,name)
            packed=fiscal.pack_reader(q['prompt'],chosen,versions,tokenizer,prompt_budget=4096,max_passages=4)
            record={'messages':packed['messages'],'reader_pin':pin,'decode':decode,'prompt_tokens':packed['prompt_tokens']}
            key=prompt_key(record)
            if key not in prompts:prompts[key]={'prompt_id':f'FP{len(prompts)+1:05d}',**record}
            logical.append({'qid':q['qid'],'condition':name,'prompt_id':prompts[key]['prompt_id'],
                'scope':q['scope'],'date_anchor':q['date_anchor'],'accepted':True,
                'evidence':[{k:v for k,v in e.items() if k!='text'} for e in packed['evidence']]})
    old=read(args.original_reader/'reader.json')
    compatible=old['complete'] and minimal_pin(old['reader_pin'])==pin and old['decode']==decode
    cached={}
    if compatible:
        for line,saved in enumerate(rows(args.original_reader/'reader-receipts.jsonl'),1):
            result=saved['result']
            if saved['model']!=pin['model_id'] or result['max_new_tokens']!=192:continue
            record={'messages':saved['messages'],'reader_pin':pin,'decode':decode,'prompt_tokens':result['prompt_tokens']}
            key=prompt_key(record)
            if key not in prompts:continue
            actual_ids=tokenizer.apply_chat_template(saved['messages'],tokenize=True,add_generation_prompt=True)
            if len(actual_ids)!=result['prompt_tokens']:raise ValueError('Old packed prompt token count differs from pinned tokenizer')
            cached.setdefault(prompts[key]['prompt_id'],{'prompt_id':prompts[key]['prompt_id'],
                'result':compact_reader_result(result),'provenance':{'kind':'old_exact_prompt_reuse',
                    'path':str(args.original_reader/'reader-receipts.jsonl'),'line':line,
                    'qid':saved['qid'],'condition':saved['condition'],
                    'input_verification':'messages plus pinned tokenizer reconstructed actual prompt length'}})
    extra_dirs=getattr(args,'reader_caches',[]) or []
    extra=compact_prompt_cache(extra_dirs,prompts,pin,decode)
    for prompt_id,record in extra.items():cached.setdefault(prompt_id,record)
    for filename,records in [('prompts.jsonl',prompts.values()),('inputs.jsonl',logical),
                             ('physical-reader-results.jsonl',cached.values())]:
        with (out/filename).open('w',encoding='utf-8') as stream:
            for record in records:save_row(stream,record)
    meta=receipt(complete=True,questions=len(questions),conditions=args.reader_conditions,
        logical_outputs=len(logical),unique_packed_prompts=len(prompts),old_exact_unique_hits=len(cached),
        old_exact_logical_hits=sum(r['prompt_id'] in cached for r in logical),
        pending_unique_prompts=len(prompts)-len(cached),reader_pin=pin,decode=decode,
        prompt_budget=4096,packing='unchanged fiscal.pack_reader: official query_window then fixed4096 tokens',
        compact_cache_dirs=[str(p) for p in extra_dirs],frozen_recipe=str(frozen) if frozen else None,
        main_method=recipe['selected_method'] if frozen else None,
        input_ID_arrays_saved=False,prompts_saved_once=True,original_reader=str(args.original_reader),
        reference_original_conditions='preserved at original reader directory; no rerun',
        support_policy='all fixed native/CE methods deliver top4; no French threshold selection')
    write_json(out/'preparation.json',meta);return meta


def compact_reader_result(result):
    return {k:v for k,v in result.items() if k not in ('input_ids','attention_mask')}


def run_reader(args,*,factory=None):
    out=reader_dir(args);prep=read(out/'preparation.json');stack=read(args.stack)
    if not prep['complete'] or prep['reader_pin']!=minimal_pin(stack['models']['reader']):
        raise ValueError('Use complete exact reader inputs and same pinned reader')
    prompts=list(rows(out/'prompts.jsonl'));saved={r['prompt_id']:r for r in rows(out/'physical-reader-results.jsonl')}
    work=Counter();reader=None
    with (out/'physical-reader-results.jsonl').open('a',encoding='utf-8') as stream:
        extra_dirs=prep.get('compact_cache_dirs',[])
        refreshed=(compact_prompt_cache(extra_dirs,{prompt_key(p):p for p in prompts},
            prep['reader_pin'],prep['decode']) if extra_dirs else {})
        for prompt_id,record in refreshed.items():
            if prompt_id not in saved:
                save_row(stream,record);stream.flush();saved[prompt_id]=record
                work['additional_exact_prompt_reuses']=work.get('additional_exact_prompt_reuses',0)+1
        for prompt in prompts:
            if prompt['prompt_id'] in saved:continue
            if reader is None:
                if factory is None:
                    import torch
                    from hidate.models import QwenReader
                    torch.manual_seed(42)
                    reader=QwenReader(stack['models']['reader']['local_path'],device='cuda',prompt_budget=4096)
                else:reader=factory(stack['models']['reader'])
            ids=reader.tokenizer.apply_chat_template(prompt['messages'],tokenize=True,add_generation_prompt=True)
            if len(ids)!=prompt['prompt_tokens']:raise ValueError('Actual pinned reader tokenizer differs from prepared prompt')
            result=reader.generate_paraphrase(prompt['messages'],max_new_tokens=192,include_inputs=False)
            if result['prompt_tokens']!=prompt['prompt_tokens']:raise ValueError('Actual reader input token count changed')
            record={'prompt_id':prompt['prompt_id'],'result':compact_reader_result(result),
                'provenance':{'kind':'new_physical_reader_call','job_id':os.environ.get('PBS_JOBID')}}
            save_row(stream,record);stream.flush();saved[prompt['prompt_id']]=record
            work.update(calls=1,prompt_tokens=result['prompt_tokens'],generated_tokens=result['generated_tokens'],
                        model_seconds=result['seconds'])
            if work['calls']%20==0:print(json.dumps({'phase':'reader','new_calls':work['calls'],
                'completed_unique':len(saved),'expected_unique':len(prompts)}),flush=True)
    count=0
    with (out/'responses.jsonl').open('w',encoding='utf-8') as stream:
        for logical in rows(out/'inputs.jsonl'):
            result=saved[logical['prompt_id']]['result']
            save_row(stream,{**logical,'response_text':result['text'],'prompt_tokens':result['prompt_tokens'],
                'generated_tokens':result['generated_tokens'],'end_status':result['end_status']});count+=1
    physical=[r for r in saved.values() if r['provenance']['kind']=='new_physical_reader_call']
    meta=receipt(complete=len(saved)==len(prompts) and count==prep['logical_outputs'],
        reader_pin=prep['reader_pin'],decode=prep['decode'],logical_outputs=count,
        unique_completed_prompts=len(saved),old_exact_unique_hits=prep['old_exact_unique_hits'],
        reader_calls=len(physical),neural_model_calls=len(physical),
        new_physical_calls_total=len(physical),work_this_job=dict(work),
        physical_work_total={k:sum(r['result'][field] for r in physical) for k,field in
            [('prompt_tokens','prompt_tokens'),('generated_tokens','generated_tokens'),('model_seconds','seconds')]},
        input_ID_arrays_saved=False)
    write_json(out/'reader.json',meta);return meta


def evaluate(args):
    out=reader_dir(args);reader=read(out/'reader.json');prep=read(out/'preparation.json')
    if not reader['complete']:raise ValueError('Real reader responses must complete before answer scoring')
    # Jobs started before this receipt-label correction already record the exact
    # physical total; align the generic counters without changing any output.
    reader.update(reader_calls=reader['new_physical_calls_total'],
                  neural_model_calls=reader['new_physical_calls_total'])
    write_json(out/'reader.json',reader)
    spec=importlib.util.spec_from_file_location('fiscal_revision_official',args.official_scorer)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    questions={q['qid']:q for q in rows(args.prepared/'questions.jsonl')};nuggets=defaultdict(list)
    for n in rows(args.prepared/'nuggets.jsonl'):nuggets[n['qid']].append(n)
    scores=[];seen=set()
    with (out/'scores.jsonl').open('w',encoding='utf-8') as stream:
        for response in rows(out/'responses.jsonl'):
            key=(response['qid'],response['condition'])
            if key in seen:raise ValueError('Duplicate logical answer response')
            seen.add(key);q=questions[key[0]]
            score={'qid':key[0],'condition':key[1],'accepted':True,
                **fiscal.score_answer(q,response['response_text'],nuggets[key[0]],response['evidence'],scorer)}
            save_row(stream,score);scores.append(score)
    expected={(qid,condition) for qid in questions for condition in prep['conditions']}
    meta=receipt(complete=seen==expected,responses=len(seen),expected_responses=len(expected),
        conditions=prep['conditions'],strata=fiscal.summarize(scores,list(questions.values())),
        official_scorer=str(args.official_scorer),evidence_stage='actual packed reader evidence',
        main_method=prep.get('main_method'),frozen_recipe=prep.get('frozen_recipe'),
        no_french_selection=True,real_reader=True)
    write_json(out/'evaluation.json',meta);return {'complete':meta['complete'],'responses':len(seen)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('score','rank','prepare-reader','read','evaluate'))
    p.add_argument('--native',type=Path,default=Path('runs/hidate/revision-aware/transfer/fiscal-native'))
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/transfer/fiscal-ce'))
    p.add_argument('--prepared',type=Path,default=Path('data/processed/fiscal'))
    p.add_argument('--protocol',type=Path,default=Path('configs/fiscal_protocol.json'))
    p.add_argument('--stack',type=Path,default=Path('models/qwen3/model-stack.json'))
    p.add_argument('--ce-models',type=Path,default=None,help='Frozen Japanese-Qwen TRAIN CE model directory; explicit opt-in')
    p.add_argument('--ce-source',type=Path,default=None,help='Read completed exact French CE cache for an independent ranking/reader cohort')
    p.add_argument('--original-reader',type=Path,default=Path('runs/hidate/fiscal-transfer/fiscal'))
    p.add_argument('--official-scorer',type=Path,default=Path('data/raw/fiscal/official/scripts/score_nuggets.py'))
    p.add_argument('--reader-conditions',nargs='+',default=list(READER_CONDITIONS))
    p.add_argument('--reader-output',type=Path,default=None,help='Separate append-only reader cohort; original eight inputs stay fixed')
    p.add_argument('--reader-caches',type=Path,nargs='*',default=[])
    p.add_argument('--frozen-recipe',type=Path,default=None)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    result={'score':score,'rank':rank,'prepare-reader':prepare_reader,'read':run_reader,'evaluate':evaluate}[args.phase](args)
    print(json.dumps({k:result[k] for k in ('complete','counts','logical_outputs','pending_unique_prompts',
        'new_physical_pairs_this_job','new_physical_pairs_total','new_physical_calls_total','responses') if k in result}),flush=True)


if __name__=='__main__':main()
