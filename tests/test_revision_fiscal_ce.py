"""Focused French CE/reader contracts, without real neural inference."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('fiscal_ce',Path(__file__).parents[1]/'scripts/run_revision_fiscal_ce.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def save(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')


def test_score_unique_literal_pairs_and_resume_compact_receipts(tmp_path):
    pin={'model_id':'qwen','revision':'frozen','conventions':{'instruction':'French'}}
    source=tmp_path/'pairs.jsonl'
    save(source,[{'qid':'a','pair_id':'p1','query_text':'q','document_text':'same','cached_score':None},
        {'qid':'b','pair_id':'p2','query_text':'q','document_text':'same','cached_score':None},
        {'qid':'a','pair_id':'p3','query_text':'q','document_text':'other',
         'cached_score':{'reranker_score':.2,'reranker_logit':-1.,'pair_tokens':3}}])
    class Model:
        def score(self,pairs,include_inputs=False):
            assert pairs==[('q','same')] and not include_inputs
            return {'scores':[.9],'logits':[2.],'token_counts':[7],'total_tokens':7,
                'input_shape':[1,7],'seconds':.1,'sdpa_backend':'default',
                'score_format':'conditional_yes_no','logit_format':'yes_minus_no'}
    result=w.score_pair_file(source,tmp_path,pin,factory=lambda pin:Model())
    assert result['complete'] and result['new_physical_pairs_total']==1
    assert result['unique_requested_pairs']==2
    assert result['logical_pair_rows']==3
    again=w.score_pair_file(source,tmp_path,pin,factory=lambda pin:pytest.fail('resume loaded model'))
    assert again['new_physical_pairs_this_job']==0
    row=next(w.rows(tmp_path/'new-ce-pair-scores.jsonl'))
    assert 'input_ids' not in row['result'] and 'document_text' not in row


def test_historical_ce_reorders_addresses_but_delivers_unchanged_current_text():
    candidates=[{'article_cid':'A','chunk_id':'a-now','historical_chunk_id':'a-old','selection_rank':1},
                {'article_cid':'B','chunk_id':'b-now','historical_chunk_id':'b-old','selection_rank':2}]
    chunks={cid:{'text':cid} for cid in ('a-now','a-old','b-now','b-old')}
    cache={('q','a-old'):{'reranker_score':.1,'reranker_logit':-2.},
           ('q','b-old'):{'reranker_score':.9,'reranker_logit':2.},
           ('q','a-now'):{'reranker_score':.8,'reranker_logit':1.},
           ('q','b-now'):{'reranker_score':.2,'reranker_logit':-1.}}
    h=w.ce_rerank(candidates,'q',chunks,cache,'historical')
    c=w.ce_rerank(candidates,'q',chunks,cache,'current')
    assert [r['chunk_id'] for r in h['evidence']]==['b-now','a-now']
    assert [r['chunk_id'] for r in c['evidence']]==['a-now','b-now']
    assert h['candidates'][0]['ce_document_chunk_id']=='b-old'


def test_reader_cache_identity_includes_exact_packed_messages_pin_and_decode():
    a={'messages':[{'role':'user','content':'same'}],'reader_pin':{'model_id':'r','revision':'v'},
       'decode':{'do_sample':False,'max_new_tokens':192},'prompt_tokens':12}
    b={**a,'condition':'different'}
    assert w.prompt_key(a)==w.prompt_key(b)
    assert w.prompt_key(a)!=w.prompt_key({**a,'decode':{'do_sample':False,'max_new_tokens':191}})
    assert w.prompt_key(a)!=w.prompt_key({**a,'messages':[{'role':'user','content':'same '} ]})
    assert w.prompt_key(a)!=w.prompt_key({**a,'reader_pin':{'model_id':'r','revision':'different'}})


def test_qwen_context_rejects_japanese_instruction_or_ruri_backend():
    wanted={'model_id':'Qwen','revision':'v','conventions':{'instruction':'French'}}
    with pytest.raises(ValueError):w.check_context({'reranker_pin':{**wanted,'conventions':{'instruction':'Japanese'}}},wanted)
    with pytest.raises(ValueError):w.validate_result({'scores':[.2],'logits':[1.],
        'token_counts':[2],'total_tokens':2,'sdpa_backend':'math','score_format':'conditional_yes_no',
        'logit_format':'yes_minus_no'},1)


def test_new_rank_cohort_reuses_complete_ce_source_without_overwriting_it(tmp_path,monkeypatch):
    native=tmp_path/'native';prepared=tmp_path/'prepared';source=tmp_path/'ce';out=tmp_path/'new'
    for path in (native,prepared,source):path.mkdir()
    pin={'model_id':'qwen','revision':'v','conventions':{'instruction':'French'}}
    monkeypatch.setattr(w,'context',lambda args:({'questions':1},{},{},pin))
    (source/'scoring.json').write_text(json.dumps({'complete':True,'reranker_pin':pin}))
    candidate={'article_cid':'A','chunk_id':'now','historical_chunk_id':'old','selection_rank':1}
    pool={'candidates':[candidate],'evidence':[candidate]}
    save(native/'predictions.jsonl',[{'qid':'q','conditions':{'RRF_native':{
        'own_pool':pool,'common_rrf10':pool}}}])
    save(native/'addresses.jsonl',[{'qid':'q','addresses':[]}])
    save(prepared/'chunks.jsonl',[{'chunk_id':'now','text':'new text'},{'chunk_id':'old','text':'old text'}])
    save(prepared/'questions.jsonl',[{'qid':'q','prompt':'question','date_anchor':'2020-01-01','scope':'in209'}])
    save(native/'ce-pairs.jsonl',[{'qid':'q','pair_id':p,'query_text':'question','document_text':text,
        'cached_score':{'reranker_score':score,'reranker_logit':score,'pair_tokens':4}}
        for p,text,score in [('p1','old text',.8),('p2','new text',.2)]])
    before=(source/'scoring.json').read_bytes()
    args=SimpleNamespace(native=native,prepared=prepared,output=out,ce_source=source,ce_models=None)
    result=w.rank(args)
    assert result['complete'] and result['ce_scoring']==str(source/'scoring.json')
    assert (source/'scoring.json').read_bytes()==before and not (source/'ranked.jsonl').exists()
    saved=next(w.rows(out/'ranked.jsonl'))
    hist=saved['reader_conditions']['RRF_historical_CE']['evidence'][0]
    assert hist['reranker_score']==.8 and hist['chunk_id']=='now'


def test_reader_runs_only_unique_missing_prompt_and_saves_compact_actual_result(tmp_path):
    output=tmp_path/'result';out=output/'reader';out.mkdir(parents=True)
    pin={'model_id':'reader','revision':'v','local_path':'unused'}
    stack=tmp_path/'stack.json';stack.write_text(json.dumps({'models':{'reader':pin}}))
    prep={'complete':True,'reader_pin':{'model_id':'reader','revision':'v'},
        'decode':{'do_sample':False,'max_new_tokens':192},'logical_outputs':3,'old_exact_unique_hits':1}
    (out/'preparation.json').write_text(json.dumps(prep))
    save(out/'prompts.jsonl',[{'prompt_id':p,'messages':[{'role':'user','content':p}],
        'prompt_tokens':6} for p in ('old','new')])
    save(out/'inputs.jsonl',[{'qid':'q','condition':str(i),'prompt_id':p,'evidence':[]}
                            for i,p in enumerate(('old','new','new'))])
    result={'text':'answer','prompt_tokens':6,'generated_tokens':1,'end_status':'eos','seconds':.1}
    save(out/'physical-reader-results.jsonl',[{'prompt_id':'old','result':result,
        'provenance':{'kind':'old_exact_prompt_reuse'}}])
    calls=[]
    class Reader:
        tokenizer=SimpleNamespace(apply_chat_template=lambda *args,**kwargs:[1]*6)
        def generate_paraphrase(self,messages,max_new_tokens,include_inputs):
            assert max_new_tokens==192 and include_inputs is False
            calls.append(messages)
            return {**result,'generated_token_ids':[2]}
    args=SimpleNamespace(output=output,stack=stack)
    actual=w.run_reader(args,factory=lambda pin:Reader())
    assert actual['complete'] and actual['new_physical_calls_total']==1 and len(calls)==1
    assert actual['reader_calls']==actual['neural_model_calls']==1
    assert len(list(w.rows(out/'responses.jsonl')))==3
    assert all('input_ids' not in r['result'] for r in w.rows(out/'physical-reader-results.jsonl'))
    resumed=w.run_reader(args,factory=lambda pin:pytest.fail('resume loaded reader'))
    assert resumed['work_this_job']=={} and resumed['new_physical_calls_total']==1


def test_frozen_main_reader_uses_saved_own_ten_historical_ce_without_reranking():
    evidence=[{'article_cid':'A','version_id':'a-now'}]
    row={'reader_conditions':{},'all_head_scope_comparisons':{
        'Full_LM18':{'own_pool':{'historical_CE':{'evidence':evidence}}}}}
    assert w.reader_evidence(row,'Full_LM18_historical_CE') is evidence


def test_incremental_reader_reuses_exact_prompt_from_existing_compact_run(tmp_path):
    source=tmp_path/'existing';source.mkdir()
    pin={'model_id':'r','revision':'v'};decode={'do_sample':False,'max_new_tokens':192}
    prompt={'messages':[{'role':'user','content':'same'}],'reader_pin':pin,'decode':decode,'prompt_tokens':6}
    (source/'preparation.json').write_text(json.dumps({'complete':True,'reader_pin':pin,'decode':decode}))
    save(source/'prompts.jsonl',[{'prompt_id':'OLD',**prompt}])
    save(source/'physical-reader-results.jsonl',[{'prompt_id':'OLD','result':{'text':'actual','prompt_tokens':6},
        'provenance':{'kind':'new_physical_reader_call'}}])
    desired={w.prompt_key(prompt):{'prompt_id':'NEW',**prompt}}
    cache=w.compact_prompt_cache([source],desired,pin,decode)
    assert cache['NEW']['result']['text']=='actual'
    assert cache['NEW']['provenance']['source_prompt_id']=='OLD'
    assert cache['NEW']['provenance']['kind']=='compact_exact_prompt_reuse'
