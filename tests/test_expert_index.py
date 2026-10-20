"""Small official-tree fixtures for the expert-date corpus extension."""
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hidate.io import write_json, write_jsonl, load_jsonl
from hidate.data import visible_provision
from hidate.retrieval import chunk_passages
from hidate.scale import add_passage
from hidate.sources import extract_provisions

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from prepare_data import target_status


SCRIPT = SCRIPTS_DIR / 'build_expert_index.py'
DATES = ['2020-06-01', '2022-06-01']


def worker():
    assert SCRIPT.is_file(), 'Expert-date index builder is not implemented'
    spec=importlib.util.spec_from_file_location('expert_index_fixture',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class OffsetTokenizer:
    is_fast=True

    def __call__(self,text,*,add_special_tokens=True,return_offsets_mapping=False,truncation=False):
        assert not truncation
        spans=[m.span() for m in re.finditer(r'\S',text)]
        result={'input_ids':list(range(len(spans)+(2 if add_special_tokens else 0)))}
        if return_offsets_mapping:result['offset_mapping']=spans
        return result


def node(tag,children=(),**attr):
    return {'tag':tag,'attr':attr,'children':list(children)}


def snapshot(law,revision,changed=False):
    def article(num,text,deleted=False):
        return node('Article',[node('ArticleTitle',[str(num)]),
            node('Paragraph',[node('ParagraphSentence',[node('Sentence',[text])])],Num='1')],
            Num=str(num),Delete=deleted)
    return {'law_info':{'law_id':law},'revision_info':{'law_revision_id':revision,'law_title':law+' law'},
        'law_full_text':node('Law',[node('LawBody',[node('MainProvision',[
            article(1,'unchanged body'),article(2,'new body' if changed else 'old body'),
            article(3,'削除' if changed else 'retained body',changed)])])])}


@pytest.fixture
def inputs(tmp_path):
    base=tmp_path/'base';census=tmp_path/'census';out=tmp_path/'index'
    pin={'model_id':'ruri-fixture','revision':'fixed','local_path':'unused',
         'conventions':{'document_prefix':'search: '}}
    stack=tmp_path/'stack.json';write_json(stack,{'complete':True,'models':{'embedding':pin}})
    preflight=tmp_path/'preflight.json';write_json(preflight,{'status':'PASS','model_pins':{'embedding':pin}})
    corpus={};targets=[];bindings=[]
    for law,role,split,dates in [('A','full','train',['2018-09-30','2021-09-30']),
            ('B','full','distractor_only',['2018-09-30']),('X','external','external',['2018-09-30'])]:
        raw=snapshot(law,law+'-old')
        for date in dates:
            for record in extract_provisions(raw):
                v=visible_provision({**record,'requested_asof':date,'raw_path':'base/'+law+'.json',
                    'source_role':role,'family_id':'F-'+law,'split':split},raw)
                add_passage(corpus,v)
                targets.append({'address_key':[law,date,v['scope'],v['article_num']],
                    'law_id':law,'requested_asof':date,'scope':v['scope'],'address_num':v['article_num'],
                    'article_num':v['article_num'],'slot_id':v['slot_id'],'status':target_status([v]),
                    'revision_id':v['revision_id'],'passage_ids':[v['passage_id']],
                    'family_id':'F-'+law,'source_role':role,'split':split,'raw_path':'base/'+law+'.json'})
            bindings.append({'law_id':law,'requested_asof':date,'revision_id':law+'-old',
                'status':'collected','source_role':role,'family_id':'F-'+law,'raw_path':'base/'+law+'.json'})
    chunks=chunk_passages(corpus.values(),OffsetTokenizer())
    chunks+=chunk_passages([{'source_id':'guide','source_role':'full','split':'shared',
        'title':'Shared guidance','text':'Static official guidance.','url':'https://example.test/guide'}],OffsetTokenizer())
    for c in chunks:c['ruri_document_tokens']=len(OffsetTokenizer()('search: '+c['text'])['input_ids'])
    write_jsonl(base/'chunks.jsonl',chunks)
    write_jsonl(base/'lexical-tokens.jsonl',[{'chunk_id':c['chunk_id'],'tokens':c['text'].split()} for c in chunks])
    write_jsonl(base/'target-addresses.jsonl',targets);write_jsonl(base/'asof-bindings.jsonl',bindings)
    write_json(base/'nested-corpora.json',{'120':['A'],'1000':['A','B']})
    write_json(base/'source-exclusions.json',{'unchanged':True})
    write_json(base/'original-family-split.json',{'F-A':'train'})
    write_json(base/'index.json',{'complete':True,'tokenizer':pin,'merged_chunk_rows':len(chunks)})
    write_json(tmp_path/'sources/new-A.json',snapshot('A','A-new',True))
    extra=[]
    for law in ['A','B']:
        for date in DATES:
            new=law=='A' and date==DATES[1]
            extra.append({'law_id':law,'family_id':'F-'+law,'source_role':'full','requested_asof':date,
                'revision_id':'A-new' if new else law+'-old','status':'collected',
                'raw_path':'sources/new-A.json' if new else 'reused-only-not-opened.json',
                'raw_source_root':'.','snapshot_requested_asof':date,
                'binding_basis':'official_census','official_url':'https://example.test/'+law})
    write_jsonl(census/'bindings.jsonl',extra)
    write_json(census/'census.json',{'complete':True,'ready_for_index':True})
    args=SimpleNamespace(original_index=base,index=out,census=census,project_root=tmp_path,
                         stack=stack,preflight=preflight,batch_size=2)
    return args,chunks,targets,bindings,pin


def prepare(inputs):
    args,*_=inputs
    return worker().prepare(args,tokenizer=OffsetTokenizer(),segmenter=lambda t:t.split(),
        package_versions={'tokenizer':'fixture'},expected_laws=2,expected_dates=2)


def test_same_pool_preserves_old_chunks_guidance_and_appends_only_new_revision(inputs):
    args,old_chunks,old_targets,old_bindings,_=inputs
    meta=prepare(inputs)
    assert meta['complete'] and meta['training_runs']==meta['model_inference_runs']==0
    chunks=load_jsonl(args.index/'chunks.jsonl')
    assert len(chunks)==len(old_chunks)+2
    for old,new in zip(old_chunks,chunks):
        assert {k:v for k,v in old.items() if k!='asof_dates'}=={k:v for k,v in new.items() if k!='asof_dates'}
        if old['source_kind']=='guidance' or old['source_role']=='external':assert old==new
        elif old['law_id']=='A':assert new['asof_dates']==['2018-09-30',DATES[0],'2021-09-30']
        else:assert new['asof_dates']==['2018-09-30',*DATES]
    assert [c['article_num'] for c in chunks[len(old_chunks):]]==['1','2']
    assert all(c['revision_id']=='A-new' and c['asof_dates']==[DATES[1]] and c['split']=='train'
               for c in chunks[len(old_chunks):])
    assert (args.index/'lexical-tokens.jsonl').read_bytes().startswith((args.original_index/'lexical-tokens.jsonl').read_bytes())
    targets=load_jsonl(args.index/'target-addresses.jsonl')
    assert targets[:len(old_targets)]==old_targets
    new_targets=targets[len(old_targets):]
    assert len(new_targets)==12 and len({tuple(r['address_key']) for r in targets})==len(targets)
    deleted=next(r for r in new_targets if r['law_id']=='A' and r['requested_asof']==DATES[1] and r['article_num']=='3')
    assert deleted['status']=='deleted' and deleted['revision_id']=='A-new'
    reused=next(r for r in new_targets if r['law_id']=='A' and r['requested_asof']==DATES[0] and r['article_num']=='1')
    assert reused['passage_ids']==old_targets[0]['passage_ids'] and reused['raw_source_root']=='.'
    assert load_jsonl(args.index/'asof-bindings.jsonl')[:len(old_bindings)]==old_bindings
    assert {r['law_id'] for r in load_jsonl(args.index/'asof-bindings.jsonl') if r['requested_asof']==DATES[1]}=={'A','B'}
    for name in ['nested-corpora.json','source-exclusions.json','original-family-split.json']:
        assert (args.index/name).read_bytes()==(args.original_index/name).read_bytes()


@pytest.mark.parametrize('change',['not_ready','missing_law_date','unresolved'])
def test_prepare_requires_completed_full_law_date_census(inputs,change):
    args,*_=inputs
    if change=='not_ready':write_json(args.census/'census.json',{'complete':True,'ready_for_index':False})
    else:
        records=load_jsonl(args.census/'bindings.jsonl')
        if change=='missing_law_date':records=records[:-1]
        else:records[0]['status']='unresolved'
        write_jsonl(args.census/'bindings.jsonl',records)
    with pytest.raises(ValueError,match='census|collected|law.date'):
        prepare(inputs)


def test_encode_reuses_old_values_and_only_encodes_new_complete_text(inputs,monkeypatch):
    args,chunks,_,_,pin=inputs;prepare(inputs)
    old=args.original_index/'embeddings';old.mkdir()
    ids=[c['chunk_id'] for c in chunks];texts=list(dict.fromkeys(c['text'] for c in chunks))
    aliases=np.asarray([texts.index(c['text']) for c in chunks],dtype=np.int64)
    vectors=np.asarray([[i+.25,i+.75] for i in range(len(texts))],dtype=np.float32)
    write_json(old/'chunk-ids.json',ids);write_json(old/'encoding.json',{'complete':True,'model_pin':pin})
    np.save(old/'vector-row-indices.npy',aliases);np.save(old/'vectors.npy',vectors)
    called=[]
    class Encoder:
        def encode(self,batch,kind):
            assert kind=='document';called.extend(batch)
            return {'embeddings':[[99.,98.] for _ in batch],'total_tokens':sum(map(len,batch)),'seconds':.01}
    import hidate.models
    monkeypatch.setattr(hidate.models,'create_encoder',lambda *a,**kw:Encoder())
    worker().encode(args)
    new=args.index/'embeddings';encoded=np.load(new/'vectors.npy');out_aliases=np.load(new/'vector-row-indices.npy')
    assert np.array_equal(encoded[:len(vectors)],vectors)
    assert np.array_equal(out_aliases[:len(aliases)],aliases)
    assert len(called)==1 and 'new body' in called[0]
    appended=load_jsonl(args.index/'chunks.jsonl')[len(chunks):]
    assert appended[0]['text'] in texts and out_aliases[len(chunks)]==texts.index(appended[0]['text'])
    assert json.loads((new/'encoding.json').read_text())['complete']


def test_cli_prepares_local_index_with_explicit_paths(inputs,monkeypatch,capsys):
    args,old_chunks,*_=inputs
    module=worker()
    prepare_index=module.prepare
    monkeypatch.setattr(module,'prepare',lambda cli_args:prepare_index(cli_args,
        tokenizer=OffsetTokenizer(),segmenter=lambda t:t.split(),
        package_versions={'tokenizer':'fixture'},expected_laws=2,expected_dates=2))
    monkeypatch.delenv('PBS_JOBID',raising=False)
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'prepare',
        '--original-index',str(args.original_index),'--index',str(args.index),
        '--census',str(args.census),'--project-root',str(args.project_root),
        '--stack',str(args.stack),'--preflight',str(args.preflight),'--batch-size','2'])
    module.main()
    result=json.loads(capsys.readouterr().out)
    assert result['complete'] is True
    assert result['training_runs']==result['model_inference_runs']==0
    assert len(load_jsonl(args.index/'chunks.jsonl'))==len(old_chunks)+2
