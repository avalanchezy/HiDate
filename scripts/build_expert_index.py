"""Extend the frozen 1000-law corpus to the completed expert-date census.

Only dated corpus metadata and new revisions are prepared. The old chunk order,
IDs and text stay fixed; the existing scale encoder reuses their saved vectors.
This worker reads no expert questions, answers or gold labels.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from hidate.io import write_json
from hidate.data import visible_provision
from hidate.retrieval import chunk_passages
from hidate.scale import add_passage
from hidate.sources import extract_provisions
from prepare_data import target_status


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def write_row(stream,row):
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def load_census(directory,law_ids,expected_dates):
    meta=read(directory/'census.json')
    if not meta.get('complete') or not meta.get('ready_for_index'):
        raise ValueError('Expert-date census must be complete and ready_for_index')
    bindings={}
    for row in rows(directory/'bindings.jsonl'):
        key=(row['law_id'],row['requested_asof'])
        if row['status']!='collected' or row['source_role']!='full' or not row['revision_id']:
            raise ValueError('Every census binding must be collected with a full-law revision')
        if key in bindings:raise ValueError('Duplicate law/date in expert census')
        bindings[key]=row
    dates=sorted({key[1] for key in bindings})
    expected={(law,date) for law in law_ids for date in dates}
    if len(dates)!=expected_dates or set(bindings)!=expected:
        raise ValueError('Expert census must cover the complete law/date product')
    return meta,dates,[bindings[law,date] for law in law_ids for date in dates]


def dated_target(template,binding):
    row={**template,'requested_asof':binding['requested_asof'],
         'address_key':[binding['law_id'],binding['requested_asof'],
                        template['scope'],template['address_num']]}
    for name in ('raw_path','raw_source_root','snapshot_requested_asof','binding_basis','official_url'):
        row[name]=binding[name]
    return row


def prepare(args,*,tokenizer=None,segmenter=None,package_versions=None,
            expected_laws=1000,expected_dates=29):
    from build_index import installed_versions,japanese_tokens,sudachi_segmenter

    started=time.perf_counter();base=Path(args.original_index);output=Path(args.index)
    if output.resolve()==base.resolve():raise ValueError('Keep the original scale index in its own directory')
    old_index=read(base/'index.json');stack=read(args.stack);pin=stack['models']['embedding']
    if not old_index['complete'] or not stack['complete'] or any(
            old_index['tokenizer'][k]!=pin[k] for k in ('model_id','revision')):
        raise ValueError('Use the complete original 1000-law index and its frozen Ruri tokenizer')
    banks=read(base/'nested-corpora.json');law_ids=banks['1000'];law_set=set(law_ids)
    if len(law_ids)!=expected_laws or len(law_set)!=expected_laws:
        raise ValueError('The expert index retains the entire frozen 1000-law bank')
    _,dates,bindings=load_census(Path(args.census),law_ids,expected_dates)
    old_bindings={(r['law_id'],r['requested_asof']):r for r in rows(base/'asof-bindings.jsonl')}
    for b in bindings:
        old=old_bindings.get((b['law_id'],b['requested_asof']))
        if old is not None and (old['status']!='collected' or old['revision_id']!=b['revision_id']):
            raise ValueError('Expert census conflicts with an existing dated binding')
    groups=defaultdict(list)
    for b in bindings:groups[b['law_id'],b['revision_id']].append(b)
    dates_by_revision={key:sorted({b['requested_asof'] for b in values}) for key,values in groups.items()}
    known_revisions={(r['law_id'],r['revision_id']) for r in old_bindings.values()
                     if r['source_role']=='full' and r['law_id'] in law_set and r['status']=='collected'}
    templates=defaultdict(dict);law_metadata={}
    for row in rows(base/'target-addresses.jsonl'):
        if row['source_role']!='full' or row['law_id'] not in law_set:continue
        key=(row['law_id'],row['revision_id'])
        templates[key].setdefault((row['scope'],row['address_num']),row)
        law_metadata.setdefault(row['law_id'],{k:row[k] for k in ('family_id','source_role','split')})
    allocation=read(base/'original-family-split.json')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(pin['local_path'],local_files_only=True,use_fast=True)
    if segmenter is None:segmenter=sudachi_segmenter()
    if package_versions is None:package_versions=installed_versions()

    output.mkdir(parents=True,exist_ok=True)
    for name in ('lexical-tokens.jsonl','target-addresses.jsonl','asof-bindings.jsonl',
                 'nested-corpora.json','source-exclusions.json','original-family-split.json'):
        shutil.copyfile(base/name,output/name)
    stats=Counter();meta={'complete':False,'phase':'extend_dates','job_id':os.environ.get('PBS_JOBID'),
        'at_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'training_runs':0,
        'ranker_fits':0,'dev_selections':0,'model_inference_runs':0,'reader_calls':0,
        'original_index':str(base),'census':str(Path(args.census)/'census.json'),
        'law_count':len(law_ids),'expert_dates':dates,'census_bindings':len(bindings),
        'tokenizer':{k:pin[k] for k in ('model_id','revision','local_path')},
        'chunking':{'max_body_tokens':448,'overlap_tokens':64},'ruri_input_limit':1024,
        'package_versions':package_versions,'expert_gold_read':False}
    write_json(output/'index.json',meta)
    with (base/'chunks.jsonl').open(encoding='utf-8',newline='') as source, \
            (output/'chunks.jsonl').open('w',encoding='utf-8',newline='') as dest:
        for line in source:
            if not line.strip():dest.write(line);continue
            row=json.loads(line);stats['original_chunk_rows']+=1
            if row['source_kind']=='guidance':stats['shared_guidance_chunks']+=1
            key=(row.get('law_id'),row.get('revision_id'))
            if row['source_kind']=='statute' and row['source_role']=='full' and key in dates_by_revision:
                dates_added=set(dates_by_revision[key])-set(row['asof_dates'])
                if dates_added:
                    row['asof_dates']=sorted(set(row['asof_dates'])|dates_added)
                    write_row(dest,row);stats['old_chunk_date_rows_updated']+=1
                    stats['old_chunk_date_memberships_added']+=len(dates_added)
                    continue
            dest.write(line)

    prefix=pin['conventions']['document_prefix']
    with (output/'chunks.jsonl').open('a',encoding='utf-8') as chunks_out, \
            (output/'lexical-tokens.jsonl').open('a',encoding='utf-8') as lexical_out, \
            (output/'target-addresses.jsonl').open('a',encoding='utf-8') as targets_out, \
            (output/'asof-bindings.jsonl').open('a',encoding='utf-8') as bindings_out, \
            (output/'extra-corpus-passages.jsonl').open('w',encoding='utf-8') as corpus_out:
        for key,dated in groups.items():
            law,revision=key
            if key in known_revisions:
                target_templates=list(templates[key].values());stats['reused_revisions']+=1
            else:
                binding=dated[0]
                snapshot=read(Path(args.project_root)/binding['raw_path'])
                if snapshot['law_info']['law_id']!=law or snapshot['revision_info']['law_revision_id']!=revision:
                    raise ValueError('Census raw snapshot differs from its law/revision binding')
                metadata=law_metadata.get(law,{'family_id':binding['family_id'],'source_role':'full',
                    'split':allocation.get(binding['family_id'],'distractor_only')})
                corpus={};by_address=defaultdict(list)
                for record in extract_provisions(snapshot):
                    visible=visible_provision({**record,**metadata,'raw_source_root':'.'},snapshot)
                    number=visible['article_num'] if visible['article_num'] is not None else visible['slot_id'].rsplit(':',1)[-1]
                    by_address[visible['scope'],number].append(visible)
                    for b in dated:
                        add_passage(corpus,{**visible,'requested_asof':b['requested_asof'],'raw_path':b['raw_path']})
                target_templates=[]
                for (scope,number),values in by_address.items():
                    first=values[0]
                    target_templates.append({'law_id':law,'scope':scope,'address_num':number,
                        'article_num':first['article_num'],'slot_id':first['slot_id'],'status':target_status(values),
                        'revision_id':revision,'passage_ids':[v['passage_id'] for v in values],**metadata})
                for passage in corpus.values():
                    passage['asof_dates'].sort();passage['raw_paths'].sort();write_row(corpus_out,passage)
                new_chunks=chunk_passages(corpus.values(),tokenizer,max_body_tokens=448,overlap_tokens=64)
                for chunk in new_chunks:
                    count=len(tokenizer(prefix+chunk['text'],add_special_tokens=True,truncation=False)['input_ids'])
                    if count>1024:raise ValueError('New prefixed Ruri chunk exceeds the fixed 1024-token limit')
                    chunk['ruri_document_tokens']=count;write_row(chunks_out,chunk)
                    write_row(lexical_out,{'chunk_id':chunk['chunk_id'],'tokens':japanese_tokens(chunk['text'],segmenter)})
                stats.update(new_revisions=1,new_unique_passages=len(corpus),new_chunks=len(new_chunks))
            for binding in dated:
                if (law,binding['requested_asof']) in old_bindings:
                    stats['existing_dated_bindings_retained']+=1;continue
                for template in target_templates:
                    write_row(targets_out,dated_target(template,binding));stats['new_target_addresses']+=1
                write_row(bindings_out,binding);stats['new_dated_bindings']+=1
            if (stats['reused_revisions']+stats['new_revisions'])%100==0:
                meta['statistics']=dict(stats);write_json(output/'index.json',meta)
                print(json.dumps({'phase':'prepare',**dict(stats)}),flush=True)
    stats['chunks']=stats['original_chunk_rows']+stats['new_chunks']
    meta.update(complete=True,phase='complete',statistics=dict(stats),
        original_chunk_rows=stats['original_chunk_rows'],merged_chunk_rows=stats['chunks'],
        original_chunk_order_ids_text_unchanged=True,only_old_chunk_field_changed='asof_dates',
        original_lexical_prefix_unchanged=True,shared_guidance_unchanged=True,
        original_target_and_binding_prefix_unchanged=True,wall_seconds=time.perf_counter()-started)
    write_json(output/'index.json',meta)
    return meta


def encode(args):
    from scale_worker import encode as scale_encode
    scale_encode(args,{})
    return read(Path(args.index)/'embeddings/encoding.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','encode'))
    parser.add_argument('--original-index',type=Path,default=Path('data/index/hidate/extended-benchmark/scale'))
    parser.add_argument('--index',type=Path,default=Path('data/index/hidate/expert-benchmark/corpus'))
    parser.add_argument('--census',type=Path,default=Path('data/raw/hidate/expert-dates'))
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    parser.add_argument('--preflight',type=Path,default=Path('runs/hidate/pbs/preflight/preflight.json'))
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    result=prepare(args) if args.phase=='prepare' else encode(args)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
