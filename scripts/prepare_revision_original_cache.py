"""Compact the saved original Ruri TEST discovery without new retrieval/model calls."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

from hidate.io import write_json
from discover_final import query_segmenter,query_tokens
from run_revision_scale import (compact_chunks,compact_discovery,original_queries,
    read,rows,save_row,receipt)


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True);path=out/'discovery.json'
    queries,_=original_queries(args,read(args.protocol))
    if path.exists() and read(path)['complete']:
        saved=read(path)
        if saved['scales']!=['120'] or set(saved['query_ids'])!=set(queries) or saved['index_path']!=str(args.index):
            raise ValueError('Existing compact cache differs from the original120-law TEST inputs')
        return saved
    chunks={c['chunk_id']:c for c in compact_chunks(args.index/'chunks.jsonl')}
    tokens={r['chunk_id']:r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    pin=read(args.stack)['models']['embedding'];segment=query_segmenter();seen=set();timing=Counter()
    manifest=receipt(complete=False,scales=['120'],query_ids=list(queries),index_path=str(args.index),
        original_prepared=str(args.original_prepared),expected_query_scale_rows=len(queries),
        completed_query_scale_rows=0,original_test_only=True,new_confirmatory_queries=0,
        source_chunks=str(args.index/'chunks.jsonl'),source_discovery_paths=[str(p) for p in args.discovery],
        new_retrieval_runs=0,new_binding_runs=0,embedding_pin=pin,discovery_top_k=50,rrf_k=60,
        cache_contract='all saved historical addresses/hits; unchanged bound-current representatives and asof hits; full-pool revision features',
        timing_definition='this run copies completed discovery/binding and computes revision features; no discovery latency is remeasured',
        outputs={'candidates':'candidates.jsonl'})
    write_json(path,manifest);started=time.perf_counter()
    with (out/'candidates.jsonl').open('w',encoding='utf-8') as stream:
        for source in args.discovery:
            meta=read(source.parent/'discovery.json')
            if (not meta['complete'] or meta['seed']!=42 or meta['evaluation_scope']!='natural'
                    or meta['top_k']!=50 or meta['rrf_k']!=60
                    or any(meta['model_pin'][k]!=pin[k] for k in ('model_id','revision'))):
                raise ValueError('Use complete original Ruri natural TEST discovery at50/60')
            for d in rows(source):
                qid=d['query_id'];q=queries[qid]
                if qid in seen or any(d[k]!=q[k] for k in ('split','query_variant','source_role','target_asof')):
                    raise ValueError('Saved original TEST discovery identity differs')
                stamp=time.perf_counter()
                cached=compact_discovery(q,d,chunks,tokens,query_tokens(q['query_text'],segment),'120')
                timing['feature_seconds']+=time.perf_counter()-stamp
                save_row(stream,cached);seen.add(qid)
                if len(seen)%200==0:
                    stream.flush();manifest.update(completed_query_scale_rows=len(seen),timing=dict(timing));write_json(path,manifest)
    if seen!=set(queries):raise ValueError('Original TEST compact cache must cover all3172 IDs once')
    manifest.update(complete=True,completed_query_scale_rows=len(seen),timing=dict(timing),wall_seconds=time.perf_counter()-started)
    write_json(path,manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=Path('data/index/hidate'))
    p.add_argument('--original-prepared',type=Path,default=Path('data/processed/hidate'))
    p.add_argument('--protocol',type=Path,default=Path('configs/scale_protocol.json'))
    p.add_argument('--stack',type=Path,default=Path('models/ruri/model-stack.json'))
    p.add_argument('--discovery',type=Path,nargs='+',default=[Path('runs/hidate/final-discovery')/('test-natural-'+k)/'candidates.jsonl' for k in ('exact','paraphrase')])
    p.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/transfer/ruri-original-native'))
    args=p.parse_args()
    prepare(args)


if __name__=='__main__':main()
