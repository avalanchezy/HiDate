"""Unlabeled retrieval cache for the frozen new-family confirmation queries.

Uses the existing 1000-law index and the original discovery budgets. No scoring
head, target label or answer metric is loaded in this stage.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from hidate.io import write_json
from hidate.scale import SharedCosines
from discover_final import vector_bank, query_segmenter, query_tokens
from run_revision_scale import (read, rows, save_row, receipt, compact_chunks,
    scale_chunks, TimedDiscoveryEngine, compact_discovery)


ROOT = Path('data/processed/hidate-holdout/confirmatory')


def load_inputs(paths):
    queries = {}
    for path in paths:
        for row in rows(path):
            q = {k: row[k] for k in ('query_id','query_text','query_variant','target_asof')}
            q.update(source_role='full', split='test')
            if not q['query_id'].startswith('HOLDOUT-') or q['query_id'] in queries:
                raise ValueError('Expected unique frozen confirmation query IDs')
            queries[q['query_id']] = q
    return queries


def release_record(query, discovery, chunks, tokens, terms):
    return compact_discovery(query,discovery,chunks,tokens,terms,'1000')


def discover(args):
    started = time.perf_counter(); out=args.output
    out.mkdir(parents=True,exist_ok=True)
    metadata=out/'discovery.json'
    if metadata.exists() and read(metadata)['complete']:
        return read(metadata)
    queries=load_inputs(args.queries)
    generation=read(ROOT/'unlabeled/paraphrase-receipt.json')
    embedding=read(args.query_index/'encoding.json')
    if not generation['complete'] or not embedding['complete'] or len(queries)!=14912:
        raise ValueError('Complete both predeclared query variants before retrieval')
    index=read(args.index/'index.json')
    if not index['complete'] or not index['original_chunk_and_lexical_prefix_unchanged']:
        raise ValueError('Use the unchanged complete 1000-law bank')
    documents=vector_bank(args.index/'embeddings','chunk-ids.json')
    bank=vector_bank(args.query_index,'query-ids.json')
    if any(bank['manifest']['model_pin'][k]!=documents['manifest']['model_pin'][k]
           for k in ('model_id','revision')):
        raise ValueError('Use the same frozen encoder for documents and queries')
    vector_rows={qid:int(alias) for qid,alias in zip(bank['ids'],bank['aliases'])}
    if set(vector_rows)!=set(queries):
        raise ValueError('Query embedding identities differ from public input rows')
    chunks=compact_chunks(args.index/'chunks.jsonl')
    if documents['ids'] != [c['chunk_id'] for c in chunks]:
        raise ValueError('Document vector aliases differ from the saved chunk order')
    chunk_map={c['chunk_id']:c for c in chunks}
    tokens={r['chunk_id']:r['tokens'] for r in rows(args.index/'lexical-tokens.jsonl')}
    targets={tuple(r['address_key']):{'status':r['status'],'passage_ids':r['passage_ids']}
        for r in rows(args.index/'target-addresses.jsonl')}
    bindings={(r['law_id'],r['requested_asof']):{'status':r['status'],'revision_id':r.get('revision_id')}
        for r in rows(args.index/'asof-bindings.jsonl')}
    nested=read(args.index/'nested-corpora.json')
    shared=SharedCosines(documents['vectors'])
    engine=TimedDiscoveryEngine(scale_chunks(chunks,set(nested['1000'])),tokens,shared,
        documents['aliases'],documents['ids'],targets,bindings,top_k=50,rrf_k=60)
    manifest=receipt(complete=False, queries=len(queries), completed_queries=0,
        variant_counts=dict(Counter(q['query_variant'] for q in queries.values())),
        index_path=str(args.index),query_index_path=str(args.query_index),
        query_sources=[str(p) for p in args.queries],embedding_pin=documents['manifest']['model_pin'],
        discovery_top_k=50,rrf_k=60,scale=1000,sealed_gold_read=False,
        ranking_models_loaded=0,evaluation_runs=0,metadata_policy='public text/date/query ID only',
        output='candidates.jsonl',binding_source='existing corpus date-to-version map',
        timing_definition={'retrieval_seconds':'dense/BM25/RRF plus bank setup; excluding binding',
            'binding_seconds':'date binding and current base feature construction',
            'feature_seconds':'revision features over full discovered address pool'})
    write_json(metadata,manifest)
    timing=Counter(); segment=query_segmenter()
    with (out/'candidates.jsonl').open('w',encoding='utf-8') as stream:
        for number,q in enumerate(sorted(queries.values(),key=lambda q:(q['target_asof'],q['query_id'])),1):
            begin=time.perf_counter(); terms=query_tokens(q['query_text'],segment)
            timing['query_tokenization_seconds']+=time.perf_counter()-begin
            vector=bank['vectors'][vector_rows[q['query_id']]]
            previous=engine.binding_seconds
            begin=time.perf_counter(); d=engine.discover(q,terms,vector)
            elapsed=time.perf_counter()-begin
            bound=engine.binding_seconds-previous
            begin=time.perf_counter(); record=release_record(q,d,chunk_map,tokens,terms)
            measured={'retrieval_seconds':elapsed-bound,'binding_seconds':bound,
                      'feature_seconds':time.perf_counter()-begin}
            record['timing']=measured; timing.update(measured)
            save_row(stream,record); engine._cache.clear()
            if number%100==0:
                stream.flush(); manifest.update(completed_queries=number,timing=dict(timing))
                write_json(metadata,manifest)
                print(json.dumps({'phase':'unlabeled_discovery','queries':number}),flush=True)
    manifest.update(complete=True,completed_queries=len(queries),timing=dict(timing),
        shared_dense_products=shared.products,retrieval_runs=engine.retrieval_runs,
        wall_seconds=time.perf_counter()-started)
    write_json(metadata,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',nargs='+',type=Path,default=[ROOT/'unlabeled/queries-exact-unlabeled.jsonl',
        ROOT/'unlabeled/queries-paraphrase-unlabeled.jsonl'])
    parser.add_argument('--query-index',type=Path,default=Path('data/index/hidate/holdout-test/queries'))
    parser.add_argument('--index',type=Path,default=Path('data/index/hidate/extended-benchmark/scale'))
    parser.add_argument('--output',type=Path,default=Path('runs/hidate/revision-aware/confirmation/discovery'))
    args=parser.parse_args()
    print(json.dumps(discover(args),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
