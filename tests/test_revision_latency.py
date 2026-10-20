"""Small checks for measured per-method latency, not shared-cache accounting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import benchmark_revision_latency as worker
from test_revision_scale import small_discovery


def test_fixed_sampling_uses_only_valid_identity_and_variant():
    queries=[{'query_id':f'{v}-{i:02}', 'query_variant':v, 'split':'valid',
              'source_role':'full', 'unread_performance':i}
             for v in worker.VARIANTS for i in range(60)]
    picked,warmup=worker.sample_queries(queries)
    assert len(picked)==100 and warmup['query_id'] not in {q['query_id'] for q in picked}
    assert all(sum(q['query_variant']==v for q in picked)==50 for v in worker.VARIANTS)
    changed=[{**q,'unread_performance':-q['unread_performance']} for q in reversed(queries)]
    again,w2=worker.sample_queries(changed)
    assert [q['query_id'] for q in picked]==[q['query_id'] for q in again]
    assert warmup['query_id']==w2['query_id']


def small_runtime():
    q,chunks,tokens,d=small_discovery();q['split']='valid'
    chunkmap={c['chunk_id']:{**c,'text':c['chunk_id']} for c in chunks}
    # The first address's two versions contain identical text: dedup within a request.
    chunkmap['a-history']['text']=chunkmap['a-current']['text']='same text'
    calls={'encode':0,'ce':0,'rank':[],'bind':0,'scored_pairs':[]}
    class Encoder:
        def encode(self,texts,kind,include_inputs):
            calls['encode']+=1
            return {'embeddings':[[1.,0.]],'token_counts':[4], 'total_tokens':4,
                    'input_shape':[1,4], 'seconds':.01, 'sdpa_backend':'math'}
    class Reranker:
        def score(self,pairs,include_inputs):
            calls['ce']+=1;calls['scored_pairs'].append(pairs)
            n=len(pairs)
            assert len(set(pairs))==n and include_inputs is False
            return {'scores':[.5]*n,'logits':[0.]*n,'token_counts':[8]*n,
                    'total_tokens':8*n,'input_shape':[n,8], 'seconds':.01,'sdpa_backend':'math'}
    class Engine:
        def _rank_bank(self,bank,terms,scores):
            calls['rank'].append(bank['mode']);return deepcopy(d[bank['mode']])
        def _addresses(self,*args):
            calls['bind']+=1;return deepcopy(d['addresses'])
    key=(q['source_role'],q['target_asof'])
    banks={key+(mode,):{'mode':mode} for mode in ('history','asof')}
    views={k:(np.arange(5),np.array([[1.,0.]]*5)) for k in banks}
    runtime=SimpleNamespace(encoder=Encoder(),reranker=Reranker(),engine=Engine(),banks=banks,
        vector_views=views,vector_count=5,chunks=chunkmap,tokens=tokens,
        parents={cid:c['parent_passage_id'] for cid,c in chunkmap.items()},
        segmenter=lambda _:['old','50'],batch_size=10)
    return q,runtime,calls


def test_each_method_reexecutes_encoder_and_ce_without_cross_method_cache():
    q,runtime,calls=small_runtime()
    model={'feature_names':['historical_CE_query_rank','current_CE_query_rank'],
           'mean':[0.,0.], 'scale':[1.,1.], 'coefficient':[0.,0.,0.]}
    method={'name':'fixture_gate','branch':'history','head':'rrf','stage':'gate','model':model}
    first=worker.measure_query(q,method,runtime)
    second=worker.measure_query(q,method,runtime)
    assert calls['encode']==2 and calls['ce']==2 and calls['bind']==2
    assert calls['rank']==['history','history']
    assert first['CE']['logical_pairs'] > first['CE']['unique_pairs']
    assert first['CE']['unique_pairs']==second['CE']['unique_pairs']
    assert first['CE']['calls']==1 and first['CE']['input_tokens']==8*first['CE']['unique_pairs']
    assert first['CE']['padded_tokens']==first['CE']['input_tokens']
    assert 'a-history' not in first['evidence_chunk_ids']
    assert first['latency_seconds']>0


def test_asof_native_only_ranks_current_bank_and_never_binds_history():
    q,runtime,calls=small_runtime()
    method={'name':'AsOf_native','branch':'asof','head':'rrf','stage':'native'}
    result=worker.measure_query(q,method,runtime)
    assert calls=={'encode':1,'ce':0,'rank':['asof'],'bind':0,'scored_pairs':[]}
    assert result['CE']['calls']==0 and result['CE']['unique_pairs']==0
