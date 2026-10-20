"""Small French transfer checks; no neural execution or benchmark selection."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'revision_fiscal', Path(__file__).parents[1] / 'scripts/run_revision_fiscal.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture():
    def chunk(cid, version, body, start, end=None):
        return {'chunk_id': version+'|0', 'article_cid': cid, 'version_id': version,
                'code_id': 'CGI', 'article_num': cid, 'text': body,
                'date_debut': start, 'date_fin': end}
    chunks = [chunk('A','a-old','Déficit agricole 50 euros','2000-01-01','2019-12-31'),
              chunk('A','a-now','Déficit agricole 10 euros','2020-01-01'),
              chunk('B','b-now','Revenus autres catégories','2010-01-01'),
              chunk('C','c-old','Déficit agricole 50 euros','2000-01-01','2001-01-01')]
    def address(cid, h, current, rrf, cosine):
        features = dict.fromkeys(worker.BASE_FEATURES, 0.)
        features.update(max_history_rrf=rrf,current_best_cosine=cosine,
                        available_target=float(current is not None))
        return {'article_cid': cid, 'bound_version_id': current.split('|')[0] if current else None,
                'current_best_chunk_id': current, 'history_chunk_ids': h, 'features': features}
    saved = {'qid':'q','date_anchor':'2021-01-01','scope':'in', 'addresses':[
        address('A',['a-old|0','a-now|0'],'a-now|0',.03,.1),
        address('B',['b-now|0'],'b-now|0',.02,.9),
        address('C',['c-old|0'],None,.01,-1.)], 'conditions':{'history_bound':{'evidence':[]}}}
    question = {'qid':'q','prompt':'Déficit agricole 50 euros ?',
                'date_anchor':'2021-01-01','scope':'in'}
    return question, saved, {c['chunk_id']:c for c in chunks}


def test_french_full_pool_features_representative_and_missing_current():
    q, saved, chunks = fixture()
    enriched = worker.enrich_fiscal_addresses(q, saved, chunks)
    assert len(enriched) == 3
    a,b,c = enriched
    assert a['historical_chunk_id'] == 'a-old|0'
    assert a['current_chunk_id'] == 'a-now|0'
    assert a['features']['query_numeric_changed'] == 1.
    assert a['historical_normalized'] == 1.
    assert b['historical_normalized'] == .5
    assert c['features']['query_score_disagreement'] == 0.
    assert c['current_chunk_id'] is None
    assert saved['addresses'][0]['features'].get('query_numeric_changed') is None


def test_native_heads_own_and_common_rrf_pool_deliver_same_legal_version():
    q, saved, chunks = fixture()
    addresses = worker.enrich_fiscal_addresses(q,saved,chunks)
    result = worker.rank_fiscal_heads(addresses, {'Original_LM10':[0.,10.,20.]}, chunks)
    assert [c['article_cid'] for c in result['Original_LM10']['own_pool']['candidates']] == ['B','A']
    assert result['RRF_native']['own_pool']['candidates'][0]['chunk_id'] == 'a-now|0'
    assert result['Original_LM10']['common_rrf10']['pool_chunk_ids'] == ['a-now|0','b-now|0']
    assert result['Original_LM10']['own_pool']['evidence'][1]['version_id'] == 'a-now'
    assert all(c['article_cid']!='C' for c in result['RRF_native']['own_pool']['candidates'])


def test_exact_pair_cache_uses_literal_text_and_distinct_french_context():
    q,saved,chunks=fixture()
    a=worker.enrich_fiscal_addresses(q,saved,chunks)
    conditions=worker.rank_fiscal_heads(a,{},chunks)
    cache={(q['prompt'],chunks['a-now|0']['text']):
           {'reranker_score':.2,'reranker_logit':-1.,'pair_tokens':20}}
    pairs=worker.prepare_ce_pairs(q,conditions,chunks,cache)
    assert any(p['document_text']==chunks['a-old|0']['text'] and p['cached_score'] is None for p in pairs)
    assert any(p['document_text']==chunks['a-now|0']['text'] and p['cached_score']['reranker_score']==.2 for p in pairs)
    assert not any('gold' in p for p in pairs)
    assert sum(p['document_text']==chunks['b-now|0']['text'] for p in pairs)==1
    assert any(set(p['views'])=={'current','historical'} for p in pairs)


def test_timely_effective_interval_preserves_old_versions_and_bound_variant():
    q,saved,chunks=fixture()
    hits=[{'chunk_id':'a-old|0','score':1.}, {'chunk_id':'a-now|0','score':.99},
          {'chunk_id':'b-now|0','score':.2}]
    addresses=worker.enrich_fiscal_addresses(q,saved,chunks)
    temporal=worker.timely_interval_conditions(hits,chunks,q['date_anchor'],addresses)
    info=temporal['scoring']
    assert info['distance_days'][0]>0 and info['distance_days'][1:]==[0.,0.]
    assert info['qit'] is None and info['dit'] is None
    assert info['valid_ratios']['QET_DIT']==0.
    assert info['date_source']=='released_legal_effective_interval'
    assert len(temporal['source_hits'])==3
    source=temporal['conditions']['TimelyRAG_interval_native']['own_pool']
    bound=temporal['conditions']['TimelyRAG_interval_bound']['own_pool']
    assert any(c['chunk_id']=='a-old|0' for c in source['candidates'])
    assert all(c['chunk_id']!='a-old|0' for c in bound['candidates'])
    assert len(source['evidence'])<=4 and len(bound['candidates'])<=10


def test_dense_pool_uses_start_eligibility_and_asof_hybrid_keeps_raw_score():
    q,saved,chunks=fixture()
    order=list(chunks)
    hits=worker.history_dense_hits(np.array([1.,.8,.4,.2]),order,chunks,q['date_anchor'])
    assert len(hits)==4 and hits[0]['chunk_id']=='a-old|0'
    conditions=worker.direct_asof_condition([{'chunk_id':'a-now|0','score':.031},
        {'chunk_id':'b-now|0','score':.02}], chunks)
    assert conditions['own_pool']['candidates'][0]['native_score']==.031
    assert 'gold' not in conditions


def test_timely_binding_can_hydrate_a_dense_cid_missing_from_rrf_pool():
    q,saved,chunks=fixture()
    selected={'A':{'id':'a-now'},'B':{'id':'b-now'}}
    hits=[{'chunk_id':'b-now|0','score':1.}]
    bound=worker.temporal_bound_addresses(hits,chunks,selected,
        {'b-now':['b-now|0'],'a-now':['a-now|0']},
        np.array([.9,.2]),{'b-now|0':0,'a-now|0':1})
    assert bound==[{'article_cid':'B','current_chunk_id':'b-now|0'}]
    result=worker.timely_interval_conditions(hits,chunks,q['date_anchor'],bound)
    assert result['conditions']['TimelyRAG_interval_bound']['own_pool']['evidence'][0]['article_cid']=='B'
