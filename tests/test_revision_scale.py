"""Small native scale checks: original pool, dated binding, unchanged RRF."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

from hidate.discovery import DiscoveryEngine
from hidate.reranking import select_candidates
from hidate.revision import normalize_scores
from hidate.scale import SharedCosines


sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location(
    'revision_scale', Path(__file__).parents[1]/'scripts/run_revision_scale.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def small_discovery():
    def statute(cid, law, revision, dates):
        return {'chunk_id': cid, 'parent_passage_id': cid+'-parent',
            'law_id': law, 'scope': 'MainProvision', 'article_num': '1',
            'revision_id': revision, 'source_role': 'full', 'source_kind': 'statute',
            'asof_dates': dates}
    old, now = '2020-01-01', '2021-01-01'
    chunks = [statute('a-history', 'A', 'A-old', [old]),
              statute('a-current', 'A', 'A-current', [now]),
              statute('b-current', 'B', 'B-current', [now]),
              statute('c-history', 'C', 'C-old', [old]),
              {'chunk_id': 'guide', 'parent_passage_id': 'guide-parent',
               'source_id': 'G', 'source_kind': 'guidance', 'source_role': 'full'}]
    tokens = {'a-history': ['old', '50'], 'a-current': ['new', '10'],
              'b-current': ['other'], 'c-history': ['old'], 'guide': ['advice']}
    targets = {(law, now, 'MainProvision', '1'):
        {'status': 'available', 'passage_ids': [law.lower()+'-current-parent']}
        for law in ('A','B')}
    bindings = {(law, now): {'status': 'collected', 'revision_id': law+'-current'}
                for law in ('A','B','C')}
    query = {'query_id': 'fixture-q', 'query_text': 'old 50', 'query_variant': 'exact_quote',
        'split': 'test', 'source_role': 'full', 'target_asof': now,
        'family_id': 'A', 'stratum': 'changed_cross_date', 'target_status': 'available',
        'target_key': ['A',now,'MainProvision','1'], 'target_passage_id': 'a-current-parent',
        'target_revision_id': 'A-current', 'source_quote_absent_from_target': True}
    shared = SharedCosines(np.array([[1.,0.],[.2,0.],[.4,0.],[.9,0.],[.3,0.]]))
    engine = DiscoveryEngine(chunks, tokens, shared, list(range(5)),
        [c['chunk_id'] for c in chunks], targets, bindings)
    discovery = engine.discover(query, ['old','50'], np.array([1.,0.]))
    return query, chunks, tokens, discovery


def test_full_pool_features_and_same_bound_parent_for_every_native_head():
    query, chunks, tokens, discovery = small_discovery()
    chunk_map = {c['chunk_id']:c for c in chunks}
    cached = worker.compact_discovery(query, discovery, chunk_map, tokens, ['old','50'], '1000')
    addresses = cached['addresses']
    assert len(addresses) == len(discovery['addresses']) == 4
    assert any(a['status']=='absent' for a in addresses)
    np.testing.assert_allclose([a['historical_normalized'] for a in addresses],
        normalize_scores([a['features']['max_history_rrf'] for a in discovery['addresses']]))
    scores = {name: np.arange(len(addresses), dtype=float) for name in worker.TREE_HEADS}
    result = worker.native_predictions(query, cached, chunk_map, scores)
    assert set(result['conditions']) == set(worker.METHODS)
    expected = {tuple(a['address']): a['current_best_chunk_id'] for a in addresses}
    for name, condition in result['conditions'].items():
        if name == 'AsOf_hybrid_native':
            continue
        for item in condition['candidates']:
            assert item['current_chunk_id'] == expected[tuple(item['address'])]
            assert item['current_chunk_id'] != 'a-history'
        assert len({c['current_parent_passage_id'] for c in condition['candidates']}) == len(condition['candidates'])
    assert result['conditions']['RRF_native']['structural']['target_recall_at_four']


def test_native_rrf_keeps_original_scores_order_and_history_cache():
    query, chunks, tokens, discovery = small_discovery()
    chunk_map = {c['chunk_id']:c for c in chunks}
    cached = worker.compact_discovery(query, discovery, chunk_map, tokens, ['old','50'], '1000')
    result = worker.native_predictions(query, cached, chunk_map,
        {name: np.zeros(len(cached['addresses'])) for name in worker.TREE_HEADS})
    reference = select_candidates(discovery, chunk_map, conditions=['history_bound'])['history_bound']['candidates']
    native = result['conditions']['RRF_native']
    assert [c['current_chunk_id'] for c in native['candidates']] == [c['chunk_id'] for c in reference]
    assert [c['native_score'] for c in native['candidates']] == [c['features']['max_history_rrf'] for c in reference]
    for original, saved in zip(discovery['addresses'], cached['addresses']):
        assert original['history_chunk_hits'] == saved['history_chunk_hits']
        assert original['features']['max_history_rrf'] == saved['features']['max_history_rrf']
    a = next(a for a in cached['addresses'] if a['address'][0]=='A')
    assert a['historical_representative_chunk_id'] == 'a-history'
    assert a['current_best_chunk_id'] == 'a-current'
    assert cached['asof']['hybrid'] == discovery['asof']['hybrid']


def test_scale_selection_handles_guidance_without_inventing_law_id():
    _, chunks, _, _ = small_discovery()
    selected = worker.scale_chunks(chunks, {'A'})
    assert {c['chunk_id'] for c in selected} == {'a-history','a-current','guide'}
    assert 'law_id' not in selected[-1]


def test_compact_cache_accepts_unlabeled_queries_without_gold_metadata():
    query, chunks, tokens, discovery = small_discovery()
    public = {k:query[k] for k in ('query_id','query_text','query_variant','split','source_role','target_asof')}
    cached = worker.compact_discovery(public,discovery,{c['chunk_id']:c for c in chunks},tokens,['old','50'],'1000')
    assert not {'family_id','stratum','target_status','quote_absent','target_key'}.intersection(cached)
    assert len(cached['addresses']) == len(discovery['addresses'])
