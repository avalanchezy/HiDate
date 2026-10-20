"""Small scientific fixtures for Qwen historical/current CE transfer."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_revision_qwen_test as worker
from test_revision_scale import small_discovery
from run_revision_scale import compact_discovery
from hidate.revision import normalize_scores


def pin():
    return {'model_id': 'Qwen/Qwen3-Reranker-0.6B', 'revision': 'fixture',
            'runtime': 'qwen3', 'conventions': {'instruction': worker.INSTRUCTION}}


def test_native_and_both_ce_views_deliver_the_same_bound_current_parents():
    query, chunks, tokens, discovery = small_discovery()
    chunk_map = {c['chunk_id']: c for c in chunks}
    cached = compact_discovery(query, discovery, chunk_map, tokens, ['old','50'], '120')
    scores = {'RRF_native': [a['features']['max_history_rrf'] for a in cached['addresses']],
              'fixture_LM': np.arange(len(cached['addresses']), dtype=float)}
    selected = worker.preselect(cached, chunk_map, scores)
    expected = {tuple(a['address']): a['current_best_chunk_id'] for a in cached['addresses']}
    for name, candidates in selected['heads'].items():
        for i, c in enumerate(candidates):
            c['historical_ce'] = {'reranker_logit': float(i), 'reranker_score': i/10}
            c['current_ce'] = {'reranker_logit': float(-i), 'reranker_score': 1-i/10}
            assert c['current_chunk_id'] == expected[tuple(c['address'])]
        for stage in ('native','historical_ce','current_ce','identity_view'):
            indices, _, _ = worker.rank_candidates(candidates, stage)
            chosen = [candidates[i] for i in indices]
            assert len({c['current_parent_passage_id'] for c in chosen}) == len(chosen)
            assert 'a-history' not in [c['current_chunk_id'] for c in chosen]
    rrf = selected['heads']['RRF_native']
    assert [c['native_score'] for c in rrf] == sorted([c['native_score'] for c in rrf], reverse=True)
    assert all('features' in c for c in rrf)
    for i,c in enumerate(selected['asof']):
        c['current_ce'] = {'reranker_score':float(i)}
    order,_,_ = worker.rank_candidates(selected['asof'],'current_ce')
    assert order == list(reversed(range(len(selected['asof']))))[:4]
    assert all('historical_chunk_id' not in c for c in selected['asof'])


def test_qwen_pair_context_cannot_reuse_ruri_or_french_scores():
    metadata = {'model_pin': pin(), 'max_length': 1024, 'sdpa_backend': 'default'}
    worker.check_context(metadata, pin())
    for change in ({'sdpa_backend':'math'}, {'instruction':'Retrieve French legal passages'},
                   {'model_pin': {'model_id':'ruri','revision':'fixture'}}):
        with pytest.raises(ValueError):
            worker.check_context({**metadata, **change}, pin())


def test_exact_pair_cache_reuses_identical_text_and_resumes_without_model(tmp_path):
    queries = {'q': {'query_text':'same raw query'}, 'q2': {'query_text':'same raw query'}}
    chunks = {'h':{'text':'historical text'}, 'c':{'text':'current text'},
              'alias':{'text':'current text'}}
    cache = {('same raw query','current text'):
             {'reranker_score':.7,'reranker_logit':1.,'pair_token_count':8}}
    calls = []
    class Model:
        def score(self, pairs, include_inputs):
            calls.append(pairs)
            assert include_inputs is False
            assert pairs == [('same raw query','historical text')]
            return {'scores':[.8], 'logits':[2.], 'token_counts':[9], 'total_tokens':9,
                    'input_shape':[1,9], 'sdpa_backend':'default', 'seconds':.01,
                    'score_format':'conditional_yes_no', 'logit_format':'yes_minus_no',
                    'yes_token_id':9693, 'no_token_id':2152, 'no_yes_logits':[[1.,3.]]}
    refs = [('q','h'),('q','c'),('q2','alias'),('q2','h')]
    work = worker.fill_missing(cache, refs, queries, chunks, pin(), tmp_path,
                               factory=lambda _: Model())
    assert work['this_job']['pairs'] == 1
    assert work['unique_requested_pairs'] == 2
    assert len(calls) == 1
    second = {('same raw query','current text'): cache[('same raw query','current text')]}
    work2 = worker.fill_missing(second, refs, queries, chunks, pin(), tmp_path,
                                factory=lambda _: pytest.fail('resume loaded model'))
    assert work2['this_job']['calls'] == 0
    assert second == cache
    record = next(worker.rows(tmp_path/'new-ce-pair-scores.jsonl'))
    assert 'input_ids' not in record['result']
    assert record['result']['logit_format'] == 'yes_minus_no'


def test_ce_gate_uses_fixed_ten_rank_features_and_saved_train_standardization():
    candidates = [{'address':[str(i),'MainProvision','1'], 'selection_rank':i+1,
        'current_chunk_id':f'c{i}', 'features':{'query_score_disagreement':v},
        'historical_ce':{'reranker_logit':h}, 'current_ce':{'reranker_logit':c}}
        for i,(h,c,v) in enumerate([(7.,1.,.15),(7.,9.,.6),(2.,4.,.95)])]
    model = {'feature_names':['query_score_disagreement','historical_CE_query_rank',
                             'current_CE_query_rank'],
             'mean':[.4,.5,.5], 'scale':[.2,.3,.3], 'coefficient':[0.,1.,0.,0.]}
    _, scores, gates = worker.rank_candidates(candidates,'gate',model=model)
    expected_gate = 1/(1+np.exp(-(np.array([.15,.6,.95])-.4)/.2))
    np.testing.assert_allclose(gates, expected_gate)
    np.testing.assert_allclose(scores, expected_gate*normalize_scores([7,7,2])+
                              (1-expected_gate)*normalize_scores([1,9,4]))
    assert [c['features']['query_score_disagreement'] for c in candidates] == [.15,.6,.95]


def test_qwen_ce_model_cannot_silently_load_a_ruri_parent_scoring_receipt(tmp_path):
    model_path=tmp_path/'ce-training/models/gate_revision20.json'
    model_path.parent.mkdir(parents=True)
    worker.write_json(model_path,{'complete':True})
    worker.write_json(model_path.parent.parent/'scoring.json',{'complete':True,
        'model_pin':{'model_id':'ruri','revision':'fixture'},'max_length':1024,'sdpa_backend':'math'})
    recipe=tmp_path/'freeze.json'
    worker.write_json(recipe,{'complete':True,'selection_split':'original_valid',
        'selected_method':'Full_LM18_historical_CE','conditions':[{
            'name':'extra_gate','head':'RRF_native','stage':'gate','model_json':str(model_path)}]})
    with pytest.raises(ValueError,match='Qwen3 reranker pin'):
        worker.load_recipe(recipe,pin())
