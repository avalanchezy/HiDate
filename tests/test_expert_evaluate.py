"""A temporal nonapplicability answer remains present without a false retrieval zero."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_evaluation_keeps_all_queries_but_counts_only_provision_targets(tmp_path, monkeypatch):
    worker = importlib.import_module('run_expert_retrieval')
    metrics = importlib.import_module('hidate.expert_metrics')
    selected = 'Full_LM18_historical_CE'
    out = tmp_path / 'final'
    out.mkdir()
    (out / 'scoring.json').write_text('{"complete": true}', encoding='utf-8')
    (tmp_path / 'fit.json').write_text('{"models": {}}', encoding='utf-8')
    (tmp_path / 'policy.json').write_text('{"alpha": 1}', encoding='utf-8')
    gold = [dict(query_id='B', family_id='F', category='changed_conditions',
                 answerability='answerable', retrieval_target_type='temporal_nonapplicability'),
            dict(query_id='A', family_id='F', category='changed_conditions',
                 answerability='answerable'),
            dict(query_id='I', family_id='F', category='unanswerable',
                 answerability='insufficient_facts')]
    (tmp_path / 'gold.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in gold), encoding='utf-8')
    (out / 'scored.jsonl').write_text(''.join(json.dumps(dict(query_id=q['query_id'],
        target_asof='2026-06-24'))+'\n' for q in gold), encoding='utf-8')
    monkeypatch.setattr(worker, 'frozen', lambda args: {'selected_method': selected})
    monkeypatch.setattr(worker, 'compact_chunks', lambda path: [{'chunk_id': 'c'}])
    monkeypatch.setattr(worker, 'lightgbm_module', lambda: SimpleNamespace())
    monkeypatch.setattr(worker, 'rank_conditions', lambda *args: {selected: ['c']})
    monkeypatch.setattr(metrics, 'retrieval_metrics', lambda q, chunks, k:
        dict.fromkeys(metrics.METRIC_NAMES, 1.0 if q['query_id']=='A' else None))
    result = worker.evaluate(SimpleNamespace(output=tmp_path, index=tmp_path,
        gold=tmp_path/'gold.jsonl', ce_models=tmp_path, original_policy=tmp_path/'policy.json'))
    assert result['complete'] and result['queries'] == 3
    panel = result['panels'][selected]['answerable']
    assert panel['n'] == 1
    assert panel['revision_correct_set_complete_at_k'] == 1.0
    assert result['answerability'] == {'answerable': 2, 'insufficient_facts': 1}
    assert result['retrieval_eligibility'] == {
        'temporal_nonapplicability': 1, 'provision_set': 1, 'insufficient': 1}
    predictions = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == 3
    assert predictions[0]['retrieval_target_type'] == 'temporal_nonapplicability'
    assert all(value is None for value in predictions[0]['conditions'][selected]['structural'].values())
