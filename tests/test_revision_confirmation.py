"""The confirmatory discovery stage needs no held-out answer fields."""
import json

from discover_revision_confirmation import load_inputs, release_record


def test_only_public_query_fields_are_passed_to_discovery(tmp_path):
    source = tmp_path/'q.jsonl'
    source.write_text(json.dumps({'query_id':'HOLDOUT-Q1-exact',
        'query_text':'old wording', 'query_variant':'exact_quote', 'target_asof':'2026-09-30',
        'source_quote_text':'not a feature', 'source_role':'full', 'split':'test',
        'evaluation_role':'confirmatory_test'})+'\n', encoding='utf-8')
    queries = load_inputs([source])
    q = next(iter(queries.values()))
    assert set(q) == {'query_id','query_text','query_variant','target_asof','source_role','split'}
    assert q['source_role'] == 'full'


def test_discovery_release_omits_gold_metadata():
    q = {'query_id':'HOLDOUT-Q1-exact','query_text':'old wording',
         'query_variant':'exact_quote','target_asof':'2026-09-30', 'source_role':'full',
         'split':'test'}
    d = {'addresses':[], 'asof':{'hybrid':[]}, 'history':{'hybrid':[]},
         'bank_counts':{'asof':3,'history':5}, 'current_feature_chunks':0}
    result = release_record(q,d,{}, {}, ['old'])
    assert result['scale']=='1000' and result['addresses']==[]
    assert not {'family_id','stratum','target_status','target_passage_id','quote_absent'} & result.keys()
    assert result['query_text']=='old wording'
