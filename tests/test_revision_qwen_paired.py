import pytest

from analyze_revision_qwen_test import merge_controls


def row(query, family, condition):
    return dict(query_id=query,family_id=family,query_variant='exact_quote',
                stratum='changed',target_asof='T',target_status='available',
                quote_absent=True,conditions=condition)


def test_controls_pair_by_query_id_not_input_order():
    main=[row('q1','f1',{'main':1,'unrelated':0}),row('q2','f2',{'main':2})]
    controls=[row('q2','f2',{'control':4}),row('q1','f1',{'control':3})]
    merged=merge_controls(main,controls,'main')
    assert [r['conditions'] for r in merged]==[{'main':1,'control':3},{'main':2,'control':4}]
    assert 'control' not in main[0]['conditions']


def test_controls_require_same_family_assignment():
    main=[row('q1','f1',{'main':1})]
    controls=[row('q1','wrong',{'control':3})]
    with pytest.raises(ValueError,match='family_id'):
        merge_controls(main,controls,'main')
