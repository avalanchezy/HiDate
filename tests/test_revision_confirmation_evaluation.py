import numpy as np

from evaluate_revision_confirmation import select_pools, result_rows


def test_head_pool_changes_do_not_change_bound_text():
    addresses=[{'address':['law','Main',str(i)],'status':'available',
        'current_best_chunk_id':'c'+str(i),'historical_representative_chunk_id':'h'+str(i),
        'features':{'max_history_rrf':1/(i+1)}} for i in range(12)]
    parents={'c'+str(i):'p'+str(i) for i in range(12)}
    pool=select_pools(addresses,{'LM':np.arange(12)},parents)
    assert pool['RRF']==list(range(10))
    assert pool['LM']==list(range(11,1,-1))
    assert addresses[pool['LM'][0]]['current_best_chunk_id']=='c11'


def test_metrics_deliver_current_even_when_historical_score_wins():
    cc=[{'address':['L','Main','1'],'current_chunk_id':'cur1','historical_chunk_id':'old1','selection_rank':1},
        {'address':['L','Main','2'],'current_chunk_id':'cur2','historical_chunk_id':'old2','selection_rank':2}]
    chunks={'cur1':{'chunk_id':'cur1','parent_passage_id':'p1','law_id':'L','scope':'Main',
                   'article_num':'1','revision_id':'new','source_kind':'law','source_role':'full'},
            'cur2':{'chunk_id':'cur2','parent_passage_id':'p2','law_id':'L','scope':'Main',
                   'article_num':'2','revision_id':'new','source_kind':'law','source_role':'full'}}
    q={'target_status':'available','target_passage_id':'p2', 'target_revision_id':'new',
       'target_key':['L','2026-09-30','Main','2'],'target_asof':'2026-09-30'}
    result=result_rows(q,cc,[.1,.9],chunks)
    assert result['evidence_chunk_ids']==['cur2','cur1']
    assert result['structural']['target_mrr']==1.
    assert result['logical_CE_views']==0
