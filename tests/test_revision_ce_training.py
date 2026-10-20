"""Small software fixtures for final-stage CE training data contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hidate.discovery import FEATURE_NAMES
from hidate.revision import REVISION_FEATURES
from hidate.reranking import PairScoreCache
from run_revision_ce_training import (CE_FEATURES,feature_sets,selected_row,
    training_arrays,fixed_scores,prepare)
from run_revision_dualview import recipes


def candidate(label,rank=1):
    return {'label':label,'features':{k:float(rank) for k in set(FEATURE_NAMES)|set(REVISION_FEATURES)},
        'historical_ce':{'reranker_logit':float(4-rank),'reranker_score':.9-.1*rank},
        'current_ce':{'reranker_logit':float(rank),'reranker_score':.1*rank},'rrf_score':1/rank}


class CETrainingContractTests(unittest.TestCase):
    def test_same_information_feature_sets(self):
        d=feature_sets({'rank_features':list(FEATURE_NAMES)})
        self.assertEqual([len(v) for v in d.values()],[13,20])
        self.assertNotIn('available_target',d['base13'])
        self.assertTrue(set(CE_FEATURES).issubset(d['base13']))
        self.assertEqual(set(d['revision20'])-set(d['base13']),
            set(REVISION_FEATURES)-{'historical_query_coverage','current_query_coverage'})

    def test_positive_group_filter_keeps_original_labels(self):
        rows=[{'split':'train','candidates':[candidate(1,1),candidate(0,2)]},
              {'split':'train','candidates':[candidate(0,1),candidate(0,2)]}]
        names=feature_sets({'rank_features':list(FEATURE_NAMES)})['base13']
        X,y,h,c,sizes,stats=training_arrays(rows,names)
        self.assertEqual(X.shape,(2,13));self.assertEqual(y.tolist(),[1,0]);self.assertEqual(sizes.tolist(),[2])
        np.testing.assert_array_equal(h,[1,0]);np.testing.assert_array_equal(c,[0,1])
        self.assertEqual(stats['no_deliverable_positive_skipped'],1)
        self.assertEqual(stats['all_train_queries'],2)
        with self.assertRaises(ValueError):training_arrays([{'split':'valid','candidates':[]}],names)

    def test_fixed_fusions_replay_original_dualview(self):
        cc=[candidate(1,i) for i in (1,2,3)]
        for c in cc:c['features']['max_history_rrf']=c['rrf_score']
        old=recipes(cc);new=fixed_scores(cc)
        for name in new:
            if name.startswith('HC_'):np.testing.assert_array_equal(new[name],old[name])

    def test_selection_filters_binding_and_duplicate_parents(self):
        addresses=[{'address':['law','MainProvision',str(i)],'status':status,
            'current_best_chunk_id':cid,'historical_representative_chunk_id':'h'+str(i),
            'features':{'max_history_rrf':1/i}} for i,status,cid in
            [(1,'unavailable',None),(2,'available','c2'),(3,'available','c3'),(4,'available','c4')]]
        selected=selected_row({'query':{'query_id':'q','split':'train'},'addresses':addresses,'labels':[1,0,0,0]},
            {'c2':'parent','c3':'parent','c4':'other'})
        self.assertEqual([c['current_chunk_id'] for c in selected['candidates']],['c2','c4'])
        self.assertEqual([c['selection_rank'] for c in selected['candidates']],[1,2])
        self.assertFalse(any(c['label'] for c in selected['candidates']))

    def test_prepare_counts_missing_and_unavailable_positive_without_scoring(self):
        def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')
        def line(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x)+'\n' for x in v),encoding='utf-8')
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);query=[];features={'train':[],'valid':[]}
            for i,split in enumerate(['train','train','valid']):
                q={'query_id':'q'+str(i),'query_text':'raw '+str(i),'split':split,'family_id':'f',
                   'query_variant':'exact_quote','stratum':'same_date','target_status':'available',
                   'target_passage_id':'p'};query.append(q)
                aa=[{'address':['law','MainProvision','1'],'status':'available','current_best_chunk_id':'c',
                     'historical_representative_chunk_id':'h','features':{'max_history_rrf':1}}]
                labels=[1]
                if i==1:
                    aa.append({'address':['law','MainProvision','2'],'status':'unavailable',
                        'current_best_chunk_id':None,'historical_representative_chunk_id':'h',
                        'features':{'max_history_rrf':.5}});labels=[0,1]
                features[split].append({'query':q,'addresses':aa,'labels':labels})
            line(r/'q.jsonl',query);line(r/'index/chunks.jsonl',[{'chunk_id':'c','parent_passage_id':'p','text':'current'},
                {'chunk_id':'h','parent_passage_id':'old','text':'history'}])
            put(r/'development/features/preparation.json',{'complete':True,'seed':42,'queries':{'train':2,'valid':1}})
            for s in features:line(r/'development/features'/(s+'.jsonl'),features[s])
            put(r/'protocol.json',{'rank_features':list(FEATURE_NAMES)})
            put(r/'stack.json',{'models':{'reranker':{'model_id':'fixture','revision':'fixed'}}})
            from types import SimpleNamespace
            args=SimpleNamespace(output=r/'out',development=r/'development',queries=[r/'q.jsonl'],
                index=r/'index',protocol=r/'protocol.json',stack=r/'stack.json')
            args.output.mkdir();cache=PairScoreCache(None)
            for text in ('current','history'):cache.cache[('raw 2',text)]={}
            with patch('run_revision_ce_training.load_cache',return_value=(cache,{'fixture':True})):
                d=prepare(args,expected_counts={'train':2,'valid':1})
            self.assertEqual(d['unique_missing_pairs'],4)
            self.assertEqual(d['unique_missing_pairs_by_split'],{'train':4})
            self.assertEqual(d['training_candidate_statistics']['original_positive_unavailable'],1)
            self.assertEqual(d['training_candidate_statistics']['selected_queries_without_positive'],1)
            self.assertEqual(d['new_neural_calls'],0)


if __name__=='__main__':unittest.main()
