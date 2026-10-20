"""Qwen-own CE TRAIN routing and exact score-template contracts."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hidate.reranking import PairScoreCache
from run_revision_ce_training import check_context
from run_revision_dualview import fill_missing
import run_revision_qwen_ce_training as qwen


class QwenCETrainingContracts(unittest.TestCase):
    def test_wrapper_uses_qwen_own_data_and_empty_cross_stack_cache(self):
        with patch.object(qwen.worker,'main') as run:
            qwen.main(['--phase','prepare'])
        args=run.call_args.args[0]
        self.assertEqual(args[args.index('--required-backend')+1],'default')
        self.assertEqual(args[args.index('--stack')+1],'models/qwen3/model-stack.json')
        self.assertIn('qwen-native-development',args[args.index('--development')+1])
        self.assertEqual(args[args.index('--compact-caches')+1],'--phase')

    def test_qwen_cache_context_rejects_backend_or_instruction_change(self):
        pin={'model_id':'Qwen/Qwen3-Reranker-0.6B','revision':'fixed'}
        meta={'model_pin':pin,'max_length':1024,'sdpa_backend':'default'}
        args=SimpleNamespace(required_backend='default')
        check_context(meta,pin,args)
        with self.assertRaises(ValueError):check_context({**meta,'sdpa_backend':'math'},pin,args)
        with self.assertRaises(ValueError):check_context(meta,{**pin,'conventions':{'instruction':'changed task'}},args)

    def test_compact_qwen_score_and_restart_do_not_load_ruri(self):
        class Scorer:
            def score(self,pairs,include_inputs):
                assert include_inputs is False
                return {'scores':[.5],'logits':[0.],'token_counts':[4],'total_tokens':4,
                    'input_shape':[1,4],'sdpa_backend':'default','seconds':.01,
                    'score_format':'conditional_yes_no','logit_format':'yes_minus_no','input_ids':['omit']}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);queries={'q':{'query_text':'query'}};chunks={'c':{'text':'document'}}
            work=fill_missing(PairScoreCache(None),[('q','c')],queries,chunks,{},out,
                factory=lambda p:Scorer(),required_backend='default')
            self.assertEqual(work['this_job']['pairs'],1)
            text=(out/'new-ce-pair-scores.jsonl').read_text()
            self.assertIn('conditional_yes_no',text);self.assertNotIn('input_ids',text)
            work=fill_missing(PairScoreCache(None),[('q','c')],queries,chunks,{},out,
                factory=lambda p:(_ for _ in ()).throw(AssertionError('do not reload model')),
                required_backend='default')
            self.assertEqual(work['this_job']['pairs'],0)


if __name__=='__main__':unittest.main()
