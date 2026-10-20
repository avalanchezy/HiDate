"""Qwen-own TRAIN CE-conditioned 13/20-feature models, original VALID only."""
import sys
import run_revision_ce_training as worker


def main(argv=None):
    root='runs/hidate/revision-aware'
    defaults=['--output',root+'/qwen-ce-training',
        '--development',root+'/qwen-native-development',
        '--index','data/index/hidate-qwen3',
        '--stack','models/qwen3/model-stack.json',
        '--dev-metadata','runs/hidate/extended-benchmark/qwen3/dev/scoring.json',
        '--dev-pairs','runs/hidate/extended-benchmark/qwen3/dev/model-pair-receipts.jsonl',
        '--native-reference',root+'/qwen-native-development/valid/evaluation.json',
        '--required-backend','default','--compact-caches']
    return worker.main(defaults+(sys.argv[1:] if argv is None else argv))


if __name__=='__main__':main()
