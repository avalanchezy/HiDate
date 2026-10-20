"""A supervised LambdaMART address baseline on the unchanged HiDate feature pool."""

from collections import Counter
from pathlib import Path

import numpy as np


LIGHTGBM_VERSION = '4.6.0'
FIXED_PARAMETERS = {
    'boosting_type': 'gbdt', 'objective': 'lambdarank', 'metric': 'ndcg',
    'n_estimators': 200, 'num_leaves': 31, 'learning_rate': .05,
    'min_child_samples': 20, 'n_jobs': 4, 'random_state': 42,
    'subsample': 1., 'subsample_freq': 0, 'colsample_bytree': 1.,
    'reg_alpha': 0., 'reg_lambda': 0., 'device_type': 'cpu',
    'deterministic': True, 'force_col_wise': True, 'label_gain': [0, 1],
    'verbosity': -1,
}


def grouped_training_data(discoveries, queries, feature_names):
    X, y, sizes, groups = [], [], [], []
    seen, families, variants = set(), set(), Counter()
    positives = positive_unavailable = 0
    for row in discoveries:
        q = queries[row['query_id']]
        if q['split'] != 'train':
            raise ValueError('LambdaMART can fit only the original TRAIN queries')
        if q['query_id'] in seen:
            raise ValueError('Duplicate TRAIN query group')
        seen.add(q['query_id']); families.add(q['family_id'])
        variants[q['query_variant']] += 1
        if not row['addresses']:
            raise ValueError('A TRAIN ranking group has no candidate addresses')
        gold = (q['law_id'], q['scope'], q['article_num'])
        positive = unavailable = 0
        for address in row['addresses']:
            label = int(tuple(address['address']) == gold)
            X.append([address['features'][name] for name in feature_names]); y.append(label)
            positive += label
            unavailable += label and address['status'] != 'available'
        size = len(row['addresses']); sizes.append(size)
        groups.append({'query_id': q['query_id'], 'family_id': q['family_id'],
            'query_variant': q['query_variant'], 'candidate_count': size,
            'positive_addresses': positive, 'positive_unavailable': int(unavailable)})
        positives += positive; positive_unavailable += unavailable
    statistics = {'query_count': len(seen), 'candidate_count': len(y),
        'positive_addresses': int(positives), 'positive_unavailable': int(positive_unavailable),
        'families': sorted(families), 'queries_by_variant': dict(variants),
        'ranking_groups': len(sizes), 'zero_positive_groups': sum(g['positive_addresses'] == 0 for g in groups)}
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int32),
            np.asarray(sizes, dtype=np.int32), groups, statistics)


def lightgbm_module():
    import lightgbm
    if lightgbm.__version__ != LIGHTGBM_VERSION:
        raise ValueError('Use the frozen lightgbm==4.6.0 baseline environment')
    return lightgbm


def fit_model(X, y, group_sizes, feature_names, model_path):
    lgb = lightgbm_module()
    if sum(group_sizes) != len(y) or len(X) != len(y):
        raise ValueError('Query group sizes must sum to the contiguous candidate rows')
    model = lgb.LGBMRanker(**FIXED_PARAMETERS)
    model.fit(X, y, group=group_sizes, feature_name=list(feature_names),
        eval_set=[(X, y)], eval_group=[group_sizes], eval_names=['train'], eval_at=[4])
    path = Path(model_path); path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    return {'lightgbm_version': lgb.__version__, 'parameters': dict(FIXED_PARAMETERS),
        'eval_at': [4], 'feature_names': list(feature_names), 'validation_used': False,
        'early_stopping': False, 'hyperparameter_search': False,
        'actual_trees': model.booster_.num_trees(), 'training_metric': model.evals_result_,
        'score_semantics': 'unbounded ranking margin, not a probability',
        'model_path': str(path)}


def load_model(path):
    return lightgbm_module().Booster(model_file=str(path))


def rank_addresses(model, addresses, feature_names):
    if not addresses:
        return []
    X = np.asarray([[a['features'][name] for name in feature_names] for a in addresses], dtype=np.float64)
    scores = np.asarray(model.predict(X, raw_score=True, num_threads=4))
    if scores.shape != (len(addresses),) or not np.isfinite(scores).all():
        raise ValueError('LambdaMART must emit one finite raw score per address')
    ordered = sorted(({**a, 'lambdamart_score': float(s)} for a, s in zip(addresses, scores)),
                     key=lambda a: (-a['lambdamart_score'], tuple(a['address'])))
    return [{**a, 'native_address_rank': rank} for rank, a in enumerate(ordered, 1)]


def canonical_candidates(ranked_addresses, chunks, limit=10):
    candidates, seen = [], set()
    for address in ranked_addresses:
        cid = address['current_best_chunk_id']
        if address['status'] != 'available' or cid is None:
            continue
        chunk = chunks[cid]
        if chunk['parent_passage_id'] in seen:
            continue
        seen.add(chunk['parent_passage_id'])
        candidates.append({**chunk, 'address': address['address'], 'features': address['features'],
            'binding_status': address['status'], 'address_rank': address['native_address_rank'],
            'selection_rank': len(candidates)+1, 'lambdamart_score': address['lambdamart_score']})
        if len(candidates) == limit:
            break
    return candidates
