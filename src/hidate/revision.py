"""Query-focused revision signals and a small query-grouped fusion head."""

from difflib import SequenceMatcher
import re
import unicodedata

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


REVISION_FEATURES = (
    'query_terms_lost', 'query_terms_gained', 'query_terms_retained',
    'historical_query_coverage', 'current_query_coverage',
    'focused_edit_fraction', 'focused_alignment_missing',
    'query_numeric_changed', 'query_score_disagreement',
)


def historical_representative(address):
    """A retrieved representative; no source/gold quotation is consulted."""
    hits = address['history_chunk_hits']
    return min(hits, key=lambda h: (-h['score'], h['chunk_id']))['chunk_id'] if hits else None


def normalize_scores(scores, mode='rank'):
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return values
    if len(values) == 1 or np.ptp(values) == 0:
        return np.full(len(values), .5)
    if mode == 'rank':
        return (rankdata(values, method='average') - 1) / (len(values) - 1)
    if mode == 'minmax':
        return (values - values.min()) / np.ptp(values)
    raise ValueError('Normalization must be rank or minmax')


def _windows(tokens, width=32, stride=16):
    if len(tokens) <= width:
        return [tokens]
    starts = list(range(0, len(tokens) - width + 1, stride))
    if starts[-1] != len(tokens) - width:
        starts.append(len(tokens) - width)
    return [tokens[start:start + width] for start in starts]


def _jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 1.


def revision_features(query_tokens, history_tokens, current_tokens):
    """Compare query-focused 32-token windows, not an entire document diff.

    No target version means no observed revision comparison; those signals are
    zero, while address identity and its positive training label remain intact.
    Query-score disagreement is filled from the common candidate pool afterward.
    """
    result = dict.fromkeys(REVISION_FEATURES, 0.)
    q = set(query_tokens)
    if current_tokens is None or not history_tokens or not q:
        return result
    h, c = set(history_tokens), set(current_tokens)
    matched_h, matched_c = q & h, q & c
    result.update(query_terms_lost=len(matched_h - c) / len(q),
                  query_terms_gained=len(matched_c - h) / len(q),
                  query_terms_retained=len(matched_h & c) / len(q),
                  historical_query_coverage=len(matched_h) / len(q),
                  current_query_coverage=len(matched_c) / len(q))
    hw = max(_windows(history_tokens), key=lambda w: len(set(w) & q))
    hs = set(hw)
    cw = max(_windows(current_tokens), key=lambda w: (_jaccard(hs, set(w)), len(set(w) & q)))
    result['focused_alignment_missing'] = 1 - _jaccard(hs, set(cw))
    result['focused_edit_fraction'] = 1 - SequenceMatcher(None, hw, cw, autojunk=False).ratio()
    numeric = {t for t in q if re.search(r'\d', unicodedata.normalize('NFKC', t))}
    result['query_numeric_changed'] = len((numeric & h) - c) / len(numeric) if numeric else 0.
    return result


def enrich_addresses(discovery, query_terms, document_tokens):
    addresses = discovery['addresses']
    h = normalize_scores([a['features']['max_history_rrf'] for a in addresses])
    c = normalize_scores([a['features']['current_best_cosine'] for a in addresses])
    enriched = []
    for index, address in enumerate(addresses):
        hid, cid = historical_representative(address), address['current_best_chunk_id']
        f = revision_features(query_terms, document_tokens.get(hid, []),
                              document_tokens[cid] if cid is not None else None)
        # Missing current evidence cannot turn a relevant address into a negative.
        current = float(c[index]) if cid is not None else float(h[index])
        f['query_score_disagreement'] = float(h[index]) - current
        enriched.append({**address, 'historical_representative_chunk_id': hid,
            'features': {**address['features'], **f},
            'historical_normalized': float(h[index]), 'current_normalized': current})
    return enriched


def address_labels(query, addresses):
    gold = (query['law_id'], query['scope'], query['article_num'])
    return [int(tuple(a['address']) == gold) for a in addresses]


def ranking_pairs(labels, sizes):
    positives, negatives, weights = [], [], []
    offset = 0
    active_groups = 0
    for size in sizes:
        group = np.asarray(labels[offset:offset + size])
        pos, neg = np.flatnonzero(group > 0), np.flatnonzero(group == 0)
        if len(pos) and len(neg):
            active_groups += 1
            positives.extend((np.repeat(pos, len(neg)) + offset).tolist())
            negatives.extend((np.tile(neg, len(pos)) + offset).tolist())
            weights.extend([1 / (len(pos) * len(neg))] * (len(pos) * len(neg)))
        offset += size
    if offset != len(labels) or not active_groups:
        raise ValueError('Need contiguous query groups with at least one positive/negative pair')
    return (np.asarray(positives), np.asarray(negatives),
            np.asarray(weights) / active_groups)


def _gate_loss(w, X, h, c, pairs, regularization):
    pos, neg, pair_weight = pairs
    gate = expit(X @ w)
    scores = 10 * (c + gate * (h - c))
    margin = scores[pos] - scores[neg]
    loss = np.dot(pair_weight, np.logaddexp(0, -margin))
    derivative = -pair_weight * expit(-margin)
    ds = (np.bincount(pos, weights=derivative, minlength=len(h))
          - np.bincount(neg, weights=derivative, minlength=len(h)))
    gradient = X.T @ (ds * 10 * (h - c) * gate * (1 - gate))
    penalty = w.copy(); penalty[0] = 0
    return (float(loss + regularization * np.dot(penalty, penalty) / 2),
            gradient + regularization * penalty)


def gate_objective(w, X, h, c, labels, sizes, regularization=.001):
    return _gate_loss(w, X, h, c, ranking_pairs(labels, sizes), regularization)


def fit_gate(X, h, c, labels, sizes, feature_names, regularization=.001):
    """Fixed small linear gate; one mean pair loss per contributing query."""
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    pairs = ranking_pairs(labels, sizes)
    result = minimize(_gate_loss, np.zeros(design.shape[1]),
        args=(design, h, c, pairs, regularization), jac=True, method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
    return {'model': 'linear revision-conditional gate', 'seed': 42,
        'feature_names': list(feature_names), 'mean': mean.tolist(), 'scale': scale.tolist(),
        'coefficient': result.x.tolist(), 'regularization': regularization,
        'score_temperature': 10., 'query_loss': 'mean positive-negative logistic loss per query',
        'iterations': int(result.nit), 'converged': bool(result.success),
        'optimizer_message': str(result.message), 'training_loss': float(result.fun),
        'normalization': 'within-query average percentile rank; ties equal',
        'score_semantics': 'ranking score, not a calibrated probability'}


def gate_scores(model, features, historical, current):
    X = np.asarray(features, dtype=float)
    design = np.column_stack([np.ones(len(X)),
        (X - np.asarray(model['mean'])) / np.asarray(model['scale'])])
    g = expit(design @ np.asarray(model['coefficient']))
    return g * np.asarray(historical) + (1 - g) * np.asarray(current), g
