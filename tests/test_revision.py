"""Small scientific checks for inference-visible revision features and group loss."""

import numpy as np
import pytest


def test_representative_uses_retrieval_score_and_chunk_tie_break():
    from hidate.revision import historical_representative
    row = {'history_chunk_hits': [
        {'chunk_id': 'source-oracle', 'score': .01},
        {'chunk_id': 'b', 'score': .03}, {'chunk_id': 'a', 'score': .03}]}
    assert historical_representative(row) == 'a'


def test_local_query_loss_detected_amid_unchanged_background():
    from hidate.revision import revision_features
    background = ['unrelated'] * 200
    old = background + ['capital', '50', 'million', 'required'] + background
    new = background + ['capital', '10', 'million', 'required'] + background
    features = revision_features(['capital', '50', 'million'], old, new)
    assert features['query_terms_lost'] == pytest.approx(1 / 3)
    assert features['query_numeric_changed'] == 1
    assert features['focused_edit_fraction'] > 0
    same = revision_features(['capital', '50', 'million'], old, old)
    assert same['query_terms_lost'] == same['focused_edit_fraction'] == 0


def test_unavailable_comparison_is_neutral_not_a_false_total_revision():
    from hidate.revision import revision_features
    assert set(revision_features(['x'], ['x', 'old'], None).values()) == {0.0}


def test_query_local_normalization_preserves_ties_and_scale():
    from hidate.revision import normalize_scores
    np.testing.assert_allclose(normalize_scores([5, 5, 5], 'rank'), [.5] * 3)
    np.testing.assert_allclose(normalize_scores([1, 2, 2, 3], 'rank'), [0, .5, .5, 1])
    for mode in ('rank', 'minmax'):
        np.testing.assert_allclose(normalize_scores([1, 2, 4], mode),
                                   normalize_scores([12, 14, 18], mode))


def test_query_pair_loss_analytic_gradient_and_equal_group_weight():
    from hidate.revision import gate_objective
    X = np.array([[1., .2], [1., -.3], [1., .4], [1., -.2], [1., .7]])
    h = np.array([.8, .1, .2, .3, .9]); c = 1 - h
    y = np.array([1, 0, 1, 0, 0]); sizes = [2, 3]
    w = np.array([.3, -.2])
    loss, gradient = gate_objective(w, X, h, c, y, sizes, regularization=.01)
    step = 1e-6
    numeric = np.array([(gate_objective(w + np.eye(2)[j] * step, X, h, c, y, sizes, .01)[0]
                      - gate_objective(w - np.eye(2)[j] * step, X, h, c, y, sizes, .01)[0])
                       / (2 * step) for j in range(2)])
    np.testing.assert_allclose(gradient, numeric, atol=1e-7)
    first = gate_objective(w, X[:2], h[:2], c[:2], y[:2], [2], 0)[0]
    second = gate_objective(w, X[2:], h[2:], c[2:], y[2:], [3], 0)[0]
    combined = gate_objective(w, X, h, c, y, sizes, 0)[0]
    assert combined == pytest.approx((first + second) / 2)
    assert np.isfinite(loss)


def test_address_label_keeps_positive_with_missing_target():
    from hidate.revision import address_labels
    q = {'law_id': 'L', 'scope': 'Main', 'article_num': '1'}
    addresses = [{'address': ['L', 'Main', '1'], 'status': 'unavailable'},
                 {'address': ['L', 'Main', '2'], 'status': 'available'}]
    assert address_labels(q, addresses) == [1, 0]


def test_small_gate_learns_conditional_switch_not_only_global_weight():
    from hidate.revision import fit_gate, gate_scores
    # History is right in changed-text groups; current is right in stable groups.
    X = np.array([[1.], [1.], [-1.], [-1.]] * 5)
    h = np.array([1., 0., 0., 1.] * 5); c = 1 - h
    labels = np.array([1, 0, 1, 0] * 5)
    fitted = fit_gate(X, h, c, labels, [2] * 10, ['revision_signal'])
    scores, weight = gate_scores(fitted, X, h, c)
    assert fitted['converged']
    assert scores[0] > scores[1] and scores[2] > scores[3]
    assert weight[0] > .5 and weight[2] < .5
