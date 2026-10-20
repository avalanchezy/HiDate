"""Address labels and portable one-fit probabilities for HiDate."""

import importlib

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("hidate.ranking")
    except ModuleNotFoundError:
        pytest.fail("HiDate address ranker is not implemented")


def query(identity="q", split="train"):
    return {"query_id": identity, "law_id": "L", "scope": "MainProvision",
            "article_num": "3_2", "split": split, "family_id": "F", "query_variant": "exact_quote"}


def row(identity="q"):
    return {"query_id": identity, "addresses": [
        {"address": ["L", "MainProvision", "3_2"], "status": "absent", "features": {"x": 2., "y": 0.}},
        {"address": ["L", "SupplProvision:1", "3_2"], "status": "available", "features": {"x": 0., "y": 1.}},
        {"address": ["L", "MainProvision", "3"], "status": "available", "features": {"x": -1., "y": 1.}}
    ]}


def test_address_training_preserves_branch_and_scope_and_unavailable_positive():
    X, y, statistics = module().training_examples([row()], {"q": query()}, ["x", "y"])
    assert X.tolist() == [[2., 0.], [0., 1.], [-1., 1.]]
    assert y.tolist() == [1, 0, 0]
    assert statistics["positive_unavailable"] == 1
    assert statistics["query_count"] == 1


def test_single_train_fit_never_accepts_valid_or_test_rows():
    with pytest.raises(ValueError, match="TRAIN"):
        module().training_examples([row()], {"q": query(split="valid")}, ["x", "y"])
    with pytest.raises(ValueError, match="TRAIN"):
        module().training_examples([row()], {"q": query(split="test")}, ["x", "y"])


def test_one_fit_requires_both_predeclared_query_variants():
    with pytest.raises(ValueError, match="query variants"):
        module().training_examples([row()], {"q": query()}, ["x", "y"],
            required_variants=["exact_quote", "generated_paraphrase"])


def test_portable_scaled_probability_matches_actual_fitted_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array([[2., 0.], [0., 1.], [-1., 1.], [3., 0.]])
    y = np.array([1, 0, 0, 1])
    fitted = module().fit_address_ranker(X, y, ["x", "y"])
    scaler = StandardScaler().fit(X)
    reference = LogisticRegression(C=1., class_weight="balanced", max_iter=1000,
                                   random_state=42).fit(scaler.transform(X), y)
    predictions = module().address_probabilities(fitted, [dict(zip(["x", "y"], r)) for r in X])
    assert predictions == pytest.approx(reference.predict_proba(scaler.transform(X))[:, 1], abs=1e-12)
    assert fitted["fit_runs"] == 1 and fitted["random_state"] == 42
    assert fitted["feature_names"] == ["x", "y"]
