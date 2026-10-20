"""One TRAIN address model, represented by its fitted numeric parameters."""

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def training_examples(discoveries, queries, feature_names, required_variants=None):
    X, y, variants, families, used_queries = [], [], Counter(), set(), set()
    positive_unavailable = 0
    for row in discoveries:
        query = queries[row["query_id"]]
        if query["split"] != "train":
            raise ValueError("The address model may fit only TRAIN queries")
        gold = (query["law_id"], query["scope"], query["article_num"])
        used_queries.add(query["query_id"])
        families.add(query["family_id"])
        variants[query["query_variant"]] += 1
        for address in row["addresses"]:
            X.append([address["features"][name] for name in feature_names])
            positive = int(tuple(address["address"]) == gold)
            y.append(positive)
            positive_unavailable += positive and address["status"] != "available"
    if required_variants is not None and set(variants) != set(required_variants):
        raise ValueError("The one TRAIN fit requires all predeclared query variants")
    labels = np.asarray(y, dtype=np.int64)
    return np.asarray(X, dtype=np.float64), labels, {
        "query_count": len(used_queries), "candidate_count": len(labels),
        "positive_addresses": int(labels.sum()), "positive_unavailable": int(positive_unavailable),
        "families": sorted(families), "queries_by_variant": dict(variants)}


def fit_address_ranker(X, y, feature_names):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1., class_weight="balanced", max_iter=1000,
                               random_state=42).fit(scaler.transform(X), y)
    return {"model": "StandardScaler then LogisticRegression", "fit_runs": 1,
        "C": 1., "class_weight": "balanced", "max_iter": 1000, "random_state": 42,
        "feature_names": list(feature_names), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "iterations": model.n_iter_.tolist(),
        "classes": model.classes_.tolist()}


def address_probabilities(model, feature_rows):
    if not feature_rows:
        return np.empty(0, dtype=np.float64)
    X = np.asarray([[row[name] for name in model["feature_names"]] for row in feature_rows], dtype=np.float64)
    scaled = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = scaled @ np.asarray(model["coefficient"]) + model["intercept"]
    return np.exp(-np.logaddexp(0, -logits))
