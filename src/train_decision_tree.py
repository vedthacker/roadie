"""Decision Tree (non-parametric, scale-invariant, high variance on 1764-d HOG vectors)."""
from __future__ import annotations

import time

from sklearn.tree import DecisionTreeClassifier

from . import config


def train_decision_tree(X_train, y_train, class_weight=None, **overrides):
    params = {**config.TREE_DEFAULT, **overrides}
    params["class_weight"] = class_weight if class_weight is not None else params.get("class_weight")
    model = DecisionTreeClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0
