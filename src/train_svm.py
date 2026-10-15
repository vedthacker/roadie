"""Support Vector Machine (the project's proposed model)."""
from __future__ import annotations

import time

from sklearn.svm import SVC

from . import config


def train_svm(X_train, y_train, class_weight=None, **overrides):
    """Fit an SVC (one-vs-one multiclass). probability=False on purpose: Platt scaling is slow and
    the app reports raw decision scores instead of pseudo-probabilities."""
    params = {**config.SVM_DEFAULT, **overrides}
    model = SVC(class_weight=class_weight, decision_function_shape="ovr", **params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0
