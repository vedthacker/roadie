"""K-Nearest Neighbours (instance-based; distance-based so it needs scaled features)."""
from __future__ import annotations

import time

from sklearn.neighbors import KNeighborsClassifier

from . import config


def train_knn(X_train, y_train, **overrides):
    """Fit KNN. Returns (model, training_time_seconds). KNN has no class_weight option."""
    params = {**config.KNN_DEFAULT, **overrides}
    model = KNeighborsClassifier(n_jobs=-1, **params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)  # 'training' = storing the data
    return model, time.perf_counter() - t0
