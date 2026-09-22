from __future__ import annotations

import numpy as np


def generate_contextual_returns(
    n_samples: int = 256,
    n_features: int = 4,
    n_assets: int = 5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    beta = rng.normal(scale=0.15, size=(n_features, n_assets))
    mu = x @ beta
    returns = mu + rng.normal(scale=0.25, size=mu.shape)
    covariance = np.cov(returns, rowvar=False) + 0.05 * np.eye(n_assets)
    return x.astype("float32"), returns.astype("float32"), covariance
