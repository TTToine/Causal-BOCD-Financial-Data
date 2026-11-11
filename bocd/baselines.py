"""Simple one-step-ahead predictive baselines, scored with the same NLPD as BOCD."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expanding_gaussian_log_pred(xs: np.ndarray, min_obs: int = 2, prior_std: float = 1.0) -> np.ndarray:
    """log p(x_t | x_1..x_{t-1}) under a Gaussian with the mean and std of all past observations.

    No regimes and thin tails: the reference point for the Student-t BOCD predictive.
    The first ``min_obs`` points use N(0, prior_std^2).
    """
    xs = np.asarray(xs, dtype=float)
    out = np.empty(len(xs))
    csum = np.concatenate([[0.0], np.cumsum(xs)])
    csq = np.concatenate([[0.0], np.cumsum(xs**2)])
    for t, x in enumerate(xs):
        if t < min_obs:
            out[t] = norm.logpdf(x, 0.0, prior_std)
            continue
        mean = csum[t] / t
        var = max((csq[t] - t * mean**2) / (t - 1), 1e-12)
        out[t] = norm.logpdf(x, mean, np.sqrt(var))
    return out
