"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with a Normal-Gamma model.

Within a regime, observations are i.i.d. N(mu, 1/tau) with a conjugate Normal-Gamma
prior on (mu, tau). The posterior predictive for the next observation, given the
last r observations, is a Student-t, so the run-length recursion is exact up to
truncation at ``r_max``.

Cost per update is O(r_max): one predictive evaluation and one sufficient-statistic
update per retained run length. It does not grow with the length of the series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .hazards import DecreasingHazard


@dataclass(frozen=True)
class NormalGammaPrior:
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 1.0


@dataclass
class BOCDResult:
    """Output of :meth:`BOCD.run` for a series of length T.

    ``log_run_length[t, r]`` is log P(r_t = r | x_1..x_t), the run-length posterior
    *after* observing x_t (row t is aligned with observation t).
    ``log_pred[t]`` is log p(x_t | x_1..x_{t-1}), the one-step-ahead predictive density.
    """

    log_run_length: np.ndarray
    log_pred: np.ndarray

    @property
    def nlpd(self) -> np.ndarray:
        """Negative log predictive density per observation."""
        return -self.log_pred

    @property
    def run_length_posterior(self) -> np.ndarray:
        return np.exp(self.log_run_length)

    @property
    def map_run_length(self) -> np.ndarray:
        return np.argmax(self.log_run_length, axis=1)

    def prob_recent_change(self, window: int = 5) -> np.ndarray:
        """P(r_t < window | x_1..x_t): probability that a regime started within the last ``window`` steps.

        Prefer this to P(r_t = 0 | x_1..x_t). The latter equals a predictive-weighted
        average of the hazard, so it never leaves the range of H and carries almost no
        information about the data (see the README).
        """
        return np.exp(logsumexp(self.log_run_length[:, :window], axis=1))


@dataclass
class BOCD:
    """Online changepoint detector. Use :meth:`update` for streaming or :meth:`run` for a batch.

    ``truncation`` decides what happens to run-length mass beyond ``r_max``:

    - ``"merge"`` (default): it is added to the last bin, which then stands for r >= r_max.
    - ``"drop"``: it is discarded and the rest renormalised, as in the original exam
      notebook. Once a regime outlives r_max this inflates the short run lengths and
      produces spurious changepoint signals.
    """

    prior: NormalGammaPrior = field(default_factory=NormalGammaPrior)
    hazard: object = field(default_factory=DecreasingHazard)
    r_max: int = 250
    truncation: str = "merge"

    def __post_init__(self) -> None:
        if self.truncation not in ("merge", "drop"):
            raise ValueError("truncation must be 'merge' or 'drop'")
        self.reset()

    def reset(self) -> None:
        n = self.r_max + 1
        p = self.prior
        self.t = 0
        self.log_r = np.full(n, -np.inf)
        self.log_r[0] = 0.0  # before any data, the run length is 0
        self.mu = np.full(n, p.mu0)
        self.kappa = np.full(n, p.kappa0)
        self.alpha = np.full(n, p.alpha0)
        self.beta = np.full(n, p.beta0)

    def update(self, x: float) -> float:
        """Process one observation; return log p(x | past). Updates the run-length posterior in place."""
        m = min(self.t, self.r_max) + 1  # run lengths 0..m-1 currently have mass
        mu, kappa = self.mu[:m], self.kappa[:m]
        alpha, beta = self.alpha[:m], self.beta[:m]

        # 1. Predictive density of x under each run length (Student-t)
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_pi = student_t.logpdf(x, df=2.0 * alpha, loc=mu, scale=scale)

        # 2. Message passing: growth (r -> r+1) and changepoint (r -> 0)
        h = self.hazard(np.arange(m))
        log_msg = self.log_r[:m] + log_pi
        log_joint = np.empty(m + 1)
        log_joint[1:] = log_msg + np.log1p(-h)
        log_joint[0] = logsumexp(log_msg + np.log(h))

        # 3. Evidence and normalised posterior, truncated at r_max
        log_evidence = logsumexp(log_joint)
        keep = min(m + 1, self.r_max + 1)
        new_log_r = np.full(self.r_max + 1, -np.inf)
        new_log_r[:keep] = log_joint[:keep]
        if m + 1 > self.r_max + 1 and self.truncation == "merge":
            # the last bin means "r >= r_max": absorb the overflow instead of discarding it
            new_log_r[-1] = np.logaddexp(new_log_r[-1], log_joint[-1])
        self.log_r = new_log_r - logsumexp(new_log_r)

        # 4. Conjugate update of the sufficient statistics, shifted by one run length
        k = min(m, self.r_max)
        new_kappa = kappa[:k] + 1.0
        self.beta[1 : k + 1] = beta[:k] + kappa[:k] * (x - mu[:k]) ** 2 / (2.0 * new_kappa)
        self.mu[1 : k + 1] = (kappa[:k] * mu[:k] + x) / new_kappa
        self.alpha[1 : k + 1] = alpha[:k] + 0.5
        self.kappa[1 : k + 1] = new_kappa
        p = self.prior
        self.mu[0], self.kappa[0], self.alpha[0], self.beta[0] = p.mu0, p.kappa0, p.alpha0, p.beta0

        self.t += 1
        return float(log_evidence)

    def run(self, xs: np.ndarray) -> BOCDResult:
        self.reset()
        xs = np.asarray(xs, dtype=float)
        log_rl = np.empty((len(xs), self.r_max + 1))
        log_pred = np.empty(len(xs))
        for i, x in enumerate(xs):
            log_pred[i] = self.update(x)
            log_rl[i] = self.log_r
        return BOCDResult(log_run_length=log_rl, log_pred=log_pred)
