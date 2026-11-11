"""Hazard functions H(r): prior probability that a regime of length r ends at the next step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstantHazard:
    """Memoryless (geometric) prior on regime length: H(r) = 1 / lam."""

    lam: float = 50.0

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return np.full(np.shape(r), 1.0 / self.lam)


@dataclass(frozen=True)
class DecreasingHazard:
    """Hazard that decays with regime age: H(r) = 1 / (lam + r**gamma).

    Older regimes are a priori less likely to end ("regime memory"). For large r
    the hazard behaves like r**(-gamma); for small r it is capped at 1 / lam.
    With the defaults (lam=50, gamma=0.5) H ranges from 0.020 (r=0) to
    about 0.015 (r=250), so the prior effect is mild.
    """

    lam: float = 50.0
    gamma: float = 0.5

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return 1.0 / (self.lam + r**self.gamma)
