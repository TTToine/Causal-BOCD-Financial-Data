"""Bayesian Online Changepoint Detection with a Normal-Gamma model, decreasing hazard and dollar bars."""

from .bars import dollar_bar_indices, dollar_bars, log_returns
from .baselines import expanding_gaussian_log_pred
from .hazards import ConstantHazard, DecreasingHazard
from .model import BOCD, BOCDResult, NormalGammaPrior

__all__ = [
    "BOCD",
    "BOCDResult",
    "ConstantHazard",
    "DecreasingHazard",
    "NormalGammaPrior",
    "dollar_bar_indices",
    "dollar_bars",
    "expanding_gaussian_log_pred",
    "log_returns",
]
