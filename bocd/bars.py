"""Dollar-bar resampling (López de Prado, 2018, ch. 2).

A new bar closes every time the traded dollar volume accumulated since the last
close reaches a threshold V. Busy periods produce more bars and quiet periods
fewer, which tends to make bar returns closer to i.i.d. Gaussian than returns
sampled at fixed time intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_bar_indices(dollar_volume: np.ndarray, threshold: float) -> np.ndarray:
    """Positions where a dollar bar closes. The first observation always opens the first bar."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    cum = np.cumsum(np.nan_to_num(np.asarray(dollar_volume, dtype=float)))
    indices = [0]
    last = 0.0
    for i in range(1, len(cum)):
        if cum[i] - last >= threshold:
            indices.append(i)
            last = cum[i]
    return np.asarray(indices)


def dollar_bars(
    prices: pd.DataFrame,
    multiplier: float = 1.5,
    burn_in: int | None = 252,
) -> tuple[pd.DataFrame, float]:
    """Resample a price series with 'Close' and 'Volume' columns into dollar bars.

    The threshold is ``multiplier`` times the mean dollar volume per row.

    - ``burn_in=k`` (default): the mean is estimated on the first k rows only and
      those rows are then discarded, so the sampling rule uses no future information.
    - ``burn_in=None``: the mean is taken over the whole sample, as in the original
      exam notebook. This leaks future volume into the sampling rule.

    Returns the bars and the threshold used.
    """
    dv = (prices["Close"] * prices["Volume"]).fillna(0.0).to_numpy()
    if burn_in is None:
        threshold = multiplier * dv.mean()
        data = prices
    else:
        if burn_in >= len(prices):
            raise ValueError("burn_in must be shorter than the series")
        threshold = multiplier * dv[:burn_in].mean()
        data = prices.iloc[burn_in:]
        dv = dv[burn_in:]
    idx = dollar_bar_indices(dv, threshold)
    return data.iloc[idx], float(threshold)


def log_returns(close: pd.Series, scale: float = 100.0) -> pd.Series:
    """Percentage log returns between consecutive rows."""
    return (scale * np.log(close / close.shift(1))).dropna()
