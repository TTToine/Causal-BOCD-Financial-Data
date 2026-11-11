"""BOCD on S&P 500 dollar bars (2000-2024 by default).

Run from the repository root:

    python -m scripts.run_sp500                       # downloads ^GSPC with yfinance
    python -m scripts.run_sp500 --csv data/gspc.csv   # offline: CSV with Date, Close, Volume

Writes figures/sp500_overview.png and results/sp500_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from bocd import BOCD, ConstantHazard, DecreasingHazard, NormalGammaPrior, dollar_bars, expanding_gaussian_log_pred
from bocd.bars import log_returns
from bocd.plotting import plot_overview


def load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf

        df = yf.download(args.ticker, start=args.start, end=args.end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if args.save_csv:
            Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
            df[["Close", "Volume"]].to_csv(args.save_csv, index_label="Date")
    return df[["Close", "Volume"]].dropna().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--csv", help="offline input with Date, Close, Volume columns")
    ap.add_argument("--save-csv", help="cache the downloaded prices to this path")
    ap.add_argument("--multiplier", type=float, default=1.5, help="bar threshold = multiplier x mean dollar volume")
    ap.add_argument("--burn-in", type=int, default=252, help="rows used to set the threshold, then discarded")
    ap.add_argument("--full-sample-threshold", action="store_true",
                    help="original notebook behaviour: threshold from the whole sample (look-ahead)")
    ap.add_argument("--r-max", type=int, default=250)
    ap.add_argument("--window", type=int, default=5, help="bars for P(r_t < window)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    prices = load_prices(args)
    burn_in = None if args.full_sample_threshold else args.burn_in
    bars, threshold = dollar_bars(prices, multiplier=args.multiplier, burn_in=burn_in)
    r_bars = log_returns(bars["Close"])
    daily = prices if burn_in is None else prices.iloc[burn_in:]
    r_daily = log_returns(daily["Close"])
    xs = r_bars.to_numpy()

    prior = NormalGammaPrior(0.0, 1.0, 2.0, 1.0)
    res = BOCD(prior=prior, hazard=DecreasingHazard(50.0, 0.5), r_max=args.r_max).run(xs)
    res_const = BOCD(prior=prior, hazard=ConstantHazard(50.0), r_max=args.r_max).run(xs)
    res_drop = BOCD(prior=prior, hazard=DecreasingHazard(50.0, 0.5), r_max=args.r_max, truncation="drop").run(xs)
    gauss = expanding_gaussian_log_pred(xs)

    recent = pd.Series(res.prob_recent_change(args.window), index=r_bars.index)
    top = recent[recent > 0.5]
    metrics = {
        "period": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "threshold_rule": "full sample (look-ahead)" if burn_in is None else f"first {burn_in} rows, then discarded",
        "dollar_threshold": threshold,
        "n_daily_returns": len(r_daily),
        "n_bar_returns": len(xs),
        "kurtosis_daily (normal = 3)": float(kurtosis(r_daily, fisher=False)),
        "kurtosis_dollar_bars (normal = 3)": float(kurtosis(xs, fisher=False)),
        "mean_nlpd_on_bars": {
            "expanding_gaussian": float(-gauss.mean()),
            "bocd_constant_hazard": float(res_const.nlpd.mean()),
            "bocd_decreasing_hazard": float(res.nlpd.mean()),
            "bocd_decreasing_hazard_drop_truncation": float(res_drop.nlpd.mean()),
        },
        f"bars_with_P(r<{args.window})>0.5": len(top),
        f"first_dates_with_P(r<{args.window})>0.5_by_year": {
            str(y): str(g.index[0].date()) for y, g in top.groupby(top.index.year)
        },
    }
    print(json.dumps(metrics, indent=2))

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    fig = plot_overview(
        r_bars.index, bars["Close"].iloc[1:].to_numpy(), res, baseline_log_pred=gauss,
        level_label=f"{args.ticker} (dollar bars)", window=args.window, rolling=50,
    )
    fig.savefig(out / "figures" / "sp500_overview.png", dpi=150)
    (out / "results" / "sp500_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved figures/sp500_overview.png and results/sp500_metrics.json")


if __name__ == "__main__":
    np.seterr(divide="ignore")
    main()
