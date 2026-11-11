"""Offline demo: BOCD on a synthetic series with known changepoints.

Run from the repository root:  python -m scripts.demo_synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bocd import BOCD, ConstantHazard, DecreasingHazard, expanding_gaussian_log_pred
from bocd.plotting import plot_overview

# (length, mean, std) of each regime; shocks are fat-tailed outliers inside regime 1
REGIMES = [(250, 0.05, 0.8), (150, -0.10, 2.5), (200, 0.05, 1.0), (120, 0.00, 4.0), (230, 0.05, 1.2)]


def make_series(seed: int = 7) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    parts, cps, t = [], [], 0
    for n, m, s in REGIMES:
        parts.append(rng.normal(m, s, n))
        t += n
        cps.append(t)
    xs = np.concatenate(parts)
    xs[[60, 140, 200]] += np.array([6.0, -7.0, 5.0])  # isolated outliers, not regime changes
    return xs, cps[:-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    xs, cps = make_series(args.seed)
    t = np.arange(len(xs))
    res = BOCD(hazard=DecreasingHazard(50.0, 0.5), r_max=250).run(xs)
    res_const = BOCD(hazard=ConstantHazard(50.0), r_max=250).run(xs)
    gauss = expanding_gaussian_log_pred(xs)

    recent = res.prob_recent_change(window=10)
    detections = {}
    for cp in cps:
        hits = np.nonzero(recent[cp : cp + 50] > 0.5)[0]
        detections[cp] = int(hits[0]) if len(hits) else None
    outlier_alarm = {o: float(recent[o : o + 3].max()) for o in (60, 140, 200)}

    metrics = {
        "n": len(xs),
        "true_changepoints": cps,
        "detection_delay_steps (P(r<10)>0.5)": detections,
        "max P(r<10) right after isolated outliers": outlier_alarm,
        "mean_nlpd": {
            "expanding_gaussian": float(-gauss.mean()),
            "bocd_constant_hazard": float(res_const.nlpd.mean()),
            "bocd_decreasing_hazard": float(res.nlpd.mean()),
        },
    }
    print(json.dumps(metrics, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = plot_overview(
        t, np.cumsum(xs), res, baseline_log_pred=gauss, true_changepoints=cps,
        level_label="Cumulative sum", window=10, rolling=30,
    )
    fig.savefig(out / "synthetic_overview.png", dpi=150)
    (out / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {out / 'synthetic_overview.png'}")


if __name__ == "__main__":
    main()
