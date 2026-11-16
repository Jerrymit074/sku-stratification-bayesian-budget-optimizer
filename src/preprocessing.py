"""Step 1: Data preprocessing — log smoothing & dynamic profit-margin ruler."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def log_smooth_score(series: pd.Series, ref_max: float | None = None) -> pd.Series:
    """Compute 0-100 log-smoothed score with zero-clipping protection for returns.

    Formula: S_score = ln(max(x, 0) + 1) / ln(ref_max + 1) * 100
    """
    clipped = np.clip(series.astype(float).fillna(0.0), a_min=0.0, a_max=None)
    max_val = float(clipped.max()) if ref_max is None else float(ref_max)
    if max_val <= 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    denom = np.log1p(max_val)
    scores = (np.log1p(clipped) / denom) * 100.0
    return pd.Series(np.clip(scores, 0.0, 100.0), index=series.index, dtype=float)


def dynamic_margin_score(
    series: pd.Series,
    floor: float = 0.0,
    cap: float = 0.85,
) -> pd.Series:
    """Map profit margin to 0-100 with floor/cap clipping.

    Negative margins clip to floor (score 0); extreme margins above cap clip to 100.
    """
    if cap <= floor:
        raise ValueError(f"cap ({cap}) must be strictly greater than floor ({floor})")
    vals = series.astype(float).fillna(floor)
    bounded = np.clip(vals, floor, cap)
    scores = ((bounded - floor) / (cap - floor)) * 100.0
    return pd.Series(np.clip(scores, 0.0, 100.0), index=series.index, dtype=float)


def preprocess_sku_features(
    df: pd.DataFrame,
    margin_floor: float = 0.0,
    margin_cap: float = 0.85,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run Step 1 preprocessing on SKU dataframe and return diagnostics."""
    out = df.copy()
    out["score_sales"] = log_smooth_score(out["sales_qty"])
    out["score_revenue"] = log_smooth_score(out["revenue"])
    out["score_profit"] = log_smooth_score(out["profit"])
    out["score_margin"] = dynamic_margin_score(
        out["profit_margin"], floor=margin_floor, cap=margin_cap
    )

    diagnostics = {
        "total_skus": int(len(out)),
        "negative_sales_clipped": int((out["sales_qty"] < 0).sum()),
        "negative_margin_count": int((out["profit_margin"] < margin_floor).sum()),
        "capped_margin_count": int((out["profit_margin"] > margin_cap).sum()),
        "capped_margin_pct": float((out["profit_margin"] > margin_cap).mean() * 100.0),
        "capped_at_60_pct": float((out["profit_margin"] > 0.60).mean() * 100.0),
        "margin_median": float(out["profit_margin"].median()),
        "margin_p95": float(out["profit_margin"].quantile(0.95)),
    }
    return out, diagnostics
