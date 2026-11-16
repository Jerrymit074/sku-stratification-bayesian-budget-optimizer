"""Step 5 (daily): Category-specific 0-100 ad scoring (with reverse inversion) & anomaly detection."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import CATEGORY_ORDER, CATEGORY_SPECS, REVERSE_METRICS


def normalize_metric_0_100(series: pd.Series, is_reverse: bool = False) -> pd.Series:
    """Robustly normalize a metric series within a cohort to [0, 100].

    Clips extreme 2nd/98th percentiles to preserve differentiation.
    Inverts reverse metrics (ACOS, CAC, inventory_turnover_days) so higher = better.
    """
    vals = series.astype(float).fillna(series.median() if not series.dropna().empty else 0.0)
    lo = float(vals.quantile(0.02))
    hi = float(vals.quantile(0.98))
    if hi - lo < 1e-9:
        return pd.Series(np.full(len(series), 50.0), index=series.index, dtype=float)
    clipped = np.clip(vals, lo, hi)
    norm = (clipped - lo) / (hi - lo) * 100.0
    if is_reverse:
        norm = 100.0 - norm
    return pd.Series(np.clip(norm, 0.0, 100.0), index=series.index, dtype=float)


def score_daily_ad_performance(
    ad_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    ad_weights_summary: Dict[str, Dict],
    target_date: str | None = None,
) -> pd.DataFrame:
    """Compute daily 4+1 composite ad score for each SKU-platform on target_date (default latest)."""
    if target_date is None:
        target_date = str(ad_df["date"].max())

    day_df = ad_df[ad_df["date"] == target_date].copy()
    day_df = day_df.merge(
        sku_df[
            [
                "sku_id",
                "product_title",
                "department",
                "strategic_role",
                "abc_tier",
                "layer1_weighted_score",
                "legacy_abc_tier",
                "profit_margin",
            ]
        ],
        on="sku_id",
        how="left",
    )

    day_df["daily_ad_score"] = 0.0
    for i in range(1, 6):
        day_df[f"z_m{i}"] = 0.0
        day_df[f"m{i}_name"] = ""
        day_df[f"w_m{i}"] = 0.0

    for role in CATEGORY_ORDER:
        mask = day_df["strategic_role"] == role
        if mask.sum() == 0:
            continue
        spec = CATEGORY_SPECS[role]
        metrics = spec["ad_metrics"]
        model_info = ad_weights_summary.get(role)
        if model_info:
            weights = [float(item["posterior_mean"]) for item in model_info["weights"]]
        else:
            weights = list(spec["layer2_priors"])
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        composite = np.zeros(int(mask.sum()), dtype=float)
        for j, m_name in enumerate(metrics):
            is_rev = m_name in REVERSE_METRICS
            z_vals = normalize_metric_0_100(day_df.loc[mask, m_name], is_reverse=is_rev).values
            day_df.loc[mask, f"z_m{j+1}"] = np.round(z_vals, 2)
            day_df.loc[mask, f"m{j+1}_name"] = m_name
            day_df.loc[mask, f"w_m{j+1}"] = round(weights[j], 4)
            composite += weights[j] * z_vals

        day_df.loc[mask, "daily_ad_score"] = np.round(composite, 2)

    return day_df


def detect_daily_anomalies(
    ad_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    target_date: str | None = None,
    drop_threshold: float = -0.35,
) -> pd.DataFrame:
    """Compare latest day ROAS/CTR/CVR against trailing 14-day mean and diagnose bottlenecks."""
    all_dates = sorted(ad_df["date"].unique())
    if target_date is None:
        target_date = all_dates[-1]

    idx_t = all_dates.index(target_date)
    hist_start = max(0, idx_t - 14)
    hist_dates = set(all_dates[hist_start:idx_t])

    latest = ad_df[ad_df["date"] == target_date].copy()
    if not hist_dates:
        return pd.DataFrame()

    hist = (
        ad_df[ad_df["date"].isin(hist_dates)]
        .groupby(["sku_id", "platform"], as_index=False)[["ctr", "cvr", "roas", "spend", "clicks"]]
        .mean()
        .rename(
            columns={
                "ctr": "hist_ctr",
                "cvr": "hist_cvr",
                "roas": "hist_roas",
                "spend": "hist_spend",
                "clicks": "hist_clicks",
            }
        )
    )

    merged = latest.merge(hist, on=["sku_id", "platform"], how="inner")
    merged = merged.merge(
        sku_df[["sku_id", "product_title", "strategic_role", "abc_tier"]],
        on="sku_id",
        how="left",
    )

    merged["ctr_delta_pct"] = (merged["ctr"] - merged["hist_ctr"]) / np.maximum(merged["hist_ctr"], 1e-6)
    merged["cvr_delta_pct"] = (merged["cvr"] - merged["hist_cvr"]) / np.maximum(merged["hist_cvr"], 1e-6)
    merged["roas_delta_pct"] = (merged["roas"] - merged["hist_roas"]) / np.maximum(merged["hist_roas"], 1e-6)

    # Filter low-volume noise (require meaningful click & spend baseline)
    meaningful_volume = (merged["hist_clicks"] >= 10.0) & (merged["hist_spend"] >= 5.0)
    is_alert = meaningful_volume & (
        (merged["ctr_delta_pct"] <= drop_threshold)
        | (merged["cvr_delta_pct"] <= drop_threshold)
        | (merged["roas_delta_pct"] <= drop_threshold)
    )
    alerts = merged[is_alert].copy()
    if alerts.empty:
        return alerts

    def _diagnose(row: pd.Series) -> Tuple[str, str, str]:
        c_drop = row["ctr_delta_pct"] <= drop_threshold
        v_drop = row["cvr_delta_pct"] <= drop_threshold
        r_drop = row["roas_delta_pct"] <= drop_threshold
        if c_drop and not v_drop:
            return ("HIGH" if row["ctr_delta_pct"] < -0.45 else "MEDIUM", "F2: Creative Fatigue (CTR Drop)", "F2：素材点击率骤降")
        if v_drop and not c_drop:
            return ("HIGH" if row["cvr_delta_pct"] < -0.45 else "MEDIUM", "F3: Listing/Offer Friction (CVR Drop)", "F3：详情页转化率异常下滑")
        if c_drop and v_drop:
            return ("HIGH", "F2+F3: Funnel Collapse", "F2+F3：曝光点击与转化双降")
        if r_drop:
            return ("HIGH" if row["roas_delta_pct"] < -0.45 else "MEDIUM", "F4: Auction Cost / ROAS Erosion", "F4：投放成本上升或回报骤降")
        return ("MEDIUM", "General Deviation", "一般指标波动")

    diag = alerts.apply(_diagnose, axis=1)
    alerts["severity"] = [d[0] for d in diag]
    alerts["diagnosis_en"] = [d[1] for d in diag]
    alerts["diagnosis_zh"] = [d[2] for d in diag]

    alerts = alerts.sort_values(["severity", "roas_delta_pct"], ascending=[True, True]).reset_index(drop=True)
    return alerts


def compute_legacy_ad_scores(
    day_df: pd.DataFrame,
    legacy_rules: Dict,
) -> pd.Series:
    """Reproduce legacy 7-metric ABC weighted scoring from arrangement_source.xlsx as benchmark."""
    df = day_df.copy()
    df["cpc"] = df["spend"] / np.maximum(df["clicks"], 1)
    df["cpa"] = df["spend"] / np.maximum(df["orders"], 0.5)
    df["cpm"] = df["spend"] / np.maximum(df["impressions"], 1) * 1000.0
    df["roi"] = (df["ad_revenue"] * df["profit_margin"].clip(lower=0.05) - df["spend"]) / np.maximum(df["spend"], 1.0)

    metric_cols = {
        "ROAS": ("roas", False),
        "ROI": ("roi", False),
        "CPM": ("cpm", True),
        "CTR": ("ctr", False),
        "CVR": ("cvr", False),
        "CPC": ("cpc", True),
        "CPA": ("cpa", True),
    }

    normed = {}
    for m_key, (col, is_rev) in metric_cols.items():
        normed[m_key] = normalize_metric_0_100(df[col], is_reverse=is_rev).values

    scores = np.zeros(len(df), dtype=float)
    tiers = df["legacy_abc_tier"].fillna("C").values
    weights_by_tier = legacy_rules["weights_by_tier"]

    for tier in ("A", "B", "C"):
        mask = tiers == tier
        if mask.sum() == 0:
            continue
        w_map = weights_by_tier[tier]
        s_vec = np.zeros(int(mask.sum()), dtype=float)
        for m_key, w_val in w_map.items():
            s_vec += float(w_val) * normed[m_key][mask]
        scores[mask] = s_vec

    return pd.Series(np.round(scores, 2), index=df.index)
