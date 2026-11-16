"""Unit & integration tests for preprocessing, reverse normalization, clustering, and optimization."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster_products
from src.config import CATEGORY_ORDER, CATEGORY_SPECS
from src.data_generator import build_anonymized_sku_master, extract_legacy_rules, generate_ad_daily_panel
from src.optimizer import optimize_daily_budget
from src.preprocessing import dynamic_margin_score, log_smooth_score, preprocess_sku_features
from src.scoring import detect_daily_anomalies, normalize_metric_0_100, score_daily_ad_performance


def test_log_smooth_clips_negative_returns():
    s = pd.Series([-50.0, 0.0, 50000.0, 100000.0])
    scores = log_smooth_score(s)
    assert scores.iloc[0] == 0.0
    assert scores.iloc[1] == 0.0
    assert 93.0 < scores.iloc[2] < 95.0
    assert pytest.approx(scores.iloc[3], abs=1e-6) == 100.0


def test_dynamic_margin_ruler_bounds():
    s = pd.Series([-0.70, 0.0, 0.30, 0.60, 0.95])
    scores_60 = dynamic_margin_score(s, floor=0.0, cap=0.60)
    assert scores_60.iloc[0] == 0.0
    assert scores_60.iloc[1] == 0.0
    assert pytest.approx(scores_60.iloc[2], abs=1e-6) == 50.0
    assert pytest.approx(scores_60.iloc[3], abs=1e-6) == 100.0
    assert pytest.approx(scores_60.iloc[4], abs=1e-6) == 100.0


def test_reverse_metric_inversion():
    acos = pd.Series([0.10, 0.25, 0.50, 0.80])
    normed = normalize_metric_0_100(acos, is_reverse=True)
    # Lowest ACOS must get the highest score
    assert normed.iloc[0] > normed.iloc[1] > normed.iloc[2] > normed.iloc[3]


def test_clustering_covers_all_eight_roles():
    sku_df = build_anonymized_sku_master(seed=42)
    prep_df, _ = preprocess_sku_features(sku_df, margin_cap=0.85)
    clustered_df, diag = cluster_products(prep_df, n_clusters=8, random_state=42)
    assert len(clustered_df) == len(sku_df)
    assert set(clustered_df["strategic_role"].unique()) == set(CATEGORY_ORDER)
    assert diag["kmeans_silhouette_k8"] > 0.1


def test_budget_optimizer_respects_bounds_and_conservation():
    sku_df = build_anonymized_sku_master(seed=42).iloc[:240].copy()
    prep_df, _ = preprocess_sku_features(sku_df, margin_cap=0.85)
    clustered_df, _ = cluster_products(prep_df, n_clusters=8, random_state=42)
    # Assign deterministic A/B/C tiers for fast unit test without running PyMC
    clustered_df["layer1_weighted_score"] = 60.0
    clustered_df["abc_tier"] = np.tile(["A", "B", "C"], len(clustered_df) // 3 + 1)[: len(clustered_df)]

    ad_df = generate_ad_daily_panel(clustered_df, role_col="strategic_role", n_days=5, seed=42)
    scored = score_daily_ad_performance(ad_df, clustered_df, ad_weights_summary={}, target_date=ad_df["date"].max())

    total_budget = 10000.0
    opt_df, cat_summary, metrics = optimize_daily_budget(scored, total_budget=total_budget, alpha=0.65)

    assert pytest.approx(opt_df["optimized_spend"].sum(), rel=1e-3) == total_budget
    assert pytest.approx(metrics["allocated_budget"], rel=1e-3) == total_budget

    for _, row in cat_summary.iterrows():
        c = row["category"]
        if row["sku_stream_count"] > 0:
            assert row["opt_share_pct"] >= CATEGORY_SPECS[c]["budget_min_pct"] * 100.0 - 0.5
            assert row["opt_share_pct"] <= CATEGORY_SPECS[c]["budget_max_pct"] * 100.0 + 0.5


def test_optimizer_rejects_infeasible_bounds():
    dummy = pd.DataFrame(
        {
            "sku_id": ["SKU-0001"],
            "strategic_role": ["P1"],
            "abc_tier": ["A"],
            "daily_ad_score": [80.0],
            "spend": [20.0],
            "ad_revenue": [100.0],
            "roas": [5.0],
            "acos": [0.20],
        }
    )
    bad_bounds = {c: (0.20, 0.30) for c in CATEGORY_ORDER}  # sum(min) = 1.60 > 1.0
    with pytest.raises(ValueError, match="Infeasible"):
        optimize_daily_budget(dummy, total_budget=1000.0, category_bounds=bad_bounds)


def test_anomaly_detector_diagnoses_ctr_drop_and_filters_low_volume():
    dates = pd.date_range("2026-10-01", periods=10, freq="D").strftime("%Y-%m-%d").tolist()
    rows = []
    for i, d in enumerate(dates):
        ctr_active = 0.030 if i < 9 else 0.010  # -66% drop on last day
        rows.append(
            {
                "date": d,
                "sku_id": "SKU-ACT",
                "platform": "Amazon Ads",
                "ctr": ctr_active,
                "cvr": 0.08,
                "roas": 3.5 if i < 9 else 1.2,
                "spend": 25.0,
                "clicks": 30,
            }
        )
        rows.append(
            {
                "date": d,
                "sku_id": "SKU-NOISE",
                "platform": "Amazon Ads",
                "ctr": 0.020 if i < 9 else 0.002,
                "cvr": 0.05,
                "roas": 2.0,
                "spend": 1.5,
                "clicks": 2,  # below hist_clicks >= 10 filter
            }
        )
    ad_df = pd.DataFrame(rows)
    sku_df = pd.DataFrame(
        {
            "sku_id": ["SKU-ACT", "SKU-NOISE"],
            "product_title": ["Active Item", "Low Vol Item"],
            "strategic_role": ["P2", "P8"],
            "abc_tier": ["A", "C"],
        }
    )
    alerts = detect_daily_anomalies(ad_df, sku_df, target_date=dates[-1])
    assert list(alerts["sku_id"]) == ["SKU-ACT"]
    assert "Creative Fatigue" in alerts.iloc[0]["diagnosis_en"]
