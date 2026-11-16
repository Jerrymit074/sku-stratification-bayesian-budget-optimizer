"""End-to-end pipeline orchestrator: Step 1 -> Step 6 + artifact serialization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.bayesian_models import (
    fit_layer1_product_models,
    fit_layer2_ad_models,
    save_bayesian_json,
)
from src.clustering import cluster_products
from src.config import ARTIFACTS_DIR, CATEGORY_ORDER, CATEGORY_SPECS, DATA_DEMO_DIR
from src.data_generator import (
    extract_legacy_rules,
    generate_ad_daily_panel,
    generate_all_demo_data,
)
from src.optimizer import optimize_daily_budget, simulate_legacy_allocation
from src.preprocessing import preprocess_sku_features
from src.scoring import (
    compute_legacy_ad_scores,
    detect_daily_anomalies,
    score_daily_ad_performance,
)


def run_full_pipeline(
    regenerate_data: bool = True,
    margin_cap: float = 0.85,
    draws: int = 300,
    tune: int = 250,
    seed: int = 42,
) -> Dict:
    """Execute the full Two-Layer Six-Step pipeline and persist demo/artifact files."""
    t0 = time.perf_counter()
    DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    sku_csv = DATA_DEMO_DIR / "sku_master_demo.csv"
    ad_gz = DATA_DEMO_DIR / "ad_daily_demo.csv.gz"
    legacy_json = DATA_DEMO_DIR / "legacy_rules_baseline.json"

    if regenerate_data or not sku_csv.exists() or not legacy_json.exists():
        print("[1/6] Building anonymized SKU master & legacy baseline...", flush=True)
        sku_raw, legacy_rules = generate_all_demo_data(seed=seed)
    else:
        sku_raw = pd.read_csv(sku_csv)
        legacy_rules = json.loads(legacy_json.read_text(encoding="utf-8"))

    print("[2/6] Step 1: Running log smoothing & dynamic profit margin preprocessing...", flush=True)
    sku_prep, prep_diag = preprocess_sku_features(sku_raw, margin_floor=0.0, margin_cap=margin_cap)

    print("[3/6] Step 2: Running K-Means strategic role clustering (P1-P8)...", flush=True)
    sku_clustered, cluster_diag = cluster_products(sku_prep, n_clusters=8, random_state=seed)

    print("[4/6] Step 3: Fitting Layer-1 PyMC Bayesian weights per category...", flush=True)
    sku_classified, l1_models = fit_layer1_product_models(
        sku_clustered, draws=draws, tune=tune, random_seed=seed
    )
    sku_classified.to_csv(ARTIFACTS_DIR / "sku_classified.csv", index=False)
    save_bayesian_json(l1_models, ARTIFACTS_DIR / "bayesian_product_weights.json")

    if regenerate_data or not ad_gz.exists():
        print("[5/6] Generating causally structured multi-platform daily ad panel...", flush=True)
        ad_df = generate_ad_daily_panel(sku_classified, role_col="strategic_role", n_days=30, seed=seed)
        ad_df.to_csv(ad_gz, index=False, compression="gzip")
    else:
        ad_df = pd.read_csv(ad_gz)

    print("[5.5/6] Step 5: Fitting Layer-2 PyMC 5-metric ad weights & scoring latest day...", flush=True)
    l2_models = fit_layer2_ad_models(ad_df, sku_classified, draws=draws, tune=tune, random_seed=seed + 100)
    save_bayesian_json(l2_models, ARTIFACTS_DIR / "bayesian_ad_weights.json")

    latest_date = str(ad_df["date"].max())
    scored_day = score_daily_ad_performance(ad_df, sku_classified, l2_models, target_date=latest_date)
    legacy_scores = compute_legacy_ad_scores(scored_day, legacy_rules)
    scored_day["legacy_ad_score"] = legacy_scores
    scored_day.to_csv(ARTIFACTS_DIR / "daily_ad_scores.csv", index=False)

    anomalies = detect_daily_anomalies(ad_df, sku_classified, target_date=latest_date)
    anomalies.to_csv(ARTIFACTS_DIR / "daily_anomalies.csv", index=False)

    # Export lightweight UTF-8 30-day drilldown panel for top anomalies + top-spend SKUs (<500 KB)
    anom_ids = anomalies["sku_id"].head(50).tolist() if not anomalies.empty else []
    top_spend_ids = scored_day.sort_values("spend", ascending=False)["sku_id"].head(30).tolist()
    drilldown_ids = set(anom_ids + top_spend_ids)
    drill_cols = ["date", "sku_id", "platform", "spend", "impressions", "clicks", "ctr", "cvr", "orders", "ad_revenue", "roas", "acos"]
    ad_df.loc[ad_df["sku_id"].isin(drilldown_ids), drill_cols].to_csv(
        ARTIFACTS_DIR / "sku_drilldown_history.csv", index=False
    )

    # Also persist compact 30-day category-platform time series summary for fast dashboard charts
    ad_with_role = ad_df.merge(sku_classified[["sku_id", "strategic_role"]], on="sku_id", how="left")
    daily_trend = (
        ad_with_role.groupby(["date", "strategic_role", "platform"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            ad_revenue=("ad_revenue", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            orders=("orders", "sum"),
        )
    )
    daily_trend["roas"] = np.round(daily_trend["ad_revenue"] / np.maximum(daily_trend["spend"], 1.0), 3)
    daily_trend["ctr"] = np.round(daily_trend["clicks"] / np.maximum(daily_trend["impressions"], 1.0), 5)
    daily_trend["cvr"] = np.round(daily_trend["orders"] / np.maximum(daily_trend["clicks"], 1.0), 5)
    daily_trend.to_csv(ARTIFACTS_DIR / "daily_trend_summary.csv", index=False)

    print("[6/6] Step 6: Solving constrained daily budget optimization...", flush=True)
    default_budget = float(round(scored_day["spend"].sum(), -2))
    opt_df, cat_summary, opt_metrics = optimize_daily_budget(
        scored_day, total_budget=default_budget, alpha=0.65
    )
    opt_df.to_csv(ARTIFACTS_DIR / "budget_allocation_baseline.csv", index=False)
    cat_summary.to_csv(ARTIFACTS_DIR / "category_budget_summary.csv", index=False)

    legacy_bench = simulate_legacy_allocation(scored_day, legacy_scores, legacy_rules, total_budget=default_budget, alpha=0.65)

    # Compute P1/P2/P5 vs P7 share in optimized plan
    opt_p7_pct = float(opt_df.loc[opt_df["strategic_role"] == "P7", "optimized_spend"].sum() / default_budget * 100.0)
    opt_p125_pct = float(opt_df.loc[opt_df["strategic_role"].isin(["P1", "P2", "P5"]), "optimized_spend"].sum() / default_budget * 100.0)

    avg_l1_rho = round(float(np.mean([m["spearman_rho"] for m in l1_models.values()])), 4)
    avg_l2_rho = round(float(np.mean([m["spearman_rho"] for m in l2_models.values()])), 4)

    summary = {
        "latest_date": latest_date,
        "margin_cap_default": margin_cap,
        "preprocessing": prep_diag,
        "clustering": cluster_diag,
        "layer1_avg_spearman_rho": avg_l1_rho,
        "layer2_avg_spearman_rho": avg_l2_rho,
        "optimization": {
            **opt_metrics,
            **legacy_bench,
            "opt_p7_spend_pct": round(opt_p7_pct, 2),
            "opt_p125_spend_pct": round(opt_p125_pct, 2),
            "roas_vs_legacy_uplift_pct": round(
                (opt_metrics["optimized_roas"] - legacy_bench["legacy_roas"])
                / max(legacy_bench["legacy_roas"], 1e-6)
                * 100.0,
                2,
            ),
        },
        "anomaly_count": int(len(anomalies)),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    (ARTIFACTS_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Pipeline complete in {summary['runtime_seconds']}s.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full stratification & budget optimization pipeline.")
    parser.add_argument("--no-regen", action="store_true", help="Reuse existing demo CSVs instead of regenerating.")
    parser.add_argument("--margin-cap", type=float, default=0.85, help="Dynamic margin upper cap (default 0.85).")
    parser.add_argument("--draws", type=int, default=300, help="PyMC posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=250, help="PyMC tuning steps per chain.")
    args = parser.parse_args()
    run_full_pipeline(
        regenerate_data=not args.no_regen,
        margin_cap=args.margin_cap,
        draws=args.draws,
        tune=args.tune,
    )
