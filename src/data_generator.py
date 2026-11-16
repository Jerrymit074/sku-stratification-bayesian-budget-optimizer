"""Mixed data generator: anonymizes real SKU snapshot & synthesizes causal multi-platform ad time series."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_ORDER,
    DATA_DEMO_DIR,
    PLATFORMS,
    RAW_ABC_PATH,
    RAW_ARRANGEMENT_PATH,
)


DEPARTMENT_MAP = {
    "CAT": "Cat Care",
    "DOG": "Dog Care",
    "DOG-CAT": "Multi-Pet",
    "SA": "Small Pet",
    "BIRD": "Avian Care",
    "OTHER": "General Pet",
    "Other": "General Pet",
    "HUMAN-P": "Lifestyle",
    "HUMAN": "Lifestyle",
}


def extract_legacy_rules(output_path: Path | None = None) -> Dict:
    """Extract legacy 7-metric ABC weights & score multipliers from arrangement_source.xlsx."""
    baseline = {
        "metrics": ["ROAS", "ROI", "CPM", "CTR", "CVR", "CPC", "CPA"],
        "weights_by_tier": {
            "A": {"ROAS": 0.20, "ROI": 0.15, "CPM": 0.10, "CTR": 0.15, "CVR": 0.10, "CPC": 0.10, "CPA": 0.20},
            "B": {"ROAS": 0.10, "ROI": 0.20, "CPM": 0.05, "CTR": 0.15, "CVR": 0.25, "CPC": 0.20, "CPA": 0.10},
            "C": {"ROAS": 0.10, "ROI": 0.10, "CPM": 0.15, "CTR": 0.10, "CVR": 0.15, "CPC": 0.20, "CPA": 0.20},
        },
        "score_multipliers": {
            "90-100": {"A": 1.25, "B": 1.10, "C": 1.00},
            "80-90": {"A": 1.05, "B": 1.00, "C": 0.85},
            "60-80": {"A": 1.00, "B": 0.90, "C": 0.60},
            "30-60": {"A": 0.90, "B": 0.65, "C": 0.40},
            "0-30": {"A": 0.70, "B": 0.40, "C": 0.30},
        },
    }
    if RAW_ARRANGEMENT_PATH.exists():
        try:
            df = pd.read_excel(RAW_ARRANGEMENT_PATH)
            parsed_weights = {"A": {}, "B": {}, "C": {}}
            for _, row in df.iloc[:7].iterrows():
                metric = str(row["KEY_METIRCS"]).strip()
                if metric in baseline["metrics"]:
                    parsed_weights["A"][metric] = float(row["A"])
                    parsed_weights["B"][metric] = float(row["B"])
                    parsed_weights["C"][metric] = float(row["C"])
            if len(parsed_weights["A"]) == 7:
                baseline["weights_by_tier"] = parsed_weights
        except Exception:
            pass

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    return baseline


def _sanitize_sub_category(raw_val: object) -> str:
    """Strip non-ASCII characters from ProductGroup2 to ensure zero proprietary text leak."""
    if pd.isna(raw_val):
        return "General"
    ascii_part = re.sub(r"[^A-Za-z0-9 &/-]+", " ", str(raw_val)).strip()
    ascii_part = re.sub(r"\s+", " ", ascii_part)
    if not ascii_part:
        return "General"
    # Keep first 2 words for clean grouping
    words = ascii_part.split(" ")[:2]
    return " ".join(words).title()


def _assign_latent_roles(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign realistic latent strategic roles P1-P8 grounded in SKU financial distributions."""
    n = len(df)
    roles = np.empty(n, dtype=object)

    sales = df["sales_qty"].values
    rev = df["revenue"].values
    margin = df["profit_margin"].values
    price = df["unit_price"].values
    profit = df["profit"].values

    q_sales_high = np.quantile(sales, 0.72)
    q_sales_low = np.quantile(sales, 0.35)
    q_price_high = np.quantile(price, 0.74)
    q_rev_high = np.quantile(rev, 0.70)
    q_margin_high = np.quantile(margin, 0.65)
    q_margin_low = np.quantile(margin, 0.16)

    for i in range(n):
        m = margin[i]
        s = sales[i]
        r = rev[i]
        p = price[i]
        prof = profit[i]

        # P7: Negative / very low margin or stagnant returns
        if m < 0.12 or s <= 0 or (m < q_margin_low and s < np.quantile(sales, 0.45)):
            roles[i] = "P7"
        # P2: High unit volume with relatively compressed margin
        elif s >= q_sales_high and m < 0.42:
            roles[i] = "P2"
        # P3: High unit volume with balanced price/margin
        elif s >= q_sales_high and p < q_price_high:
            roles[i] = "P3" if rng.random() < 0.80 else "P6"
        # P4: High ticket price & high revenue
        elif p >= q_price_high and r >= np.quantile(rev, 0.55):
            roles[i] = "P4" if rng.random() < 0.82 else "P6"
        # P5: High margin & strong profit contribution
        elif m >= q_margin_high and prof >= np.quantile(profit, 0.45):
            roles[i] = "P5" if rng.random() < 0.82 else "P6"
        # P6: Mature solid revenue items facing competitive pressure
        elif r >= q_rev_high * 0.80:
            roles[i] = "P6" if rng.random() < 0.65 else "P3"
        # P1 vs P8: Lower cumulative sales split between recent launches (P1) and steady long-tail (P8)
        elif s <= q_sales_low:
            roles[i] = "P1" if rng.random() < 0.42 else "P8"
        else:
            # Remaining middle band distributed naturally across P8, P1, P5, P3
            roles[i] = rng.choice(["P8", "P1", "P5", "P3"], p=[0.42, 0.22, 0.20, 0.16])

    return roles


def _synthesize_lifecycle_metrics(df: pd.DataFrame, roles: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Generate coherent lifecycle & operational fields conditioned on role & SKU financials."""
    n = len(df)
    days_since_launch = np.zeros(n, dtype=int)
    sales_mom_growth = np.zeros(n, dtype=float)
    revenue_stability = np.zeros(n, dtype=float)
    inv_turnover_days = np.zeros(n, dtype=float)

    norm_s = np.clip(np.log1p(np.maximum(df["sales_qty"].values, 0)) / 11.6, 0, 1)
    norm_r = np.clip(np.log1p(np.maximum(df["revenue"].values, 0)) / 15.0, 0, 1)
    norm_m = np.clip(df["profit_margin"].values, 0, 1)

    for i, role in enumerate(roles):
        if role == "P1":
            days_since_launch[i] = int(rng.integers(14, 86))
            sales_mom_growth[i] = float(
                0.15 + 0.35 * norm_s[i] + 0.12 * norm_m[i] + rng.normal(0.0, 0.08)
            )
            revenue_stability[i] = float(np.clip(0.42 + rng.normal(0, 0.10), 0.15, 0.75))
            inv_turnover_days[i] = float(np.clip(48 - 10 * norm_s[i] + rng.normal(0, 8), 22, 85))
        elif role == "P2":
            days_since_launch[i] = int(rng.integers(110, 420))
            sales_mom_growth[i] = float(np.clip(0.12 + rng.normal(0, 0.06), -0.02, 0.32))
            revenue_stability[i] = float(np.clip(0.76 + rng.normal(0, 0.07), 0.55, 0.94))
            inv_turnover_days[i] = float(np.clip(22 - 8 * norm_s[i] + rng.normal(0, 4), 8, 38))
        elif role == "P3":
            days_since_launch[i] = int(rng.integers(130, 450))
            sales_mom_growth[i] = float(np.clip(0.08 + rng.normal(0, 0.05), -0.05, 0.25))
            revenue_stability[i] = float(np.clip(0.80 + rng.normal(0, 0.06), 0.60, 0.95))
            inv_turnover_days[i] = float(np.clip(28 - 8 * norm_s[i] + rng.normal(0, 5), 12, 48))
        elif role == "P4":
            days_since_launch[i] = int(rng.integers(140, 460))
            sales_mom_growth[i] = float(np.clip(0.04 + rng.normal(0, 0.06), -0.08, 0.22))
            revenue_stability[i] = float(np.clip(0.78 + rng.normal(0, 0.07), 0.55, 0.94))
            inv_turnover_days[i] = float(np.clip(42 - 8 * norm_r[i] + rng.normal(0, 8), 18, 72))
        elif role == "P5":
            days_since_launch[i] = int(rng.integers(150, 480))
            sales_mom_growth[i] = float(np.clip(0.05 + rng.normal(0, 0.05), -0.05, 0.20))
            revenue_stability[i] = float(np.clip(0.82 + rng.normal(0, 0.05), 0.65, 0.96))
            inv_turnover_days[i] = float(np.clip(38 + rng.normal(0, 7), 18, 65))
        elif role == "P6":
            days_since_launch[i] = int(rng.integers(260, 520))
            sales_mom_growth[i] = float(np.clip(-0.02 + rng.normal(0, 0.03), -0.10, 0.06))
            revenue_stability[i] = float(
                np.clip(0.72 + 0.14 * norm_r[i] + 0.08 * norm_m[i] + rng.normal(0, 0.015), 0.72, 0.98)
            )
            inv_turnover_days[i] = float(np.clip(42 + rng.normal(0, 7), 22, 68))
        elif role == "P7":
            days_since_launch[i] = int(rng.integers(210, 540))
            sales_mom_growth[i] = float(np.clip(-0.26 + rng.normal(0, 0.09), -0.60, -0.05))
            revenue_stability[i] = float(np.clip(0.45 + rng.normal(0, 0.10), 0.18, 0.70))
            inv_turnover_days[i] = float(
                np.clip(155 - 48 * norm_s[i] - 20 * norm_m[i] + rng.normal(0, 8.0), 85, 260)
            )
        else:  # P8
            days_since_launch[i] = int(rng.integers(125, 430))
            sales_mom_growth[i] = float(np.clip(0.01 + rng.normal(0, 0.04), -0.09, 0.12))
            revenue_stability[i] = float(np.clip(0.75 + rng.normal(0, 0.07), 0.55, 0.92))
            inv_turnover_days[i] = float(np.clip(52 + rng.normal(0, 10), 26, 85))

    out = df.copy()
    out["days_since_launch"] = days_since_launch
    out["sales_mom_growth"] = np.round(sales_mom_growth, 4)
    out["revenue_stability"] = np.round(revenue_stability, 4)
    out["inventory_turnover_days"] = np.round(inv_turnover_days, 2)
    out["inventory_turnover_rate"] = np.round(365.0 / np.maximum(inv_turnover_days, 1.0), 4)
    out["profit_x_margin"] = np.round(
        np.maximum(out["profit"].values, 0.0) * np.maximum(out["profit_margin"].values, 0.0), 2
    )
    return out


def build_anonymized_sku_master(seed: int = 42) -> pd.DataFrame:
    """Build anonymized 2,511-SKU master dataset from Classify_ABC_2024-03-14.xlsx."""
    rng = np.random.default_rng(seed)

    if RAW_ABC_PATH.exists():
        raw = pd.read_excel(RAW_ABC_PATH)
        n = len(raw)

        # Deterministic anonymization maps
        suppliers = {v: f"SUP-{idx+1:03d}" for idx, v in enumerate(sorted(raw["SupplierCode"].astype(str).unique()))}
        brands = {v: f"Brand-{idx+1:03d}" for idx, v in enumerate(sorted(raw["Brand"].astype(str).unique()))}

        departments = raw["ProductGroup"].map(lambda x: DEPARTMENT_MAP.get(str(x), "General")).values
        sub_cats = raw["ProductGroup2"].map(_sanitize_sub_category).values

        # Light 2.5% jitter so exact private ledger values aren't exposed while distributions stay faithful
        jitter_s = rng.normal(1.0, 0.025, n)
        jitter_p = rng.normal(1.0, 0.025, n)

        raw_sales = raw["SALES"].astype(float).values
        sales_qty = np.where(raw_sales > 0, np.maximum(1, np.round(raw_sales * jitter_s)), raw_sales).astype(int)

        unit_price = np.round(np.maximum(0.0, raw["UNIT_PRICE"].astype(float).values * jitter_p), 2)
        unit_cost = np.round(np.maximum(0.0, raw["UNIT_COSE"].astype(float).values * jitter_p), 2)

        raw_rev = raw["Revenue"].astype(float).values
        revenue = np.where(
            raw_rev >= 0,
            np.round(sales_qty * unit_price, 2),
            np.round(raw_rev * jitter_p, 2),
        )
        profit_margin = np.round(
            np.where(
                raw["Profit_Margin"].astype(float).values >= 0,
                np.clip(raw["Profit_Margin"].astype(float).values + rng.normal(0, 0.008, n), 0.0, 1.0),
                raw["Profit_Margin"].astype(float).values,
            ),
            4,
        )
        profit = np.round(
            np.where(revenue >= 0, revenue * profit_margin, raw["Profit"].astype(float).values * jitter_p),
            2,
        )

        df = pd.DataFrame(
            {
                "sku_id": [f"SKU-{i+1:04d}" for i in range(n)],
                "supplier_id": raw["SupplierCode"].astype(str).map(suppliers).values,
                "brand_id": raw["Brand"].astype(str).map(brands).values,
                "brand_tier": raw["BrandLevel"].astype(str).values,
                "department": departments,
                "sub_category": sub_cats,
                "product_title": [f"{d} · {sc} #{i+1:04d}" for i, (d, sc) in enumerate(zip(departments, sub_cats))],
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "sales_qty": sales_qty,
                "revenue": revenue,
                "profit_margin": profit_margin,
                "profit": profit,
                "legacy_score_margin_w": np.round(raw["overall_score_sales&margin&Profit_margin"].astype(float).values, 4),
                "legacy_score_profit_w": np.round(raw["overall_score_sales&margin&Profit"].astype(float).values, 4),
                "legacy_abc_tier": raw["Confirm_level"].astype(str).values,
            }
        )
    else:
        # Fallback distribution-matched synthetic catalog if raw Excel is unavailable
        n = 2511
        sales_qty = np.round(np.exp(rng.normal(6.0, 1.8, n))).astype(int)
        unit_price = np.round(np.exp(rng.normal(4.3, 1.2, n)), 2)
        profit_margin = np.round(np.clip(rng.beta(4.0, 2.5, n), -0.4, 0.95), 4)
        revenue = np.round(sales_qty * unit_price, 2)
        profit = np.round(revenue * profit_margin, 2)
        unit_cost = np.round(unit_price * (1.0 - np.clip(profit_margin, -0.5, 0.99)), 2)
        tiers = np.where(np.argsort(np.argsort(-profit)) < 500, "A", np.where(np.argsort(np.argsort(-profit)) < 1500, "B", "C"))
        depts = rng.choice(["Cat Care", "Dog Care", "Multi-Pet", "Small Pet"], size=n, p=[0.41, 0.32, 0.16, 0.11])
        df = pd.DataFrame(
            {
                "sku_id": [f"SKU-{i+1:04d}" for i in range(n)],
                "supplier_id": [f"SUP-{rng.integers(1, 60):03d}" for _ in range(n)],
                "brand_id": [f"Brand-{rng.integers(1, 80):03d}" for _ in range(n)],
                "brand_tier": rng.choice(["L1", "L2", "L3", "L4", "STAR"], size=n, p=[0.31, 0.17, 0.15, 0.34, 0.03]),
                "department": depts,
                "sub_category": rng.choice(["Food", "Litter", "Hygiene", "Toys", "Accessories"], size=n),
                "product_title": [f"{d} Item #{i+1:04d}" for i, d in enumerate(depts)],
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "sales_qty": sales_qty,
                "revenue": revenue,
                "profit_margin": profit_margin,
                "profit": profit,
                "legacy_score_margin_w": np.round(rng.uniform(0.05, 0.95, n), 4),
                "legacy_score_profit_w": np.round(rng.uniform(0.05, 0.95, n), 4),
                "legacy_abc_tier": tiers,
            }
        )

    latent_roles = _assign_latent_roles(df, rng)
    df = _synthesize_lifecycle_metrics(df, latent_roles, rng)
    return df


def generate_ad_daily_panel(
    sku_df: pd.DataFrame,
    role_col: str = "strategic_role",
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate causally structured multi-platform daily ad records for all SKUs.

    Ensures strict funnel identities:
      clicks = impressions * ctr
      orders = clicks * cvr
      ad_revenue = orders * aov
      roas = ad_revenue / spend, acos = spend / ad_revenue
    And encodes realistic role-specific causal relationships for Layer-2 Bayesian learning.
    """
    rng = np.random.default_rng(seed + 100)
    dates = pd.date_range(end="2026-10-31", periods=n_days, freq="D").strftime("%Y-%m-%d").tolist()

    # Role-specific advertising priors (base spend, ctr, cvr, cpm, imp_share, ntb_pct)
    role_params = {
        "P1": {"spend": 18.0, "ctr": 0.018, "cvr": 0.055, "cpm": 12.0, "share": 0.18, "ntb": 0.68},
        "P2": {"spend": 52.0, "ctr": 0.028, "cvr": 0.095, "cpm": 10.5, "share": 0.34, "ntb": 0.54},
        "P3": {"spend": 36.0, "ctr": 0.022, "cvr": 0.088, "cpm": 11.5, "share": 0.28, "ntb": 0.36},
        "P4": {"spend": 34.0, "ctr": 0.017, "cvr": 0.052, "cpm": 14.5, "share": 0.42, "ntb": 0.32},
        "P5": {"spend": 30.0, "ctr": 0.021, "cvr": 0.085, "cpm": 13.5, "share": 0.26, "ntb": 0.28},
        "P6": {"spend": 24.0, "ctr": 0.024, "cvr": 0.092, "cpm": 15.0, "share": 0.52, "ntb": 0.20},
        "P7": {"spend": 14.0, "ctr": 0.015, "cvr": 0.068, "cpm": 8.5,  "share": 0.15, "ntb": 0.30},
        "P8": {"spend": 11.0, "ctr": 0.019, "cvr": 0.072, "cpm": 10.0, "share": 0.16, "ntb": 0.25},
    }
    plat_mult = {
        "Amazon Ads": {"cpm": 1.25, "ctr": 1.15, "cvr": 1.35, "spend_w": 0.55},
        "TikTok Ads": {"cpm": 0.72, "ctr": 1.08, "cvr": 0.72, "spend_w": 0.25},
        "Meta Ads":   {"cpm": 0.95, "ctr": 0.92, "cvr": 0.85, "spend_w": 0.20},
    }

    # Assign active platforms per SKU: A -> 3 platforms, B -> 2 platforms, C -> 1 platform
    tier_col = "abc_tier" if "abc_tier" in sku_df.columns else "legacy_abc_tier"
    rows = []

    # Pre-sample SKU-level latent quality so Bayesian regression detects true signal across SKUs
    n_skus = len(sku_df)
    sku_creative_q = rng.normal(1.0, 0.22, n_skus)
    sku_page_q = rng.normal(1.0, 0.20, n_skus)

    # Pick ~4% SKUs for injected recent-day funnel anomalies (Page 5 monitoring showcase)
    anomaly_skus = set(rng.choice(sku_df["sku_id"].values, size=max(40, int(n_skus * 0.04)), replace=False))
    anomaly_type_map = {
        sid: rng.choice(["CTR_DROP", "CVR_DROP", "ROAS_CRASH"]) for sid in anomaly_skus
    }

    for idx, row in enumerate(sku_df.itertuples(index=False)):
        sid = row.sku_id
        role = getattr(row, role_col)
        tier = getattr(row, tier_col)
        rp = role_params.get(role, role_params["P8"])
        unit_price = max(float(row.unit_price), 8.0)
        profit_margin = float(np.clip(row.profit_margin, 0.05, 0.90))
        inv_days_base = float(row.inventory_turnover_days)
        mom_growth_base = float(row.sales_mom_growth)

        if tier == "A":
            active_plats = PLATFORMS
        elif tier == "B":
            active_plats = ["Amazon Ads", "TikTok Ads"] if (idx % 2 == 0) else ["Amazon Ads", "Meta Ads"]
        else:
            active_plats = ["Amazon Ads"] if (idx % 5 != 0) else ["TikTok Ads"]

        cq = np.clip(sku_creative_q[idx], 0.55, 1.65)
        pq = np.clip(sku_page_q[idx], 0.55, 1.65)

        for plat in active_plats:
            pm = plat_mult[plat]
            day_arr = np.arange(n_days)
            season = 1.0 + 0.08 * np.sin(2 * np.pi * day_arr / 7.0 + (idx % 7))

            spend_base = rp["spend"] * pm["spend_w"] * (1.25 if tier == "A" else (0.90 if tier == "B" else 0.60))
            spend = np.round(np.maximum(2.0, spend_base * season * rng.normal(1.0, 0.12, n_days)), 2)

            cpm = np.maximum(3.0, rp["cpm"] * pm["cpm"] * rng.normal(1.0, 0.08, n_days))
            impressions = np.maximum(100, np.round(spend / cpm * 1000.0)).astype(int)

            ctr = np.clip(rp["ctr"] * pm["ctr"] * cq * rng.normal(1.0, 0.10, n_days), 0.003, 0.09)
            cvr = np.clip(rp["cvr"] * pm["cvr"] * pq * rng.normal(1.0, 0.12, n_days), 0.008, 0.28)

            # Inject realistic anomaly in last 3 days for flagged SKUs
            if sid in anomaly_type_map and plat == active_plats[0]:
                atype = anomaly_type_map[sid]
                if atype == "CTR_DROP":
                    ctr[-3:] *= 0.38
                elif atype == "CVR_DROP":
                    cvr[-3:] *= 0.40
                else:
                    cvr[-3:] *= 0.45
                    cpm[-3:] *= 1.45
                    impressions[-3:] = np.maximum(80, np.round(spend[-3:] / cpm[-3:] * 1000.0)).astype(int)

            clicks = np.maximum(1, np.round(impressions * ctr)).astype(int)
            orders = np.maximum(0, np.round(clicks * cvr)).astype(int)
            aov = np.round(unit_price * rng.uniform(1.02, 1.38, n_days), 2)
            ad_revenue = np.round(orders * aov, 2)
            ad_profit = np.round(ad_revenue * profit_margin - spend, 2)

            roas = np.round(ad_revenue / np.maximum(spend, 0.01), 4)
            acos = np.round(
                np.where(ad_revenue > 0, spend / np.maximum(ad_revenue, 1e-6), np.minimum(2.5, spend / np.maximum(aov * 0.5, 1.0))),
                4,
            )
            imp_share = np.round(np.clip(rp["share"] * cq * rng.normal(1.0, 0.08, n_days), 0.03, 0.92), 4)
            ntb_pct = np.round(np.clip(rp["ntb"] * rng.normal(1.0, 0.09, n_days), 0.08, 0.92), 4)
            new_cust = np.maximum(0.4, orders * ntb_pct)
            cac = np.round(np.minimum(300.0, spend / new_cust), 2)

            sales_growth_rate = np.round(mom_growth_base + 0.15 * (ctr - rp["ctr"]) / rp["ctr"] + rng.normal(0, 0.03, n_days), 4)
            margin_change_pct = np.round(0.02 * (roas - 3.0) + rng.normal(0, 0.015, n_days), 4)
            brand_search_growth = np.round(
                0.08 * (imp_share - 0.25) + 0.04 * (ctr - 0.02) * 10 + rng.normal(0, 0.02, n_days),
                4,
            )
            inv_days = np.round(np.clip(inv_turnover_days_step(inv_days_base, impressions, cvr, rng, n_days), 5.0, 280.0), 2)
            repeat_rate = np.round(np.clip(0.28 + 0.12 * (cvr - 0.05) + rng.normal(0, 0.03, n_days), 0.05, 0.72), 4)

            # Causally aligned Layer-2 regression target variables
            # P1 -> Day T+30 sales; P2 -> Total Traffic; P3 -> Total Sales; P4 -> Revenue; P5/P8 -> Profit; P6 -> Stability; P7 -> Clearance speed
            sales_t30 = np.round(
                np.maximum(
                    0.0,
                    0.0018 * impressions * (1 + 8 * ctr) * (1 + 4 * cvr) * (1 + 0.8 * ntb_pct) * (1.2 - 0.3 * np.clip(acos, 0, 1.5))
                    + rng.normal(0, 2.0, n_days),
                ),
                2,
            )
            total_traffic = np.round(clicks + 0.05 * impressions, 2)
            clearance_speed = np.round(
                np.maximum(0.1, (120.0 / np.maximum(inv_days, 5.0)) * (1 + 0.0001 * impressions) * (1 + 3 * cvr) + rng.normal(0, 0.2, n_days)),
                4,
            )
            rev_stab_idx = np.round(
                np.clip(0.55 + 0.35 * imp_share + 0.45 * brand_search_growth + 0.8 * ctr + rng.normal(0, 0.03, n_days), 0.10, 0.99),
                4,
            )

            for d_i in range(n_days):
                rows.append(
                    {
                        "date": dates[d_i],
                        "sku_id": sid,
                        "platform": plat,
                        "spend": float(spend[d_i]),
                        "impressions": int(impressions[d_i]),
                        "impression_share": float(imp_share[d_i]),
                        "clicks": int(clicks[d_i]),
                        "ctr": float(round(ctr[d_i], 5)),
                        "cvr": float(round(cvr[d_i], 5)),
                        "orders": int(orders[d_i]),
                        "aov": float(aov[d_i]),
                        "ad_revenue": float(ad_revenue[d_i]),
                        "ad_profit": float(ad_profit[d_i]),
                        "roas": float(roas[d_i]),
                        "acos": float(acos[d_i]),
                        "cac": float(cac[d_i]),
                        "ntb_pct": float(ntb_pct[d_i]),
                        "sales_growth_rate": float(sales_growth_rate[d_i]),
                        "margin_change_pct": float(margin_change_pct[d_i]),
                        "brand_search_growth_pct": float(brand_search_growth[d_i]),
                        "inventory_turnover_days": float(inv_days[d_i]),
                        "repeat_purchase_rate": float(repeat_rate[d_i]),
                        "sales_t30": float(sales_t30[d_i]),
                        "total_traffic": float(total_traffic[d_i]),
                        "total_sales": float(orders[d_i]),
                        "total_revenue": float(ad_revenue[d_i]),
                        "total_profit": float(ad_profit[d_i]),
                        "revenue_stability_idx": float(rev_stab_idx[d_i]),
                        "inventory_clearance_speed": float(clearance_speed[d_i]),
                    }
                )

    return pd.DataFrame(rows)


def inv_turnover_days_step(
    base_days: float, impressions: np.ndarray, cvr: np.ndarray, rng: np.random.Generator, n_days: int
) -> np.ndarray:
    """Adjust inventory turnover days smoothly as function of ad demand."""
    demand_pull = (impressions / np.maximum(impressions.mean(), 1.0)) * (cvr / np.maximum(cvr.mean(), 0.01))
    return base_days / np.clip(0.75 + 0.25 * demand_pull, 0.5, 1.6) + rng.normal(0, 3.5, n_days)


def generate_all_demo_data(seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """Generate and persist sku_master_demo.csv & legacy_rules_baseline.json."""
    DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    legacy = extract_legacy_rules(DATA_DEMO_DIR / "legacy_rules_baseline.json")
    sku_df = build_anonymized_sku_master(seed=seed)
    sku_df.to_csv(DATA_DEMO_DIR / "sku_master_demo.csv", index=False)
    return sku_df, legacy
