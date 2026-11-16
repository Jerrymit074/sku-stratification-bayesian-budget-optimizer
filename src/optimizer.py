"""Step 6: Hierarchical constrained budget optimization via SciPy (SLSQP + KKT water-filling)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from src.config import CATEGORY_ORDER, CATEGORY_SPECS, TIER_MULTIPLIERS


def _kkt_within_category(
    utilities: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    target_budget: float,
    alpha: float,
) -> np.ndarray:
    """Exact global optimum of max sum(U_k * b_k^alpha) s.t. sum(b_k)=B, l_k<=b_k<=u_k."""
    n = len(utilities)
    if n == 0:
        return np.array([], dtype=float)
    if target_budget <= lowers.sum() + 1e-9:
        return lowers.copy()
    if target_budget >= uppers.sum() - 1e-9:
        return uppers.copy()

    exp_power = 1.0 / (1.0 - alpha)
    scaled_u = np.maximum(utilities, 1e-9) * alpha

    def _sum_spend(lam: float) -> float:
        raw = (scaled_u / lam) ** exp_power
        return float(np.sum(np.clip(raw, lowers, uppers)) - target_budget)

    # Bracket lambda cleanly
    lam_min = float(np.min(scaled_u / (np.maximum(uppers, 1e-6) ** (1.0 - alpha)))) * 0.1
    lam_max = float(np.max(scaled_u / (np.maximum(lowers, 1e-6) ** (1.0 - alpha)))) * 10.0
    lam_min = max(lam_min, 1e-12)
    lam_max = max(lam_max, lam_min * 100.0)

    while _sum_spend(lam_min) < 0:
        lam_min *= 0.1
        if lam_min < 1e-18:
            break
    while _sum_spend(lam_max) > 0:
        lam_max *= 10.0
        if lam_max > 1e18:
            break

    try:
        lam_star = brentq(_sum_spend, lam_min, lam_max, xtol=1e-10, maxiter=200)
        alloc = np.clip((scaled_u / lam_star) ** exp_power, lowers, uppers)
    except Exception:
        # Fallback proportional allocation within bounds
        weights = utilities / max(utilities.sum(), 1e-9)
        alloc = np.clip(target_budget * weights, lowers, uppers)

    # Fine-adjust rounding residual on non-saturated items
    diff = target_budget - float(alloc.sum())
    if abs(diff) > 1e-6:
        slack = (uppers - alloc) if diff > 0 else (alloc - lowers)
        slack_sum = float(slack.sum())
        if slack_sum > 1e-9:
            alloc += diff * (slack / slack_sum)
            alloc = np.clip(alloc, lowers, uppers)
    return alloc


def optimize_daily_budget(
    scored_day_df: pd.DataFrame,
    total_budget: float,
    category_bounds: Dict[str, Tuple[float, float]] | None = None,
    alpha: float = 0.65,
    max_acos_p1: float = 0.65,
    min_roas_core: float = 1.50,
    max_sku_share_in_cat: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Solve two-level constrained nonlinear budget optimization using SciPy SLSQP + KKT."""
    if category_bounds is None:
        category_bounds = {
            c: (CATEGORY_SPECS[c]["budget_min_pct"], CATEGORY_SPECS[c]["budget_max_pct"])
            for c in CATEGORY_ORDER
        }

    sum_min = sum(b[0] for b in category_bounds.values())
    sum_max = sum(b[1] for b in category_bounds.values())
    if sum_min > 1.0 + 1e-6 or sum_max < 1.0 - 1e-6:
        raise ValueError(
            f"Infeasible category bounds: sum(min)={sum_min:.2%}, sum(max)={sum_max:.2%}"
        )

    df = scored_day_df.copy().reset_index(drop=True)
    alpha = float(np.clip(alpha, 0.25, 0.90))

    # Tier baseline multiplier (Layer 1)
    tier_mult = df["abc_tier"].map(lambda x: TIER_MULTIPLIERS.get(str(x), TIER_MULTIPLIERS["C"])["default"]).values
    ad_score = np.clip(df["daily_ad_score"].astype(float).values, 1.0, 100.0)

    # Guardrail compliance factor
    guardrail_mult = np.ones(len(df), dtype=float)
    is_p1 = df["strategic_role"].values == "P1"
    is_p7 = df["strategic_role"].values == "P7"
    is_core = ~(is_p1 | is_p7)

    # Penalize P1 exceeding max ACOS or core categories falling below min ROAS
    acos_vals = df["acos"].astype(float).values
    roas_vals = df["roas"].astype(float).values
    guardrail_mult[is_p1 & (acos_vals > max_acos_p1)] *= 0.60
    guardrail_mult[is_core & (roas_vals < min_roas_core)] *= 0.65

    # Strategic category priority weights (aligned with Section 6.1 policy tiers)
    role_priority_map = {
        "P5": 1.35,  # High-Margin Harvester (efficiency)
        "P2": 1.25,  # Hero Traffic Driver (expansion)
        "P4": 1.15,  # Revenue Driver (cash flow return)
        "P1": 1.10,  # New Product Incubation (protected growth)
        "P3": 1.05,  # Volume Scale Driver (stability)
        "P6": 1.00,  # Share Defender (defense)
        "P8": 0.72,  # Balanced Long-Tail (baseline)
        "P7": 0.38,  # Clearance Exit (controlled taper)
    }
    role_priority = df["strategic_role"].map(lambda c: role_priority_map.get(str(c), 1.0)).values

    # Commercial return anchor combined with 5-metric strategic ad score
    roas_anchor = np.clip(roas_vals / max(float(np.median(roas_vals)), 1.0), 0.35, 2.50)

    # Composite strategic utility U_{i,p}
    utilities = (
        tier_mult
        * role_priority
        * (0.30 + 0.45 * (ad_score / 100.0) + 0.25 * roas_anchor)
        * guardrail_mult
    )
    df["utility_weight"] = np.round(utilities, 4)

    # Prepare per-category indices and per-SKU lower/upper bounds
    cat_indices = {c: np.where(df["strategic_role"].values == c)[0] for c in CATEGORY_ORDER}
    cat_b_bounds = []
    for c in CATEGORY_ORDER:
        lo_pct, hi_pct = category_bounds[c]
        if len(cat_indices[c]) == 0:
            cat_b_bounds.append((0.0, 0.0))
        else:
            cat_b_bounds.append((total_budget * lo_pct, total_budget * hi_pct))

    # Pre-compute SKU-level bounds inside each category given a candidate category budget B_c
    def _sku_bounds_for_cat(c_idx: np.ndarray, B_c: float) -> Tuple[np.ndarray, np.ndarray]:
        m = len(c_idx)
        if m == 0:
            return np.array([]), np.array([])
        curr = np.maximum(df["spend"].values[c_idx], 1.0)
        lo = np.minimum(np.full(m, 1.5), (B_c / m) * 0.40)
        # Upper bound prevents single-SKU concentration while respecting guardrails
        cap_share = max(max_sku_share_in_cat, 1.5 / m)
        hi = np.maximum(lo * 1.2, np.minimum(B_c * cap_share, np.maximum(curr * 2.8, (B_c / m) * 3.5)))
        # Ensure sum(hi) >= B_c and sum(lo) <= B_c
        if hi.sum() < B_c:
            hi = hi * (B_c / max(hi.sum(), 1e-9)) * 1.05
        if lo.sum() > B_c:
            lo = lo * (B_c / max(lo.sum(), 1e-9)) * 0.90
        return lo, hi

    # Upper-level optimization over 8 category budgets using scipy.optimize.minimize (SLSQP)
    def _neg_portfolio_objective(B_vec: np.ndarray) -> float:
        total_obj = 0.0
        for ci, c in enumerate(CATEGORY_ORDER):
            idx = cat_indices[c]
            if len(idx) == 0:
                continue
            B_c = float(B_vec[ci])
            lo, hi = _sku_bounds_for_cat(idx, B_c)
            alloc_c = _kkt_within_category(utilities[idx], lo, hi, B_c, alpha)
            total_obj += float(np.sum(utilities[idx] * (np.maximum(alloc_c, 1e-9) ** alpha)))
        return -total_obj

    # Initial feasible point at midpoint of category bounds scaled to total_budget
    b_init = np.array([0.5 * (lo + hi) for lo, hi in cat_b_bounds], dtype=float)
    # Project onto feasible box & sum constraint
    for _ in range(25):
        b_init = np.clip(b_init, [b[0] for b in cat_b_bounds], [b[1] for b in cat_b_bounds])
        diff = total_budget - float(b_init.sum())
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            room = np.array([hi - b_init[i] for i, (_, hi) in enumerate(cat_b_bounds)])
        else:
            room = np.array([b_init[i] - lo for i, (lo, _) in enumerate(cat_b_bounds)])
        if room.sum() > 1e-9:
            b_init += diff * (room / room.sum())

    res = minimize(
        _neg_portfolio_objective,
        x0=b_init,
        method="SLSQP",
        bounds=cat_b_bounds,
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)}],
        options={"maxiter": 80, "ftol": 1e-6},
    )

    opt_cat_budgets = res.x if res.success else b_init
    # Final projection to exact total_budget inside bounds
    opt_cat_budgets = np.clip(opt_cat_budgets, [b[0] for b in cat_b_bounds], [b[1] for b in cat_b_bounds])
    if abs(float(opt_cat_budgets.sum()) - total_budget) > 1e-4:
        for _ in range(20):
            diff = total_budget - float(opt_cat_budgets.sum())
            if abs(diff) < 1e-5:
                break
            room = np.array(
                [hi - opt_cat_budgets[i] if diff > 0 else opt_cat_budgets[i] - lo for i, (lo, hi) in enumerate(cat_b_bounds)]
            )
            if room.sum() > 1e-9:
                opt_cat_budgets = np.clip(
                    opt_cat_budgets + diff * (room / room.sum()),
                    [b[0] for b in cat_b_bounds],
                    [b[1] for b in cat_b_bounds],
                )

    # Solve final within-category allocations
    opt_spend = np.zeros(len(df), dtype=float)
    for ci, c in enumerate(CATEGORY_ORDER):
        idx = cat_indices[c]
        if len(idx) == 0:
            continue
        B_c = float(opt_cat_budgets[ci])
        lo, hi = _sku_bounds_for_cat(idx, B_c)
        opt_spend[idx] = _kkt_within_category(utilities[idx], lo, hi, B_c, alpha)

    df["optimized_spend"] = np.round(opt_spend, 2)
    df["spend_delta"] = np.round(df["optimized_spend"] - df["spend"], 2)
    df["spend_delta_pct"] = np.round((df["optimized_spend"] - df["spend"]) / np.maximum(df["spend"], 1.0) * 100.0, 2)

    # Concave response projection for expected revenue/ROAS comparison under equal total budget
    base_spend = np.maximum(df["spend"].values, 1.0)
    scale_to_budget = total_budget / max(float(base_spend.sum()), 1.0)
    baseline_spend_norm = base_spend * scale_to_budget

    # Estimated revenue under diminishing returns exponent alpha
    base_rev = np.maximum(df["ad_revenue"].values, 0.0)
    est_opt_rev = base_rev * ((np.maximum(opt_spend, 0.1) / base_spend) ** alpha)
    est_base_rev = base_rev * ((baseline_spend_norm / base_spend) ** alpha)

    df["baseline_spend_scaled"] = np.round(baseline_spend_norm, 2)
    df["est_opt_revenue"] = np.round(est_opt_rev, 2)
    df["est_base_revenue"] = np.round(est_base_rev, 2)

    # Category summary table
    cat_rows = []
    for ci, c in enumerate(CATEGORY_ORDER):
        spec = CATEGORY_SPECS[c]
        sub = df[df["strategic_role"] == c]
        lo_pct, hi_pct = category_bounds[c]
        opt_b = float(sub["optimized_spend"].sum())
        base_b = float(sub["baseline_spend_scaled"].sum())
        cat_rows.append(
            {
                "category": c,
                "name_zh": spec["name_zh"],
                "name_en": spec["name_en"],
                "policy_zh": spec["budget_policy_zh"],
                "policy_en": spec["budget_policy_en"],
                "min_pct": round(lo_pct * 100.0, 1),
                "max_pct": round(hi_pct * 100.0, 1),
                "opt_budget": round(opt_b, 2),
                "opt_share_pct": round(opt_b / max(total_budget, 1.0) * 100.0, 2),
                "baseline_budget": round(base_b, 2),
                "baseline_share_pct": round(base_b / max(total_budget, 1.0) * 100.0, 2),
                "avg_ad_score": round(float(sub["daily_ad_score"].mean()) if len(sub) else 0.0, 2),
                "sku_stream_count": int(len(sub)),
            }
        )
    cat_summary = pd.DataFrame(cat_rows)

    opt_total_rev = float(df["est_opt_revenue"].sum())
    base_total_rev = float(df["est_base_revenue"].sum())
    opt_util_score = float(np.sum(utilities * (np.maximum(opt_spend, 1e-6) ** alpha)))
    base_util_score = float(np.sum(utilities * (np.maximum(baseline_spend_norm, 1e-6) ** alpha)))

    metrics = {
        "solver_success": bool(res.success),
        "total_budget": round(float(total_budget), 2),
        "allocated_budget": round(float(df["optimized_spend"].sum()), 2),
        "baseline_roas": round(base_total_rev / max(total_budget, 1.0), 3),
        "optimized_roas": round(opt_total_rev / max(total_budget, 1.0), 3),
        "roas_uplift_pct": round((opt_total_rev - base_total_rev) / max(base_total_rev, 1.0) * 100.0, 2),
        "utility_uplift_pct": round((opt_util_score - base_util_score) / max(base_util_score, 1.0) * 100.0, 2),
    }

    return df, cat_summary, metrics


def simulate_legacy_allocation(
    scored_day_df: pd.DataFrame,
    legacy_scores: pd.Series,
    legacy_rules: Dict,
    total_budget: float,
    alpha: float = 0.65,
) -> Dict:
    """Allocate budget using legacy arrangement_source.xlsx rules for portfolio benchmark."""
    df = scored_day_df.copy()
    df["legacy_ad_score"] = legacy_scores.values
    mult_table = legacy_rules["score_multipliers"]

    def _lookup_mult(score: float, tier: str) -> float:
        t_key = tier if tier in ("A", "B", "C") else "C"
        if score >= 90:
            band = "90-100"
        elif score >= 80:
            band = "80-90"
        elif score >= 60:
            band = "60-80"
        elif score >= 30:
            band = "30-60"
        else:
            band = "0-30"
        return float(mult_table[band][t_key])

    mults = np.array(
        [_lookup_mult(s, t) for s, t in zip(df["legacy_ad_score"].values, df["legacy_abc_tier"].values)],
        dtype=float,
    )
    raw_alloc = np.maximum(df["spend"].values, 1.0) * mults
    legacy_spend = raw_alloc * (total_budget / max(float(raw_alloc.sum()), 1.0))

    base_spend = np.maximum(df["spend"].values, 1.0)
    est_legacy_rev = np.maximum(df["ad_revenue"].values, 0.0) * ((legacy_spend / base_spend) ** alpha)

    # Share of budget going to P7 clearance drain vs P1/P2/P5 strategic priority
    p7_share = float(legacy_spend[df["strategic_role"].values == "P7"].sum() / total_budget * 100.0)
    p125_share = float(legacy_spend[df["strategic_role"].isin(["P1", "P2", "P5"]).values].sum() / total_budget * 100.0)

    return {
        "legacy_roas": round(float(est_legacy_rev.sum()) / max(total_budget, 1.0), 3),
        "legacy_total_revenue": round(float(est_legacy_rev.sum()), 2),
        "legacy_p7_spend_pct": round(p7_share, 2),
        "legacy_p125_spend_pct": round(p125_share, 2),
    }
