"""Step 3 & Step 5: PyMC Bayesian regression for Layer-1 & Layer-2 weight learning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    CATEGORY_ORDER,
    CATEGORY_SPECS,
    LAYER1_FEATURE_COLS,
    REVERSE_METRICS,
)

# Force PyTensor to avoid failing on missing C++ compiler on minimal Windows environments
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


def compute_hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
    """Compute exact 1D Highest Density Interval (HDI) from posterior draws."""
    sorted_pts = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(sorted_pts)
    if n < 2:
        val = float(sorted_pts[0]) if n == 1 else 0.0
        return val, val
    interval_idx_inc = max(1, int(np.floor(credible_mass * n)))
    n_intervals = n - interval_idx_inc
    interval_width = sorted_pts[interval_idx_inc:] - sorted_pts[:n_intervals]
    min_idx = int(np.argmin(interval_width))
    return float(sorted_pts[min_idx]), float(sorted_pts[min_idx + interval_idx_inc])


def _prepare_target(series: pd.Series, target_name: str) -> np.ndarray:
    """Transform skewed count/monetary targets onto log scale before standardizing."""
    vals = series.astype(float).fillna(0.0).values
    skewed_targets = {
        "sales_qty",
        "revenue",
        "profit",
        "profit_x_margin",
        "inventory_turnover_rate",
        "sales_t30",
        "total_traffic",
        "total_sales",
        "total_revenue",
        "total_profit",
        "inventory_clearance_speed",
    }
    if target_name in skewed_targets:
        vals = np.log1p(np.maximum(vals, 0.0))
    std = float(np.std(vals))
    if std < 1e-8:
        return np.zeros_like(vals)
    return (vals - float(np.mean(vals))) / std


def _prepare_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> np.ndarray:
    """Align reverse metrics and z-score predictors within category."""
    mat = np.zeros((len(df), len(feature_cols)), dtype=float)
    for j, col in enumerate(feature_cols):
        vals = df[col].astype(float).fillna(df[col].median()).values
        if col in REVERSE_METRICS:
            vals = -vals
        std = float(np.std(vals))
        mat[:, j] = (vals - float(np.mean(vals))) / std if std > 1e-8 else 0.0
    return mat


def fit_dirichlet_bayesian_regression(
    X_std: np.ndarray,
    y_std: np.ndarray,
    prior_weights: List[float],
    feature_names: List[str],
    prior_concentration: float = 8.0,
    draws: int = 300,
    tune: int = 250,
    chains: int = 2,
    random_seed: int = 42,
) -> Dict:
    """Fit PyMC Bayesian linear model with Dirichlet simplex weight priors."""
    import arviz as az
    import pymc as pm

    p_arr = np.asarray(prior_weights, dtype=float)
    p_arr = p_arr / np.sum(p_arr)
    alpha = prior_concentration * p_arr

    # Analytical prior 95% HDI via Beta marginal of Dirichlet(alpha)
    rng = np.random.default_rng(random_seed)
    prior_draws = rng.dirichlet(alpha, size=4000)

    with pm.Model():
        w = pm.Dirichlet("w", a=alpha)
        beta0 = pm.Normal("beta0", mu=0.0, sigma=0.5)
        scale = pm.HalfNormal("scale", sigma=1.5)
        # Positive noise floor prevents singular likelihood ridges on near-collinear targets
        sigma_raw = pm.HalfNormal("sigma_raw", sigma=0.75)
        sigma = 0.12 + sigma_raw
        mu = beta0 + scale * pm.math.dot(X_std, w)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)

        sample_kwargs = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": 1,
            "random_seed": random_seed,
            "progressbar": False,
            "compute_convergence_checks": False,
            "init": "adapt_diag",
            "target_accept": 0.85,
            "nuts": {"max_treedepth": 7},
        }
        try:
            import numba  # noqa: F401
            sample_kwargs["compile_kwargs"] = {"mode": "NUMBA"}
        except ImportError:
            pass

        idata = pm.sample(**sample_kwargs)

    posterior = idata["posterior"] if hasattr(idata, "__getitem__") and "posterior" in idata else idata.posterior
    w_samples = np.asarray(posterior["w"].values, dtype=float)  # shape: (chains, draws, k)
    flat_w = w_samples.reshape(-1, len(feature_names))
    scale_mean = float(np.asarray(posterior["scale"].values).mean())
    beta0_mean = float(np.asarray(posterior["beta0"].values).mean())

    # Convergence diagnostics via ArviZ with exact split-Rhat fallback
    try:
        rhat_obj = az.rhat(idata, var_names=["w"])
        rhat_vals = np.asarray(rhat_obj["w"].values, dtype=float)
        ess_obj = az.ess(idata, var_names=["w"])
        ess_vals = np.asarray(ess_obj["w"].values, dtype=float)
    except Exception:
        # Exact split-Rhat across half-chains
        c_cnt, d_cnt, k_cnt = w_samples.shape
        half = d_cnt // 2
        if c_cnt >= 1 and half >= 10:
            splits = np.concatenate([w_samples[:, :half, :], w_samples[:, half : 2 * half, :]], axis=0)
            m_chains = splits.shape[0]
            chain_means = splits.mean(axis=1)
            chain_vars = splits.var(axis=1, ddof=1)
            W = chain_vars.mean(axis=0)
            B = half * chain_means.var(axis=0, ddof=1)
            var_hat = ((half - 1.0) / half) * W + (1.0 / half) * B
            rhat_vals = np.sqrt(np.maximum(var_hat, 1e-12) / np.maximum(W, 1e-12))
        else:
            rhat_vals = np.ones(k_cnt, dtype=float)
        ess_vals = np.full(len(feature_names), float(len(flat_w)))

    w_means = flat_w.mean(axis=0)
    w_means = w_means / np.sum(w_means)

    y_pred = beta0_mean + scale_mean * (X_std @ w_means)
    rho, p_val = spearmanr(y_std, y_pred)
    if np.isnan(rho):
        rho, p_val = 0.0, 1.0

    weights_detail = []
    for idx, fname in enumerate(feature_names):
        post_lo, post_hi = compute_hdi(flat_w[:, idx], 0.95)
        prior_lo, prior_hi = compute_hdi(prior_draws[:, idx], 0.95)
        weights_detail.append(
            {
                "feature": fname,
                "prior_mean": round(float(p_arr[idx]), 4),
                "prior_hdi_low": round(prior_lo, 4),
                "prior_hdi_high": round(prior_hi, 4),
                "posterior_mean": round(float(w_means[idx]), 4),
                "posterior_sd": round(float(flat_w[:, idx].std()), 4),
                "hdi_low": round(post_lo, 4),
                "hdi_high": round(post_hi, 4),
                "shift": round(float(w_means[idx] - p_arr[idx]), 4),
                "rhat": round(float(rhat_vals[idx]), 4),
                "ess": int(round(float(ess_vals[idx]))),
            }
        )

    return {
        "n_samples": int(len(y_std)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_val),
        "max_rhat": round(float(np.max(rhat_vals)), 4),
        "min_ess": int(round(float(np.min(ess_vals)))),
        "scale_mean": round(scale_mean, 4),
        "weights": weights_detail,
    }


def fit_layer1_product_models(
    sku_df: pd.DataFrame,
    draws: int = 300,
    tune: int = 250,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Step 3: Fit PyMC 4-dimension weights per category & assign internal A/B/C sub-tiers."""
    out = sku_df.copy()
    out["layer1_weighted_score"] = 0.0
    out["abc_tier"] = "C"

    models_summary: Dict[str, Dict] = {}

    for i, role in enumerate(CATEGORY_ORDER):
        spec = CATEGORY_SPECS[role]
        mask = out["strategic_role"] == role
        sub = out.loc[mask]
        if len(sub) < 10:
            continue

        X_std = _prepare_features(sub, LAYER1_FEATURE_COLS)
        y_std = _prepare_target(sub[spec["layer1_target"]], spec["layer1_target"])

        res = fit_dirichlet_bayesian_regression(
            X_std=X_std,
            y_std=y_std,
            prior_weights=spec["layer1_priors"],
            feature_names=LAYER1_FEATURE_COLS,
            prior_concentration=10.0,
            draws=draws,
            tune=tune,
            random_seed=random_seed + i,
        )
        res["category"] = role
        res["target_col"] = spec["layer1_target"]
        res["target_zh"] = spec["layer1_target_zh"]
        res["target_en"] = spec["layer1_target_en"]
        models_summary[role] = res

        w_vec = np.array([item["posterior_mean"] for item in res["weights"]], dtype=float)
        scores = sub[LAYER1_FEATURE_COLS].values @ w_vec
        out.loc[mask, "layer1_weighted_score"] = np.round(scores, 2)

        # Within-category ranking: Top 20% = A, 20%-60% = B, Bottom 40% = C
        ranks = sub[LAYER1_FEATURE_COLS].values @ w_vec
        pct_rank = pd.Series(ranks, index=sub.index).rank(pct=True, ascending=False)
        tiers = np.where(pct_rank <= 0.20, "A", np.where(pct_rank <= 0.60, "B", "C"))
        out.loc[mask, "abc_tier"] = tiers

    return out, models_summary


def fit_layer2_ad_models(
    ad_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    draws: int = 300,
    tune: int = 250,
    random_seed: int = 142,
) -> Dict:
    """Step 5: Fit PyMC 5-metric ad regression weights for each strategic category P1-P8."""
    merged = ad_df.merge(
        sku_df[["sku_id", "strategic_role"]], on="sku_id", how="left"
    )
    # Aggregate SKU-platform level over training window (exclude last 3 days reserved for anomaly eval)
    all_dates = sorted(merged["date"].unique())
    train_dates = set(all_dates[:-3]) if len(all_dates) > 10 else set(all_dates)
    train_df = merged[merged["date"].isin(train_dates)]

    models_summary: Dict[str, Dict] = {}

    for i, role in enumerate(CATEGORY_ORDER):
        spec = CATEGORY_SPECS[role]
        metrics = spec["ad_metrics"]
        target_col = spec["layer2_target"]
        sub = train_df[train_df["strategic_role"] == role]
        if len(sub) < 15:
            continue

        # Aggregate per SKU-platform so each unit is independent and training stays brisk
        agg_cols = {m: "mean" for m in metrics}
        agg_cols[target_col] = "mean"
        grouped = sub.groupby(["sku_id", "platform"], as_index=False).agg(agg_cols)

        # Subsample up to 600 rows if huge so NUTS finishes within seconds per category
        if len(grouped) > 600:
            grouped = grouped.sample(n=600, random_seed=random_seed + i)

        X_std = _prepare_features(grouped, metrics)
        y_std = _prepare_target(grouped[target_col], target_col)

        res = fit_dirichlet_bayesian_regression(
            X_std=X_std,
            y_std=y_std,
            prior_weights=spec["layer2_priors"],
            feature_names=metrics,
            prior_concentration=10.0,
            draws=draws,
            tune=tune,
            random_seed=random_seed + i,
        )
        res["category"] = role
        res["target_col"] = target_col
        res["target_zh"] = spec["layer2_target_zh"]
        res["target_en"] = spec["layer2_target_en"]
        models_summary[role] = res

    return models_summary


def save_bayesian_json(models_summary: Dict, path: Path) -> None:
    """Persist fitted Bayesian weights & diagnostics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models_summary, indent=2, ensure_ascii=False), encoding="utf-8")
