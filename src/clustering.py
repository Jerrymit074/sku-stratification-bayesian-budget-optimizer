"""Step 2: Strategic product role clustering (K-Means + Hungarian archetype alignment + PCA)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import CATEGORY_ORDER


CLUSTER_FEATURES = [
    "score_sales",
    "score_revenue",
    "score_margin",
    "score_profit",
    "log_days_since_launch",
    "sales_mom_growth",
    "revenue_stability",
    "log_inv_turnover_days",
    "log_unit_price",
]

# Canonical z-score direction profiles for optimal 1-to-1 cluster-to-role assignment
# Columns correspond to CLUSTER_FEATURES
ARCHETYPE_PROFILES = {
    "P1": np.array([-0.8, -0.7,  0.0, -0.6, -2.2,  1.8, -1.5,  0.1, -0.2]),
    "P2": np.array([ 1.4,  0.8, -1.6,  0.2,  0.1,  0.6,  0.3, -1.2, -0.5]),
    "P3": np.array([ 1.2,  0.9, -0.1,  0.7,  0.3,  0.4,  0.6, -0.7, -0.3]),
    "P4": np.array([ 0.2,  1.2,  0.1,  0.9,  0.3,  0.1,  0.4,  0.0,  1.6]),
    "P5": np.array([-0.1,  0.4,  1.1,  0.9,  0.4,  0.1,  0.7, -0.1,  0.4]),
    "P6": np.array([ 0.5,  1.0,  0.2,  0.7,  1.1, -0.4,  1.3,  0.0,  0.2]),
    "P7": np.array([-0.7, -0.8, -1.5, -0.9,  0.7, -2.0, -1.4,  2.2, -0.2]),
    "P8": np.array([-0.5, -0.5,  0.2, -0.4,  0.2, -0.1,  0.2,  0.3, -0.2]),
}


def build_clustering_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Build standardized feature matrix for K-Means strategic role clustering."""
    work = pd.DataFrame(index=df.index)
    work["score_sales"] = df["score_sales"].astype(float)
    work["score_revenue"] = df["score_revenue"].astype(float)
    work["score_margin"] = df["score_margin"].astype(float)
    work["score_profit"] = df["score_profit"].astype(float)
    work["log_days_since_launch"] = np.log1p(np.maximum(df["days_since_launch"].astype(float), 0.0))
    work["sales_mom_growth"] = np.clip(df["sales_mom_growth"].astype(float), -0.8, 1.0)
    work["revenue_stability"] = np.clip(df["revenue_stability"].astype(float), 0.0, 1.0)
    work["log_inv_turnover_days"] = np.log1p(np.maximum(df["inventory_turnover_days"].astype(float), 1.0))
    raw_up = np.log1p(np.maximum(df["unit_price"].astype(float), 0.0))
    work["log_unit_price"] = np.clip(raw_up, None, float(np.quantile(raw_up, 0.97)))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work[CLUSTER_FEATURES].values)
    return X_scaled, scaler


def cluster_products(
    df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Run lifecycle-gated GMM clustering, align to P1-P8 archetypes, and compute PCA + silhouette."""
    out = df.copy()
    X_scaled, _ = build_clustering_matrix(out)

    # Evaluate global silhouette curve across k=4..10 for model selection diagnostics
    silhouette_curve = {}
    for k in range(4, 11):
        km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init=15)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        silhouette_curve[str(k)] = round(float(silhouette_score(X_scaled, labels_tmp)), 4)

    # Stage 2a: Lifecycle-gated roles (P1: launch < 90d; P7: clearance / negative margin)
    is_p1 = (out["days_since_launch"] < 90) & (out["sales_mom_growth"] > -0.05)
    is_p7 = (~is_p1) & (
        (out["profit_margin"] < 0.08)
        | ((out["inventory_turnover_days"] > 90) & (out["sales_mom_growth"] < -0.04))
    )
    is_mature = ~(is_p1 | is_p7)

    mature_roles = ["P2", "P3", "P4", "P5", "P6", "P8"]
    raw_cluster = np.full(len(out), -1, dtype=int)
    aligned_roles = pd.Series("", index=out.index, dtype=object)
    aligned_roles.loc[is_p1] = "P1"
    aligned_roles.loc[is_p7] = "P7"

    X_mature = X_scaled[is_mature.values]
    n_mature_clusters = min(len(mature_roles), max(1, len(X_mature)))
    gmm = GaussianMixture(
        n_components=n_mature_clusters,
        covariance_type="diag",
        random_state=random_state,
        n_init=10,
    )
    mature_labels = gmm.fit_predict(X_mature)
    centroids = gmm.means_

    # Optimal 1-to-1 Hungarian matching between mature GMM centroids and [P2, P3, P4, P5, P6, P8]
    cost_matrix = np.zeros((n_mature_clusters, len(mature_roles)))
    for ci in range(n_mature_clusters):
        for ri, role in enumerate(mature_roles):
            target = ARCHETYPE_PROFILES[role]
            c_norm = centroids[ci] / (np.linalg.norm(centroids[ci]) + 1e-9)
            t_norm = target / (np.linalg.norm(target) + 1e-9)
            cost_matrix[ci, ri] = -float(np.dot(c_norm, t_norm))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_role = {int(c): mature_roles[r] for c, r in zip(row_ind, col_ind)}

    mature_mapped = pd.Series(mature_labels, index=out.index[is_mature]).map(cluster_to_role)
    aligned_roles.loc[is_mature] = mature_mapped.values

    # Assign unified 0..7 cluster_id for inspection
    raw_cluster[is_p1.values] = 0
    raw_cluster[is_p7.values] = 6
    role_to_id = {"P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 5, "P8": 7}
    raw_cluster[is_mature.values] = [role_to_id[r] for r in mature_mapped.values]
    rule_adjusted = (is_p1 | is_p7).values

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    out["cluster_id"] = raw_cluster
    out["strategic_role"] = aligned_roles.values
    out["pca_1"] = np.round(coords[:, 0], 4)
    out["pca_2"] = np.round(coords[:, 1], 4)

    # Lifecycle transition watchlist indicator (Section 6.2)
    transition_flag = np.full(len(out), "Stable", dtype=object)
    transition_flag[(out["strategic_role"] == "P1") & (out["days_since_launch"] >= 70) & (out["score_sales"] >= 50)] = "P1 -> P2/P3 Candidate"
    transition_flag[(out["strategic_role"] == "P6") & (out["sales_mom_growth"] < -0.05)] = "P6 -> P7 Watch"
    transition_flag[(out["strategic_role"] == "P8") & (out["sales_mom_growth"] > 0.07) & (out["score_margin"] >= 75)] = "P8 -> P5 Candidate"
    out["transition_signal"] = transition_flag

    # Compute silhouette on final strategic labels
    final_sil = float(silhouette_score(X_scaled, pd.Categorical(out["strategic_role"]).codes))
    kmeans_sil = silhouette_curve[str(n_clusters)]

    diagnostics = {
        "kmeans_silhouette_k8": round(kmeans_sil, 4),
        "final_role_silhouette": round(final_sil, 4),
        "silhouette_curve": silhouette_curve,
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "rule_confirmed_pct": round(float(rule_adjusted.mean() * 100.0), 2),
        "role_counts": {r: int((out["strategic_role"] == r).sum()) for r in CATEGORY_ORDER},
    }
    return out, diagnostics
