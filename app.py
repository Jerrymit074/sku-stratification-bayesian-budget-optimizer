"""Bilingual (EN / 中文) Business-First Streamlit Dashboard for Product Grouping & Ad Budget Planning."""

from __future__ import annotations

import io
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.bootstrap import ensure_artifacts
from src.config import (
    ARTIFACTS_DIR,
    CATEGORY_ORDER,
    CATEGORY_SPECS,
    DATA_DEMO_DIR,
    METRIC_LABELS,
    PLATFORMS,
)
from src.i18n import t
from src.optimizer import optimize_daily_budget, simulate_legacy_allocation
from src.preprocessing import preprocess_sku_features


st.set_page_config(
    page_title="Smart Product Grouping & Ad Budget Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> Dict:
    """Load precomputed pipeline outputs and legacy baseline."""
    summary = json.loads((ARTIFACTS_DIR / "pipeline_summary.json").read_text(encoding="utf-8"))
    sku_df = pd.read_csv(ARTIFACTS_DIR / "sku_classified.csv")
    l1_weights = json.loads((ARTIFACTS_DIR / "bayesian_product_weights.json").read_text(encoding="utf-8"))
    l2_weights = json.loads((ARTIFACTS_DIR / "bayesian_ad_weights.json").read_text(encoding="utf-8"))
    daily_scores = pd.read_csv(ARTIFACTS_DIR / "daily_ad_scores.csv")
    anomalies = pd.read_csv(ARTIFACTS_DIR / "daily_anomalies.csv")
    daily_trend = pd.read_csv(ARTIFACTS_DIR / "daily_trend_summary.csv")
    cat_budget = pd.read_csv(ARTIFACTS_DIR / "category_budget_summary.csv")
    budget_base = pd.read_csv(ARTIFACTS_DIR / "budget_allocation_baseline.csv")
    legacy_rules = json.loads((DATA_DEMO_DIR / "legacy_rules_baseline.json").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "sku_df": sku_df,
        "l1_weights": l1_weights,
        "l2_weights": l2_weights,
        "daily_scores": daily_scores,
        "anomalies": anomalies,
        "daily_trend": daily_trend,
        "cat_budget": cat_budget,
        "budget_base": budget_base,
        "legacy_rules": legacy_rules,
    }


@st.cache_data(show_spinner=False)
def load_full_ad_panel() -> pd.DataFrame:
    """Load 30-day SKU drilldown panel."""
    drill_csv = ARTIFACTS_DIR / "sku_drilldown_history.csv"
    if drill_csv.exists():
        return pd.read_csv(drill_csv)
    return pd.read_csv(DATA_DEMO_DIR / "ad_daily_demo.csv.gz")


def cat_name(code: str, lang: str) -> str:
    spec = CATEGORY_SPECS.get(code)
    if not spec:
        return code
    return f"{code} · {spec['name_en'] if lang == 'en' else spec['name_zh']}"


def m_label(metric: str, lang: str) -> str:
    return METRIC_LABELS.get(metric, {}).get(lang, metric)


def to_excel_bytes(opt_df: pd.DataFrame, cat_summary: pd.DataFrame) -> bytes:
    """Export actionable daily allocation plan and category summary to multi-sheet Excel."""
    buf = io.BytesIO()
    out_df = opt_df.copy()
    out_df["action"] = np.where(
        out_df["spend_delta_pct"] >= 10.0,
        "Scale Up",
        np.where(out_df["spend_delta_pct"] <= -10.0, "Trim Spend", "Hold"),
    )
    export_cols = [
        "date",
        "sku_id",
        "product_title",
        "strategic_role",
        "abc_tier",
        "platform",
        "action",
        "daily_ad_score",
        "baseline_spend_scaled",
        "optimized_spend",
        "spend_delta",
        "spend_delta_pct",
        "est_opt_revenue",
        "roas",
        "acos",
        "ctr",
        "cvr",
    ]
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        cat_summary.to_excel(writer, sheet_name="Category_Budget_Summary", index=False)
        out_df[export_cols].sort_values(["strategic_role", "optimized_spend"], ascending=[True, False]).to_excel(
            writer, sheet_name="SKU_Daily_Action_Plan", index=False
        )
    return buf.getvalue()


def main() -> None:
    ensure_artifacts()
    if not (ARTIFACTS_DIR / "pipeline_summary.json").exists():
        st.error("Artifacts not found. Please run `python -m src.run_pipeline` first.")
        return

    data = load_artifacts()
    summary = data["summary"]
    sku_df = data["sku_df"]
    l1_weights = data["l1_weights"]
    l2_weights = data["l2_weights"]
    daily_scores = data["daily_scores"]
    anomalies = data["anomalies"]
    daily_trend = data["daily_trend"]
    cat_budget = data["cat_budget"]
    budget_base = data["budget_base"]
    legacy_rules = data["legacy_rules"]

    with st.sidebar:
        lang_choice = st.radio(
            "Language / 语言",
            options=["English", "简体中文"],
            horizontal=True,
        )
        lang = "en" if lang_choice == "English" else "zh"

        st.markdown("---")
        nav_options = ["overview", "clustering", "bayesian", "allocator", "monitor"]
        nav_labels = {
            "overview": t("nav_overview", lang),
            "clustering": t("nav_clustering", lang),
            "bayesian": t("nav_bayesian", lang),
            "allocator": t("nav_allocator", lang),
            "monitor": t("nav_monitor", lang),
        }
        page = st.radio(
            t("nav_title", lang),
            options=nav_options,
            format_func=lambda k: nav_labels[k],
        )
        st.markdown("---")
        st.caption(t("sidebar_note", lang))
        st.caption(f"Snapshot: `{summary['latest_date']}`")

    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    # =========================================================================
    # PAGE 1: BUSINESS SUMMARY
    # =========================================================================
    if page == "overview":
        st.info(t("overview_intro", lang))

        opt = summary["optimization"]
        proj_opt_rev = float(budget_base["est_opt_revenue"].sum())
        rev_gain_vs_legacy = proj_opt_rev - float(opt["legacy_total_revenue"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            t("kpi_skus", lang),
            f"{summary['preprocessing']['total_skus']:,}",
            delta="8 Business Roles (P1–P8)" if lang == "en" else "8 大业务角色 (P1–P8)",
            delta_color="off",
        )
        c2.metric(
            t("kpi_budget", lang),
            f"${opt['total_budget']:,.0f}",
            delta=f"Clearance (P7) capped at {opt['opt_p7_spend_pct']:.1f}%" if lang == "en" else f"清仓品预算控制仅 {opt['opt_p7_spend_pct']:.1f}%",
            delta_color="off",
        )
        c3.metric(
            t("kpi_rev", lang),
            f"${proj_opt_rev:,.0f}",
            delta=f"+${rev_gain_vs_legacy:,.0f}/day (+{opt['roas_vs_legacy_uplift_pct']:.1f}% vs Old Rules)",
        )
        c4.metric(
            t("kpi_focus", lang),
            f"{opt['opt_p125_spend_pct']:.1f}%",
            delta=f"+{opt['opt_p125_spend_pct'] - opt['legacy_p125_spend_pct']:.1f}% vs Old ({opt['legacy_p125_spend_pct']:.1f}%)",
        )

        col_left, col_right = st.columns([1.05, 1.2])

        with col_left:
            title_cmp = "Where Ad Dollars Shift: Old Baseline vs. Smart Plan ($)" if lang == "en" else "预算怎么挪：旧基线 vs 智能分配 ($)"
            st.subheader(title_cmp)
            fig_cmp = go.Figure()
            x_cats = [cat_name(c, lang) for c in CATEGORY_ORDER]
            fig_cmp.add_trace(
                go.Bar(
                    name="Old Baseline ($)" if lang == "en" else "旧基线花费 ($)",
                    x=x_cats,
                    y=cat_budget["baseline_budget"],
                    marker_color="#CBD5E1",
                )
            )
            fig_cmp.add_trace(
                go.Bar(
                    name="Smart Recommended ($)" if lang == "en" else "智能推荐预算 ($)",
                    x=x_cats,
                    y=cat_budget["opt_budget"],
                    marker_color="#1F4E79",
                )
            )
            fig_cmp.update_layout(
                barmode="group",
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Daily Budget ($)" if lang == "en" else "每日预算 ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cmp, width="stretch")

        with col_right:
            st.subheader(t("sankey_title", lang))
            legacy_nodes = ["Old A" if lang == "en" else "旧版 A级", "Old B" if lang == "en" else "旧版 B级", "Old C" if lang == "en" else "旧版 C级"]
            role_nodes = [cat_name(c, lang) for c in CATEGORY_ORDER]
            new_tier_nodes = ["Tier A (Top 20%)" if lang == "en" else "类内 A级 (前20%)", "Tier B (Mid 40%)" if lang == "en" else "类内 B级 (中40%)", "Tier C (Bot 40%)" if lang == "en" else "类内 C级 (后40%)"]
            all_nodes = legacy_nodes + role_nodes + new_tier_nodes

            legacy_idx_map = {"A": 0, "B": 1, "C": 2}
            role_idx_map = {c: 3 + i for i, c in enumerate(CATEGORY_ORDER)}
            new_idx_map = {"A": 11, "B": 12, "C": 13}

            sources, targets, values, link_colors = [], [], [], []
            g1 = sku_df.groupby(["legacy_abc_tier", "strategic_role"]).size().reset_index(name="cnt")
            for _, r in g1.iterrows():
                sources.append(legacy_idx_map[r["legacy_abc_tier"]])
                targets.append(role_idx_map[r["strategic_role"]])
                values.append(int(r["cnt"]))
                link_colors.append("rgba(100, 116, 139, 0.25)")

            g2 = sku_df.groupby(["strategic_role", "abc_tier"]).size().reset_index(name="cnt")
            for _, r in g2.iterrows():
                sources.append(role_idx_map[r["strategic_role"]])
                targets.append(new_idx_map[r["abc_tier"]])
                values.append(int(r["cnt"]))
                col_hex = CATEGORY_SPECS[r["strategic_role"]]["color"]
                link_colors.append(f"rgba({int(col_hex[1:3],16)}, {int(col_hex[3:5],16)}, {int(col_hex[5:7],16)}, 0.35)")

            node_colors = ["#1E293B", "#475569", "#94A3B8"] + [CATEGORY_SPECS[c]["color"] for c in CATEGORY_ORDER] + ["#10B981", "#3B82F6", "#CBD5E1"]
            fig_sankey = go.Figure(
                go.Sankey(
                    node=dict(pad=14, thickness=18, line=dict(color="white", width=0.5), label=all_nodes, color=node_colors),
                    link=dict(source=sources, target=targets, value=values, color=link_colors),
                )
            )
            fig_sankey.update_layout(height=400, margin=dict(l=10, r=10, t=15, b=10))
            st.plotly_chart(fig_sankey, width="stretch")

        st.subheader(t("role_table_title", lang))
        role_stats = []
        for c in CATEGORY_ORDER:
            sub = sku_df[sku_df["strategic_role"] == c]
            cb_row = cat_budget[cat_budget["category"] == c].iloc[0]
            spec = CATEGORY_SPECS[c]
            role_stats.append(
                {
                    "Code": c,
                    "Business Role" if lang == "en" else "产品业务角色": spec["name_en"] if lang == "en" else spec["name_zh"],
                    "SKUs": int(len(sub)),
                    "Avg Margin" if lang == "en" else "平均毛利率": f"{sub['profit_margin'].mean()*100:.1f}%",
                    "Typical Units Sold" if lang == "en" else "中位销量": f"{int(sub['sales_qty'].median()):,}",
                    "Recommended Budget" if lang == "en" else "建议日预算": f"${cb_row['opt_budget']:,.0f} ({cb_row['opt_share_pct']:.1f}%)",
                    "Plain-English Playbook" if lang == "en" else "业务打法说明": spec["goal_en"] if lang == "en" else spec["goal_zh"],
                }
            )
        st.dataframe(pd.DataFrame(role_stats), width="stretch", hide_index=True)

    # =========================================================================
    # PAGE 2: HOW PRODUCTS ARE GROUPED (P1-P8)
    # =========================================================================
    elif page == "clustering":
        col_map, col_bar = st.columns([1.4, 1.0])

        with col_map:
            st.subheader(t("cluster_biz_title", lang))
            plot_df = sku_df.copy()
            plot_df["role_display"] = plot_df["strategic_role"].map(lambda c: cat_name(c, lang))
            plot_df["margin_pct"] = np.round(plot_df["profit_margin"] * 100.0, 1)
            plot_df["sales_pos"] = np.maximum(plot_df["sales_qty"], 1)
            plot_df["rev_bubble"] = np.sqrt(np.maximum(plot_df["revenue"], 100.0))
            color_map = {cat_name(c, lang): CATEGORY_SPECS[c]["color"] for c in CATEGORY_ORDER}

            fig_biz = px.scatter(
                plot_df,
                x="sales_pos",
                y="margin_pct",
                size="rev_bubble",
                size_max=18,
                color="role_display",
                color_discrete_map=color_map,
                log_x=True,
                hover_data=["sku_id", "abc_tier", "sales_qty", "revenue", "margin_pct", "days_since_launch", "inventory_turnover_days"],
                opacity=0.72,
                height=430,
            )
            fig_biz.update_layout(
                margin=dict(l=10, r=10, t=15, b=10),
                legend_title="Business Role" if lang == "en" else "产品角色",
                xaxis_title="Total Units Sold (Log Scale)" if lang == "en" else "累计销量规模（对数坐标）",
                yaxis_title="Profit Margin (%)" if lang == "en" else "毛利率 (%)",
            )
            st.plotly_chart(fig_biz, width="stretch")

        with col_bar:
            st.subheader(t("cluster_bar_title", lang))
            tier_counts = []
            for c in CATEGORY_ORDER:
                for tr in ["A", "B", "C"]:
                    cnt = int(((sku_df["strategic_role"] == c) & (sku_df["abc_tier"] == tr)).sum())
                    tier_counts.append({"Role": c, "Tier": f"Tier {tr}" if lang == "en" else f"{tr} 级", "Count": cnt})
            tc_df = pd.DataFrame(tier_counts)
            fig_tc = px.bar(
                tc_df,
                x="Role",
                y="Count",
                color="Tier",
                barmode="stack",
                color_discrete_sequence=["#1F4E79", "#3B82F6", "#CBD5E1"],
                height=430,
            )
            fig_tc.update_layout(
                margin=dict(l=10, r=10, t=15, b=10),
                xaxis_title="Product Role" if lang == "en" else "产品角色",
                yaxis_title="Number of SKUs" if lang == "en" else "SKU 数量",
            )
            st.plotly_chart(fig_tc, width="stretch")

        st.markdown("---")
        st.subheader(t("margin_cap_title", lang))
        st.info(t("margin_cap_alert", lang))

        cap_pct = st.slider(t("margin_cap_slider", lang), min_value=50, max_value=100, value=85, step=5)
        prep_live, diag_live = preprocess_sku_features(sku_df, margin_floor=0.0, margin_cap=cap_pct / 100.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Catalog Median Margin" if lang == "en" else "全店毛利率中位数", f"{diag_live['margin_median']*100:.1f}%")
        m2.metric(
            "Tied at 100 pts (Old Cap=60%)" if lang == "en" else "旧 60% 上限导致满分扎堆占比",
            f"{diag_live['capped_at_60_pct']:.1f}%",
        )
        m3.metric(
            f"Tied at 100 pts (Cap={cap_pct}%)" if lang == "en" else f"当前 Cap={cap_pct}% 满分占比",
            f"{diag_live['capped_margin_pct']:.1f}%",
            delta=f"{diag_live['capped_margin_pct'] - diag_live['capped_at_60_pct']:+.1f}% vs 60%",
            delta_color="inverse",
        )

        fig_mnorm = px.histogram(
            prep_live,
            x="score_margin",
            nbins=25,
            title=f"Resulting Margin Score Spread (0–100 @ Cap={cap_pct}%)" if lang == "en" else f"调整后毛利率得分分布（0–100分，Cap={cap_pct}%）",
            color_discrete_sequence=["#1F4E79"],
            height=260,
        )
        fig_mnorm.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Margin Score (0-100)" if lang == "en" else "毛利率标准化得分 (0-100)",
            yaxis_title="SKU Count" if lang == "en" else "产品数量",
        )
        st.plotly_chart(fig_mnorm, width="stretch")

        st.subheader(t("sku_lookup_title", lang))
        trans_map_en = {
            "Stable": "Steady in Current Role",
            "P1 -> P2/P3 Candidate": "Ready to Graduate to Bestseller",
            "P6 -> P7 Watch": "Declining — Watch for Clearance Risk",
            "P8 -> P5 Candidate": "Margin Rising — Star Candidate",
        }
        trans_map_zh = {
            "Stable": "表现稳定",
            "P1 -> P2/P3 Candidate": "新品起量 — 可晋升爆款/销量款",
            "P6 -> P7 Watch": "下滑预警 — 注意是否转入清仓",
            "P8 -> P5 Candidate": "高毛利潜力 — 可重点挖掘",
        }
        view_df = sku_df.copy()
        view_df["watch_label"] = view_df["transition_signal"].map(lambda s: (trans_map_en if lang == "en" else trans_map_zh).get(s, s))

        f1, f2, f3 = st.columns([1, 1, 1.5])
        role_filter = f1.multiselect("Role" if lang == "en" else "业务角色", CATEGORY_ORDER, default=CATEGORY_ORDER)
        watch_opts = sorted(view_df["watch_label"].unique())
        watch_filter = f2.multiselect("Lifecycle Watch" if lang == "en" else "阶段流转提醒", watch_opts, default=watch_opts)
        search_txt = f3.text_input("Search SKU ID" if lang == "en" else "搜索 SKU 编号", placeholder="SKU-0001")

        filtered = view_df[view_df["strategic_role"].isin(role_filter) & view_df["watch_label"].isin(watch_filter)]
        if search_txt.strip():
            filtered = filtered[filtered["sku_id"].str.contains(search_txt.strip(), case=False, na=False)]

        show_cols = [
            "sku_id",
            "department",
            "strategic_role",
            "abc_tier",
            "legacy_abc_tier",
            "sales_qty",
            "revenue",
            "profit_margin",
            "days_since_launch",
            "inventory_turnover_days",
            "watch_label",
        ]
        st.dataframe(filtered[show_cols].head(250), width="stretch", hide_index=True)

        with st.expander("For Data Teams: Clustering Validation Details (PCA & Silhouette)" if lang == "en" else "给技术人员看的校验细节（PCA 降维与轮廓系数）", expanded=False):
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                fig_pca = px.scatter(
                    sku_df,
                    x="pca_1",
                    y="pca_2",
                    color="strategic_role",
                    color_discrete_map={c: CATEGORY_SPECS[c]["color"] for c in CATEGORY_ORDER},
                    opacity=0.65,
                    height=300,
                    title=f"PCA Projection (Silhouette: {summary['clustering']['final_role_silhouette']:.3f})",
                )
                st.plotly_chart(fig_pca, width="stretch")
            with c_t2:
                sil_map = summary["clustering"]["silhouette_curve"]
                fig_sil = go.Figure(
                    go.Scatter(x=[int(k) for k in sil_map], y=[float(v) for v in sil_map.values()], mode="lines+markers")
                )
                fig_sil.add_vline(x=8, line_dash="dash", line_color="#10B981")
                fig_sil.update_layout(height=300, title="Silhouette Score Across k=4..10")
                st.plotly_chart(fig_sil, width="stretch")

    # =========================================================================
    # PAGE 3: WHAT DRIVES EACH ROLE (DATA-DRIVEN PRIORITIES)
    # =========================================================================
    elif page == "bayesian":
        selected_cat = st.selectbox(
            t("select_category", lang),
            options=CATEGORY_ORDER,
            format_func=lambda c: cat_name(c, lang),
        )
        spec = CATEGORY_SPECS[selected_cat]
        st.info(f"**{cat_name(selected_cat, lang)}** — {spec['goal_en'] if lang == 'en' else spec['goal_zh']}")
        st.caption(t("bayes_explain", lang))

        m_l1 = l1_weights[selected_cat]
        m_l2 = l2_weights[selected_cat]

        k1, k2, k3 = st.columns(3)
        k1.metric("Products in This Group" if lang == "en" else "该类别产品数量", f"{m_l1['n_samples']:,} SKUs")
        k2.metric("Product Grading Goal" if lang == "en" else "产品评优看什么结果", m_l1["target_en"] if lang == "en" else m_l1["target_zh"])
        k3.metric("Ad Optimization Goal" if lang == "en" else "广告打分看什么结果", m_l2["target_en"] if lang == "en" else m_l2["target_zh"])

        def _plot_priority_bars(model_dict: Dict, title_txt: str) -> go.Figure:
            w_df = pd.DataFrame(model_dict["weights"]).sort_values("posterior_mean", ascending=True).reset_index(drop=True)
            w_df["label"] = w_df["feature"].map(lambda f: m_label(f, lang))
            w_df["post_pct"] = w_df["posterior_mean"] * 100.0
            w_df["prior_pct"] = w_df["prior_mean"] * 100.0
            w_df["err_hi"] = (w_df["hdi_high"] - w_df["posterior_mean"]) * 100.0
            w_df["err_lo"] = (w_df["posterior_mean"] - w_df["hdi_low"]) * 100.0

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=w_df["label"],
                    x=w_df["post_pct"],
                    orientation="h",
                    name="Data-Proven Weight (%)" if lang == "en" else "数据算出实际权重 (%)",
                    marker_color=CATEGORY_SPECS[selected_cat]["color"],
                    error_x=dict(type="data", symmetric=False, array=w_df["err_hi"].tolist(), arrayminus=w_df["err_lo"].tolist(), thickness=1.5, color="#334155"),
                    text=[f"{v:.1f}%" for v in w_df["post_pct"]],
                    textposition="outside",
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=w_df["label"],
                    x=w_df["prior_pct"],
                    mode="markers",
                    name="Initial Guess (%)" if lang == "en" else "人工经验初始猜想 (%)",
                    marker=dict(symbol="diamond-open", size=11, color="#0F172A", line=dict(width=2)),
                )
            )
            fig.update_layout(
                title=title_txt,
                height=340,
                xaxis_title="Relative Importance (%)" if lang == "en" else "指标重要度占比 (%)",
                xaxis=dict(range=[0, max(65.0, float(w_df["hdi_high"].max() * 100.0) + 12.0)]),
                margin=dict(l=10, r=20, t=45, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            return fig

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.plotly_chart(_plot_priority_bars(m_l1, t("bayes_l1_title", lang)), width="stretch")
            top1 = sorted(m_l1["weights"], key=lambda x: x["posterior_mean"], reverse=True)[:2]
            msg1 = (
                f"**Plain takeaway**: In `{selected_cat}`, **{m_label(top1[0]['feature'], 'en')} ({top1[0]['posterior_mean']*100:.0f}%)** and **{m_label(top1[1]['feature'], 'en')} ({top1[1]['posterior_mean']*100:.0f}%)** explain most of the difference between Tier A and Tier C SKUs."
                if lang == "en"
                else f"**大白话结论**：在 `{selected_cat}` 类产品里，**{m_label(top1[0]['feature'], 'zh')} ({top1[0]['posterior_mean']*100:.0f}%)** 和 **{m_label(top1[1]['feature'], 'zh')} ({top1[1]['posterior_mean']*100:.0f}%)** 最能决定谁是 A 级好货。"
            )
            st.success(msg1)

        with col_b2:
            st.plotly_chart(_plot_priority_bars(m_l2, t("bayes_l2_title", lang)), width="stretch")
            top2 = sorted(m_l2["weights"], key=lambda x: x["posterior_mean"], reverse=True)[:2]
            msg2 = (
                f"**Plain takeaway**: For daily ads in `{selected_cat}`, focus budget on SKUs strong in **{m_label(top2[0]['feature'], 'en')} ({top2[0]['posterior_mean']*100:.0f}%)** and **{m_label(top2[1]['feature'], 'en')} ({top2[1]['posterior_mean']*100:.0f}%)**."
                if lang == "en"
                else f"**大白话结论**：这类产品的广告，重点给 **{m_label(top2[0]['feature'], 'zh')} ({top2[0]['posterior_mean']*100:.0f}%)** 和 **{m_label(top2[1]['feature'], 'zh')} ({top2[1]['posterior_mean']*100:.0f}%)** 表现好的 SKU 加预算。"
            )
            st.success(msg2)

        with st.expander("For Data Teams: PyMC MCMC Diagnostics (Spearman ρ, R-hat, ESS, 95% HDI)" if lang == "en" else "给技术人员看的统计诊断（Spearman ρ、R-hat、ESS、95% HDI）", expanded=False):
            st.write(f"Layer-1 Spearman ρ = `{m_l1['spearman_rho']:.4f}` (max R-hat `{m_l1['max_rhat']:.4f}`) | Layer-2 Spearman ρ = `{m_l2['spearman_rho']:.4f}` (max R-hat `{m_l2['max_rhat']:.4f}`)")
            d1, d2 = st.columns(2)
            with d1:
                st.dataframe(pd.DataFrame(m_l1["weights"]), width="stretch", hide_index=True)
            with d2:
                st.dataframe(pd.DataFrame(m_l2["weights"]), width="stretch", hide_index=True)

    # =========================================================================
    # PAGE 4: SMART BUDGET PLANNER
    # =========================================================================
    elif page == "allocator":
        st.subheader(t("allocator_title", lang))

        default_total = float(round(daily_scores["spend"].sum(), -2))
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        total_budget = p_col1.number_input(
            t("total_budget_label", lang),
            min_value=5000.0,
            max_value=500000.0,
            value=default_total,
            step=2000.0,
        )

        spread_options = ["Balanced Spread (Recommended)", "Concentrate on Top Winners", "Broad Even Coverage"] if lang == "en" else ["均衡聚焦（推荐）", "激进集中头部爆款", "广撒网分散覆盖"]
        spread_alpha_map = {0: 0.65, 1: 0.78, 2: 0.45}
        spread_idx = p_col2.selectbox(
            t("spread_label", lang),
            options=[0, 1, 2],
            format_func=lambda i: spread_options[i],
        )
        alpha = spread_alpha_map[spread_idx]

        max_acos_p1 = p_col3.slider("New Product Max Ad Cost % (ACOS)" if lang == "en" else "新品广告费率安全上限 (ACOS)", 0.25, 1.20, 0.65, 0.05)
        min_roas_core = p_col4.slider("Mature Product Min Return (ROAS)" if lang == "en" else "成熟品最低回报要求 (ROAS)", 0.80, 4.00, 1.50, 0.10)

        preset_opts = ["Balanced (Default)", "Growth Push (More to P1/P2)", "Profit Protection (More to P5/P6)"] if lang == "en" else ["均衡默认（架构推荐）", "扩张抢量（倾斜 P1 新品/P2 引流）", "利润防守（倾斜 P5 高毛利/P6 防守）"]
        preset_idx = st.radio(
            "Business Goal Preset" if lang == "en" else "经营目标预设",
            options=[0, 1, 2],
            format_func=lambda i: preset_opts[i],
            horizontal=True,
        )
        preset_bounds_map = {
            0: {c: (int(CATEGORY_SPECS[c]["budget_min_pct"] * 100), int(CATEGORY_SPECS[c]["budget_max_pct"] * 100)) for c in CATEGORY_ORDER},
            1: {"P1": (10, 20), "P2": (22, 38), "P3": (10, 20), "P4": (8, 18), "P5": (8, 18), "P6": (4, 12), "P7": (1, 5), "P8": (3, 8)},
            2: {"P1": (4, 12), "P2": (12, 22), "P3": (10, 20), "P4": (12, 22), "P5": (18, 32), "P6": (10, 20), "P7": (2, 6), "P8": (5, 10)},
        }

        custom_bounds: Dict[str, Tuple[float, float]] = {}
        with st.expander("Fine-Tune Category Min/Max Budget Limits (%)" if lang == "en" else "自定义调整 8 大类别的最低保障与最高上限 (%)", expanded=False):
            b_cols = st.columns(4)
            for idx, c in enumerate(CATEGORY_ORDER):
                with b_cols[idx % 4]:
                    lo_hi = st.slider(
                        cat_name(c, lang),
                        min_value=0,
                        max_value=50,
                        value=preset_bounds_map[preset_idx][c],
                        step=1,
                        key=f"bnd_{preset_idx}_{c}",
                    )
                    custom_bounds[c] = (lo_hi[0] / 100.0, lo_hi[1] / 100.0)

        sum_min_pct = sum(v[0] for v in custom_bounds.values()) * 100.0
        sum_max_pct = sum(v[1] for v in custom_bounds.values()) * 100.0
        if sum_min_pct > 100.0 + 1e-6 or sum_max_pct < 100.0 - 1e-6:
            st.error(f"{t('opt_bounds_invalid', lang)} (Min sum = {sum_min_pct:.0f}%, Max sum = {sum_max_pct:.0f}%)")
            return

        opt_df, cat_sum, opt_metrics = optimize_daily_budget(
            daily_scores,
            total_budget=total_budget,
            category_bounds=custom_bounds,
            alpha=alpha,
            max_acos_p1=max_acos_p1,
            min_roas_core=min_roas_core,
        )
        leg_bench = simulate_legacy_allocation(
            daily_scores,
            daily_scores["legacy_ad_score"],
            legacy_rules,
            total_budget=total_budget,
            alpha=alpha,
        )

        opt_rev_total = float(opt_df["est_opt_revenue"].sum())
        extra_rev_vs_legacy = opt_rev_total - float(leg_bench["legacy_total_revenue"])
        p7_opt_share = float(cat_sum.loc[cat_sum["category"] == "P7", "opt_share_pct"].iloc[0])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Planned Daily Budget" if lang == "en" else "当日分配总预算", f"${opt_metrics['allocated_budget']:,.0f}")
        k2.metric(
            "Projected Ad Revenue" if lang == "en" else "预期产生广告营收",
            f"${opt_rev_total:,.0f}",
            delta=f"{opt_metrics['optimized_roas']:.2f}x ROAS (+{opt_metrics['roas_uplift_pct']:.1f}%)",
        )
        k3.metric(
            "Extra Revenue vs Old Rules" if lang == "en" else "比旧规则多赚营收",
            f"+${extra_rev_vs_legacy:,.0f}/day",
            delta=f"{(opt_metrics['optimized_roas'] - leg_bench['legacy_roas']) / max(leg_bench['legacy_roas'], 1e-6) * 100:+.1f}% vs {leg_bench['legacy_roas']:.2f}x",
        )
        k4.metric(
            "Clearance (P7) Budget Cap" if lang == "en" else "清仓品预算控制占比",
            f"{p7_opt_share:.1f}%",
            delta=f"vs Old {leg_bench['legacy_p7_spend_pct']:.1f}%",
            delta_color="inverse",
        )

        ch1, ch2 = st.columns([1.35, 1.0])
        with ch1:
            fig_b = go.Figure()
            fig_b.add_trace(
                go.Bar(
                    name="Old Baseline ($)" if lang == "en" else "旧基线预算 ($)",
                    x=[cat_name(c, lang) for c in cat_sum["category"]],
                    y=cat_sum["baseline_budget"],
                    marker_color="#CBD5E1",
                )
            )
            fig_b.add_trace(
                go.Bar(
                    name="Recommended ($)" if lang == "en" else "智能推荐预算 ($)",
                    x=[cat_name(c, lang) for c in cat_sum["category"]],
                    y=cat_sum["opt_budget"],
                    marker_color="#1F4E79",
                )
            )
            fig_b.update_layout(
                title="Daily Budget Shift by Business Role ($)" if lang == "en" else "各业务角色每日预算分配对比 ($)",
                barmode="group",
                height=370,
                yaxis_title="Budget ($)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_b, width="stretch")

        with ch2:
            plat_df = opt_df.groupby("platform", as_index=False)[["optimized_spend"]].sum()
            fig_p = px.pie(
                plat_df,
                names="platform",
                values="optimized_spend",
                hole=0.45,
                title="Budget Share Across Ad Platforms" if lang == "en" else "各广告平台预算占比",
                color_discrete_sequence=["#1F4E79", "#10B981", "#3B82F6"],
            )
            fig_p.update_layout(height=370, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_p, width="stretch")

        st.download_button(
            label=t("download_excel", lang),
            data=to_excel_bytes(opt_df, cat_sum),
            file_name="daily_budget_allocation_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        opt_view = opt_df.copy()
        opt_view["action"] = np.where(
            opt_view["spend_delta_pct"] >= 10.0,
            "Scale Up ↑" if lang == "en" else "建议加投 ↑",
            np.where(opt_view["spend_delta_pct"] <= -10.0, "Trim Spend ↓" if lang == "en" else "建议缩减 ↓", "Hold →" if lang == "en" else "维持不变 →"),
        )
        disp_cols = ["sku_id", "strategic_role", "abc_tier", "platform", "action", "daily_ad_score", "baseline_spend_scaled", "optimized_spend", "spend_delta", "est_opt_revenue", "roas"]
        tab_gain, tab_cut = st.tabs([
            "Top Products to Scale Up ↑" if lang == "en" else "建议重点加投的 SKU ↑",
            "Top Products to Trim ↓" if lang == "en" else "建议缩减止损的 SKU ↓",
        ])
        with tab_gain:
            st.dataframe(opt_view.sort_values("spend_delta", ascending=False)[disp_cols].head(25), width="stretch", hide_index=True)
        with tab_cut:
            st.dataframe(opt_view.sort_values("spend_delta", ascending=True)[disp_cols].head(25), width="stretch", hide_index=True)

    # =========================================================================
    # PAGE 5: DAILY ACTION ALERTS
    # =========================================================================
    elif page == "monitor":
        st.subheader(t("monitor_title", lang))
        st.info(t("reverse_norm_note", lang))

        high_cnt = int((anomalies["severity"] == "HIGH").sum()) if not anomalies.empty else 0
        med_cnt = int((anomalies["severity"] == "MEDIUM").sum()) if not anomalies.empty else 0
        at_risk_spend = float(anomalies["spend"].sum()) if not anomalies.empty else 0.0

        a1, a2, a3 = st.columns(3)
        a1.metric("Urgent Alerts (Fix Today)" if lang == "en" else "高优止损告警（今日处理）", high_cnt)
        a2.metric("Watchlist Alerts" if lang == "en" else "一般波动提醒", med_cnt)
        a3.metric("Daily Spend on Flagged Ads" if lang == "en" else "异常广告涉及日花费", f"${at_risk_spend:,.0f}")

        if not anomalies.empty:
            action_map_en = {
                "F2: Creative Fatigue (CTR Drop)": "Refresh ad image / headline",
                "F3: Listing/Offer Friction (CVR Drop)": "Check price, reviews, or stock",
                "F2+F3: Funnel Collapse": "Pause ad & audit product + creative",
                "F4: Auction Cost / ROAS Erosion": "Lower bids or trim weak keywords",
            }
            action_map_zh = {
                "F2：素材点击率骤降": "更换主图或广告文案，防素材疲劳",
                "F3：详情页转化率异常下滑": "排查价格、差评或断货问题",
                "F2+F3：曝光点击与转化双降": "先暂停止损，同时排查素材与详情页",
                "F4：投放成本上升或回报骤降": "降低竞价出价，剔除高耗低转词",
            }
            sev_filter = st.multiselect("Priority" if lang == "en" else "紧急程度", ["HIGH", "MEDIUM"], default=["HIGH", "MEDIUM"])
            al_view = anomalies[anomalies["severity"].isin(sev_filter)].copy()
            diag_col = "diagnosis_en" if lang == "en" else "diagnosis_zh"
            al_view["recommended_fix"] = al_view[diag_col].map(lambda s: (action_map_en if lang == "en" else action_map_zh).get(s, "Review campaign"))
            al_cols = [
                "severity",
                "sku_id",
                "strategic_role",
                "platform",
                diag_col,
                "recommended_fix",
                "spend",
                "ctr_delta_pct",
                "cvr_delta_pct",
                "roas_delta_pct",
            ]
            st.dataframe(al_view[al_cols].head(100), width="stretch", hide_index=True)

            st.subheader("Inspect One Flagged Product (30-Day History)" if lang == "en" else "查看单个异常产品的近 30 天走势")
            panel = load_full_ad_panel()
            avail_drill_skus = set(panel["sku_id"].unique().tolist())
            flagged_avail = [s for s in al_view["sku_id"].unique().tolist() if s in avail_drill_skus]
            sku_opts = flagged_avail if flagged_avail else sorted(avail_drill_skus)[:40]
            pick_sku = st.selectbox("Select Product" if lang == "en" else "选择产品编号", options=sku_opts)
            sku_hist = panel[panel["sku_id"] == pick_sku].sort_values("date")
            fig_sku = px.line(
                sku_hist,
                x="date",
                y=["ctr", "cvr"],
                facet_row="platform",
                title=f"{pick_sku}: Click Rate (CTR) & Conversion Rate (CVR) Over 30 Days" if lang == "en" else f"{pick_sku}：近 30 天点击率与转化率变化",
                height=420,
            )
            fig_sku.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sku, width="stretch")

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        trend_metric = col_t1.selectbox(
            "30-Day Portfolio Trend" if lang == "en" else "查看 30 天整体走势指标",
            options=["ad_revenue", "spend", "roas", "ctr", "cvr"],
            format_func=lambda m: m.upper(),
        )
        trend_plat = col_t2.multiselect("Platform" if lang == "en" else "筛选平台", PLATFORMS, default=PLATFORMS)
        tr_sub = daily_trend[daily_trend["platform"].isin(trend_plat)].groupby(["date", "strategic_role"], as_index=False).agg(
            spend=("spend", "sum"),
            ad_revenue=("ad_revenue", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            orders=("orders", "sum"),
        )
        tr_sub["roas"] = tr_sub["ad_revenue"] / np.maximum(tr_sub["spend"], 1.0)
        tr_sub["ctr"] = tr_sub["clicks"] / np.maximum(tr_sub["impressions"], 1.0)
        tr_sub["cvr"] = tr_sub["orders"] / np.maximum(tr_sub["clicks"], 1.0)

        fig_tr = px.line(
            tr_sub,
            x="date",
            y=trend_metric,
            color="strategic_role",
            color_discrete_map={c: CATEGORY_SPECS[c]["color"] for c in CATEGORY_ORDER},
            height=320,
        )
        fig_tr.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_tr, width="stretch")


if __name__ == "__main__":
    main()
