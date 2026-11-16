"""Generate a self-contained business-friendly interactive HTML report in outputs/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ARTIFACTS_DIR, CATEGORY_ORDER, CATEGORY_SPECS, OUTPUTS_DIR


def export_html_report(output_path: Path | None = None) -> Path:
    """Build a standalone executive HTML report that opens directly in any browser."""
    if output_path is None:
        output_path = OUTPUTS_DIR / "interactive_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads((ARTIFACTS_DIR / "pipeline_summary.json").read_text(encoding="utf-8"))
    sku_df = pd.read_csv(ARTIFACTS_DIR / "sku_classified.csv")
    cat_budget = pd.read_csv(ARTIFACTS_DIR / "category_budget_summary.csv")
    budget_base = pd.read_csv(ARTIFACTS_DIR / "budget_allocation_baseline.csv")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. How 2,511 Products Group into 8 Roles (P1-P8)",
            "2. Product Business Map: Log Sales vs. Profit Margin (%)",
            "3. Daily Budget Allocation: Old Baseline vs. Smart Plan ($)",
            "4. Expected Daily Ad Revenue & ROAS Comparison",
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    # 1. Role counts stacked by A/B/C
    for tier, color in [("A", "#1F4E79"), ("B", "#3B82F6"), ("C", "#94A3B8")]:
        counts = [int(((sku_df["strategic_role"] == c) & (sku_df["abc_tier"] == tier)).sum()) for c in CATEGORY_ORDER]
        fig.add_trace(go.Bar(name=f"Tier {tier}", x=CATEGORY_ORDER, y=counts, marker_color=color), row=1, col=1)

    # 2. Intuitive business map: log10(sales) vs profit_margin %
    for c in CATEGORY_ORDER:
        sub = sku_df[sku_df["strategic_role"] == c]
        fig.add_trace(
            go.Scatter(
                x=np.log10(np.maximum(sub["sales_qty"], 1)),
                y=sub["profit_margin"] * 100.0,
                mode="markers",
                name=f"{c} {CATEGORY_SPECS[c]['name_en']}",
                marker=dict(size=5, opacity=0.70, color=CATEGORY_SPECS[c]["color"]),
                text=sub["sku_id"],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. Dollar budget comparison
    fig.add_trace(
        go.Bar(
            name="Old Baseline ($)",
            x=cat_budget["category"],
            y=cat_budget["baseline_budget"],
            marker_color="#94A3B8",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Smart Plan ($)",
            x=cat_budget["category"],
            y=cat_budget["opt_budget"],
            marker_color="#10B981",
        ),
        row=2,
        col=1,
    )

    # 4. Expected Daily Ad Revenue comparison across 3 methods
    opt = summary["optimization"]
    base_rev = float(budget_base["est_base_revenue"].sum())
    legacy_rev = float(opt["legacy_total_revenue"])
    opt_rev = float(budget_base["est_opt_revenue"].sum())

    fig.add_trace(
        go.Bar(
            x=["Unoptimized Baseline", "Legacy 7-Metric Rules", "Smart Role + Data Optimizer"],
            y=[base_rev, legacy_rev, opt_rev],
            marker_color=["#CBD5E1", "#64748B", "#1F4E79"],
            text=[
                f"${base_rev:,.0f}<br>({opt['baseline_roas']:.2f}x ROAS)",
                f"${legacy_rev:,.0f}<br>({opt['legacy_roas']:.2f}x ROAS)",
                f"${opt_rev:,.0f}<br>({opt['optimized_roas']:.2f}x ROAS)",
            ],
            textposition="auto",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Log10(Unit Sales)", row=1, col=2)
    fig.update_yaxes(title_text="Profit Margin (%)", row=1, col=2)
    fig.update_yaxes(title_text="Daily Budget ($)", row=2, col=1)
    fig.update_yaxes(title_text="Projected Daily Revenue ($)", row=2, col=2)

    fig.update_layout(
        height=880,
        barmode="group",
        template="plotly_white",
        title_text=(
            f"Smart Product Grouping & Ad Budget Optimization — Executive Summary | "
            f"2,511 SKUs · ${opt['total_budget']:,.0f}/day Budget · "
            f"+{opt['roas_vs_legacy_uplift_pct']:.1f}% Revenue Uplift vs Legacy Rules"
        ),
    )

    html_body = fig.to_html(full_html=True, include_plotlyjs="cdn")
    output_path.write_text(html_body, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    p = export_html_report()
    print(f"Exported interactive HTML report to {p}")
