"""Bilingual (English / Simplified Chinese) business-friendly copy for the UI."""

from typing import Dict


TEXTS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "en": "Smart Product Grouping & Ad Budget Planner",
        "zh": "智能产品分层与广告预算分配系统",
    },
    "app_subtitle": {
        "en": "Classify 2,511 products into 8 practical business roles and automatically allocate daily ad spend to top performers.",
        "zh": "基于销售与利润数据将 2,511 个产品分为 8 类业务角色，并结合每日广告表现自动算出最优预算分配方案。",
    },
    "nav_title": {
        "en": "Menu",
        "zh": "功能菜单",
    },
    "nav_overview": {
        "en": "1. Business Summary",
        "zh": "1. 业务总览：分类与预算全景",
    },
    "nav_clustering": {
        "en": "2. Product Roles (P1–P8)",
        "zh": "2. 产品怎么分类：8大业务角色",
    },
    "nav_bayesian": {
        "en": "3. What Drives Each Role",
        "zh": "3. 各类别看重什么指标",
    },
    "nav_allocator": {
        "en": "4. Smart Budget Planner",
        "zh": "4. 广告预算怎么分（沙盘导出）",
    },
    "nav_monitor": {
        "en": "5. Daily Action Alerts",
        "zh": "5. 每日投放诊断与止损",
    },
    "sidebar_note": {
        "en": "Demo uses anonymized distribution from 2,511 real SKUs paired with structured multi-platform ad data.",
        "zh": "演示基于 2,511 个脱敏真实产品分布 + 多平台结构化广告数据。",
    },
    "overview_intro": {
        "en": (
            "**How it works in 3 plain steps:**  \n"
            "1. **Group by Business Role**: Instead of scoring new launches and clearance items on the same scale, each SKU is placed into one of 8 practical roles (`P1–P8`).  \n"
            "2. **Learn What Matters**: Historical data automatically measures which metrics drive success for each role—replacing guesswork.  \n"
            "3. **Put Budget Where It Pays**: Daily ad spend is shifted away from low-return clearance traps into high-profit and high-growth winners."
        ),
        "zh": (
            "**三步看懂这套系统怎么帮你赚钱：**  \n"
            "1. **先分角色再打分**：不再让新品、引流款、高毛利款和清仓品在同一把尺子上硬比，先把 2,511 个产品分成 **8 种清晰的业务角色（P1–P8）**。  \n"
            "2. **用数据代替拍脑袋**：用历史销售与广告结果自动算出每类产品真正看重什么指标，并排出每类里的 **A / B / C 优劣档位**。  \n"
            "3. **把钱花在刀刃上**：每日自动把预算从低效清仓品挪向高增长和高毛利爆款，同时给每类产品留足安全边界。"
        ),
    },
    "kpi_skus": {
        "en": "Products Managed",
        "zh": "纳管产品总数",
    },
    "kpi_budget": {
        "en": "Daily Ad Budget",
        "zh": "每日广告总预算",
    },
    "kpi_rev": {
        "en": "Projected Daily Ad Revenue",
        "zh": "优化后预期日广告营收",
    },
    "kpi_focus": {
        "en": "Budget in Growth & Profit Roles",
        "zh": "高价值角色预算占比 (P1+P2+P5)",
    },
    "sankey_title": {
        "en": "Why One-Size-Fits-All 'ABC' Fails: How Old Ranks Split Across 8 Roles",
        "zh": "为什么旧版 ABC 一刀切不准：旧等级在新 8 类角色中的重新洗牌",
    },
    "role_table_title": {
        "en": "8 Business Product Roles at a Glance",
        "zh": "八大产品业务角色一览表",
    },
    "margin_cap_title": {
        "en": "Fair Scoring Check: Fixing the Profit Margin Ceiling",
        "zh": "公平打分检查：为什么毛利率上限不能死守 60%",
    },
    "margin_cap_slider": {
        "en": "Adjust Margin Cap (%)",
        "zh": "拖动调整毛利率满分上限 Cap (%)",
    },
    "margin_cap_alert": {
        "en": "Data check: 56.4% of your products already have margins above 60% (median 62.9%). If capped at 60%, over half the catalog ties at 100 points! Raising the cap to 85% lets real high-margin stars stand out.",
        "zh": "数据实测发现：全店有 56.4% 的产品毛利率超过 60%（中位数 62.9%）。如果按旧规 60% 封顶，一半以上产品全拿满分、拉不开差距！把上限调到 85% 就能把真正的高毛利明星款区分出来。",
    },
    "cluster_biz_title": {
        "en": "Product Map: Unit Sales Volume vs. Profit Margin (%)",
        "zh": "业务全景图：销量规模 vs 毛利率（气泡大小 = 销售额）",
    },
    "cluster_bar_title": {
        "en": "SKU Count & Tier Breakdown by Role",
        "zh": "各类别产品数量与 A/B/C 分布",
    },
    "sku_lookup_title": {
        "en": "Product Lookup & Next-Step Watchlist",
        "zh": "单品分层明细检索与流转预警",
    },
    "select_category": {
        "en": "Choose a Product Role to Inspect",
        "zh": "选择要查看的产品业务角色",
    },
    "bayes_l1_title": {
        "en": "What Defines a 'Tier A' Product in This Role?",
        "zh": "在这个角色里，靠什么评出 A / B / C 级好产品？",
    },
    "bayes_l2_title": {
        "en": "Which Daily Ad Metrics Matter Most Here?",
        "zh": "在这个角色里，日常广告投放最看重哪 5 个指标？",
    },
    "bayes_explain": {
        "en": "Gray diamonds show the team's initial assumption; colored bars show what historical performance data actually proved important (with uncertainty range).",
        "zh": "灰色空心菱形代表“人工经验猜的初始权重”，彩色实心点代表“用历史数据真实算出来的有效权重”（横线为可信波动范围）。",
    },
    "allocator_title": {
        "en": "Interactive Daily Ad Budget Planner",
        "zh": "广告预算分配实时模拟与导出",
    },
    "total_budget_label": {
        "en": "Daily Ad Budget ($)",
        "zh": "每日广告总预算 ($)",
    },
    "spread_label": {
        "en": "Allocation Style",
        "zh": "资金集中度策略",
    },
    "opt_bounds_invalid": {
        "en": "Budget limits don't add up: sum of Min% must be <= 100% and sum of Max% must be >= 100%.",
        "zh": "各类别预算比例边界冲突：最低保障之和必须 ≤ 100%，最高上限之和必须 ≥ 100%。",
    },
    "download_excel": {
        "en": "Download Ready-to-Use Excel Budget Plan (.xlsx)",
        "zh": "一键导出可执行 Excel 预算分配表 (.xlsx)",
    },
    "monitor_title": {
        "en": "Daily Ad Health & Action Alerts",
        "zh": "每日广告健康度监控与止损建议",
    },
    "reverse_norm_note": {
        "en": "Note: Cost & speed metrics (ACOS, Customer Acquisition Cost, Inventory Days) work in reverse—lower numbers earn higher performance scores.",
        "zh": "提示：成本与周转类指标（ACOS 广告费率、CAC 获客成本、库存周转天数）采用“越低得分越高”的逆向计分方式，确保所有分数统一为“越高越好”。",
    },
}


def t(key: str, lang: str) -> str:
    """Lookup localized string."""
    item = TEXTS.get(key)
    if not item:
        return key
    return item.get(lang, item.get("en", key))
