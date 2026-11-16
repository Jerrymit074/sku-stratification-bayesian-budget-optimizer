"""Canonical architecture configuration for Product Stratification & Ad Budget Optimization."""

from pathlib import Path
from typing import Dict, List, TypedDict


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DEMO_DIR = ROOT_DIR / "data" / "demo"
ARTIFACTS_DIR = ROOT_DIR / "data" / "artifacts"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RAW_ABC_PATH = ROOT_DIR / "Classify_ABC_2024-03-14.xlsx"
RAW_ARRANGEMENT_PATH = ROOT_DIR / "arrangement_source.xlsx"


class CategorySpec(TypedDict):
    code: str
    name_zh: str
    name_en: str
    goal_zh: str
    goal_en: str
    layer1_target: str
    layer1_target_zh: str
    layer1_target_en: str
    layer1_priors: List[float]  # [score_sales, score_revenue, score_margin, score_profit]
    ad_metrics: List[str]
    layer2_target: str
    layer2_target_zh: str
    layer2_target_en: str
    layer2_priors: List[float]
    budget_min_pct: float
    budget_max_pct: float
    budget_policy_zh: str
    budget_policy_en: str
    color: str


LAYER1_FEATURE_COLS = [
    "score_sales",
    "score_revenue",
    "score_margin",
    "score_profit",
]

REVERSE_METRICS = {"acos", "cac", "inventory_turnover_days"}

PLATFORMS = ["Amazon Ads", "TikTok Ads", "Meta Ads"]

TIER_MULTIPLIERS = {
    "A": {"share": 0.20, "min": 1.00, "max": 1.50, "default": 1.25},
    "B": {"share": 0.40, "min": 0.70, "max": 1.00, "default": 0.85},
    "C": {"share": 0.40, "min": 0.30, "max": 0.70, "default": 0.50},
}

CATEGORY_SPECS: Dict[str, CategorySpec] = {
    "P1": {
        "code": "P1",
        "name_zh": "新品孵化型",
        "name_en": "New Product Incubation",
        "goal_zh": "快速验证市场响应，积累初始数据",
        "goal_en": "Validate market fit rapidly and gather initial traction",
        "layer1_target": "sales_mom_growth",
        "layer1_target_zh": "销量环比增长率",
        "layer1_target_en": "MoM Sales Growth Rate",
        "layer1_priors": [0.40, 0.25, 0.15, 0.20],
        "ad_metrics": ["impressions", "ctr", "cvr", "acos", "ntb_pct"],
        "layer2_target": "sales_t30",
        "layer2_target_zh": "第 T+30 天销量",
        "layer2_target_en": "Day T+30 Unit Sales",
        "layer2_priors": [0.22, 0.20, 0.20, 0.15, 0.23],
        "budget_min_pct": 0.05,
        "budget_max_pct": 0.15,
        "budget_policy_zh": "保护性预算",
        "budget_policy_en": "Protected Sandbox",
        "color": "#3B82F6",
    },
    "P2": {
        "code": "P2",
        "name_zh": "爆款引流型",
        "name_en": "Hero Traffic Driver",
        "goal_zh": "最大化曝光和流量获取，降低获客成本",
        "goal_en": "Maximize traffic acquisition and lower CAC",
        "layer1_target": "sales_qty",
        "layer1_target_zh": "实际总销量",
        "layer1_target_en": "Total Unit Sales",
        "layer1_priors": [0.55, 0.25, 0.05, 0.15],
        "ad_metrics": ["clicks", "ctr", "cvr", "roas", "cac"],
        "layer2_target": "total_traffic",
        "layer2_target_zh": "总流量 (点击量+展示量)",
        "layer2_target_en": "Total Traffic Volume",
        "layer2_priors": [0.30, 0.22, 0.12, 0.14, 0.22],
        "budget_min_pct": 0.15,
        "budget_max_pct": 0.30,
        "budget_policy_zh": "扩张性预算",
        "budget_policy_en": "Aggressive Growth",
        "color": "#EF4444",
    },
    "P3": {
        "code": "P3",
        "name_zh": "销量驱动型",
        "name_en": "Volume Scale Driver",
        "goal_zh": "维持规模增长，扩大市场份额",
        "goal_en": "Sustain unit scale and expand market share",
        "layer1_target": "sales_qty",
        "layer1_target_zh": "实际总销量",
        "layer1_target_en": "Total Unit Sales",
        "layer1_priors": [0.45, 0.25, 0.15, 0.15],
        "ad_metrics": ["clicks", "ctr", "cvr", "roas", "sales_growth_rate"],
        "layer2_target": "total_sales",
        "layer2_target_zh": "总销量",
        "layer2_target_en": "Total Ad-Driven Units",
        "layer2_priors": [0.25, 0.18, 0.22, 0.15, 0.20],
        "budget_min_pct": 0.10,
        "budget_max_pct": 0.20,
        "budget_policy_zh": "稳定性预算",
        "budget_policy_en": "Core Stability",
        "color": "#F59E0B",
    },
    "P4": {
        "code": "P4",
        "name_zh": "销售额驱动型",
        "name_en": "Revenue Driver",
        "goal_zh": "贡献总营收，维持现金流健康",
        "goal_en": "Drive top-line revenue and healthy cash flow",
        "layer1_target": "revenue",
        "layer1_target_zh": "实际总销售额",
        "layer1_target_en": "Total Revenue",
        "layer1_priors": [0.20, 0.50, 0.10, 0.20],
        "ad_metrics": ["impression_share", "ctr", "cvr", "roas", "aov"],
        "layer2_target": "total_revenue",
        "layer2_target_zh": "总销售额",
        "layer2_target_en": "Total Ad Revenue",
        "layer2_priors": [0.20, 0.15, 0.20, 0.25, 0.20],
        "budget_min_pct": 0.10,
        "budget_max_pct": 0.20,
        "budget_policy_zh": "回报性预算",
        "budget_policy_en": "Revenue Return",
        "color": "#8B5CF6",
    },
    "P5": {
        "code": "P5",
        "name_zh": "高利润收割型",
        "name_en": "High-Margin Harvester",
        "goal_zh": "最大化单品利润，提升整体盈利质量",
        "goal_en": "Maximize unit profit and portfolio margin quality",
        "layer1_target": "profit_x_margin",
        "layer1_target_zh": "利润额 × 利润率",
        "layer1_target_en": "Profit × Margin Index",
        "layer1_priors": [0.10, 0.20, 0.35, 0.35],
        "ad_metrics": ["clicks", "ctr", "cvr", "roas", "margin_change_pct"],
        "layer2_target": "total_profit",
        "layer2_target_zh": "总利润额",
        "layer2_target_en": "Total Ad Profit",
        "layer2_priors": [0.14, 0.16, 0.22, 0.28, 0.20],
        "budget_min_pct": 0.10,
        "budget_max_pct": 0.25,
        "budget_policy_zh": "效率性预算",
        "budget_policy_en": "Profit Efficiency",
        "color": "#10B981",
    },
    "P6": {
        "code": "P6",
        "name_zh": "份额防守型",
        "name_en": "Share Defender",
        "goal_zh": "保卫现有市场份额，防止被竞品蚕食",
        "goal_en": "Defend category share and brand mindshare",
        "layer1_target": "revenue_stability",
        "layer1_target_zh": "销售额稳定性指数",
        "layer1_target_en": "Revenue Stability Index",
        "layer1_priors": [0.25, 0.35, 0.20, 0.20],
        "ad_metrics": ["impression_share", "ctr", "cvr", "roas", "brand_search_growth_pct"],
        "layer2_target": "revenue_stability_idx",
        "layer2_target_zh": "销售额稳定性指数",
        "layer2_target_en": "Revenue Stability Index",
        "layer2_priors": [0.28, 0.16, 0.16, 0.16, 0.24],
        "budget_min_pct": 0.05,
        "budget_max_pct": 0.15,
        "budget_policy_zh": "防御性预算",
        "budget_policy_en": "Share Defense",
        "color": "#06B6D4",
    },
    "P7": {
        "code": "P7",
        "name_zh": "清仓退出型",
        "name_en": "Clearance Exit",
        "goal_zh": "快速出清库存，回收现金流，腾出货架",
        "goal_en": "Accelerate inventory clearance and recover cash",
        "layer1_target": "inventory_turnover_rate",
        "layer1_target_zh": "库存周转率",
        "layer1_target_en": "Inventory Turnover Rate",
        "layer1_priors": [0.40, 0.25, 0.20, 0.15],
        "ad_metrics": ["impressions", "ctr", "cvr", "acos", "inventory_turnover_days"],
        "layer2_target": "inventory_clearance_speed",
        "layer2_target_zh": "库存下降速度",
        "layer2_target_en": "Inventory Clearance Speed",
        "layer2_priors": [0.25, 0.15, 0.18, 0.17, 0.25],
        "budget_min_pct": 0.02,
        "budget_max_pct": 0.08,
        "budget_policy_zh": "收缩性预算",
        "budget_policy_en": "Controlled Taper",
        "color": "#64748B",
    },
    "P8": {
        "code": "P8",
        "name_zh": "均衡长尾型",
        "name_en": "Balanced Long-Tail",
        "goal_zh": "维持运营效率，作为产品矩阵底座",
        "goal_en": "Maintain baseline efficiency across catalog tail",
        "layer1_target": "profit",
        "layer1_target_zh": "实际总利润额",
        "layer1_target_en": "Total Profit",
        "layer1_priors": [0.25, 0.25, 0.25, 0.25],
        "ad_metrics": ["clicks", "ctr", "cvr", "roas", "repeat_purchase_rate"],
        "layer2_target": "total_profit",
        "layer2_target_zh": "总利润额",
        "layer2_target_en": "Total Ad Profit",
        "layer2_priors": [0.18, 0.18, 0.22, 0.24, 0.18],
        "budget_min_pct": 0.05,
        "budget_max_pct": 0.10,
        "budget_policy_zh": "基础性预算",
        "budget_policy_en": "Baseline Maintenance",
        "color": "#EC4899",
    },
}

CATEGORY_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

METRIC_LABELS = {
    "score_sales": {"zh": "销量得分", "en": "Sales Score"},
    "score_revenue": {"zh": "销售额得分", "en": "Revenue Score"},
    "score_margin": {"zh": "利润率得分", "en": "Margin Score"},
    "score_profit": {"zh": "利润额得分", "en": "Profit Score"},
    "impressions": {"zh": "展示量", "en": "Impressions"},
    "impression_share": {"zh": "展示份额", "en": "Impression Share"},
    "clicks": {"zh": "点击量", "en": "Clicks"},
    "ctr": {"zh": "CTR 点击率", "en": "CTR"},
    "cvr": {"zh": "CVR 转化率", "en": "CVR"},
    "roas": {"zh": "ROAS", "en": "ROAS"},
    "acos": {"zh": "ACOS (逆向)", "en": "ACOS (Rev.)"},
    "cac": {"zh": "CAC 获客成本 (逆向)", "en": "CAC (Rev.)"},
    "ntb_pct": {"zh": "新客占比 NTB%", "en": "New-to-Brand %"},
    "sales_growth_rate": {"zh": "销量环比增长率", "en": "Sales Growth %"},
    "aov": {"zh": "平均客单价 AOV", "en": "Avg Order Value"},
    "margin_change_pct": {"zh": "利润率变化 %", "en": "Margin Change %"},
    "brand_search_growth_pct": {"zh": "品牌搜索量增长 %", "en": "Brand Search Growth %"},
    "inventory_turnover_days": {"zh": "库存周转天数 (逆向)", "en": "Inv. Turnover Days (Rev.)"},
    "repeat_purchase_rate": {"zh": "复购率", "en": "Repeat Purchase Rate"},
}
