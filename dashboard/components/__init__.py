"""
Components package - reusable Streamlit components.
"""
from dashboard.components.charts import (
    bar_chart_comparison,
    line_chart_dp,
    heatmap_consistency,
    horizontal_bar_importance,
    scatter_equivalence,
)
from dashboard.components.tables import (
    styled_dataframe,
    display_df_with_download,
)
from dashboard.components.metrics import (
    metric_card,
    metric_row,
)
from dashboard.components.rules import (
    rule_card,
    parse_clinical_rules,
)
from dashboard.components.architecture import (
    render_architecture_diagram,
    render_co_mapping,
)

__all__ = [
    "bar_chart_comparison",
    "line_chart_dp",
    "heatmap_consistency",
    "horizontal_bar_importance",
    "scatter_equivalence",
    "styled_dataframe",
    "display_df_with_download",
    "metric_card",
    "metric_row",
    "rule_card",
    "parse_clinical_rules",
    "render_architecture_diagram",
    "render_co_mapping",
]