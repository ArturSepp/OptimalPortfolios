"""Plot exact instrument P&L for the best-relative futures cluster result.

The owner-frozen selected cell is M1-star, q=25%, monthly ROSAA momentum with a
12-month long span, no short/reversal span, volatility span 13, and EWMA mean
adjustment.  Both the cluster and same-signal global-rank books use +1/-1
long-short exposure, 30/30/30/10 sleeve budgets, one W-WED implementation lag,
10 bp one-way costs, and the U1 headline calendar.  The owner-frozen eleven-name
low-liquidity exclusion set is applied at every decision date.  The requested ``MMR1 Curncy``
does not exist in the source and is explicitly resolved to ``BMR1 Curncy``
(``BTC MINI``).  Cached cluster partitions are reused without refitting.

Instrument P&L is exact currency holding P&L less realised instrument costs:
prior units times the price change, minus the cost booked to that instrument.
It therefore reconciles to each portfolio's NAV change.  The scatter divides
each contribution by beginning NAV, placing M1-star cluster P&L on x and global
rank P&L on y for every contract eligible at least once in the analysis window.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs_version
from plotly.utils import PlotlyJSONEncoder
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_commodity_pnl_attribution as attribution
import papers.cluster_lineage_2026.replication.run_futures_prod_signal_grid_30303010_10bp as grid
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as construction
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as u1_prod


SPEC = grid.SignalSpec(short_span=None, vol_span=13, mean_adj_type="EWMA")
CLUSTER_METHOD = "sleeve_cluster_M1_star"
GLOBAL_METHOD = "sleeve_global"
Q = 0.25
CLUSTER_FALLBACK = 5
COST_BPS = 10.0
ACCOUNTING_TOLERANCE = 1e-10
REGRESSION_TOLERANCE = 1e-12
PLOT_FILE = "best_relative_cluster_vs_global_instrument_pnl.html"
EXPECTED_FUTURES_EXCLUSIONS = frozenset(
    {
        "BMR1 Curncy",
        "CUA1 Comdty",
        "IJ1 Comdty",
        "KC1 Comdty",
        "KM1 Index",
        "MES1 Index",
        "QC1 Index",
        "RS1 Comdty",
        "ST1 Index",
        "UXY1 Comdty",
        "WN1 Comdty",
    }
)
BEST_METHOD_STATUS = "OWNER_FROZEN_2026-08-15"
FROZEN_BEST_METHOD_SPEC = {
    "analysis_window": matched.WINDOW,
    "cluster_method": CLUSTER_METHOD,
    "global_benchmark": GLOBAL_METHOD,
    "strategy": "long_short",
    "q": Q,
    "signal_frequency": grid.SIGNAL_FREQUENCY,
    "momentum_long_span": grid.MOMENTUM_LONG_SPAN,
    "momentum_short_span": SPEC.short_span,
    "momentum_vol_span": SPEC.vol_span,
    "momentum_mean_adj_type": SPEC.mean_adj_type,
    "cluster_fallback": CLUSTER_FALLBACK,
    "implementation_lag_periods": 1,
    "cost_bps_one_way": COST_BPS,
    "sleeve_budgets_per_side": dict(construction.TARGET),
}
PERFORMANCE_METRICS = (
    "net_total_return",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "gross_return_annualized",
)
FROZEN_PERFORMANCE = {
    CLUSTER_METHOD: {
        "net_total_return": 0.005001341884408594,
        "net_return_annualized": 0.0002968655612269888,
        "volatility_annualized": 0.044196101171229,
        "sharpe_rf0": 0.01786713605973425,
        "one_way_turnover_annualized": 3.033029253130443,
        "cost_drag_bp_per_year": 122.04555214484047,
        "gross_return_annualized": 0.012501420775711036,
    },
    GLOBAL_METHOD: {
        "net_total_return": -0.0027331295501789032,
        "net_return_annualized": -0.00016282145030643846,
        "volatility_annualized": 0.08212519254414496,
        "sharpe_rf0": 0.02957235969914262,
        "one_way_turnover_annualized": 3.4414109147418026,
        "cost_drag_bp_per_year": 138.45758857912838,
        "gross_return_annualized": 0.0136829374076064,
    },
}
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_best_relative_pnl_scatter.py"
)


def _root() -> Path:
    """Return and create the external output directory beside the signal grid."""
    root = grid._root() / "best_relative_instrument_pnl_owner_exclusions_20260815"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _instrument_table(
    *,
    tickers: pd.Index,
    cluster_net_pnl: pd.Series,
    global_net_pnl: pd.Series,
    cluster_beginning_nav: float,
    global_beginning_nav: float,
    taxonomy: pd.DataFrame,
    sleeves: pd.Series,
    eligibility: pd.DataFrame | None = None,
    cluster_weights: pd.DataFrame | None = None,
    global_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the requested cluster-x/global-y instrument scatter data."""
    metadata = taxonomy.reindex(tickers)
    names = metadata["name"].where(metadata["name"].notna(), tickers.to_series())
    cluster_currency = cluster_net_pnl.reindex(tickers).fillna(0.0)
    global_currency = global_net_pnl.reindex(tickers).fillna(0.0)
    cluster_pct = 100.0 * cluster_currency / cluster_beginning_nav
    global_pct = 100.0 * global_currency / global_beginning_nav
    table = pd.DataFrame(
        {
            "ticker": tickers,
            "name": names.astype(str).str.replace("_", " ", regex=False).to_numpy(),
            "broad_asset_class": sleeves.reindex(tickers).to_numpy(),
            "source_asset_class": metadata["asset_class"].to_numpy(),
            "cluster_net_pnl_currency": cluster_currency.to_numpy(),
            "global_net_pnl_currency": global_currency.to_numpy(),
            "cluster_net_pnl_pct_of_start": cluster_pct.to_numpy(),
            "global_net_pnl_pct_of_start": global_pct.to_numpy(),
            "cluster_minus_global_pnl_pct_of_start": (
                cluster_pct - global_pct
            ).to_numpy(),
        }
    )
    table["eligible_decision_count"] = (
        np.nan
        if eligibility is None
        else eligibility.reindex(columns=tickers).sum(axis=0).to_numpy()
    )
    for prefix, weights in (
        ("cluster", cluster_weights),
        ("global", global_weights),
    ):
        if weights is None:
            table[f"{prefix}_long_decision_count"] = np.nan
            table[f"{prefix}_short_decision_count"] = np.nan
        else:
            aligned = weights.reindex(columns=tickers).fillna(0.0)
            table[f"{prefix}_long_decision_count"] = aligned.gt(0.0).sum().to_numpy()
            table[f"{prefix}_short_decision_count"] = aligned.lt(0.0).sum().to_numpy()
    table["hover_name"] = table["ticker"] + " — " + table["name"]
    label_count = min(4, len(table))
    label_tickers = set(
        table.nlargest(label_count, "cluster_minus_global_pnl_pct_of_start", keep="all")[
            "ticker"
        ]
    )
    label_tickers.update(
        table.nsmallest(label_count, "cluster_minus_global_pnl_pct_of_start", keep="all")[
            "ticker"
        ]
    )
    table["plot_label"] = table["ticker"].where(
        table["ticker"].isin(label_tickers), ""
    )
    return table.sort_values(["broad_asset_class", "ticker"]).reset_index(drop=True)


def _build_weights_and_portfolios() -> tuple[dict, dict[str, pd.DataFrame], dict]:
    """Reconstruct the two frozen portfolios without refitting any cluster model."""
    context = grid._build_context()
    eligibility = context["eligibility"]
    sleeve_panel = context["sleeve_panel"]
    groups_by_method = context["groups_by_method"]
    prices = context["performance_prices"]
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("eligibility is not a DataFrame")
    if not isinstance(sleeve_panel, pd.DataFrame):
        raise AssertionError("sleeve panel is not a DataFrame")
    if not isinstance(groups_by_method, dict):
        raise AssertionError("group panels are not a dictionary")
    if not isinstance(prices, pd.DataFrame):
        raise AssertionError("performance prices are not a DataFrame")

    global_scores, raw_source, timestamps, signal_diagnostic = grid._signal_for_spec(
        SPEC, context
    )
    global_groups = groups_by_method[GLOBAL_METHOD]
    global_weights, global_diagnostic = construction._build_constrained_weights(
        "long_short",
        global_scores,
        eligibility,
        sleeve_panel,
        global_groups,
        Q,
    )

    cluster_groups = groups_by_method[CLUSTER_METHOD]
    cluster_source = score_within_clusters(
        raw_signal=raw_source,
        rolling_clusters=u1_prod._panel_dict(cluster_groups),
        min_cluster_size=CLUSTER_FALLBACK,
    )
    cluster_scores, cluster_timestamps = u1_prod._asof_panel(
        cluster_source, context["dates"]
    )
    if not cluster_timestamps.equals(timestamps):
        raise AssertionError("cluster and global signal timestamps differ")
    cluster_scores = cluster_scores.reindex(columns=eligibility.columns).where(
        eligibility
    )
    cluster_weights, cluster_diagnostic = construction._build_constrained_weights(
        "long_short",
        cluster_scores,
        eligibility,
        sleeve_panel,
        cluster_groups,
        Q,
    )

    cost_rate = COST_BPS / 10000.0
    global_net, global_gross = equal._backtest(
        prices,
        global_weights,
        cost_rate,
        "futures_best_relative_global_rank_q_0.25",
    )
    cluster_net, cluster_gross = equal._backtest(
        prices,
        cluster_weights,
        cost_rate,
        "futures_best_relative_M1_star_q_0.25",
    )
    portfolios = {
        "cluster": cluster_net,
        "cluster_gross": cluster_gross,
        "global": global_net,
        "global_gross": global_gross,
    }
    weights = {"cluster": cluster_weights, "global": global_weights}
    diagnostics = {
        "signal": signal_diagnostic,
        "cluster_weights": cluster_diagnostic,
        "global_weights": global_diagnostic,
    }
    return portfolios, weights, {"context": context, **diagnostics}


def _net_attribution(portfolio) -> tuple[pd.DataFrame, dict]:
    """Return exact bounded-window net instrument P&L and reconciliation data."""
    nav = matched._bounded_panel(portfolio.get_portfolio_nav()).dropna()
    return attribution._net_currency_pnl(portfolio, nav.index.min(), nav.index.max())


def _performance_table(portfolios: Mapping[str, object], ew_nav: pd.Series) -> pd.DataFrame:
    """Return the accepted non-EW payoff metrics for the two updated books."""
    rows = []
    for key, method in (("cluster", CLUSTER_METHOD), ("global", GLOBAL_METHOD)):
        net = matched._WindowedPortfolio(portfolios[key])
        gross = matched._WindowedPortfolio(portfolios[f"{key}_gross"])
        payload = equal._performance_payload(net, gross, ew_nav)
        rows.append(
            {
                "method": method,
                "q": Q,
                "signal_id": SPEC.signal_id,
                "cost_bps_one_way": COST_BPS,
                **{metric: payload[metric] for metric in PERFORMANCE_METRICS},
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _plot(table: pd.DataFrame, reconciliation: pd.DataFrame):
    """Create the interactive Plotly instrument-contribution scatter."""
    category_order = [
        sleeve
        for sleeve in equal.SLEEVES
        if sleeve in set(table["broad_asset_class"])
    ]
    fig = px.scatter(
        table,
        x="cluster_net_pnl_pct_of_start",
        y="global_net_pnl_pct_of_start",
        color="broad_asset_class",
        category_orders={"broad_asset_class": category_order},
        text="plot_label",
        hover_name="hover_name",
        hover_data={
            "ticker": False,
            "name": False,
            "broad_asset_class": True,
            "source_asset_class": True,
            "cluster_net_pnl_pct_of_start": ":.3f",
            "global_net_pnl_pct_of_start": ":.3f",
            "cluster_minus_global_pnl_pct_of_start": ":+.3f",
            "eligible_decision_count": True,
            "cluster_long_decision_count": True,
            "cluster_short_decision_count": True,
            "global_long_decision_count": True,
            "global_short_decision_count": True,
            "plot_label": False,
        },
        labels={
            "broad_asset_class": "Asset class",
            "source_asset_class": "Source class",
            "cluster_net_pnl_pct_of_start": "M1-star cluster net P&L (pp of start NAV)",
            "global_net_pnl_pct_of_start": "Global-rank net P&L (pp of start NAV)",
            "cluster_minus_global_pnl_pct_of_start": "Cluster minus global (pp)",
            "eligible_decision_count": "Eligible decisions",
            "cluster_long_decision_count": "Cluster long decisions",
            "cluster_short_decision_count": "Cluster short decisions",
            "global_long_decision_count": "Global long decisions",
            "global_short_decision_count": "Global short decisions",
        },
        title="Best-relative futures cell: instrument net P&L",
    )
    values = table[
        ["cluster_net_pnl_pct_of_start", "global_net_pnl_pct_of_start"]
    ].to_numpy()
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    padding = 0.08 * span if span > 0.0 else 1.0
    domain = [low - padding, high + padding]
    fig.add_shape(
        type="line",
        x0=domain[0],
        y0=domain[0],
        x1=domain[1],
        y1=domain[1],
        line={"color": "#667085", "dash": "dash", "width": 1.5},
        name="Equal P&L",
    )
    fig.add_annotation(
        x=domain[1],
        y=domain[1],
        text="equal P&L",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"color": "#667085"},
    )
    row = reconciliation.iloc[0]
    summary = (
        f"Contribution correlation {row['contribution_correlation']:.3f} · "
        f"cluster higher for {int(row['cluster_higher_instruments'])}/"
        f"{int(row['eligible_instruments'])} instruments<br>"
        f"Σ cluster {row['cluster_attributed_total_return_pct']:.2f}% · "
        f"Σ global {row['global_attributed_total_return_pct']:.2f}%"
    )
    fig.update_traces(
        marker={"size": 10, "opacity": 0.82, "line": {"width": 0.6, "color": "#ffffff"}},
        textposition="top center",
        textfont={"size": 10},
    )
    fig.update_layout(
        template="plotly_white",
        height=760,
        title={
            "text": f"Best-relative futures cell: instrument net P&L<br><sup>{summary}</sup>",
            "x": 0.05,
            "xanchor": "left",
        },
        margin={"l": 80, "r": 35, "t": 145, "b": 150},
        legend={"title": {"text": "Asset class"}, "orientation": "h", "y": -0.20},
        hovermode="closest",
    )
    fig.update_xaxes(
        range=domain,
        title_text="M1-star cluster net P&L<br>(pp of start NAV)",
        zeroline=True,
        zerolinewidth=1,
        constrain="domain",
    )
    fig.update_yaxes(
        range=domain,
        title_text="Global-rank net P&L<br>(pp of start NAV)",
        zeroline=True,
        zerolinewidth=1,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    return fig


def _inline_fragment(fig) -> str:
    """Return a theme-aware Plotly fragment for the in-conversation chart."""
    figure = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder).replace("</", "<\\/")
    plotly_version = get_plotlyjs_version()
    return f"""<div id="futures-best-relative-pnl-root">
  <div id="futures-best-relative-pnl-chart"
       role="img"
       aria-label="Scatter plot comparing M1-star cluster and global-rank net P&amp;L
                   contributions for every eligible futures instrument."></div>
</div>
<style>
#futures-best-relative-pnl-root {{ width: 100%; color: var(--foreground); }}
#futures-best-relative-pnl-chart {{ width: 100%; height: 760px; }}
@media (max-width: 519px) {{
  #futures-best-relative-pnl-chart {{ height: 620px; }}
}}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{plotly_version}/plotly.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('futures-best-relative-pnl-root');
  const chart = document.getElementById('futures-best-relative-pnl-chart');
  const figure = {figure};
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const draw = () => {{
    const series = ['--viz-series-1', '--viz-series-2', '--viz-series-3', '--viz-series-4'];
    const data = JSON.parse(JSON.stringify(figure.data));
    data.forEach((trace, index) => {{
      trace.marker = trace.marker || {{}};
      trace.marker.color = token(series[index % series.length]);
      trace.marker.line = {{ color: token('--background'), width: 0.6 }};
      trace.textfont = {{ color: token('--foreground'), size: 11 }};
    }});
    const layout = JSON.parse(JSON.stringify(figure.layout));
    if (root.getBoundingClientRect().width < 520) {{
      data.forEach((trace) => {{ trace.text = trace.text.map(() => ''); }});
      layout.height = 620;
      layout.margin = {{ l: 66, r: 14, t: 130, b: 115 }};
      layout.legend.title = {{ text: '' }};
      layout.legend.y = -0.25;
      layout.title.text = layout.title.text
        .replace(
          'Best-relative futures cell: instrument net P&L',
          'Cluster vs global: instrument net P&L'
        )
        .replace('Contribution correlation', 'Corr')
        .replace('cluster higher for', 'cluster higher')
        .replace(' instruments', '');
      layout.title.x = 0;
      layout.title.font = {{ size: 13 }};
      layout.xaxis.nticks = 4;
      layout.yaxis.nticks = 4;
    }}
    layout.paper_bgcolor = 'rgba(0,0,0,0)';
    layout.plot_bgcolor = 'rgba(0,0,0,0)';
    layout.font = {{ color: token('--foreground') }};
    layout.xaxis.gridcolor = token('--border');
    layout.xaxis.linecolor = token('--border');
    layout.xaxis.zerolinecolor = token('--muted-foreground');
    layout.yaxis.gridcolor = token('--border');
    layout.yaxis.linecolor = token('--border');
    layout.yaxis.zerolinecolor = token('--muted-foreground');
    (layout.shapes || []).forEach((shape) => {{
      shape.line.color = token('--muted-foreground');
    }});
    (layout.annotations || []).forEach((annotation) => {{
      annotation.font = {{
        ...(annotation.font || {{}}),
        color: token('--foreground')
      }};
    }});
    Plotly.react(
      chart,
      data,
      layout,
      {{ responsive: true, displaylogo: false, scrollZoom: false }}
    );
  }};
  const start = () => {{
    let narrow = root.getBoundingClientRect().width < 520;
    draw();
    new ResizeObserver(() => {{
      const nextNarrow = root.getBoundingClientRect().width < 520;
      if (nextNarrow !== narrow) {{
        narrow = nextNarrow;
        draw();
      }} else {{
        Plotly.Plots.resize(chart);
      }}
    }}).observe(root);
    new MutationObserver(draw).observe(
      document.documentElement,
      {{ attributes: true, attributeFilter: ['class', 'style'] }}
    );
  }};
  if (window.Plotly) start();
  else window.addEventListener('load', start, {{ once: true }});
}})();
</script>
"""


def run() -> Mapping[str, pd.DataFrame]:
    """Create exact attribution data, validations, and the Plotly scatter."""
    started = time.perf_counter()
    portfolios, weights, diagnostics = _build_weights_and_portfolios()
    context = diagnostics["context"]
    eligibility = context["eligibility"]
    data = context["data"]
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("eligibility is not a DataFrame")

    cluster_pnl, cluster_diag = _net_attribution(portfolios["cluster"])
    global_pnl, global_diag = _net_attribution(portfolios["global"])
    tickers = eligibility.columns[eligibility.any(axis=0)]
    sleeves = equal._broad_sleeves(data.taxonomy, eligibility.columns)
    table = _instrument_table(
        tickers=tickers,
        cluster_net_pnl=cluster_pnl.sum(axis=0),
        global_net_pnl=global_pnl.sum(axis=0),
        cluster_beginning_nav=float(cluster_diag["beginning_nav"]),
        global_beginning_nav=float(global_diag["beginning_nav"]),
        taxonomy=data.taxonomy,
        sleeves=sleeves,
        eligibility=eligibility,
        cluster_weights=weights["cluster"],
        global_weights=weights["global"],
    )

    ew_nav = context["ew_nav"]
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("EW reference is not a Series")
    performance = _performance_table(portfolios, ew_nav)
    indexed_performance = performance.set_index("method")
    regression_rows = []
    for method, expected in FROZEN_PERFORMANCE.items():
        errors = {
            metric: abs(float(indexed_performance.loc[method, metric]) - value)
            for metric, value in expected.items()
        }
        maximum = max(errors.values())
        regression_rows.append(
            {
                "method": method,
                "compared_metrics": len(errors),
                "max_abs_error": maximum,
                "tolerance": REGRESSION_TOLERANCE,
                "status": "PASS" if maximum <= REGRESSION_TOLERANCE else "FAIL",
            }
        )
    performance_regression = pd.DataFrame(regression_rows).sort_values("method")
    cluster_regression_error = float(
        performance_regression.set_index("method").loc[
            CLUSTER_METHOD, "max_abs_error"
        ]
    )
    global_regression_error = float(
        performance_regression.set_index("method").loc[
            GLOBAL_METHOD, "max_abs_error"
        ]
    )
    comparison_row = {
        "cluster_method": CLUSTER_METHOD,
        "global_method": GLOBAL_METHOD,
        "q": Q,
        "signal_id": SPEC.signal_id,
    }
    for metric in PERFORMANCE_METRICS:
        cluster_value = float(indexed_performance.loc[CLUSTER_METHOD, metric])
        global_value = float(indexed_performance.loc[GLOBAL_METHOD, metric])
        comparison_row[f"cluster_{metric}"] = cluster_value
        comparison_row[f"global_{metric}"] = global_value
        comparison_row[f"delta_{metric}"] = cluster_value - global_value
    comparison_row["cluster_beats_global_net_return"] = (
        comparison_row["delta_net_return_annualized"] > 0.0
    )
    comparison_row["cluster_beats_global_sharpe"] = (
        comparison_row["delta_sharpe_rf0"] > 0.0
    )
    performance_comparison = pd.DataFrame([comparison_row])
    cluster_sum_error = abs(
        float(table["cluster_net_pnl_pct_of_start"].sum())
        - 100.0 * float(cluster_diag["attributed_net_total_return"])
    )
    global_sum_error = abs(
        float(table["global_net_pnl_pct_of_start"].sum())
        - 100.0 * float(global_diag["attributed_net_total_return"])
    )
    excluded_rows = int(
        table["ticker"].isin(e5.FUTURES_INVESTABILITY_EXCLUSIONS).sum()
    )
    present_exclusions = eligibility.columns.intersection(
        e5.FUTURES_INVESTABILITY_EXCLUSIONS
    )
    excluded_eligible_observations = int(
        eligibility.reindex(columns=present_exclusions).sum().sum()
    )
    exclusion_set_match = (
        e5.FUTURES_INVESTABILITY_EXCLUSIONS == EXPECTED_FUTURES_EXCLUSIONS
    )
    alias_match = e5.FUTURES_INVESTABILITY_EXCLUSION_ALIASES == {
        "MMR1 Curncy": "BMR1 Curncy"
    }
    freeze_match = (
        e5.FUTURES_ELIGIBLE_UNIVERSE_STATUS == "OWNER_FROZEN_2026-08-15"
        and BEST_METHOD_STATUS == "OWNER_FROZEN_2026-08-15"
        and set(e5.FUTURES_INVESTABILITY_EXCLUSION_REASONS)
        == set(EXPECTED_FUTURES_EXCLUSIONS)
        and set(e5.FUTURES_INVESTABILITY_EXCLUSION_REASONS.values())
        == {"low_liquidity_owner_ruling"}
    )
    excluded_weight = max(
        float(
            panel.reindex(columns=e5.FUTURES_INVESTABILITY_EXCLUSIONS)
            .fillna(0.0)
            .abs()
            .to_numpy()
            .max()
        )
        for panel in weights.values()
    )
    status = (
        float(cluster_diag["max_step_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and float(global_diag["max_step_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and float(cluster_diag["cumulative_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and float(global_diag["cumulative_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and cluster_regression_error <= REGRESSION_TOLERANCE
        and global_regression_error <= REGRESSION_TOLERANCE
        and cluster_sum_error <= ACCOUNTING_TOLERANCE
        and global_sum_error <= ACCOUNTING_TOLERANCE
        and exclusion_set_match
        and alias_match
        and freeze_match
        and len(present_exclusions) == len(EXPECTED_FUTURES_EXCLUSIONS)
        and excluded_eligible_observations == 0
        and excluded_rows == 0
        and excluded_weight <= ACCOUNTING_TOLERANCE
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "strategy": "long_short",
                "cluster_method": CLUSTER_METHOD,
                "global_method": GLOBAL_METHOD,
                "q": Q,
                "signal_id": SPEC.signal_id,
                "cluster_fallback": CLUSTER_FALLBACK,
                "cost_bps_one_way": COST_BPS,
                "eligible_instruments": len(table),
                "cluster_higher_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].gt(0.0).sum()
                ),
                "global_higher_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].lt(0.0).sum()
                ),
                "equal_contribution_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].eq(0.0).sum()
                ),
                "actual_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "eligible_universe_status": e5.FUTURES_ELIGIBLE_UNIVERSE_STATUS,
                "exclusion_reason": "low_liquidity_owner_ruling",
                "best_method_status": BEST_METHOD_STATUS,
                "requested_alias_resolution": "MMR1 Curncy->BMR1 Curncy",
                "source_exclusions_present": len(present_exclusions),
                "excluded_eligible_observations": excluded_eligible_observations,
                "contribution_correlation": float(
                    table[
                        [
                            "cluster_net_pnl_pct_of_start",
                            "global_net_pnl_pct_of_start",
                        ]
                    ].corr().iloc[0, 1]
                ),
                "cluster_attributed_total_return_pct": 100.0
                * float(cluster_diag["attributed_net_total_return"]),
                "global_attributed_total_return_pct": 100.0
                * float(global_diag["attributed_net_total_return"]),
                "cluster_max_step_reconciliation_abs_error": cluster_diag[
                    "max_step_reconciliation_abs_error"
                ],
                "global_max_step_reconciliation_abs_error": global_diag[
                    "max_step_reconciliation_abs_error"
                ],
                "cluster_cumulative_reconciliation_abs_error": cluster_diag[
                    "cumulative_reconciliation_abs_error"
                ],
                "global_cumulative_reconciliation_abs_error": global_diag[
                    "cumulative_reconciliation_abs_error"
                ],
                "cluster_frozen_performance_max_abs_error": cluster_regression_error,
                "global_frozen_performance_max_abs_error": global_regression_error,
                "cluster_table_sum_abs_error": cluster_sum_error,
                "global_table_sum_abs_error": global_sum_error,
                "owner_excluded_rows": excluded_rows,
                "max_owner_excluded_weight_abs": excluded_weight,
                "accounting_tolerance": ACCOUNTING_TOLERANCE,
                "regression_tolerance": REGRESSION_TOLERANCE,
                "status": "PASS" if status else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not status:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])

    design = pd.DataFrame(
        [
            {
                "x_axis": (
                    "M1-star cluster net P&L contribution, percentage points "
                    "of beginning NAV"
                ),
                "y_axis": (
                    "same-signal global-rank net P&L contribution, percentage "
                    "points of beginning NAV"
                ),
                "identity_line": "y=x; below line means cluster contribution is higher",
                "instruments": "eligible at least once in the U1 headline window",
                "signal_frequency": grid.SIGNAL_FREQUENCY,
                "momentum_long_span": grid.MOMENTUM_LONG_SPAN,
                "momentum_short_span": np.nan,
                "momentum_vol_span": SPEC.vol_span,
                "momentum_mean_adj_type": SPEC.mean_adj_type,
                "q": Q,
                "cluster_method": CLUSTER_METHOD,
                "cluster_fallback": CLUSTER_FALLBACK,
                "cost_bps_one_way": COST_BPS,
                "sleeve_budgets": "Equity 30%|Fixed Income 30%|Commodities 30%|FX 10% per side",
                "pnl_accounting": "prior units times price change less realised instrument cost",
                "owner_exclusions": "|".join(sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)),
                "eligible_universe_status": e5.FUTURES_ELIGIBLE_UNIVERSE_STATUS,
                "exclusion_reason": "low_liquidity_owner_ruling",
                "best_method_status": BEST_METHOD_STATUS,
                "frozen_best_method_spec": json.dumps(
                    FROZEN_BEST_METHOD_SPEC, sort_keys=True
                ),
                "requested_alias_resolution": "MMR1 Curncy->BMR1 Curncy",
                "cluster_partitions": "cached; no refit",
                "runner": RUNNER,
            }
        ]
    )
    fig = _plot(table, reconciliation)
    outputs = {
        "instrument_pnl": table,
        "performance": performance,
        "performance_comparison": performance_comparison,
        "performance_regression": performance_regression,
        "reconciliation": reconciliation,
        "design": design,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    fig.write_html(
        _root() / PLOT_FILE,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        div_id="futures-best-relative-pnl-scatter",
    )
    visual_path = os.environ.get("CLUSTER_LINEAGE_VISUAL_PATH")
    if visual_path:
        Path(visual_path).write_text(_inline_fragment(fig), encoding="utf-8")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic numerical and Plotly outputs, excluding timing/replay."""
    paths = sorted(_root().glob("*.csv")) + [_root() / PLOT_FILE]
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay attribution and require byte-identical data and Plotly output."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run, replay, and print the attribution reconciliation."""
    replay = verify_determinism()
    reconciliation = pd.read_csv(
        _root() / "reconciliation.csv", float_precision="round_trip"
    )
    print(reconciliation.to_string(index=False))
    print(
        f"Best-relative futures P&L scatter: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
