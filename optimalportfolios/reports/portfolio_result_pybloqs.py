import pandas as pd
import pybloqs as p
import pybloqs.block.table_formatters as tf
from optimalportfolios.optimization import PortfolioOptimisationResult


def generate_portfolio_report(
    result: PortfolioOptimisationResult,
    name: str = None,
    report_name: str = 'Portfolio Optimisation Report',
    fontsize: int = 5,
) -> p.VStack:
    """
    Generate one-page pybloqs report with weights, risk summary, and factor exposures.

    Args:
        result: PortfolioOptimisationResult instance.
        name: Portfolio name for weight summary / turnover. Defaults to first portfolio.
        report_name: Report title override.
        fontsize: Base font size for table content.
    """
    name = name or result.portfolio_names[0]

    date_str = f"{result.optimisation_date:%d%b%Y}" if result.optimisation_date else ''
    report_name = result.portfolio_id or report_name
    report_header = f"{report_name} — {date_str}"

    KWARGS_SUPTITLE = {'styles': {'font-size': '14px', 'font-weight': 'bold'}}
    KWARGS_TITLE = {'styles': {'font-size': '10px', 'font-weight': 'bold'}}
    KWARGS_FOOTNOTE = {'styles': {'font-size': '6px'}}
    KWARGS_TEXT = {'styles': {'font-size': f'{fontsize}px'}}

    blocks = [p.Paragraph(report_header, **KWARGS_SUPTITLE)]

    # 1. Weights tables
    is_multi_portfolio = name is None and result.n_portfolios > 1

    if is_multi_portfolio:
        # Multi-portfolio: tickers as rows, portfolio names as columns
        all_weights = result.compute_all_weights_summary()
        for table_name, df in all_weights.items():
            portfolio_cols = df.columns.to_list()
            b_wt = p.Block(
                [p.Paragraph(f"{table_name.capitalize()} weights", **KWARGS_TITLE),
                 p.Block(df,
                         formatters=[
                             tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                             tf.FmtReplaceNaN(value=''),
                             tf.FmtHeatmap(columns=portfolio_cols),
                             tf.FmtAddCellBorder(each=1.0,
                                                 columns=portfolio_cols,
                                                 color=tf.colors.GREY,
                                                 apply_to_header_and_index=True),
                         ]),
                 p.Paragraph("  ", **KWARGS_FOOTNOTE)],
                **KWARGS_TEXT
            )
            blocks.append(b_wt)
    else:
        # Single portfolio: standard asset-level weights table
        report_portfolio = name or result.portfolio_names[0]
        weights_df = result.compute_weight_summary(name=report_portfolio)
        numeric_cols = ['CMA'] + weights_df.columns.to_list()
        weights = result.get_combined_asset_weight_table()

        weight_heatmap_cols = ['new', 'benchmark']
        if 'current' in weights.columns:
            weight_heatmap_cols.append('current')
        diff_heatmap_cols = ['active']
        if 'trade' in weights.columns:
            diff_heatmap_cols.append('trade')

        b_weights = p.Block(
            [p.Paragraph(f"Optimised portfolio weights", **KWARGS_TITLE),
             p.Block(weights,
                     formatters=[
                         tf.FmtPercent(n_decimals=2, columns=numeric_cols, apply_to_header_and_index=False),
                         tf.FmtReplaceNaN(value=''),
                         tf.FmtHeatmap(columns=['CMA']),
                         tf.FmtHeatmap(columns=weight_heatmap_cols),
                         tf.FmtHeatmap(columns=diff_heatmap_cols),
                         tf.FmtAddCellBorder(each=1.0,
                                             columns=weights.columns.to_list(),
                                             color=tf.colors.GREY,
                                             apply_to_header_and_index=True),
                     ]),
             p.Paragraph("  ", **KWARGS_FOOTNOTE)],
            **KWARGS_TEXT
        )
        blocks.append(b_weights)

    # 2. Turnover summary (only if current_weights available and single portfolio view)
    if result.has_current_weights and not is_multi_portfolio:
        report_portfolio = name or result.portfolio_names[0]
        turnover = result.compute_turnover_analysis(name=report_portfolio)
        turnover_df = turnover.to_frame().T
        pct_cols = ['turnover', 'buys', 'sells', 'avg_trade_size', 'max_trade_size']
        b_turnover = p.Block(
            [p.Paragraph(f"Turnover analysis", **KWARGS_TITLE),
             p.Block(turnover_df,
                     formatters=[
                         tf.FmtPercent(n_decimals=2, columns=pct_cols, apply_to_header_and_index=False),
                         tf.FmtDecimals(n=0, columns=['n_trades'], apply_to_header_and_index=False),
                         tf.FmtReplaceNaN(value=''),
                         tf.FmtAddCellBorder(each=1.0,
                                             columns=turnover_df.columns.to_list(),
                                             color=tf.colors.GREY,
                                             apply_to_header_and_index=True),
                     ]),
             p.Paragraph("  ", **KWARGS_FOOTNOTE)],
            **KWARGS_TEXT
        )
        blocks.append(b_turnover)

    # 3. Group attribution tables — separate table per metric with heatmaps
    group_attribs = result.compute_group_attribution()
    group_blocks = []
    for group_name, metric_dfs in group_attribs.items():
        metric_blocks = []

        # Bounds table if present
        if 'bounds' in metric_dfs:
            bounds_df = metric_dfs['bounds']
            bounds_cols = bounds_df.columns.to_list()
            metric_blocks.append(p.Block(
                [p.Paragraph(f"{group_name} — bounds", **KWARGS_TITLE),
                 p.Block(bounds_df,
                         formatters=[
                             tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                             tf.FmtReplaceNaN(value=''),
                             tf.FmtAddCellBorder(each=1.0, columns=bounds_cols,
                                                 color=tf.colors.GREY,
                                                 apply_to_header_and_index=True),
                         ]),
                 p.Paragraph("  ", **KWARGS_FOOTNOTE)],
                **KWARGS_TEXT
            ))

        metric_titles = {'weight': 'Weight', 'rc': 'Risk contribution', 'rc_pct': 'Risk contribution %'}
        for metric in ['weight', 'rc', 'rc_pct']:
            if metric not in metric_dfs:
                continue
            df = metric_dfs[metric]
            cols = df.columns.to_list()
            metric_blocks.append(p.Block(
                [p.Paragraph(f"{group_name} — {metric_titles[metric]}", **KWARGS_TITLE),
                 p.Block(df,
                         formatters=[
                             tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                             tf.FmtReplaceNaN(value=''),
                             tf.FmtHeatmap(columns=cols),
                             tf.FmtAddCellBorder(each=1.0, columns=cols,
                                                 color=tf.colors.GREY,
                                                 apply_to_header_and_index=True),
                         ]),
                 p.Paragraph("  ", **KWARGS_FOOTNOTE)],
                **KWARGS_TEXT
            ))

        group_blocks.extend(metric_blocks)

    blocks.append(p.VStack(group_blocks))

    # 4. Risk decomposition — separate tables for portfolio and benchmark
    risk_summary_dict = result.compute_risk_summary()

    risk_table_configs = [
        ('Risk decomposition — Portfolio', 'portfolio'),
        ('Risk decomposition — Benchmark', 'benchmark'),
    ]
    for title_str, key in risk_table_configs:
        df = risk_summary_dict[key]
        cols = df.columns.to_list()
        decimal_cols = [c for c in ['sharpe_ratio'] if c in cols]
        pct_cols = [c for c in cols if c not in decimal_cols]

        formatters = [
            tf.FmtPercent(n_decimals=2, columns=pct_cols, apply_to_header_and_index=False),
            tf.FmtDecimals(n=2, columns=decimal_cols, apply_to_header_and_index=False),
            tf.FmtReplaceNaN(value=''),
        ]
        for col in cols:
            formatters.append(tf.FmtHeatmap(columns=[col]))
        formatters.append(
            tf.FmtAddCellBorder(each=1.0, columns=cols,
                                color=tf.colors.GREY, apply_to_header_and_index=True)
        )

        blocks.append(p.Block(
            [p.Paragraph(title_str, **KWARGS_TITLE),
             p.Block(df, formatters=formatters),
             p.Paragraph("  ", **KWARGS_FOOTNOTE)],
            **KWARGS_TEXT
        ))

    # 5. Factor exposures and risk contribution — 6 tables
    factor_exp_dict = result.compute_factor_exposures_summary()

    exposure_tables = [
        ('Factor exposures — Portfolio', 'exposure_portfolio'),
        ('Factor exposures — Benchmark', 'exposure_benchmark'),
        ('Factor exposures — Active', 'exposure_active'),
    ]
    for title_str, key in exposure_tables:
        df = factor_exp_dict[key]
        cols = df.columns.to_list()
        blocks.append(p.Block(
            [p.Paragraph(title_str, **KWARGS_TITLE),
             p.Block(df,
                     formatters=[
                         tf.FmtDecimals(n=2, apply_to_header_and_index=False),
                         tf.FmtReplaceNaN(value=''),
                         tf.FmtHeatmap(columns=cols),
                         tf.FmtAddCellBorder(each=1.0, columns=cols,
                                             color=tf.colors.GREY,
                                             apply_to_header_and_index=True),
                     ]),
             p.Paragraph("  ", **KWARGS_FOOTNOTE)],
            **KWARGS_TEXT
        ))

    risk_pct_tables = [
        ('Factor risk contribution — Portfolio', 'risk_pct_portfolio'),
        ('Factor risk contribution — Benchmark', 'risk_pct_benchmark'),
        ('Factor risk contribution — Active', 'risk_pct_active'),
    ]
    for title_str, key in risk_pct_tables:
        df = factor_exp_dict[key]
        cols = df.columns.to_list()
        blocks.append(p.Block(
            [p.Paragraph(title_str, **KWARGS_TITLE),
             p.Block(df,
                     formatters=[
                         tf.FmtPercent(n_decimals=1, apply_to_header_and_index=False),
                         tf.FmtReplaceNaN(value=''),
                         tf.FmtHeatmap(columns=cols),
                         tf.FmtAddCellBorder(each=1.0, columns=cols,
                                             color=tf.colors.GREY,
                                             apply_to_header_and_index=True),
                     ]),
             p.Paragraph("  ", **KWARGS_FOOTNOTE)],
            **KWARGS_TEXT
        ))

    # 6. Asset risk snapshot table
    asset_risk_table = result.get_asset_betas_table()
    factor_cols = result.covar_data.x_covar.columns
    decimal_cols = factor_cols
    pct_cols = ['CMA', 'r2', 'stat_alpha', 'insample_alpha', 'total_vol', 'sys_vol', 'resid_vol']

    b_asset_risk = p.Block(
        [p.Paragraph(f"Asset factor exposures and risk", **KWARGS_TITLE),
         p.Block(asset_risk_table,
                 formatters=[
                     tf.FmtDecimals(n=2, columns=decimal_cols, apply_to_header_and_index=False),
                     tf.FmtPercent(n_decimals=1, columns=pct_cols, apply_to_header_and_index=False),
                     tf.FmtReplaceNaN(value=''),
                     tf.FmtHeatmap(columns=factor_cols),
                     tf.FmtHeatmap(columns=['CMA']),
                     tf.FmtHeatmap(columns=['r2']),
                     tf.FmtHeatmap(columns=['stat_alpha']),
                     tf.FmtHeatmap(columns=['total_vol']),
                     tf.FmtTruncateContentsWithEllipsis(columns=asset_risk_table.columns.to_list(),
                                                        rows=asset_risk_table.index[1:].to_list(),
                                                        apply_to_header_and_index=True),
                     tf.FmtAddCellBorder(each=1.0,
                                         columns=asset_risk_table.columns.to_list(),
                                         color=tf.colors.GREY,
                                         apply_to_header_and_index=True),
                 ]),
         p.Paragraph("  ", **KWARGS_FOOTNOTE)],
        **KWARGS_TEXT
    )
    blocks.append(b_asset_risk)

    report = p.VStack(blocks, cascade_cfg=False)
    return report