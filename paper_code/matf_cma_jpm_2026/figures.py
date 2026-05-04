"""
Paper figures — produce 7 LocalTests, each saving via ``qis.save_fig``.

This module reads only ``paper_inputs.xlsx`` via the ``PaperInputs`` container.
No dependency on ``rosaa``.

LocalTests
----------
- FACTOR_PERFORMANCES → risk_factors_perf, risk_factors_annual, risk_factors_corr
- ALL_FACTOR_CMAS     → all_factor_cmas
- EQUITY_FACTOR_CMAS  → equity_factor_cmas
- RATES_FACTOR_CMAS   → rates_factor_cmas
- FACTOR_ATTRIBUTION  → factor_attribution
- CMA_SCENARIOS       → cma_scenarios
- UNIVERSE_SNAPSHOT   → universe_cmas (and writes universe_snapshot.xlsx)

Run a single test:
    python figures.py --test UNIVERSE_SNAPSHOT --output-path figures/

Run all:
    python figures.py --all --output-path figures/
"""
from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import qis as qis

from paper_inputs import PaperInputs
from universe import load_paper_assets_short


# ─────────────────────────────────────────────────────────────────────
# Generic bar-plot helper (lifted unchanged from the original figures.py
# except for losing the rosaa dependency)
# ─────────────────────────────────────────────────────────────────────
def plot_bar_cma(df: Union[pd.DataFrame, pd.Series],
                 stacked: bool = False,
                 title: Optional[str] = None,
                 legend_loc: Optional[str] = None,
                 add_bar_values: bool = False,
                 add_totals: bool = False,
                 yvar_format: str = '{:.1%}',
                 reverse_columns: bool = True,
                 colors: List[str] = None,
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> Optional[plt.Figure]:
    """Wrapper around ``qis.plot_bars`` with the paper's default styling."""
    if colors is None:
        if isinstance(df, pd.Series):
            colors = qis.get_n_sns_colors(n=1)
        else:
            colors = qis.get_n_sns_colors(n=len(df.columns))
    totals = df.sum(axis=1) if add_totals else None

    with sns.axes_style('darkgrid'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        else:
            fig = None
        qis.plot_bars(df,
                      stacked=stacked,
                      yvar_format=yvar_format,
                      legend_loc=legend_loc,
                      add_bar_values=add_bar_values,
                      is_top_totals=False,
                      reverse_columns=reverse_columns,
                      colors=colors,
                      totals=totals,
                      title=title,
                      ax=ax,
                      **kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Per-figure builders
# ─────────────────────────────────────────────────────────────────────
def plot_all_factor_cmas(inputs: PaperInputs) -> Tuple[plt.Figure, pd.DataFrame]:
    """Bar chart of all 9 factor excess CMAs."""
    df = inputs.get_factor_cmas()
    fig = plot_bar_cma(df=df,
                       legend_loc='upper center',
                       add_bar_values=True,
                       ncols=3)
    return fig, df.to_frame()


def plot_equity_factor_cmas(inputs: PaperInputs) -> Tuple[plt.Figure, pd.DataFrame]:
    """Regional equity excess CMAs (Global, US, Europe, ...)."""
    df = inputs.get_equity_excess_cmas()
    fig = plot_bar_cma(df=df, add_bar_values=True)
    return fig, df.to_frame()


def plot_rates_factor_cmas(inputs: PaperInputs) -> Tuple[plt.Figure, pd.DataFrame]:
    """Stacked regional rates total CMAs (real, inflation premium, term, default)."""
    df = inputs.get_rates_total_cmas()
    fig = plot_bar_cma(df=df,
                       stacked=True,
                       legend_loc='upper left',
                       add_bar_values=True,
                       add_totals=True)
    return fig, df


def plot_factor_attribution(inputs: PaperInputs,
                            assets: Optional[pd.Series] = None
                            ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Per-asset CMA attribution (stacked bar, vertical separators between
    Bonds / Equities / Alternatives groupings)."""
    if assets is None:
        assets = load_paper_assets_short()
    df_attrib = inputs.compose_displayed_attribution(assets=assets)

    # Vertical-line column positions correspond to group boundaries
    # (Bonds=7, +Equities=12, +Alts=17 in the canonical 17-asset order)
    vline_columns = [7, 12, 17]
    fig = plot_bar_cma(df=df_attrib,
                       stacked=True,
                       legend_loc='upper left',
                       add_totals=True,
                       reverse_columns=False,
                       y_limits=(-0.02, None),
                       vline_columns=vline_columns)
    return fig, df_attrib


def plot_scenario_cmas(inputs: PaperInputs,
                       assets: Optional[pd.Series] = None
                       ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Per-asset stress / base / upside total CMAs (3 bars per asset)."""
    if assets is None:
        assets = load_paper_assets_short()

    df = inputs.cma_metadata.loc[list(assets.index), :].copy()
    df = df.rename(assets.to_dict(), axis=0)
    cols = ['stress_total_cma', 'base_total_cma', 'upside_total_cma']
    rename_map = {'base_total_cma': 'BaseCMA',
                  'stress_total_cma': 'StressCMA',
                  'upside_total_cma': 'UpsideCMA'}
    df = df[cols].rename(rename_map, axis=1)
    colors = ['darkred', 'slateblue', 'darkgreen']

    fig = plot_bar_cma(df=df,
                       stacked=False,
                       legend_loc='upper center',
                       add_totals=False,
                       reverse_columns=False,
                       ncols=3,
                       y_limits=(-0.02, None),
                       colors=colors)
    return fig, df


def plot_universe_snapshot(inputs: PaperInputs,
                           assets: Optional[pd.Series] = None,
                           plot_table: bool = True
                           ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Scatter plot of base / stress / upside CMA vs realised volatility,
    with asset labels. Optional companion: the per-asset CMA + beta table.
    """
    if assets is None:
        assets = load_paper_assets_short()

    df = inputs.get_universe_snapshot(assets=assets)

    if plot_table:
        var_formats = {x: '{:,.2f}' for x in
                       ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                        'Commodities', 'Private Equity', 'Rates Vol', 'Fx']}
        var_formats.update({x: '{:,.1%}' for x in
                            ['R2', 'Alpha', 'Vol', 'SystVol', 'ResidVol']})
        dfs = qis.df_to_str(df=df, var_formats=var_formats)
        qis.plot_df_table(df=dfs, heatmap_columns=[4, 5, 6, 7, 8])

    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
        df1 = qis.melt_scatter_data_with_xvar(
            df=df[['BaseCMA', 'StressCMA', 'UpsideCMA', 'Vol']],
            xvar_str='Vol', y_column='CMAs')

        n = len(df.index)
        colors = qis.get_n_sns_colors(n=3)
        annotation_labels = df.index.to_list() + [' '] * (2 * n)
        annotation_colors = n * [colors[0]] + n * [colors[1]] + n * [colors[2]]

        qis.plot_scatter(df=df1, x='Vol', y='CMAs', hue='hue',
                         y_limits=(0.0, 0.12),
                         full_sample_order=None,
                         annotation_labels=annotation_labels,
                         annotation_colors=annotation_colors,
                         colors=colors,
                         order=2,
                         fontsize=12,
                         legend_loc='upper left',
                         add_hue_model_label=False,
                         ax=ax)
    return fig, df


def plot_risk_factors(inputs: PaperInputs,
                      time_period: Optional[qis.TimePeriod] = None,
                      ) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Three-figure factor-performance exhibit:
    (a) factor NAV cumulative growth, (b) annual returns table, (c) factor
    correlation heatmap.

    Parameters
    ----------
    time_period : qis.TimePeriod, optional
        If given, restricts the NAV / annual-returns plots to that window.
        Default behaviour falls back to the full NAV history.
    """
    prices = inputs.factors_prices.copy()
    if time_period is not None:
        prices = time_period.locate(prices)

    # (a) NAV / cumulative performance
    fig_perf, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    with sns.axes_style('darkgrid'):
        qis.plot_prices(prices=prices,
                        title='Risk Factor Cumulative Performance',
                        ax=ax)

    # (b) Annual returns table
    fig_annual, ax = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
    qis.plot_periodic_returns_table(prices=prices, freq='YE',
                                    title='Annual Returns',
                                    add_total=True,
                                    ax=ax)

    # (c) Correlation heatmap from x_covar
    fig_corr = qis.plot_corr_matrix_from_covar(covar=inputs.x_covar)

    return fig_perf, fig_annual, fig_corr


# ─────────────────────────────────────────────────────────────────────
# LocalTests entry points
# ─────────────────────────────────────────────────────────────────────
class LocalTests(Enum):
    FACTOR_PERFORMANCES = 1
    ALL_FACTOR_CMAS     = 2
    EQUITY_FACTOR_CMAS  = 3
    RATES_FACTOR_CMAS   = 4
    FACTOR_ATTRIBUTION  = 5
    CMA_SCENARIOS       = 6
    UNIVERSE_SNAPSHOT   = 7


@qis.timer
def run_local_test(local_test: LocalTests,
                   paper_inputs_xlsx: Path,
                   output_path: Path) -> None:
    """Run one LocalTest and save the produced figure(s) via ``qis.save_fig``."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    inputs = PaperInputs.load(paper_inputs_xlsx)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if local_test == LocalTests.FACTOR_PERFORMANCES:
        time_period = qis.TimePeriod('31Dec2004', '31Mar2026')
        fig_perf, fig_annual, fig_corr = plot_risk_factors(
            inputs=inputs, time_period=time_period)
        qis.save_fig(fig=fig_perf,   file_name='risk_factors_perf',   local_path=str(output_path))
        qis.save_fig(fig=fig_annual, file_name='risk_factors_annual', local_path=str(output_path))
        qis.save_fig(fig=fig_corr,   file_name='risk_factors_corr',   local_path=str(output_path))

    elif local_test == LocalTests.ALL_FACTOR_CMAS:
        fig, _ = plot_all_factor_cmas(inputs=inputs)
        qis.save_fig(fig=fig, file_name='all_factor_cmas', local_path=str(output_path))

    elif local_test == LocalTests.EQUITY_FACTOR_CMAS:
        fig, _ = plot_equity_factor_cmas(inputs=inputs)
        qis.save_fig(fig=fig, file_name='equity_factor_cmas', local_path=str(output_path))

    elif local_test == LocalTests.RATES_FACTOR_CMAS:
        fig, _ = plot_rates_factor_cmas(inputs=inputs)
        qis.save_fig(fig=fig, file_name='rates_factor_cmas', local_path=str(output_path))

    elif local_test == LocalTests.FACTOR_ATTRIBUTION:
        fig, _ = plot_factor_attribution(inputs=inputs)
        qis.save_fig(fig=fig, file_name='factor_attribution', local_path=str(output_path))

    elif local_test == LocalTests.CMA_SCENARIOS:
        fig, _ = plot_scenario_cmas(inputs=inputs)
        qis.save_fig(fig=fig, file_name='cma_scenarios', local_path=str(output_path))

    elif local_test == LocalTests.UNIVERSE_SNAPSHOT:
        fig, df = plot_universe_snapshot(inputs=inputs)
        snapshot_xlsx = output_path / 'universe_snapshot.xlsx'
        snapshot_mode = 'a' if snapshot_xlsx.exists() else 'w'
        qis.save_df_to_excel(
            data=df,
            file_name='universe_snapshot',
            local_path=str(output_path),
            sheet_names=f"universe_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
            mode=snapshot_mode)
        qis.save_fig(fig=fig, file_name='universe_cmas', local_path=str(output_path))


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-inputs-xlsx",
                   default=Path("data") / "paper_inputs.xlsx",
                   type=Path)
    p.add_argument("--output-path", default=Path("figures"), type=Path)
    p.add_argument("--test", choices=[t.name for t in LocalTests],
                   default=None)
    p.add_argument("--all", action='store_true',
                   help="Run every LocalTest in sequence.")
    args = p.parse_args()

    if args.all:
        tests = list(LocalTests)
    elif args.test is not None:
        tests = [LocalTests[args.test]]
    else:
        tests = [LocalTests.UNIVERSE_SNAPSHOT]   # default for back-compat

    for t in tests:
        print(f"\n===== {t.name} =====")
        try:
            run_local_test(t, args.paper_inputs_xlsx, args.output_path)
        except ValueError as exc:
            print(f"[skip] {t.name}: {exc}")


if __name__ == "__main__":
    _cli()
