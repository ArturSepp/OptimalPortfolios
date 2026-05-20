"""
Signal-quality diagnostics for optimalportfolios alpha signals.

Wraps ``qis.estimate_signal_diagnostics`` for the rosaa /
optimalportfolios pipeline. The diagnostic operates on the per-frequency
``asset_returns_dict`` produced by ``compute_fx_adjusted_returns`` —
returns are already FX-adjusted, unsmoothed, and at each asset's native
cadence. Horizons are interpreted in **per-asset native cadence units**:
horizon=1 means one month for monthly assets and one quarter for
quarterly assets.

The module is split into pure-compute and pure-plot functions. No file
I/O — plot functions return Figures; the caller decides what to do with
them.

    run_signal_diagnostics
        Compute. Returns a ``qis.SignalDiagnosticsResult``.

    plot_signal_diagnostics
        Compute + render. Returns the Figure.

    run_signal_diagnostics_per_component
        Component-wise compute. Loops over every populated score panel
        in ``AlphasData`` and returns a dict of results keyed by
        component name.

    plot_signal_diagnostics_per_component
        Component-wise plot. Same loop, returns a dict of Figures.

    compare_signal_diagnostics
        Side-by-side comparison: aggregates pooled regression rows from
        multiple signals into one DataFrame.
"""
from __future__ import annotations

import logging
from dataclasses import fields
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import qis

from optimalportfolios.alphas.alpha_data import AlphasData


logger = logging.getLogger(__name__)


# Score panels in AlphasData that can be passed as signals to the diagnostic.
# Ordered roughly from "combined" to "individual components" — preserves
# this order in component-wise sweeps.
_SCORE_ATTRIBUTES: Tuple[str, ...] = (
    'alpha_scores',
    'momentum_score',
    'momentum_cluster_score',
    'beta_score',
    'beta_cluster_score',
    'residual_momentum_score',
    'residual_momentum_cluster_score',
    'managers_scores',
)


def _resolve_signal(
        signal: Union[pd.DataFrame, AlphasData],
        signal_attribute: str = 'alpha_scores',
) -> pd.DataFrame:
    """Extract the signal DataFrame from either a raw panel or AlphasData."""
    if isinstance(signal, AlphasData):
        if not hasattr(signal, signal_attribute):
            raise AttributeError(
                f"AlphasData has no attribute '{signal_attribute}'. "
                f"Available: {[f.name for f in fields(signal)]}"
            )
        df = getattr(signal, signal_attribute)
        if df is None:
            raise ValueError(
                f"AlphasData.{signal_attribute} is None — this score panel was "
                f"not populated during alpha aggregation."
            )
        return df
    if isinstance(signal, pd.DataFrame):
        return signal
    raise TypeError(f"signal must be pd.DataFrame or AlphasData, got {type(signal)}")


def _auto_title(signal_label: str,
                result: qis.SignalDiagnosticsResult) -> str:
    """Auto-generated figure title from signal label and sample window."""
    date_part = ''
    if result.start_date is not None and result.end_date is not None:
        date_part = (f"  ({result.start_date.strftime('%b-%Y')} → "
                     f"{result.end_date.strftime('%b-%Y')})")
    return f"Signal diagnostic — '{signal_label}' panel{date_part}"


# ───────────────────────────────────────────────────────────────────────────────
# Compute
# ───────────────────────────────────────────────────────────────────────────────


def run_signal_diagnostics(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: Union[pd.DataFrame, AlphasData],
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        signal_attribute: str = 'alpha_scores',
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
) -> qis.SignalDiagnosticsResult:
    """Cross-sectional predictive regression of forward returns on lagged signal.

    Wraps ``qis.estimate_signal_diagnostics`` with optional ``AlphasData``
    handling. See the qis docstring for the regression methodology and
    cadence rules.

    Args:
        asset_returns_dict: Per-frequency returns dict as produced by
            ``compute_fx_adjusted_returns`` — already FX-adjusted, already
            unsmoothed, and at each asset's native cadence. Keys are
            pandas frequency strings (e.g. 'ME', 'QE'); values are return
            DataFrames at that frequency.
        signal: T x N signal panel or an ``AlphasData`` instance. When an
            ``AlphasData`` is passed, ``signal_attribute`` is used (default
            ``alpha_scores``).
        group_data: Optional asset -> group-label Series.
        horizons: Forward-return horizons in **per-asset native cadence
            units**. Default ``(1, 2, 3, 6)``. String codes like ``'YE'``
            override per-asset cadence and resample uniformly.
        signal_attribute: ``AlphasData`` field to use.
        group_order: Explicit ordering for the per-group rows.
        is_log_returns: True if ``asset_returns_dict`` contains log
            returns (default), False for arithmetic.

    Returns:
        ``qis.SignalDiagnosticsResult``.
    """
    signal_df = _resolve_signal(signal, signal_attribute)
    return qis.estimate_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=signal_df,
        group_data=group_data,
        horizons=horizons,
        group_order=group_order,
        is_log_returns=is_log_returns,
    )


def run_signal_diagnostics_per_component(
        asset_returns_dict: Dict[str, pd.DataFrame],
        alphas_data: AlphasData,
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        components: Optional[Sequence[str]] = None,
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
) -> Dict[str, qis.SignalDiagnosticsResult]:
    """Run the diagnostic separately for each populated score panel.

    Loops over ``AlphasData`` score attributes (``alpha_scores``,
    ``momentum_score``, ``beta_score``, ``residual_momentum_score``,
    ``managers_scores``, plus cluster variants) and produces one
    ``SignalDiagnosticsResult`` per non-None component.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        alphas_data: ``AlphasData`` whose populated fields are looped over.
        group_data: Optional asset -> group-label Series.
        horizons: Forward-return horizons.
        components: Restrict the sweep to this subset (in order). When
            None, every populated score panel is included.
        group_order: Explicit ordering for the per-group rows.
        is_log_returns: True if ``asset_returns_dict`` contains log returns.

    Returns:
        dict keyed by component name, values are ``SignalDiagnosticsResult``.
    """
    if components is None:
        components = _SCORE_ATTRIBUTES

    results: Dict[str, qis.SignalDiagnosticsResult] = {}
    for attr in components:
        df = getattr(alphas_data, attr, None)
        if df is None:
            logger.info("Skipping '%s' — not populated in AlphasData", attr)
            continue
        results[attr] = run_signal_diagnostics(
            asset_returns_dict=asset_returns_dict,
            signal=df,
            group_data=group_data,
            horizons=horizons,
            group_order=group_order,
            is_log_returns=is_log_returns,
        )
    return results


def compare_signal_diagnostics(
        results: Dict[str, qis.SignalDiagnosticsResult],
        horizon: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate pooled regression rows from multiple signals into one table.

    Args:
        results: dict keyed by signal name, values are
            ``SignalDiagnosticsResult``.
        horizon: Restrict to this single horizon label. When None,
            the output keeps the (signal, horizon) MultiIndex.

    Returns:
        DataFrame indexed by signal (and horizon if not restricted).
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for name, res in results.items():
        if horizon is not None:
            if horizon not in res.pooled_universe.index:
                logger.warning("Signal '%s' has no horizon %r; skipping",
                               name, horizon)
                continue
            row = res.pooled_universe.loc[horizon].copy()
            row.name = name
            rows.append(row)
        else:
            for h in res.pooled_universe.index:
                row = res.pooled_universe.loc[h].copy()
                row.name = (name, h)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if horizon is None:
        out.index = pd.MultiIndex.from_tuples(out.index.tolist(),
                                              names=['signal', 'horizon'])
    else:
        out.index.name = 'signal'
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: Union[pd.DataFrame, AlphasData],
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        signal_attribute: str = 'alpha_scores',
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (22, 12),
        group_colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Run the diagnostic and render the composite figure.

    Composes ``run_signal_diagnostics`` and ``qis.plot_signal_diagnostics``.
    Caller is responsible for the returned Figure (save, embed, close).

    Args:
        asset_returns_dict: Per-frequency returns dict.
        signal: DataFrame or ``AlphasData``.
        group_data: asset -> group-label Series.
        horizons: forward-return horizons.
        signal_attribute: ``AlphasData`` field to use as the signal.
        group_order: explicit ordering for per-group bars.
        is_log_returns: True if returns dict contains log returns.
        title: figure title; auto-generated when None.
        figsize: figure dimensions in inches.
        group_colors: optional dict mapping group label -> hex colour.

    Returns:
        Matplotlib Figure.
    """
    result = run_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=signal,
        group_data=group_data,
        horizons=horizons,
        signal_attribute=signal_attribute,
        group_order=group_order,
        is_log_returns=is_log_returns,
    )
    if title is None:
        signal_label = (signal_attribute if isinstance(signal, AlphasData)
                        else 'signal')
        title = _auto_title(signal_label, result)
    return qis.plot_signal_diagnostics(
        result=result,
        figsize=figsize,
        group_colors=group_colors,
        title=title,
    )


def plot_signal_diagnostics_per_component(
        asset_returns_dict: Dict[str, pd.DataFrame],
        alphas_data: AlphasData,
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        components: Optional[Sequence[str]] = None,
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
        figsize: Tuple[float, float] = (22, 12),
        group_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, plt.Figure]:
    """Run the per-component sweep and render one figure per component.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        alphas_data: ``AlphasData``.
        group_data: asset -> group-label Series.
        horizons: forward-return horizons.
        components: restrict the sweep to this subset.
        group_order: explicit ordering for per-group bars.
        is_log_returns: True if returns dict contains log returns.
        figsize: figure dimensions in inches.
        group_colors: optional dict mapping group label -> hex colour.

    Returns:
        dict keyed by component name, values are Matplotlib Figures.
    """
    results = run_signal_diagnostics_per_component(
        asset_returns_dict=asset_returns_dict,
        alphas_data=alphas_data,
        group_data=group_data,
        horizons=horizons,
        components=components,
        group_order=group_order,
        is_log_returns=is_log_returns,
    )

    figures: Dict[str, plt.Figure] = {}
    for component, result in results.items():
        title = _auto_title(component, result)
        figures[component] = qis.plot_signal_diagnostics(
            result=result,
            figsize=figsize,
            group_colors=group_colors,
            title=title,
        )
    return figures


# ───────────────────────────────────────────────────────────────────────────────
# Per-asset β boxplot
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics_beta_boxplot(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: Union[pd.DataFrame, AlphasData],
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        signal_attribute: str = 'alpha_scores',
        is_log_returns: bool = True,
        min_obs_per_asset: int = 12,
        hue: Optional[str] = 'asset_freq',
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 7),
) -> plt.Figure:
    """Boxplot of per-asset β across horizons.

    For each horizon, estimates one β per asset (no-intercept regression
    on that asset's (z, r_norm_univ) pairs) and visualises the
    cross-asset distribution as boxes — x = horizon, y = β. Optionally
    colours boxes by the asset's native cadence (``asset_freq``), which
    is informative for mixed-cadence universes where monthly and
    quarterly assets may show different signal-quality regimes.

    Args:
        asset_returns_dict: Per-frequency returns dict (same shape the
            pipeline produces).
        signal: DataFrame or ``AlphasData``.
        group_data: Optional asset → group-label Series. Forwarded to the
            underlying regression; doesn't affect the boxplot output but
            populates the ``group`` column of the per-asset table.
        horizons: Forward-return horizons. Default (1, 2, 3, 6).
        signal_attribute: AlphasData field to use as signal.
        is_log_returns: True if returns dict contains log returns.
        min_obs_per_asset: Minimum (z, r) pair count per asset per
            horizon required to report a β. Default 12 (~1 year of
            monthly obs, or ~3 years of quarterly).
        hue: Column to use for box colouring. Default ``'asset_freq'``;
            also supports ``'group'``. Pass ``None`` to disable grouping
            and show a single box per horizon.
        title: Figure title. Auto-generated when None.
        figsize: Figure dimensions in inches.

    Returns:
        Matplotlib Figure.
    """
    result = run_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=signal,
        group_data=group_data,
        horizons=horizons,
        signal_attribute=signal_attribute,
        is_log_returns=is_log_returns,
    )

    df = qis.compute_per_asset_betas(
        result=result, min_obs_per_asset=min_obs_per_asset,
    )

    if title is None:
        signal_label = (signal_attribute if isinstance(signal, AlphasData)
                        else 'signal')
        title = (f"Per-asset β by horizon — '{signal_label}' panel "
                 f"(min {min_obs_per_asset} obs)")

    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5,
                f"No assets meet min_obs_per_asset={min_obs_per_asset}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # qis.plot_box accepts a long-format DataFrame with x/y column names.
    # When hue is requested and the column has >1 unique value, it groups;
    # otherwise we pass hue=None to avoid spurious legends.
    use_hue = hue
    if use_hue is not None:
        if use_hue not in df.columns or df[use_hue].nunique() <= 1:
            use_hue = None

    # qis.plot_box's internal palette helper does df.groupby(x).mean() on
    # the WHOLE frame, not just `y` — pandas then fails trying to "average"
    # the asset string column. Project to the columns plot_box actually
    # consumes so the groupby has only numeric data to reduce over.
    keep_cols = ['horizon', 'beta']
    if use_hue is not None and use_hue not in keep_cols:
        keep_cols.append(use_hue)
    df_plot = df[keep_cols]

    qis.plot_box(
        df=df_plot, x='horizon', y='beta', hue=use_hue,
        xlabel='Horizon (native cadence units)',
        ylabel=r'$\beta$ (per unit of signal)',
        title=title,
        showmedians=True,
        add_zero_line=True,
        yvar_format='{:+.3f}',
        ax=ax,
    )
    return fig