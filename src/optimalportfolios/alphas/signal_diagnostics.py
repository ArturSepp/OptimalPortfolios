"""
Signal-quality diagnostics for optimalportfolios alpha signals.

Thin AlphasData-aware layer on top of ``qis.estimate_signal_diagnostics``
and the ``qis.plot_signal_diagnostics*`` family. The numerics live in
qis; this module is the AlphasData adapter:

    signal_diagnostics_panel
        Enumerate populated score panels in an ``AlphasData`` as a
        ``{name: DataFrame}`` dict. The canonical way to discover what
        signal components are available for per-component diagnostics.

    run_signal_diagnostics
        Resolve a ``Union[DataFrame, AlphasData]`` signal argument to a
        DataFrame, then call ``qis.estimate_signal_diagnostics``. Five
        lines, but keeps the AlphasData type-fanout out of qis.

    run_signal_diagnostics_per_component
        Component-wise compute. Loops every populated score panel in an
        ``AlphasData`` and returns a dict of results keyed by component
        name.

    compare_signal_diagnostics
        Side-by-side comparison: aggregates pooled regression rows from
        multiple results into one DataFrame.

The ``plot_signal_diagnostics`` and ``plot_signal_diagnostics_beta_boxplot``
"compute+plot" wrappers previously in this module now live in qis as
``qis.plot_signal_diagnostics_for_returns`` and
``qis.plot_signal_diagnostics_beta_boxplot``; consumers should call
those directly once they have a DataFrame signal.
"""
from __future__ import annotations

import logging
from dataclasses import fields
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import qis

from optimalportfolios.alphas.alpha_data import AlphasData


logger = logging.getLogger(__name__)


# Score panels in AlphasData enumerated for per-component sweeps.
# Order matters — ``signal_diagnostics_panel`` preserves it so the
# downstream report has a stable section order.
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
    """Extract a DataFrame signal panel from ``DataFrame | AlphasData``.

    Args:
        signal: Either a DataFrame (used as-is) or an ``AlphasData``
            (the specified ``signal_attribute`` is read off and
            returned).
        signal_attribute: Which ``AlphasData`` attribute to use when
            ``signal`` is an ``AlphasData``. Ignored when ``signal`` is
            already a DataFrame.

    Returns:
        The signal as a DataFrame.

    Raises:
        AttributeError: ``signal_attribute`` not found on ``AlphasData``.
        ValueError: ``AlphasData.<signal_attribute>`` is None.
    """
    if isinstance(signal, AlphasData):
        if not hasattr(signal, signal_attribute):
            raise AttributeError(
                f"AlphasData has no attribute '{signal_attribute}'. "
                f"Available: {[f.name for f in fields(signal)]}"
            )
        df = getattr(signal, signal_attribute)
        if df is None:
            raise ValueError(
                f"AlphasData.{signal_attribute} is None — this score "
                f"panel was not populated during alpha aggregation."
            )
        return df
    return signal


def signal_diagnostics_panel(
        alphas_data: AlphasData,
        components: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Enumerate populated score panels in ``AlphasData`` as a dict.

    The standard way to discover what signal components are available
    for diagnostics. Returns a dict keyed by component name (the
    ``AlphasData`` attribute name), values are the underlying score
    DataFrames. None-valued attributes are skipped.

    Args:
        alphas_data: ``AlphasData`` instance.
        components: Restrict to this subset of attribute names. Defaults
            to ``_SCORE_ATTRIBUTES`` — all score panels enumerated
            in this module.

    Returns:
        ``{component_name: DataFrame}`` in iteration order.
    """
    attrs = components if components is not None else _SCORE_ATTRIBUTES
    out: Dict[str, pd.DataFrame] = {}
    for attr in attrs:
        df = getattr(alphas_data, attr, None)
        if df is None:
            logger.info(
                "signal_diagnostics_panel: '%s' not populated — skipping",
                attr,
            )
            continue
        out[attr] = df
    return out


def run_signal_diagnostics(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: Union[pd.DataFrame, AlphasData],
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        signal_attribute: str = 'alpha_scores',
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
) -> qis.SignalDiagnosticsResult:
    """Compute signal diagnostics with AlphasData resolution.

    Five-line shim wrapping ``qis.estimate_signal_diagnostics``: resolves
    AlphasData → DataFrame via ``_resolve_signal``, then delegates. Use
    qis directly when the caller already has a DataFrame.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        signal: DataFrame or ``AlphasData``.
        group_data: Optional asset → group-label Series.
        horizons: Forward-return horizons.
        signal_attribute: ``AlphasData`` field to use when ``signal``
            is an AlphasData. Ignored otherwise.
        group_order: Explicit group ordering for the per-group panel.
        is_log_returns: True if ``asset_returns_dict`` is log returns.

    Returns:
        ``qis.SignalDiagnosticsResult``.
    """
    df = _resolve_signal(signal, signal_attribute=signal_attribute)
    return qis.estimate_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=df,
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

    Loops over the score panels in ``alphas_data`` (via
    ``signal_diagnostics_panel``) and produces one
    ``SignalDiagnosticsResult`` per component.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        alphas_data: ``AlphasData`` whose populated fields are looped.
        group_data: Optional asset → group-label Series.
        horizons: Forward-return horizons.
        components: Restrict the sweep to this subset (in order). When
            None, every populated score panel is included.
        group_order: Explicit group ordering.
        is_log_returns: True if returns dict contains log returns.

    Returns:
        ``{component_name: SignalDiagnosticsResult}``.
    """
    panels = signal_diagnostics_panel(alphas_data, components=components)
    results: Dict[str, qis.SignalDiagnosticsResult] = {}
    for name, df in panels.items():
        results[name] = qis.estimate_signal_diagnostics(
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
        results: ``{signal_name: SignalDiagnosticsResult}``.
        horizon: Restrict to this single horizon label. When None, the
            output keeps the (signal, horizon) MultiIndex.

    Returns:
        DataFrame indexed by signal (and horizon if not restricted).
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for name, res in results.items():
        if horizon is not None:
            if horizon not in res.pooled_universe.index:
                logger.warning(
                    "Signal '%s' has no horizon %r; skipping", name, horizon,
                )
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
        out.index = pd.MultiIndex.from_tuples(
            out.index.tolist(), names=['signal', 'horizon'],
        )
    else:
        out.index.name = 'signal'
    return out
def compare_signal_ic_ir(
        results: Dict[str, qis.SignalDiagnosticsResult],
        horizon: Optional[str] = None,
        method: str = 'spearman',
        return_col: str = 'r_norm_univ',
        periods_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """Aggregate IC-IR rows from multiple signals into one table.

    The IC-IR sibling of ``compare_signal_diagnostics``: calls
    ``qis.estimate_ic_ir`` on each result and stacks the per-horizon rows
    in the same (signal[, horizon]) index shape, so the time-series IC-IR
    table sits next to the pooled-regression table on the signal-
    diagnostics page. The numerics live in ``qis.estimate_ic_ir``; this
    only does the multi-signal assembly.

    Args:
        results: ``{signal_name: SignalDiagnosticsResult}`` (e.g. from
            ``run_signal_diagnostics_per_component``).
        horizon: Restrict to this single horizon label. When None, the
            output keeps the (signal, horizon) MultiIndex.
        method: 'spearman' (rank IC, default) or 'pearson' for the
            per-date IC.
        return_col: 'r_norm_univ' (universe cross-section, default) or
            'r_norm_group' (within-group — requires ``group_data`` to have
            been passed to the diagnostic).
        periods_per_year: Annualisation factor forwarded to
            ``qis.estimate_ic_ir`` (inferred from IC-date spacing when
            None).

    Returns:
        DataFrame indexed by signal (and horizon if not restricted), with
        the IC-IR columns from ``qis.estimate_ic_ir``.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for name, res in results.items():
        ic_ir = qis.estimate_ic_ir(
            res, method=method, return_col=return_col,
            periods_per_year=periods_per_year,
        )
        if horizon is not None:
            if horizon not in ic_ir.index:
                logger.warning(
                    "Signal '%s' has no horizon %r; skipping", name, horizon,
                )
                continue
            row = ic_ir.loc[horizon].copy()
            row.name = name
            rows.append(row)
        else:
            for h in ic_ir.index:
                row = ic_ir.loc[h].copy()
                row.name = (name, h)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if horizon is None:
        out.index = pd.MultiIndex.from_tuples(
            out.index.tolist(), names=['signal', 'horizon'],
        )
    else:
        out.index.name = 'signal'
    return out


def build_signal_diagnostics_table(
        results: Dict[str, qis.SignalDiagnosticsResult],
        horizon: Optional[str] = None,
        method: str = 'spearman',
        return_col: str = 'r_norm_univ',
        periods_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """One wide table for the signal-diagnostics report page.

    Joins the pooled-regression stats (``n, beta, se, t_stat, IC_pearson,
    IC_spearman`` from ``compare_signal_diagnostics``) with the IC-IR stats
    (``n_dates, mean_IC, std_IC, IC_IR, IC_IR_an, hit_rate`` and the IC-IR
    significance, renamed ``IC_t_stat`` to avoid colliding with the
    regression ``t_stat``). Rows are (signal[, horizon]).

    This is the table the separate report page renders — the pooled β/IC
    on the left, the time-series IC-IR on the right.

    Args:
        results: ``{signal_name: SignalDiagnosticsResult}``.
        horizon: Restrict to this single horizon label. When None, the
            output keeps the (signal, horizon) MultiIndex.
        method: 'spearman' (default) or 'pearson' for the per-date IC.
        return_col: 'r_norm_univ' (default) or 'r_norm_group'.
        periods_per_year: Forwarded to ``qis.estimate_ic_ir``.

    Returns:
        DataFrame indexed by signal (and horizon if not restricted) with
        the pooled-regression and IC-IR columns side by side.
    """
    pooled = compare_signal_diagnostics(results, horizon=horizon)
    ic_ir = compare_signal_ic_ir(
        results, horizon=horizon, method=method,
        return_col=return_col, periods_per_year=periods_per_year,
    )
    if pooled.empty:
        return ic_ir
    if ic_ir.empty:
        return pooled
    ic_ir = ic_ir.rename(columns={'t_stat': 'IC_t_stat'})
    return pooled.join(ic_ir, how='left')