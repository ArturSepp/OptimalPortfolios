"""Search and diagnose BlackRock-fund cluster long-short specifications.

The search keeps the payoff comparison fair: every cluster leg is compared with a
same-signal, same-quantile, same-sleeve-budget global rank.  Both sides use monthly
decisions, one implementation-period lag, +1/-1 exposure and 20 bp one-way costs.
No covariance model is estimated here; all 28 partitions come from the frozen cache.

The experiment is deliberately staged.  Marginal grids vary one previously exercised
dimension around the owner-frozen 50/30/20 base.  The two best pre-2018 values in each
dimension are then crossed, with all three budget constructions retained.  Selection
uses 2009-08-31 through 2017-12-31 only; 2018-01-31 through 2026-06-30 is evaluation.
Full-window and evaluation-oracle leaders are labelled descriptive, never selected.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_three_universe_signal_comparison as three
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_signal_comparison as signals
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_blackrock_long_short_search.py"
)
UNIVERSE = funds.UNIVERSE
COST_BPS = 20.0
CLUSTER_FALLBACK = 5
MOMENTUM_LONG_SPAN = 12
SIGNAL_FREQUENCY = "ME"
QUANTILES = (0.10, 0.15, 0.20, 0.25, 0.30)
CONSTRUCTIONS = ("group_equal", "sqrt_group_size", "asset_equal")
HYBRID_VARIANTS = ("global_long_cluster_short", "cluster_long_global_short")
HOLDING_SCHEDULES = ("monthly", "every_two_months", "quarterly")
COVARIANCE_CELLS = tuple(funds._cells())
WEIGHT_GRID = sleeves._weight_grid()
SLEEVES = tuple(sleeves.SLEEVES)
WEIGHT_TOLERANCE = 1e-12
SIGNAL_TOLERANCE = 1e-12
TRAIN_WINDOW = sleeves.TRAIN_WINDOW
EVALUATION_WINDOW = sleeves.TEST_WINDOW
FULL_WINDOW = funds.HEADLINE_WINDOW
WINDOWS = {
    TRAIN_WINDOW: (sleeves.TRAIN_START, sleeves.TRAIN_END),
    EVALUATION_WINDOW: (sleeves.TEST_START, sleeves.TEST_END),
    FULL_WINDOW: (funds.HEADLINE_START, funds.HEADLINE_END),
}
COMPARISON_METRICS = (
    "gross_return_annualized",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "net_total_return",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
)


@dataclass(frozen=True)
class SignalSpec:
    """Identify one production-style or classic momentum signal."""

    kind: str
    short_span: int | None = None
    vol_span: int | None = None
    mean_adj_type: str | None = None

    @property
    def signal_id(self) -> str:
        """Return a stable identifier for the signal definition."""
        if self.kind == "classic":
            return "classic_12m_skip1"
        short = "none" if self.short_span is None else str(self.short_span)
        return (
            f"rosaa_short_{short}_vol_{self.vol_span}_mean_"
            f"{self.mean_adj_type}"
        )


SIGNAL_SPECS = tuple(
    SignalSpec("rosaa", short_span, vol_span, mean_adj_type)
    for short_span in (None, 1, 2, 3)
    for vol_span in (13, 26, 52)
    for mean_adj_type in ("NONE", "EWMA")
) + (SignalSpec("classic"),)
SIGNAL_BY_ID = {spec.signal_id: spec for spec in SIGNAL_SPECS}
BASE_SIGNAL_ID = "rosaa_short_none_vol_13_mean_EWMA"
ABSOLUTE_SIGNAL_ID = "rosaa_short_3_vol_13_mean_EWMA"


@dataclass(frozen=True)
class CandidateSpec:
    """Identify one complete cluster and matched-global portfolio specification."""

    signal_id: str
    frequency: str
    span: int
    q: float
    weight_id: str
    construction: str
    stage: str = "marginal"

    @property
    def candidate_id(self) -> str:
        """Return a stable identifier excluding the descriptive search stage."""
        frequency = self.frequency.replace("-", "_")
        return (
            f"{self.signal_id}__{frequency}_{self.span}__q_{self.q:.2f}__"
            f"{self.weight_id}__{self.construction}"
        )


BASE_CANDIDATE = CandidateSpec(
    signal_id=BASE_SIGNAL_ID,
    frequency="W-THU",
    span=156,
    q=0.25,
    weight_id="E50_F30_R20",
    construction="group_equal",
)


def _root() -> Path:
    """Return and create the external search-output directory."""
    root = funds._root() / "long_short_spec_search_20260815"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mean_adj_enum(label: str) -> qis.MeanAdjType:
    """Resolve a persisted mean-adjustment label."""
    if label == "NONE":
        return qis.MeanAdjType.NONE
    if label == "EWMA":
        return qis.MeanAdjType.EWMA
    raise KeyError(label)


def _target_map(weight_id: str) -> dict[str, float]:
    """Return one frozen strategic sleeve allocation."""
    row = WEIGHT_GRID.loc[WEIGHT_GRID["weight_id"].eq(weight_id)]
    if len(row) != 1:
        raise KeyError(weight_id)
    item = row.iloc[0]
    return {
        "Equity": float(item["equity_weight"]),
        "Fixed Income": float(item["fixed_income_weight"]),
        "Rest": float(item["rest_weight"]),
    }


def _sqrt_group_weights(
    ranks: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
) -> pd.DataFrame:
    """Allocate group budgets in proportion to square-root selected breadth."""
    selected = e5._weights_from_ranks(
        ranks, eligibility, q, u1_grid.UNIVERSE
    ).gt(0.0)
    output = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for date in ranks.index:
        labels = groups.loc[date]
        selected_labels = labels.loc[selected.loc[date] & labels.notna()]
        counts = selected_labels.value_counts(sort=False)
        if counts.empty:
            continue
        group_budgets = np.sqrt(counts.astype(float))
        group_budgets = group_budgets / group_budgets.sum()
        for label, count in counts.items():
            assets = selected_labels.index[selected_labels.eq(label)]
            output.loc[date, assets] = float(group_budgets.loc[label]) / int(count)
    return output


def _side_from_ranks(
    ranks: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    target: Mapping[str, float],
    q: float,
    construction: str,
) -> pd.DataFrame:
    """Build one positive side with exact top-level sleeve budgets."""
    combined = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for sleeve in SLEEVES:
        budget = float(target[sleeve])
        if budget <= 0.0:
            continue
        sleeve_eligibility = eligibility & sleeve_panel.eq(sleeve)
        if construction == "group_equal":
            side, _, _ = _group_equal_from_ranks(
                ranks,
                sleeve_eligibility,
                groups,
                q,
                u1_grid.UNIVERSE,
            )
        elif construction == "asset_equal":
            side = e5._weights_from_ranks(
                ranks, sleeve_eligibility, q, u1_grid.UNIVERSE
            )
        elif construction == "sqrt_group_size":
            side = _sqrt_group_weights(ranks, sleeve_eligibility, groups, q)
        else:
            raise KeyError(construction)
        if side.sum(axis=1).le(0.0).any():
            raise AssertionError(f"{sleeve} has an empty selected side")
        combined = combined.add(side.mul(budget), fill_value=0.0)
    return combined


def _long_short_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    target: Mapping[str, float],
    *,
    q: float,
    construction: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a sleeve-neutral +1/-1 long-short portfolio."""
    long_ranks = e5._rank_panel(scores, groups)
    short_ranks = e5._rank_panel(-scores, groups)
    long_raw = _side_from_ranks(
        long_ranks, eligibility, sleeve_panel, groups, target, q, construction
    )
    short_raw = _side_from_ranks(
        short_ranks, eligibility, sleeve_panel, groups, target, q, construction
    )
    overlap = long_raw.gt(0.0) & short_raw.gt(0.0)
    long_book = sleeves._renormalize_side_by_sleeve(
        long_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    short_book = sleeves._renormalize_side_by_sleeve(
        short_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    weights = long_book - short_book
    sleeve_errors = []
    for sleeve in SLEEVES:
        expected = float(target[sleeve])
        mask = sleeve_panel.eq(sleeve)
        measured_long = long_book.where(mask, 0.0).sum(axis=1)
        measured_short = short_book.where(mask, 0.0).sum(axis=1)
        sleeve_errors.extend(
            [
                float(measured_long.sub(expected).abs().max()),
                float(measured_short.sub(expected).abs().max()),
            ]
        )
    diagnostics = {
        "max_long_exposure_abs_error": float(
            long_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_short_exposure_abs_error": float(
            short_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_net_exposure_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "max_sleeve_budget_abs_error": max(sleeve_errors),
        "max_overlap_assets_removed": int(overlap.sum(axis=1).max()),
    }
    return weights, diagnostics


def _hybrid_weights(
    cluster_weights: pd.DataFrame,
    global_weights: pd.DataFrame,
    variant: str,
    sleeve_panel: pd.DataFrame | None = None,
    target: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Substitute one side, remove overlap, and restore exact signed budgets."""
    if variant == "global_long_cluster_short":
        long_raw = global_weights.clip(lower=0.0)
        short_raw = -cluster_weights.clip(upper=0.0)
    elif variant == "cluster_long_global_short":
        long_raw = cluster_weights.clip(lower=0.0)
        short_raw = -global_weights.clip(upper=0.0)
    else:
        raise KeyError(variant)
    overlap = long_raw.gt(0.0) & short_raw.gt(0.0)
    long_raw = long_raw.mask(overlap, 0.0)
    short_raw = short_raw.mask(overlap, 0.0)
    if sleeve_panel is not None and target is not None:
        long_book = sleeves._renormalize_side_by_sleeve(
            long_raw, sleeve_panel, target
        )
        short_book = sleeves._renormalize_side_by_sleeve(
            short_raw, sleeve_panel, target
        )
    else:
        long_book = long_raw.div(long_raw.sum(axis=1), axis=0)
        short_book = short_raw.div(short_raw.sum(axis=1), axis=0)
    return long_book - short_book


def _rebalance_dates(
    dates: pd.DatetimeIndex,
    schedule: str,
) -> pd.DatetimeIndex:
    """Return a deterministic decision subset that always includes the window start."""
    if schedule == "monthly":
        return dates
    if schedule == "every_two_months":
        return dates[::2]
    if schedule == "quarterly":
        quarterly = dates[dates.month.isin((3, 6, 9, 12))]
        return pd.DatetimeIndex([dates[0]]).union(quarterly).sort_values()
    raise KeyError(schedule)


def _signal_panels(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    monthly_eligibility: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Build all predeclared signals once and enforce point-in-time timing."""
    monthly_returns = funds._native_returns(daily, SIGNAL_FREQUENCY)
    if monthly_eligibility is None:
        monthly_eligibility = funds._eligibility_for_dates(
            daily, monthly_returns.index
        )
    monthly_eligibility = monthly_eligibility.reindex(
        index=monthly_returns.index, columns=monthly_returns.columns
    ).fillna(False).astype(bool)
    signal_prices = qis.returns_to_nav(np.expm1(monthly_returns))
    benchmark_returns = np.expm1(monthly_returns).where(monthly_eligibility).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
    rebuilt = qis.to_returns(
        signal_prices, freq=SIGNAL_FREQUENCY, is_log_returns=True
    ).reindex_like(monthly_returns)
    finite = rebuilt.subtract(monthly_returns).abs().to_numpy()
    finite = finite[np.isfinite(finite)]
    roundtrip_error = float(finite.max()) if finite.size else 0.0

    output: dict[str, dict[str, pd.DataFrame]] = {}
    rows = []
    for spec in SIGNAL_SPECS:
        if spec.kind == "classic":
            raw_decision = signals._classic_scores(monthly_returns, dates)
            global_decision = qis.df_to_cross_sectional_score(df=raw_decision)
            raw_source = raw_decision
            lookahead_days = 0
            reconstruction = signals.classic._independent_score_regression(
                monthly_returns, dates, raw_decision
            )
            reconstruction_error = float(reconstruction.loc[0, "max_abs_error"])
            reconstruction_tolerance = signals.classic.SCORE_REGRESSION_TOLERANCE
        else:
            global_source, raw_source = compute_momentum_alpha(
                prices=signal_prices,
                benchmark_price=benchmark,
                returns_freq=SIGNAL_FREQUENCY,
                group_data=None,
                long_span=MOMENTUM_LONG_SPAN,
                short_span=spec.short_span,
                vol_span=int(spec.vol_span),
                mean_adj_type=_mean_adj_enum(str(spec.mean_adj_type)),
            )
            global_decision, timestamps = prod._asof_panel(global_source, dates)
            raw_decision, raw_timestamps = prod._asof_panel(raw_source, dates)
            if not timestamps.equals(raw_timestamps):
                raise AssertionError(f"raw/global timestamps differ: {spec.signal_id}")
            lookahead_days = int(timestamps.sub(timestamps.index).dt.days.max())
            reconstruction_error = roundtrip_error
            reconstruction_tolerance = SIGNAL_TOLERANCE
        global_decision = global_decision.reindex(
            index=dates, columns=eligibility.columns
        ).where(eligibility)
        raw_decision = raw_decision.reindex(
            index=dates, columns=eligibility.columns
        ).where(eligibility)
        valid = global_decision.loc[funds.HEADLINE_START:funds.HEADLINE_END].notna().sum(axis=1)
        passed = (
            lookahead_days <= 0
            and reconstruction_error <= reconstruction_tolerance
            and int(valid.min()) > 0
        )
        output[spec.signal_id] = {
            "global": global_decision,
            "raw_source": raw_source,
            "raw_decision": raw_decision,
        }
        rows.append(
            {
                "signal_id": spec.signal_id,
                "kind": spec.kind,
                "short_span": spec.short_span,
                "vol_span": spec.vol_span,
                "mean_adj_type": spec.mean_adj_type,
                "max_signal_lookahead_days": lookahead_days,
                "max_reconstruction_abs_error": reconstruction_error,
                "reconstruction_tolerance": reconstruction_tolerance,
                "valid_assets_min": int(valid.min()),
                "valid_assets_median": float(valid.median()),
                "valid_assets_max": int(valid.max()),
                "status": "PASS" if passed else "FAIL",
            }
        )
    diagnostics = pd.DataFrame(rows)
    if not diagnostics["status"].eq("PASS").all():
        raise AssertionError(diagnostics.loc[~diagnostics["status"].eq("PASS")])
    return output, diagnostics


def _context() -> dict[str, object]:
    """Load invariant fund data, signals, eligibility, and payoff panels."""
    dates = funds._dates()
    headline_dates = dates[(dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)]
    daily = funds._read_daily()
    eligibility_all = funds._eligibility_for_dates(daily, dates)
    eligibility = eligibility_all.reindex(index=headline_dates).astype(bool)
    signal_panels, signal_diagnostics = _signal_panels(
        daily, dates, eligibility_all
    )
    broad_sleeves = sleeves._broad_sleeves(eligibility.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, broad_sleeves)
    prices_all = funds._performance_prices(daily)
    windows = {}
    for name, (start, end) in WINDOWS.items():
        window_dates = headline_dates[(headline_dates >= start) & (headline_dates <= end)]
        windows[name] = {
            "dates": window_dates,
            "prices": funds._window_prices(prices_all, window_dates),
            "ew_nav": funds._ew_reference(
                prices_all, eligibility_all, window_dates, name
            ),
        }
    return {
        "dates": headline_dates,
        "eligibility": eligibility,
        "signals": signal_panels,
        "signal_diagnostics": signal_diagnostics,
        "sleeves": broad_sleeves,
        "sleeve_panel": sleeve_panel,
        "windows": windows,
    }


def _marginal_candidates() -> tuple[list[CandidateSpec], dict[str, set[str]]]:
    """Return the union of all one-dimension-at-a-time grids and their tags."""
    candidates: dict[str, CandidateSpec] = {}
    tags: dict[str, set[str]] = {}

    def add(candidate: CandidateSpec, tag: str) -> None:
        candidates[candidate.candidate_id] = candidate
        tags.setdefault(candidate.candidate_id, set()).add(tag)

    for spec in SIGNAL_SPECS:
        add(replace(BASE_CANDIDATE, signal_id=spec.signal_id), "signal")
    for frequency, span in COVARIANCE_CELLS:
        add(replace(BASE_CANDIDATE, frequency=frequency, span=span), "covariance")
    for q in QUANTILES:
        add(replace(BASE_CANDIDATE, q=q), "quantile")
    for weight_id in WEIGHT_GRID["weight_id"]:
        add(replace(BASE_CANDIDATE, weight_id=str(weight_id)), "sleeve_weight")
    for construction in CONSTRUCTIONS:
        add(replace(BASE_CANDIDATE, construction=construction), "construction")
    return list(candidates.values()), tags


def _top_interaction_candidates(
    phase_one: pd.DataFrame,
    marginal_tags: Mapping[str, set[str]],
    candidates: Mapping[str, CandidateSpec],
) -> tuple[list[CandidateSpec], pd.DataFrame]:
    """Cross only pre-2018 marginal finalists, retaining every construction."""
    training = phase_one.loc[phase_one["analysis_window"].eq(TRAIN_WINDOW)].copy()
    dimensions = {
        "signal": "signal_id",
        "covariance": "covariance",
        "quantile": "q",
        "sleeve_weight": "weight_id",
    }
    selected_values: dict[str, list[object]] = {}
    selection_rows = []
    for tag, value_name in dimensions.items():
        ids = [candidate_id for candidate_id, values in marginal_tags.items() if tag in values]
        panel = training.loc[training["candidate_id"].isin(ids)].copy()
        if tag == "covariance":
            panel["covariance"] = list(zip(panel["frequency"], panel["span"]))
        panel = panel.sort_values(
            ["delta_net_return_annualized", "delta_sharpe_rf0"],
            ascending=[False, False],
        ).drop_duplicates(value_name)
        leaders = panel.head(2)
        selected_values[tag] = leaders[value_name].tolist()
        for rank, (_, row) in enumerate(leaders.iterrows(), start=1):
            selection_rows.append(
                {
                    "dimension": tag,
                    "training_rank": rank,
                    "selected_value": str(row[value_name]),
                    "source_candidate_id": row["candidate_id"],
                    "training_delta_net_return_annualized": row[
                        "delta_net_return_annualized"
                    ],
                    "training_delta_sharpe_rf0": row["delta_sharpe_rf0"],
                }
            )

    output = {}
    for signal_id, covariance, q, weight_id, construction in product(
        selected_values["signal"],
        selected_values["covariance"],
        selected_values["quantile"],
        selected_values["sleeve_weight"],
        CONSTRUCTIONS,
    ):
        frequency, span = covariance
        candidate = CandidateSpec(
            signal_id=str(signal_id),
            frequency=str(frequency),
            span=int(span),
            q=float(q),
            weight_id=str(weight_id),
            construction=str(construction),
            stage="interaction",
        )
        if candidate.candidate_id not in candidates:
            output[candidate.candidate_id] = candidate
    return list(output.values()), pd.DataFrame(selection_rows)


def _weight_diagnostics(
    cluster_weights: pd.DataFrame,
    global_weights: pd.DataFrame,
    groups: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> dict[str, float]:
    """Summarise breadth, overlap, and small-cluster budget channels."""
    cluster_long = cluster_weights.gt(0.0)
    cluster_short = cluster_weights.lt(0.0)
    global_long = global_weights.gt(0.0)
    global_short = global_weights.lt(0.0)

    def jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
        intersection = (left & right).sum(axis=1)
        union = (left | right).sum(axis=1).replace(0, np.nan)
        return float(intersection.div(union).mean())

    effective_long = 1.0 / cluster_weights.clip(lower=0.0).pow(2).sum(axis=1)
    effective_short = 1.0 / cluster_weights.clip(upper=0.0).pow(2).sum(axis=1)
    group_counts = []
    singleton_shares = []
    small_shares = []
    for date in cluster_weights.index:
        labels = groups.loc[date]
        eligible_labels = labels.loc[eligibility.loc[date] & labels.notna()]
        sizes = eligible_labels.value_counts(sort=False)
        group_counts.append(len(sizes))
        asset_sizes = labels.map(sizes)
        absolute = cluster_weights.loc[date].abs()
        singleton_shares.append(float(absolute.loc[asset_sizes.eq(1)].sum() / 2.0))
        small_shares.append(float(absolute.loc[asset_sizes.le(5)].sum() / 2.0))
    return {
        "selected_long_assets_mean": float(cluster_long.sum(axis=1).mean()),
        "selected_short_assets_mean": float(cluster_short.sum(axis=1).mean()),
        "effective_long_positions_mean": float(effective_long.mean()),
        "effective_short_positions_mean": float(effective_short.mean()),
        "available_hierarchical_groups_mean": float(np.mean(group_counts)),
        "available_hierarchical_groups_std": float(np.std(group_counts, ddof=1)),
        "long_selection_jaccard_vs_global_mean": jaccard(cluster_long, global_long),
        "short_selection_jaccard_vs_global_mean": jaccard(cluster_short, global_short),
        "singleton_group_gross_budget_share_mean": float(np.mean(singleton_shares)),
        "small_group_le5_gross_budget_share_mean": float(np.mean(small_shares)),
    }


def _performance_payload(
    weights: pd.DataFrame,
    window: Mapping[str, object],
    ticker: str,
    cost_bps: float = COST_BPS,
) -> dict[str, float]:
    """Run the canonical QIS net/gross paths for one window."""
    dates = window["dates"]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("window dates are not a DatetimeIndex")
    net, gross = funds._backtest(
        window["prices"],
        weights.reindex(index=dates),
        cost_bps / 10000.0,
        ticker,
    )
    return sleeves._performance_payload(net, gross, window["ew_nav"])


def _run_candidates(
    candidate_specs: list[CandidateSpec],
    context: Mapping[str, object],
    score_cache: dict[tuple[str, str, int], pd.DataFrame],
    partition_cache: dict[tuple[str, int], pd.DataFrame],
    cluster_weight_cache: dict[str, pd.DataFrame],
    global_weight_cache: dict[tuple[str, float, str], pd.DataFrame],
    global_performance_cache: dict[tuple[str, float, str, str], dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run candidate cluster books and memoized matched-global controls."""
    rows = []
    comparisons = []
    diagnostics_rows = []
    dates = context["dates"]
    eligibility = context["eligibility"]
    sleeve_panel = context["sleeve_panel"]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("context dates are not a DatetimeIndex")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("context eligibility is not a DataFrame")
    if not isinstance(sleeve_panel, pd.DataFrame):
        raise AssertionError("context sleeve panel is not a DataFrame")

    for number, candidate in enumerate(candidate_specs, start=1):
        target = _target_map(candidate.weight_id)
        signal = context["signals"][candidate.signal_id]
        cell = (candidate.frequency, candidate.span)
        if cell not in partition_cache:
            loader = context.get("partition_loader", funds._load_partition)
            partition_cache[cell] = loader(*cell)[0].reindex(
                index=dates, columns=eligibility.columns
            )
        clusters = partition_cache[cell]
        missing = int((eligibility & clusters.isna()).sum().sum())
        if missing:
            raise AssertionError(f"{cell} misses {missing} eligible memberships")
        hierarchical_groups = sleeves._hierarchical_groups(clusters, sleeve_panel)
        score_key = (candidate.signal_id, candidate.frequency, candidate.span)
        if score_key not in score_cache:
            source = score_within_clusters(
                raw_signal=signal["raw_source"],
                rolling_clusters=funds._panel_dict(clusters),
                min_cluster_size=CLUSTER_FALLBACK,
            )
            decision, _ = prod._asof_panel(source, dates)
            score_cache[score_key] = decision.reindex(
                index=dates, columns=eligibility.columns
            ).where(eligibility)
        cluster_scores = score_cache[score_key]
        cluster_weights, exact = _long_short_weights(
            cluster_scores,
            eligibility,
            sleeve_panel,
            hierarchical_groups,
            target,
            q=candidate.q,
            construction=candidate.construction,
        )
        cluster_weight_cache[candidate.candidate_id] = cluster_weights

        global_key = (candidate.signal_id, candidate.q, candidate.weight_id)
        if global_key not in global_weight_cache:
            global_groups = sleeve_panel
            global_weight_cache[global_key], _ = _long_short_weights(
                signal["global"].reindex(index=dates).where(eligibility),
                eligibility,
                sleeve_panel,
                global_groups,
                target,
                q=candidate.q,
                construction="asset_equal",
            )
        global_weights = global_weight_cache[global_key]
        breadth = _weight_diagnostics(
            cluster_weights, global_weights, hierarchical_groups, eligibility
        )
        maximum_error = max(
            abs(value) for key, value in exact.items() if "error" in key
        )
        diagnostics_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "stage": candidate.stage,
                "signal_id": candidate.signal_id,
                "frequency": candidate.frequency,
                "span": candidate.span,
                "q": candidate.q,
                "weight_id": candidate.weight_id,
                "construction": candidate.construction,
                "eligible_memberships_missing": missing,
                **exact,
                **breadth,
                "maximum_exact_weight_error": maximum_error,
                "weight_tolerance": WEIGHT_TOLERANCE,
                "status": "PASS" if maximum_error <= WEIGHT_TOLERANCE else "FAIL",
            }
        )

        short_hash = hashlib.sha256(candidate.candidate_id.encode()).hexdigest()[:12]
        for window_name, window in context["windows"].items():
            cluster_payload = _performance_payload(
                cluster_weights,
                window,
                f"fund_cluster_{short_hash}_{window_name}",
            )
            global_perf_key = (*global_key, window_name)
            if global_perf_key not in global_performance_cache:
                global_performance_cache[global_perf_key] = _performance_payload(
                    global_weights,
                    window,
                    f"fund_global_{hash(global_perf_key)}_{window_name}",
                )
            global_payload = global_performance_cache[global_perf_key]
            common = {
                "candidate_id": candidate.candidate_id,
                "stage": candidate.stage,
                "analysis_window": window_name,
                "signal_id": candidate.signal_id,
                "frequency": candidate.frequency,
                "span": candidate.span,
                "q": candidate.q,
                "weight_id": candidate.weight_id,
                "construction": candidate.construction,
                "cost_bps_one_way": COST_BPS,
                "runner": str(context.get("runner", RUNNER)),
            }
            rows.extend(
                [
                    {**common, "leg": "cluster", **cluster_payload},
                    {**common, "leg": "global", **global_payload},
                ]
            )
            comparison = {**common, "cluster_leg": "cluster", "benchmark_leg": "global"}
            for metric in COMPARISON_METRICS:
                comparison[f"cluster_{metric}"] = cluster_payload[metric]
                comparison[f"global_{metric}"] = global_payload[metric]
                comparison[f"delta_{metric}"] = (
                    cluster_payload[metric] - global_payload[metric]
                )
            comparison["beats_global_net_return"] = (
                comparison["delta_net_return_annualized"] > 0.0
            )
            comparison["beats_global_sharpe"] = comparison["delta_sharpe_rf0"] > 0.0
            comparison["beats_global_both"] = (
                comparison["beats_global_net_return"]
                and comparison["beats_global_sharpe"]
            )
            comparisons.append(comparison)
        if number % 10 == 0 or number == len(candidate_specs):
            print(
                f"fund search {candidate_specs[0].stage}: "
                f"{number}/{len(candidate_specs)} candidates",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame(diagnostics_rows)


def _selection_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Identify the honest train-selected candidate and descriptive oracles."""
    indexed = comparison.set_index(["candidate_id", "analysis_window"])
    training = comparison.loc[comparison["analysis_window"].eq(TRAIN_WINDOW)]
    evaluation = comparison.loc[comparison["analysis_window"].eq(EVALUATION_WINDOW)]
    full = comparison.loc[comparison["analysis_window"].eq(FULL_WINDOW)]
    train_leader = training.sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"], ascending=[False, False]
    ).iloc[0]["candidate_id"]
    evaluation_oracle = evaluation.sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"], ascending=[False, False]
    ).iloc[0]["candidate_id"]
    full_oracle = full.sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"], ascending=[False, False]
    ).iloc[0]["candidate_id"]
    train_positive = set(
        training.loc[
            training["delta_net_return_annualized"].gt(0), "candidate_id"
        ]
    )
    evaluation_positive = set(
        evaluation.loc[evaluation["delta_net_return_annualized"].gt(0), "candidate_id"]
    )
    stable = sorted(train_positive & evaluation_positive)
    stable_leader = None
    if stable:
        stability = pd.DataFrame(
            {
                candidate_id: {
                    "minimum_half_delta": min(
                        float(
                            indexed.loc[
                                (candidate_id, TRAIN_WINDOW),
                                "delta_net_return_annualized",
                            ]
                        ),
                        float(
                            indexed.loc[
                                (candidate_id, EVALUATION_WINDOW),
                                "delta_net_return_annualized",
                            ]
                        ),
                    )
                }
                for candidate_id in stable
            }
        ).T
        stable_leader = stability["minimum_half_delta"].idxmax()
    roles = {
        "train_selected": train_leader,
        "evaluation_oracle_descriptive": evaluation_oracle,
        "full_oracle_descriptive": full_oracle,
    }
    if stable_leader is not None:
        roles["positive_both_halves_descriptive"] = stable_leader
    rows = []
    for role, candidate_id in roles.items():
        for window in WINDOWS:
            row = indexed.loc[(candidate_id, window)].to_dict()
            rows.append(
                {
                    "selection_role": role,
                    "candidate_id": candidate_id,
                    "analysis_window": window,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def _driver_table(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose net underperformance into gross selection and incremental costs."""
    role_map = {
        BASE_CANDIDATE.candidate_id: "owner_base",
        **selection.drop_duplicates("selection_role").set_index("candidate_id")[
            "selection_role"
        ].to_dict(),
    }
    panel = comparison.loc[comparison["candidate_id"].isin(role_map)].copy()
    panel["selection_role"] = panel["candidate_id"].map(role_map)
    panel["gross_selection_gap_bp"] = panel["delta_gross_return_annualized"] * 10000.0
    panel["incremental_cost_penalty_bp"] = (
        panel["cluster_cost_drag_bp_per_year"]
        - panel["global_cost_drag_bp_per_year"]
    )
    panel["net_gap_bp"] = panel["delta_net_return_annualized"] * 10000.0
    panel["gross_minus_cost_reconciliation_bp"] = (
        panel["net_gap_bp"]
        - panel["gross_selection_gap_bp"]
        + panel["incremental_cost_penalty_bp"]
    )
    return panel.merge(diagnostics, on=[
        "candidate_id", "stage", "signal_id", "frequency", "span", "q", "weight_id", "construction"
    ], how="left", suffixes=("", "_diagnostic"))


def _cost_sensitivity(
    selection: pd.DataFrame,
    context: Mapping[str, object],
    cluster_weights: Mapping[str, pd.DataFrame],
    global_weights: Mapping[tuple[str, float, str], pd.DataFrame],
    candidate_map: Mapping[str, CandidateSpec],
) -> pd.DataFrame:
    """Reprice selected and oracle candidates at the frozen cost grid."""
    role_ids = selection.drop_duplicates("selection_role")[
        ["selection_role", "candidate_id"]
    ]
    role_ids = pd.concat(
        [
            pd.DataFrame(
                [{"selection_role": "owner_base", "candidate_id": BASE_CANDIDATE.candidate_id}]
            ),
            role_ids,
        ],
        ignore_index=True,
    ).drop_duplicates("candidate_id")
    rows = []
    for _, role in role_ids.iterrows():
        candidate = candidate_map[str(role["candidate_id"])]
        global_key = (candidate.signal_id, candidate.q, candidate.weight_id)
        for window_name in (EVALUATION_WINDOW, FULL_WINDOW):
            window = context["windows"][window_name]
            for cost_bps in (0.0, 10.0, 20.0, 50.0):
                payloads = {}
                for leg, weights in (
                    ("cluster", cluster_weights[candidate.candidate_id]),
                    ("global", global_weights[global_key]),
                ):
                    payload = _performance_payload(
                        weights,
                        window,
                        f"cost_{role['selection_role']}_{leg}_{window_name}_{cost_bps}",
                        cost_bps=cost_bps,
                    )
                    payloads[leg] = payload
                    rows.append(
                        {
                            "selection_role": role["selection_role"],
                            "candidate_id": candidate.candidate_id,
                            "analysis_window": window_name,
                            "cost_bps_one_way": cost_bps,
                            "leg": leg,
                            **payload,
                        }
                    )
                rows.append(
                    {
                        "selection_role": role["selection_role"],
                        "candidate_id": candidate.candidate_id,
                        "analysis_window": window_name,
                        "cost_bps_one_way": cost_bps,
                        "leg": "cluster_minus_global",
                        "net_return_annualized": (
                            payloads["cluster"]["net_return_annualized"]
                            - payloads["global"]["net_return_annualized"]
                        ),
                        "gross_return_annualized": (
                            payloads["cluster"]["gross_return_annualized"]
                            - payloads["global"]["gross_return_annualized"]
                        ),
                        "sharpe_rf0": (
                            payloads["cluster"]["sharpe_rf0"]
                            - payloads["global"]["sharpe_rf0"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _component_attribution(
    selection: pd.DataFrame,
    context: Mapping[str, object],
    cluster_weights: Mapping[str, pd.DataFrame],
    global_weights: Mapping[tuple[str, float, str], pd.DataFrame],
    candidate_map: Mapping[str, CandidateSpec],
) -> pd.DataFrame:
    """Backtest static sleeves and signed sides as canonical standalone components."""
    selected = selection.loc[selection["selection_role"].eq("train_selected")].iloc[0]
    roles = {
        "owner_base": BASE_CANDIDATE.candidate_id,
        "train_selected": str(selected["candidate_id"]),
    }
    sleeve_panel = context["sleeve_panel"]
    rows = []
    for role, candidate_id in roles.items():
        candidate = candidate_map[candidate_id]
        global_key = (candidate.signal_id, candidate.q, candidate.weight_id)
        for leg, weights in (
            ("cluster", cluster_weights[candidate_id]),
            ("global", global_weights[global_key]),
        ):
            components = {
                **{
                    f"sleeve:{sleeve}": weights.where(sleeve_panel.eq(sleeve), 0.0)
                    for sleeve in SLEEVES
                },
                "side:long": weights.clip(lower=0.0),
                "side:short": weights.clip(upper=0.0),
            }
            for window_name in (EVALUATION_WINDOW, FULL_WINDOW):
                for component, component_weights in components.items():
                    payload = _performance_payload(
                        component_weights,
                        context["windows"][window_name],
                        f"component_{role}_{leg}_{component}_{window_name}",
                    )
                    rows.append(
                        {
                            "selection_role": role,
                            "candidate_id": candidate_id,
                            "analysis_window": window_name,
                            "leg": leg,
                            "component": component,
                            **payload,
                        }
                    )
    return pd.DataFrame(rows)


def _acceptance(
    performance: pd.DataFrame,
    diagnostics: pd.DataFrame,
    signal_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate timing, exact weights, and the owner-base payoff regression."""
    rows = []
    max_weight_error = float(diagnostics["maximum_exact_weight_error"].max())
    rows.append(
        {
            "check": "all_candidate_weight_and_exposure_errors",
            "measured": max_weight_error,
            "tolerance": WEIGHT_TOLERANCE,
            "status": "PASS" if max_weight_error <= WEIGHT_TOLERANCE else "FAIL",
        }
    )
    rows.append(
        {
            "check": "all_signal_timing_and_reconstruction_rows",
            "measured": int(signal_diagnostics["status"].eq("PASS").sum()),
            "tolerance": len(signal_diagnostics),
            "status": "PASS"
            if signal_diagnostics["status"].eq("PASS").all()
            else "FAIL",
        }
    )
    accepted_path = three._root() / "performance.csv"
    accepted = pd.read_csv(accepted_path, float_precision="round_trip")
    accepted = accepted.loc[
        accepted["universe"].eq("U2_BlackRock_funds")
        & accepted["signal"].eq(three.ROSAA)
    ].set_index("leg")
    current = performance.loc[
        performance["candidate_id"].eq(BASE_CANDIDATE.candidate_id)
        & performance["analysis_window"].eq(FULL_WINDOW)
    ].set_index("leg")
    regression_metrics = [
        "gross_return_annualized",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    ]
    errors = []
    for leg in ("cluster", "global"):
        errors.extend(
            abs(float(current.loc[leg, metric]) - float(accepted.loc[leg, metric]))
            for metric in regression_metrics
        )
    regression_error = max(errors)
    rows.append(
        {
            "check": "owner_base_regression_to_three_universe_run",
            "measured": regression_error,
            "tolerance": WEIGHT_TOLERANCE,
            "status": "PASS" if regression_error <= WEIGHT_TOLERANCE else "FAIL",
        }
    )
    result = pd.DataFrame(rows)
    if not result["status"].eq("PASS").all():
        raise AssertionError(result)
    return result


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the staged search, diagnostics, and selected-candidate repricing."""
    started = time.perf_counter()
    context = _context()
    marginal, marginal_tags = _marginal_candidates()
    candidate_map = {candidate.candidate_id: candidate for candidate in marginal}
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    cluster_weight_cache: dict[str, pd.DataFrame] = {}
    global_weight_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    global_performance_cache: dict[tuple[str, float, str, str], dict[str, float]] = {}

    phase_one_perf, phase_one_comparison, phase_one_diag = _run_candidates(
        marginal,
        context,
        score_cache,
        partition_cache,
        cluster_weight_cache,
        global_weight_cache,
        global_performance_cache,
    )
    interaction, marginal_selection = _top_interaction_candidates(
        phase_one_comparison, marginal_tags, candidate_map
    )
    candidate_map.update({candidate.candidate_id: candidate for candidate in interaction})
    if interaction:
        phase_two_perf, phase_two_comparison, phase_two_diag = _run_candidates(
            interaction,
            context,
            score_cache,
            partition_cache,
            cluster_weight_cache,
            global_weight_cache,
            global_performance_cache,
        )
    else:
        phase_two_perf = pd.DataFrame()
        phase_two_comparison = pd.DataFrame()
        phase_two_diag = pd.DataFrame()

    performance = pd.concat([phase_one_perf, phase_two_perf], ignore_index=True)
    comparison = pd.concat(
        [phase_one_comparison, phase_two_comparison], ignore_index=True
    ).drop_duplicates(["candidate_id", "analysis_window"])
    diagnostics = pd.concat([phase_one_diag, phase_two_diag], ignore_index=True).drop_duplicates(
        "candidate_id"
    )
    selection = _selection_table(comparison)
    drivers = _driver_table(comparison, diagnostics, selection)
    costs = _cost_sensitivity(
        selection,
        context,
        cluster_weight_cache,
        global_weight_cache,
        candidate_map,
    )
    components = _component_attribution(
        selection,
        context,
        cluster_weight_cache,
        global_weight_cache,
        candidate_map,
    )
    acceptance = _acceptance(
        performance, diagnostics, context["signal_diagnostics"]
    )
    grid_tags = pd.DataFrame(
        [
            {"candidate_id": candidate_id, "marginal_dimension": tag}
            for candidate_id, tags in marginal_tags.items()
            for tag in sorted(tags)
        ]
    )
    runtime = pd.DataFrame(
        [
            {
                "marginal_candidates": len(marginal),
                "interaction_candidates": len(interaction),
                "unique_candidates": int(comparison["candidate_id"].nunique()),
                "qis_performance_rows": len(performance),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    output = {
        "signal_diagnostics": context["signal_diagnostics"],
        "marginal_grid_tags": grid_tags,
        "marginal_finalist_selection": marginal_selection,
        "performance": performance,
        "comparison": comparison,
        "selection": selection,
        "weight_diagnostics": diagnostics,
        "driver_decomposition": drivers,
        "cost_sensitivity": costs,
        "component_attribution": components,
        "acceptance": acceptance,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _candidate_from_row(row: pd.Series) -> CandidateSpec:
    """Reconstruct a candidate specification from a persisted comparison row."""
    return CandidateSpec(
        signal_id=str(row["signal_id"]),
        frequency=str(row["frequency"]),
        span=int(row["span"]),
        q=float(row["q"]),
        weight_id=str(row["weight_id"]),
        construction=str(row["construction"]),
        stage=str(row["stage"]),
    )


def _build_weight_pair(
    candidate: CandidateSpec,
    context: Mapping[str, object],
    score_cache: dict[tuple[str, str, int], pd.DataFrame],
    partition_cache: dict[tuple[str, int], pd.DataFrame],
    global_cache: dict[tuple[str, float, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild one cluster book and its exact matched-global control."""
    dates = context["dates"]
    eligibility = context["eligibility"]
    sleeve_panel = context["sleeve_panel"]
    signal = context["signals"][candidate.signal_id]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("context dates are not a DatetimeIndex")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("context eligibility is not a DataFrame")
    if not isinstance(sleeve_panel, pd.DataFrame):
        raise AssertionError("context sleeve panel is not a DataFrame")
    cell = (candidate.frequency, candidate.span)
    if cell not in partition_cache:
        loader = context.get("partition_loader", funds._load_partition)
        partition_cache[cell] = loader(*cell)[0].reindex(
            index=dates, columns=eligibility.columns
        )
    clusters = partition_cache[cell]
    if int((eligibility & clusters.isna()).sum().sum()):
        raise AssertionError(f"eligible memberships missing for {cell}")
    groups = sleeves._hierarchical_groups(clusters, sleeve_panel)
    score_key = (candidate.signal_id, candidate.frequency, candidate.span)
    if score_key not in score_cache:
        source = score_within_clusters(
            raw_signal=signal["raw_source"],
            rolling_clusters=funds._panel_dict(clusters),
            min_cluster_size=CLUSTER_FALLBACK,
        )
        decision, _ = prod._asof_panel(source, dates)
        score_cache[score_key] = decision.reindex(
            index=dates, columns=eligibility.columns
        ).where(eligibility)
    target = _target_map(candidate.weight_id)
    cluster_weights, _ = _long_short_weights(
        score_cache[score_key],
        eligibility,
        sleeve_panel,
        groups,
        target,
        q=candidate.q,
        construction=candidate.construction,
    )
    global_key = (candidate.signal_id, candidate.q, candidate.weight_id)
    if global_key not in global_cache:
        global_cache[global_key], _ = _long_short_weights(
            signal["global"].reindex(index=dates).where(eligibility),
            eligibility,
            sleeve_panel,
            sleeve_panel,
            target,
            q=candidate.q,
            construction="asset_equal",
        )
    return cluster_weights, global_cache[global_key]


def _hybrid_comparison_row(
    *,
    candidate: CandidateSpec,
    variant: str,
    window_name: str,
    payload: Mapping[str, float],
    reference: pd.Series,
) -> dict[str, object]:
    """Compare one hybrid side substitution with its matched global book."""
    row: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "stage": candidate.stage,
        "hybrid_variant": variant,
        "analysis_window": window_name,
        "signal_id": candidate.signal_id,
        "frequency": candidate.frequency,
        "span": candidate.span,
        "q": candidate.q,
        "weight_id": candidate.weight_id,
        "construction": candidate.construction,
        "cost_bps_one_way": COST_BPS,
    }
    for metric in COMPARISON_METRICS:
        row[f"hybrid_{metric}"] = payload[metric]
        row[f"global_{metric}"] = reference[f"global_{metric}"]
        row[f"delta_{metric}"] = payload[metric] - reference[f"global_{metric}"]
    row["beats_global_net_return"] = row["delta_net_return_annualized"] > 0.0
    row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
    row["beats_global_both"] = (
        row["beats_global_net_return"] and row["beats_global_sharpe"]
    )
    return row


def run_hybrid_followup() -> Mapping[str, pd.DataFrame]:
    """Select and evaluate side-specific cluster hybrids without re-running the grid."""
    started = time.perf_counter()
    base_comparison = pd.read_csv(
        _root() / "comparison.csv", float_precision="round_trip"
    )
    unique_rows = base_comparison.drop_duplicates("candidate_id")
    candidates = {
        str(row["candidate_id"]): _candidate_from_row(row)
        for _, row in unique_rows.iterrows()
    }
    reference = base_comparison.set_index(["candidate_id", "analysis_window"])
    context = _context()
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    global_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    cluster_cache: dict[str, pd.DataFrame] = {}
    hybrid_cache: dict[tuple[str, str], pd.DataFrame] = {}
    training_rows = []
    acceptance_rows = []

    for number, candidate in enumerate(candidates.values(), start=1):
        cluster_weights, global_weights = _build_weight_pair(
            candidate, context, score_cache, partition_cache, global_cache
        )
        cluster_cache[candidate.candidate_id] = cluster_weights
        for variant in HYBRID_VARIANTS:
            hybrid = _hybrid_weights(
                cluster_weights,
                global_weights,
                variant,
                context["sleeve_panel"],
                _target_map(candidate.weight_id),
            )
            hybrid_cache[(candidate.candidate_id, variant)] = hybrid
            long_error = float(hybrid.clip(lower=0.0).sum(axis=1).sub(1.0).abs().max())
            short_error = float(
                (-hybrid.clip(upper=0.0)).sum(axis=1).sub(1.0).abs().max()
            )
            net_error = float(hybrid.sum(axis=1).abs().max())
            gross_error = float(hybrid.abs().sum(axis=1).sub(2.0).abs().max())
            maximum = max(long_error, short_error, net_error, gross_error)
            acceptance_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "hybrid_variant": variant,
                    "max_long_exposure_abs_error": long_error,
                    "max_short_exposure_abs_error": short_error,
                    "max_net_exposure_abs_error": net_error,
                    "max_gross_exposure_abs_error": gross_error,
                    "maximum_error": maximum,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS" if maximum <= WEIGHT_TOLERANCE else "FAIL",
                }
            )
            payload = _performance_payload(
                hybrid,
                context["windows"][TRAIN_WINDOW],
                f"hybrid_train_{number}_{variant}",
            )
            training_rows.append(
                _hybrid_comparison_row(
                    candidate=candidate,
                    variant=variant,
                    window_name=TRAIN_WINDOW,
                    payload=payload,
                    reference=reference.loc[(candidate.candidate_id, TRAIN_WINDOW)],
                )
            )
        if number % 20 == 0 or number == len(candidates):
            print(
                f"fund hybrid training screen: {number}/{len(candidates)} candidates",
                flush=True,
            )

    training = pd.DataFrame(training_rows).sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"],
        ascending=[False, False],
    )
    finalists = training.head(5)[["candidate_id", "hybrid_variant"]]
    owner_base = pd.DataFrame(
        {
            "candidate_id": [BASE_CANDIDATE.candidate_id] * len(HYBRID_VARIANTS),
            "hybrid_variant": HYBRID_VARIANTS,
        }
    )
    evaluated_keys = pd.concat([finalists, owner_base], ignore_index=True).drop_duplicates()
    evaluation_rows = []
    for _, key in evaluated_keys.iterrows():
        candidate_id = str(key["candidate_id"])
        variant = str(key["hybrid_variant"])
        candidate = candidates[candidate_id]
        weights = hybrid_cache[(candidate_id, variant)]
        for window_name in (EVALUATION_WINDOW, FULL_WINDOW):
            payload = _performance_payload(
                weights,
                context["windows"][window_name],
                f"hybrid_eval_{hash((candidate_id, variant, window_name))}",
            )
            evaluation_rows.append(
                _hybrid_comparison_row(
                    candidate=candidate,
                    variant=variant,
                    window_name=window_name,
                    payload=payload,
                    reference=reference.loc[(candidate_id, window_name)],
                )
            )
    comparison = pd.concat(
        [training, pd.DataFrame(evaluation_rows)], ignore_index=True
    )
    leader_key = tuple(training.iloc[0][["candidate_id", "hybrid_variant"]])
    selected = comparison.loc[
        comparison["candidate_id"].eq(leader_key[0])
        & comparison["hybrid_variant"].eq(leader_key[1])
    ].copy()
    selected.insert(0, "selection_role", "pre2018_selected_hybrid")
    selected["positive_net_edge_both_halves"] = bool(
        selected.set_index("analysis_window")
        .loc[[TRAIN_WINDOW, EVALUATION_WINDOW], "delta_net_return_annualized"]
        .gt(0.0)
        .all()
    )

    cost_rows = []
    selected_candidate = candidates[str(leader_key[0])]
    selected_global_key = (
        selected_candidate.signal_id,
        selected_candidate.q,
        selected_candidate.weight_id,
    )
    for window_name in (EVALUATION_WINDOW, FULL_WINDOW):
        for cost_bps in (0.0, 10.0, 20.0, 50.0):
            payloads = {}
            for leg, weights in (
                ("hybrid", hybrid_cache[leader_key]),
                ("global", global_cache[selected_global_key]),
            ):
                payloads[leg] = _performance_payload(
                    weights,
                    context["windows"][window_name],
                    f"hybrid_cost_{leg}_{window_name}_{cost_bps}",
                    cost_bps=cost_bps,
                )
            cost_rows.append(
                {
                    "candidate_id": leader_key[0],
                    "hybrid_variant": leader_key[1],
                    "analysis_window": window_name,
                    "cost_bps_one_way": cost_bps,
                    "hybrid_net_return_annualized": payloads["hybrid"][
                        "net_return_annualized"
                    ],
                    "global_net_return_annualized": payloads["global"][
                        "net_return_annualized"
                    ],
                    "delta_net_return_annualized": (
                        payloads["hybrid"]["net_return_annualized"]
                        - payloads["global"]["net_return_annualized"]
                    ),
                    "hybrid_sharpe_rf0": payloads["hybrid"]["sharpe_rf0"],
                    "global_sharpe_rf0": payloads["global"]["sharpe_rf0"],
                }
            )
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    summary = pd.DataFrame(
        [
            {
                "training_hybrid_rows": len(training),
                "training_net_edge_wins": int(training["beats_global_net_return"].sum()),
                "training_sharpe_wins": int(training["beats_global_sharpe"].sum()),
                "training_both_wins": int(training["beats_global_both"].sum()),
                "evaluated_finalists": len(evaluated_keys),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    output = {
        "hybrid_comparison": comparison,
        "hybrid_finalists": finalists,
        "hybrid_selection": selected,
        "hybrid_cost_sensitivity": pd.DataFrame(cost_rows),
        "hybrid_acceptance": acceptance,
        "hybrid_summary": summary,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _scheduled_performance_payload(
    weights: pd.DataFrame,
    window: Mapping[str, object],
    schedule: str,
    ticker: str,
) -> dict[str, float]:
    """Run QIS using only the selected rebalance rows and hold units between them."""
    dates = window["dates"]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("window dates are not a DatetimeIndex")
    scheduled_dates = _rebalance_dates(dates, schedule)
    net, gross = funds._backtest(
        window["prices"],
        weights.reindex(index=scheduled_dates),
        COST_BPS / 10000.0,
        ticker,
    )
    return sleeves._performance_payload(net, gross, window["ew_nav"])


def run_holding_period_followup() -> Mapping[str, pd.DataFrame]:
    """Test whether slower implementation rescues the base cluster-assisted spread."""
    context = _context()
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    global_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    group_cluster, global_weights = _build_weight_pair(
        BASE_CANDIDATE, context, score_cache, partition_cache, global_cache
    )
    asset_candidate = replace(BASE_CANDIDATE, construction="asset_equal")
    asset_cluster, _ = _build_weight_pair(
        asset_candidate, context, score_cache, partition_cache, global_cache
    )
    hybrid = _hybrid_weights(
        group_cluster,
        global_weights,
        "global_long_cluster_short",
        context["sleeve_panel"],
        _target_map(BASE_CANDIDATE.weight_id),
    )
    methods = {
        "global": global_weights,
        "cluster_group_equal": group_cluster,
        "cluster_asset_equal": asset_cluster,
        "hybrid_global_long_cluster_short": hybrid,
    }
    rows = []
    for window_name, window in context["windows"].items():
        for schedule in HOLDING_SCHEDULES:
            for method, weights in methods.items():
                payload = _scheduled_performance_payload(
                    weights,
                    window,
                    schedule,
                    f"holding_{window_name}_{schedule}_{method}",
                )
                rows.append(
                    {
                        "analysis_window": window_name,
                        "schedule": schedule,
                        "rebalance_dates": len(
                            _rebalance_dates(window["dates"], schedule)
                        ),
                        "method": method,
                        "signal_id": BASE_CANDIDATE.signal_id,
                        "frequency": BASE_CANDIDATE.frequency,
                        "span": BASE_CANDIDATE.span,
                        "q": BASE_CANDIDATE.q,
                        "weight_id": BASE_CANDIDATE.weight_id,
                        "cost_bps_one_way": COST_BPS,
                        **payload,
                    }
                )
    performance = pd.DataFrame(rows)
    global_rows = performance.loc[performance["method"].eq("global")].set_index(
        ["analysis_window", "schedule"]
    )
    comparisons = []
    for _, row in performance.loc[performance["method"].ne("global")].iterrows():
        reference = global_rows.loc[(row["analysis_window"], row["schedule"])]
        output = row.to_dict()
        for metric in COMPARISON_METRICS:
            output[f"global_{metric}"] = reference[metric]
            output[f"delta_{metric}"] = row[metric] - reference[metric]
        output["beats_global_net_return"] = output[
            "delta_net_return_annualized"
        ] > 0.0
        output["beats_global_sharpe"] = output["delta_sharpe_rf0"] > 0.0
        output["beats_global_both"] = (
            output["beats_global_net_return"]
            and output["beats_global_sharpe"]
        )
        comparisons.append(output)
    comparison = pd.DataFrame(comparisons)
    original = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    original = original.loc[
        original["candidate_id"].eq(BASE_CANDIDATE.candidate_id)
        & original["analysis_window"].eq(FULL_WINDOW)
    ].set_index("leg")
    monthly = performance.loc[
        performance["analysis_window"].eq(FULL_WINDOW)
        & performance["schedule"].eq("monthly")
        & performance["method"].isin(("global", "cluster_group_equal"))
    ].set_index("method")
    metrics = (
        "gross_return_annualized",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    )
    errors = []
    for method, leg in (("global", "global"), ("cluster_group_equal", "cluster")):
        errors.extend(
            abs(float(monthly.loc[method, metric]) - float(original.loc[leg, metric]))
            for metric in metrics
        )
    regression_error = max(errors)
    acceptance = pd.DataFrame(
        [
            {
                "check": "monthly_base_regression",
                "measured": regression_error,
                "tolerance": WEIGHT_TOLERANCE,
                "status": "PASS"
                if regression_error <= WEIGHT_TOLERANCE
                else "FAIL",
            },
            {
                "check": "declared_schedule_rows",
                "measured": len(performance),
                "tolerance": len(WINDOWS) * len(HOLDING_SCHEDULES) * len(methods),
                "status": "PASS"
                if len(performance)
                == len(WINDOWS) * len(HOLDING_SCHEDULES) * len(methods)
                else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    output = {
        "holding_period_performance": performance,
        "holding_period_comparison": comparison,
        "holding_period_acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def run_short_reversal_followup() -> Mapping[str, pd.DataFrame]:
    """Test the stable absolute-return signal leader with cluster-assisted shorts."""
    context = _context()
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    global_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    candidates = {
        construction: replace(
            BASE_CANDIDATE,
            signal_id=ABSOLUTE_SIGNAL_ID,
            construction=construction,
            stage="targeted_absolute_signal",
        )
        for construction in CONSTRUCTIONS
    }
    cluster_weights = {}
    global_weights = None
    for construction, candidate in candidates.items():
        cluster, global_book = _build_weight_pair(
            candidate, context, score_cache, partition_cache, global_cache
        )
        cluster_weights[construction] = cluster
        global_weights = global_book
    if global_weights is None:
        raise AssertionError("the targeted signal global book was not built")
    target = _target_map(BASE_CANDIDATE.weight_id)
    methods = {"global": global_weights}
    for construction, cluster in cluster_weights.items():
        methods[f"cluster_{construction}"] = cluster
        methods[f"hybrid_global_long_cluster_short_{construction}"] = (
            _hybrid_weights(
                cluster,
                global_weights,
                "global_long_cluster_short",
                context["sleeve_panel"],
                target,
            )
        )

    rows = []
    for window_name, window in context["windows"].items():
        for schedule in HOLDING_SCHEDULES:
            for method, weights in methods.items():
                payload = _scheduled_performance_payload(
                    weights,
                    window,
                    schedule,
                    f"short3_{window_name}_{schedule}_{method}",
                )
                rows.append(
                    {
                        "analysis_window": window_name,
                        "schedule": schedule,
                        "method": method,
                        "signal_id": ABSOLUTE_SIGNAL_ID,
                        "frequency": BASE_CANDIDATE.frequency,
                        "span": BASE_CANDIDATE.span,
                        "q": BASE_CANDIDATE.q,
                        "weight_id": BASE_CANDIDATE.weight_id,
                        "cost_bps_one_way": COST_BPS,
                        **payload,
                    }
                )
    performance = pd.DataFrame(rows)
    global_rows = performance.loc[performance["method"].eq("global")].set_index(
        ["analysis_window", "schedule"]
    )
    comparisons = []
    for _, row in performance.loc[performance["method"].ne("global")].iterrows():
        reference = global_rows.loc[(row["analysis_window"], row["schedule"])]
        output = row.to_dict()
        for metric in COMPARISON_METRICS:
            output[f"global_{metric}"] = reference[metric]
            output[f"delta_{metric}"] = row[metric] - reference[metric]
        output["beats_global_net_return"] = output[
            "delta_net_return_annualized"
        ] > 0.0
        output["beats_global_sharpe"] = output["delta_sharpe_rf0"] > 0.0
        output["beats_global_both"] = (
            output["beats_global_net_return"]
            and output["beats_global_sharpe"]
        )
        output["positive_net_return"] = output["net_return_annualized"] > 0.0
        comparisons.append(output)
    comparison = pd.DataFrame(comparisons)
    original = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    expected_id = replace(
        BASE_CANDIDATE, signal_id=ABSOLUTE_SIGNAL_ID
    ).candidate_id
    original = original.loc[
        original["candidate_id"].eq(expected_id)
        & original["analysis_window"].eq(FULL_WINDOW)
    ].set_index("leg")
    monthly = performance.loc[
        performance["analysis_window"].eq(FULL_WINDOW)
        & performance["schedule"].eq("monthly")
        & performance["method"].isin(("global", "cluster_group_equal"))
    ].set_index("method")
    metrics = (
        "gross_return_annualized",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    )
    errors = []
    for method, leg in (("global", "global"), ("cluster_group_equal", "cluster")):
        errors.extend(
            abs(float(monthly.loc[method, metric]) - float(original.loc[leg, metric]))
            for metric in metrics
        )
    regression_error = max(errors)
    acceptance = pd.DataFrame(
        [
            {
                "check": "short3_monthly_grid_regression",
                "measured": regression_error,
                "tolerance": WEIGHT_TOLERANCE,
                "status": "PASS"
                if regression_error <= WEIGHT_TOLERANCE
                else "FAIL",
            },
            {
                "check": "short3_declared_rows",
                "measured": len(performance),
                "tolerance": len(WINDOWS) * len(HOLDING_SCHEDULES) * len(methods),
                "status": "PASS"
                if len(performance)
                == len(WINDOWS) * len(HOLDING_SCHEDULES) * len(methods)
                else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    output = {
        "short3_performance": performance,
        "short3_comparison": comparison,
        "short3_acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def main() -> None:
    """Run the search and print the compact selection table."""
    output = run()
    columns = [
        "selection_role",
        "analysis_window",
        "signal_id",
        "frequency",
        "span",
        "q",
        "weight_id",
        "construction",
        "cluster_net_return_annualized",
        "global_net_return_annualized",
        "delta_net_return_annualized",
        "delta_sharpe_rf0",
    ]
    print(output["selection"][columns].to_string(index=False))
    print(output["acceptance"].to_string(index=False))


if __name__ == "__main__":
    main()
