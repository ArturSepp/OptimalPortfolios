"""Public momentum alpha, OP constrained allocation and QIS executed portfolio accounting."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import factormodels as fm
import optimalportfolios as op
import qis

from examples.oss_portfolio_pipeline.config import PipelineConfig
from examples.oss_portfolio_pipeline.covariance import RollingCovariances
from examples.oss_portfolio_pipeline.data import external_directory, validate_prices
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs


@dataclass(frozen=True)
class BacktestConfig:
    """Illustrative public policy, fixed before observing results; rates are fractional."""

    tracking_error: float = 0.03
    max_weight: float = 0.30
    group_band: float = 0.10
    turnover_limit: float = 0.50
    transaction_cost: float = 0.0005
    implementation_lag: int = 0
    momentum_span: int = 12
    volatility_span: int = 13
    min_cluster_size: int = 5
    solver: str = 'MOSEK'

    def __post_init__(self) -> None:
        """Validate explicit risk, trading and signal settings without silently clipping them."""
        for name in ('tracking_error', 'max_weight', 'group_band', 'turnover_limit'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite in (0, 1]')
        if not np.isfinite(self.transaction_cost) or not 0 <= self.transaction_cost < 0.01:
            raise ValueError('transaction_cost must be finite in [0, 0.01)')
        for name, minimum in (('implementation_lag', 0), ('momentum_span', 2),
                              ('volatility_span', 2), ('min_cluster_size', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.solver not in ('MOSEK', 'CLARABEL'):
            raise ValueError('solver must be MOSEK or CLARABEL')


@dataclass
class PortfolioBacktest:
    """Completed OP decisions, QIS portfolios, alpha inputs and auditable risk results."""

    executed: op.ExecutedRollingResult
    benchmark: qis.PortfolioData
    gross_portfolio: qis.PortfolioData
    gross_benchmark: qis.PortfolioData
    navs: pd.DataFrame
    raw_alpha: pd.DataFrame
    alpha_scores: pd.DataFrame
    risk_model: qis.RiskModel
    diagnostics: pd.DataFrame
    constraint_audit: pd.DataFrame
    realised_tracking_error: pd.Series
    te_ir: pd.DataFrame
    settings: BacktestConfig


def build_alpha_scores(prices: pd.DataFrame, risk: RollingCovariances,
                       settings: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score OP's causal excess momentum within FM clusters, excluding cash from scoring.

    Scores are dimensionless rankings, not calibrated forecasts of annual returns.
    No full-sample parameters, fitted regression intercepts or future clusters are used.
    Both momentum horizons use the complete downloaded prehistory, independently of
    the shorter factor-estimation panel. Completed calendar MEs use observed closes.
    """
    prices = validate_prices(prices, tuple(risk.asset_excess_returns.columns))
    relative_prices = prices.div(prices['BIL'], axis=0).drop(columns='BIL')
    returns = qis.to_returns(relative_prices, freq='ME', is_log_returns=True,
                             include_start_date=False, include_end_date=False)
    if not np.isfinite(returns.iloc[1:]).all().all():
        raise ValueError('Alpha inputs need complete monthly excess returns after the anchor')
    navs = qis.returns_to_nav(returns, is_log_returns=True, init_period=0)
    _, raw = op.compute_momentum_alpha(
        prices=navs, returns_freq='ME', long_span=settings.momentum_span,
        short_span=None, vol_span=settings.volatility_span,
        mean_adj_type=qis.MeanAdjType.NONE)
    raw = raw.reindex(risk.schedule.index)
    if not np.isfinite(raw).all().all():
        raise ValueError('Momentum warm-up is incomplete at a model date')
    clusters = fm.extract_rolling_clusters(risk.model, assets=returns.columns)
    if set(clusters) != set(raw.index):
        raise ValueError('Every alpha observation needs its completed FM cluster assignment')
    scores = fm.prepare_cluster_scores(
        raw_signal=raw, rolling_clusters=clusters,
        min_cluster_size=settings.min_cluster_size, pooling_type='none')
    if not np.isfinite(scores).all().all():
        raise ValueError('Cluster scoring returned nonfinite alpha scores')
    raw['BIL'], scores['BIL'] = 0.0, 0.0
    raw = raw.reindex(columns=risk.asset_excess_returns.columns)
    scores = scores.reindex(columns=raw.columns)
    raw.index = scores.index = pd.DatetimeIndex(risk.schedule.observation_date)
    return raw, scores


def portfolio_constraints(inputs: PipelineInputs, settings: BacktestConfig) -> op.Constraints:
    """Create long-only policy with benchmark-relative economic sleeve bands and hard TE."""
    benchmark = inputs.benchmark_weights
    if benchmark.max() > settings.max_weight:
        raise ValueError('max_weight must admit the fixed benchmark')
    loadings = pd.get_dummies(inputs.universe.economic_group, dtype=float)
    group_weights = loadings.T @ benchmark
    groups = op.GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=(group_weights - settings.group_band).clip(lower=0.),
        group_max_allocation=(group_weights + settings.group_band).clip(upper=1.))
    return op.Constraints(
        is_long_only=True, min_exposure=1., max_exposure=1.,
        min_weights=pd.Series(0., index=benchmark.index),
        max_weights=pd.Series(settings.max_weight, index=benchmark.index),
        benchmark_weights=benchmark, tracking_err_vol_constraint=settings.tracking_error,
        turnover_constraint=settings.turnover_limit, group_lower_upper_constraints=groups,
        constraint_enforcement_type=op.ConstraintEnforcementType.FORCED_CONSTRAINTS)


# op.round_weights_to_pct quantises allocations for display; it does not reconcile solver
# round-off with QIS execution. Bound the correction and re-audit using OP's public evaluator.
def execution_outcome(outcome: op.OptimizationOutcome) -> op.OptimizationOutcome:
    """Reconcile at most one part per million of solver round-off, then recheck all policy rows."""
    values = outcome.weights
    if (not np.isfinite(values).all() or values.min() < -1e-8
            or abs(values.sum() - 1.) > 1e-6):
        raise RuntimeError('Solver weights require more than a bounded round-off correction')
    target = np.maximum(values, 0.)
    target = target / target.sum()
    if np.abs(target - values).sum() > 1e-6:
        raise RuntimeError('Execution correction exceeds one part per million of portfolio weight')
    residuals = op.evaluate_constraint_residuals(
        weights=target, constraints=outcome.constraints,
        covar_factorization=outcome.covar_factorization, tolerance=1e-6)
    corrected = replace(outcome, weights=target, constraint_residuals=residuals)
    if not corrected.compliant:
        raise RuntimeError('Execution weights fail the post-correction constraint audit')
    return corrected


# OP's rolling_maximise_alpha_over_tre filters the zero-variance BIL row and hides outcomes.
# Use OP's public single-date solver in its QIS state driver to retain cash and audit each solve.
def run_backtest(inputs: PipelineInputs, risk: RollingCovariances, config: PipelineConfig,
                 settings: BacktestConfig | None = None) -> PortfolioBacktest:
    """Maximise active alpha under a TE budget and execute ME targets through QIS.

    Zero lag is the stated same-model-close convention. Positive lag moves entry prices
    for both portfolios; constraints are assessed at the decision close. A rejected or
    noncompliant solve raises instead of being presented as an optimised allocation.
    """
    settings = settings or BacktestConfig()
    if settings.solver not in cp.installed_solvers():
        raise RuntimeError(f'{settings.solver} is unavailable; select a solver explicitly')
    columns = inputs.universe.index
    prices = validate_prices(inputs.prices, tuple(columns))
    prices = prices.loc[(prices.index >= config.backtest_start) & (prices.index < config.end)]
    if prices.empty or prices.index[0] != pd.Timestamp(config.backtest_start):
        raise ValueError('Backtest prices must begin at the configured inception close')
    covariances = risk.covariances
    dates = pd.DatetimeIndex(covariances)
    if (dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates.isin(prices.index).all()):
        raise ValueError('Covariances must have unique ordered observed decision dates')
    if not dates.equals(inputs.benchmark_targets.index):
        raise ValueError('Risk and benchmark must share the complete monthly schedule')
    if not all(matrix.index.equals(columns) and matrix.columns.equals(columns)
               for matrix in covariances.values()):
        raise ValueError('Covariance axes must match the complete ordered investment universe')
    # A terminal close has no subsequent return to backtest. Retain its model in risk,
    # but submit only decisions whose executions have a later valuation observation.
    positions = prices.index.get_indexer(dates) + settings.implementation_lag
    decisions = dates[positions < len(prices) - 1]
    if decisions.empty or decisions[0] != pd.Timestamp(config.backtest_start):
        raise ValueError('Need a tradable inception and at least one subsequent price')
    raw, scores = build_alpha_scores(inputs.prices, risk, settings)
    policy = portfolio_constraints(inputs, settings)
    risk_model = fm.build_risk_model({date: covariances[date] for date in decisions})
    diagnostic_rows, audits = {}, {}

    def decide(history: pd.DataFrame, state: op.ExecutedState):
        """Solve with the dated risk/alpha and QIS holdings before the proposed trade."""
        date = state.date
        if history.index[-1] != date:
            raise ValueError('Decision history extends beyond its cutoff')
        # Initial purchase invests new cash; subsequent L1 budgets use actual holdings.
        previous = None if date == decisions[0] else state.weights
        constraints = policy.copy(weights_0=previous)
        outcome = op.cvx_maximise_alpha_over_tre(
            covar=covariances[date].to_numpy(), alphas=scores.loc[date].to_numpy(),
            constraints=constraints, solver=settings.solver, context=str(date.date()),
            factorize_covar=True, diagnose=True)
        if not outcome.accepted or not outcome.compliant:
            raise RuntimeError(f'{date.date()}: optimisation failed ({outcome.status}): '
                               f'{outcome.reason}; no fallback portfolio accepted')
        raw_weights = outcome.weights.copy()
        outcome = execution_outcome(outcome)
        target = pd.Series(outcome.weights, index=columns)
        audits[date] = outcome.residuals_frame()
        diagnostic_rows[date] = {
            'solver': outcome.solver, 'status': outcome.status, 'accepted': outcome.accepted,
            'compliant': outcome.compliant,
            'raw_weight_sum': float(raw_weights.sum()),
            'execution_adjustment_l1': float(np.abs(outcome.weights - raw_weights).sum()),
            'ex_ante_tracking_error': risk_model.compute_tre_at_date(
                inputs.benchmark_weights, target, date),
            'active_alpha_score': float(scores.loc[date] @ (target - inputs.benchmark_weights)),
        }
        return target, outcome

    executed = op.backtest_with_qis_state(
        prices, decisions, decide, implementation_lag=settings.implementation_lag,
        transaction_costs=settings.transaction_cost, initial_nav=100.)
    benchmark_targets = inputs.benchmark_targets.loc[decisions]
    common = dict(prices=prices, weight_implementation_lag=settings.implementation_lag,
                  initial_nav=100.)
    benchmark = qis.backtest_model_portfolio(
        **common, weights=benchmark_targets, rebalancing_costs=settings.transaction_cost,
        ticker='Benchmark (net)')
    gross = qis.backtest_model_portfolio(
        **common, weights=executed.targets, rebalancing_costs=0., ticker='Active (gross)')
    gross_benchmark = qis.backtest_model_portfolio(
        **common, weights=benchmark_targets, rebalancing_costs=0., ticker='Benchmark (gross)')
    navs = pd.concat({
        'Active (net)': executed.portfolio.nav, 'Benchmark (net)': benchmark.nav,
        'Active (gross)': gross.nav, 'Benchmark (gross)': gross_benchmark.nav}, axis=1)
    realised_te = qis.compute_ewma_realised_tracking_error(
        navs['Active (net)'], navs['Benchmark (net)'], ewma_span=36,
        freq='ME', is_log_returns=False)
    monthly = qis.to_returns(navs, freq='ME', is_log_returns=False, drop_first=True)
    differences = pd.DataFrame({
        basis: monthly[f'Active ({basis})'] - monthly[f'Benchmark ({basis})']
        for basis in ('net', 'gross')})
    if len(differences) >= 3:
        te, ir = qis.compute_te_ir_errors(differences)
    else:
        # QIS infers annualisation from dates; fewer than three ME points cannot
        # establish that cadence. Do not present a daily-scaled short-sample statistic.
        te = pd.Series(np.nan, index=differences.columns, name='TE')
        ir = pd.Series(np.nan, index=differences.columns, name='IR')
    return PortfolioBacktest(
        executed, benchmark, gross, gross_benchmark, navs, raw, scores, risk_model,
        pd.DataFrame.from_dict(diagnostic_rows, orient='index'),
        pd.concat(audits, names=['decision_date', 'constraint_row']),
        realised_te, pd.concat([te, ir], axis=1), settings)


def save_backtest(result: PortfolioBacktest, output_dir: Path | str) -> Path:
    """Persist QIS results, OP decisions and audits with a hash link to the risk bundle."""
    output = external_directory(output_dir)
    covariance_manifest = output / 'covariance_manifest.json'
    if not covariance_manifest.is_file():
        raise ValueError('Save the covariance bundle before the portfolio backtest')
    portfolio = result.executed.portfolio
    tables = {
        'portfolio_navs.csv': result.navs,
        'portfolio_targets.csv': result.executed.targets,
        'portfolio_realised_weights.csv': portfolio.weights,
        'portfolio_units.csv': portfolio.units,
        'portfolio_trade_costs.csv': portfolio.realized_costs,
        'benchmark_net_realised_weights.csv': result.benchmark.weights,
        'benchmark_trade_costs.csv': result.benchmark.realized_costs,
        'portfolio_turnover.csv': portfolio.get_turnover(is_agg=True, roll_period=None),
        'benchmark_turnover.csv': result.benchmark.get_turnover(is_agg=True, roll_period=None),
        'alpha_raw_momentum.csv': result.raw_alpha, 'alpha_scores.csv': result.alpha_scores,
        'optimisation_diagnostics.csv': result.diagnostics,
        'constraint_audit.csv': result.constraint_audit,
        'execution_events.csv': result.executed.events,
        'pretrade_weights.csv': pd.DataFrame(
            {state.date: state.weights for state in result.executed.states}).T,
        'realised_tracking_error.csv': result.realised_tracking_error,
        'tracking_error_information_ratio.csv': result.te_ir,
    }
    hashes = {}
    for name, table in tables.items():
        content = table.to_csv(float_format='%.17g').encode('utf-8')
        (output / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'signal_history': 'complete downloaded adjusted prices / BIL; completed ME log returns',
        'covariance_manifest_sha256': hashlib.sha256(covariance_manifest.read_bytes()).hexdigest(),
        'file_sha256': hashes, 'settings': asdict(result.settings),
        'objective': 'maximise alpha_score dot active_weights under hard annual TE cap',
        'alpha': 'OP excess momentum; FM/FL causal cluster scores; cash score zero',
        'alpha_units': 'dimensionless cross-sectional scores, not annual expected returns',
        'policy': 'fully invested long-only; max ETF weight; economic group benchmark +/- band',
        'turnover': 'L1 target minus QIS actual pretrade weights; initial funding exempt',
        'execution_correction': 'at most 1e-6 L1 weight change; all OP constraints re-audited',
        'execution': 'ME observed close; configured observation lag for both portfolios',
        'costs': 'fraction of bought/sold notional, including initial purchase; both portfolios',
        'gross_comparison': 'replay the same net-policy targets without costs; not reoptimised',
        'realised_risk': 'completed ME simple-return differentials; 36-month EWMA TE',
        'summary': 'whole-sample simple-return TE/IR; QIS annualisation inferred from ME grid',
        'versions': {name: version(name) for name in
                     ('optimalportfolios', 'factormodels', 'factorlasso', 'qis', 'cvxpy',
                      result.settings.solver.lower())},
    }
    (output / 'backtest_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return output
