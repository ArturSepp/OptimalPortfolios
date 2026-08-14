"""
input validation across the three minimum-tracking-error entry points.

Every guard here stands between a caller mistake and a solve that succeeds on the wrong
problem. That is the specific hazard of a TRE objective: ``(w - w_b)' Sigma (w - w_b)`` is
perfectly well-defined for a benchmark that is silently misaligned, truncated, or carrying a
NaN filled in as zero -- CVXPY returns weights, ``validate_solution`` accepts them, and the
portfolio tracks something nobody asked for. So each of these raises rather than repairing.

The three layers validate different things and are tested separately for that reason. The
rolling layer owns benchmark *shape* across time (Series broadcast vs DataFrame forward-fill);
the wrapper owns the covariance's own well-formedness after NaN filtering; the CVXPY layer
re-checks the raw array it is handed, because it is public and callable without either wrapper.

The solver-failure fallback is covered too: a ``SolverError`` must come back as a non-accepted
outcome with a recorded status, never as an exception escaping into the backtest loop.
"""
# packages
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.general.minimum_tracking_error import (
    cvx_minimise_tracking_error,
    rolling_minimise_tracking_error,
    wrapper_minimise_tracking_error,
)

TICKERS = pd.Index(['A', 'B', 'C', 'D'])
DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30'])


def covar_frame() -> pd.DataFrame:
    """A well-conditioned 4-asset covariance in the canonical ticker order."""
    vols = np.array([0.20, 0.12, 0.08, 0.05])
    corr = np.array([
        [1.00, 0.30, 0.10, 0.00],
        [0.30, 1.00, 0.20, 0.10],
        [0.10, 0.20, 1.00, 0.25],
        [0.00, 0.10, 0.25, 1.00],
    ])
    return pd.DataFrame(np.outer(vols, vols) * corr, index=TICKERS, columns=TICKERS)


def benchmark() -> pd.Series:
    """An equal-weight benchmark over the four tickers."""
    return pd.Series(0.25, index=TICKERS)


def make_constraints(**overrides) -> Constraints:
    """Long-only bounds carrying the benchmark, with optional field overrides."""
    kwargs = dict(min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS),
                  benchmark_weights=benchmark())
    kwargs.update(overrides)
    return Constraints(**kwargs)


def prices_frame() -> pd.DataFrame:
    """A price panel spanning the rebalancing dates, used only for column alignment."""
    return pd.DataFrame(100.0, index=pd.date_range('2024-01-31', periods=8, freq='ME'),
                        columns=TICKERS)


# --------------------------------------------------------------------------- #
# rolling_minimise_tracking_error
# --------------------------------------------------------------------------- #
def test_an_empty_rebalancing_schedule_returns_an_empty_frame() -> None:
    """No covariance dates means no weights -- an empty frame, not an error or a NaN row."""
    weights = rolling_minimise_tracking_error(prices=prices_frame(),
                                              constraints=make_constraints(),
                                              benchmark_weights=benchmark(),
                                              covar_dict={})
    assert weights.empty
    assert list(weights.columns) == list(TICKERS)
    assert isinstance(weights.index, pd.DatetimeIndex)


def test_a_static_benchmark_series_is_broadcast_over_every_rebalance() -> None:
    """A Series benchmark applies unchanged at each date; a DataFrame is forward-filled."""
    covar_dict = {date: covar_frame() for date in DATES}
    weights = rolling_minimise_tracking_error(prices=prices_frame(),
                                              constraints=make_constraints(),
                                              benchmark_weights=benchmark(),
                                              covar_dict=covar_dict)
    assert list(weights.index) == list(DATES)
    assert not weights.isna().any().any()


def test_a_time_varying_benchmark_frame_is_forward_filled() -> None:
    """A DataFrame benchmark is reindexed onto the rebalancing dates by ffill."""
    frame = pd.DataFrame([[0.25, 0.25, 0.25, 0.25]],
                         index=pd.DatetimeIndex(['2024-01-31']), columns=TICKERS)
    covar_dict = {date: covar_frame() for date in DATES}
    weights = rolling_minimise_tracking_error(prices=prices_frame(),
                                              constraints=make_constraints(),
                                              benchmark_weights=frame,
                                              covar_dict=covar_dict)
    assert list(weights.index) == list(DATES)


def test_inclusion_indicators_are_aligned_onto_the_rebalancing_dates() -> None:
    """An eligibility panel is forward-filled and column-aligned like the benchmark."""
    indicators = pd.DataFrame(1.0, index=pd.DatetimeIndex(['2024-01-31']), columns=TICKERS)
    covar_dict = {date: covar_frame() for date in DATES}
    weights = rolling_minimise_tracking_error(prices=prices_frame(),
                                              constraints=make_constraints(),
                                              benchmark_weights=benchmark(),
                                              covar_dict=covar_dict,
                                              inclusion_indicators=indicators)
    assert list(weights.index) == list(DATES)


def test_a_benchmark_that_is_neither_series_nor_frame_raises() -> None:
    """A dict or array benchmark would broadcast in some silently wrong way."""
    with pytest.raises(TypeError, match='must be a Series or DataFrame'):
        rolling_minimise_tracking_error(prices=prices_frame(),
                                        constraints=make_constraints(),
                                        benchmark_weights={'A': 0.25},
                                        covar_dict={DATES[0]: covar_frame()})


def test_a_benchmark_missing_a_ticker_raises_rather_than_filling_zero() -> None:
    """Reindexing onto the price columns introduces NaN, which must not become a 0% target."""
    partial = benchmark().drop('D')
    with pytest.raises(ValueError, match='finite and complete at every rebalance'):
        rolling_minimise_tracking_error(prices=prices_frame(),
                                        constraints=make_constraints(),
                                        benchmark_weights=partial,
                                        covar_dict={DATES[0]: covar_frame()})


# --------------------------------------------------------------------------- #
# wrapper_minimise_tracking_error
# --------------------------------------------------------------------------- #
def test_the_wrapper_solves_and_returns_an_accepted_outcome() -> None:
    """The happy path returns weights over the ticker index and an accepted outcome."""
    weights, outcome = wrapper_minimise_tracking_error(pd_covar=covar_frame(),
                                                       benchmark_weights=benchmark(),
                                                       constraints=make_constraints())
    assert list(weights.index) == list(TICKERS)
    assert outcome.accepted


@pytest.mark.parametrize('bad_covar', [
    pd.DataFrame(),
    'not a frame',
])
def test_a_missing_or_empty_covariance_raises(bad_covar) -> None:
    """An empty frame is the shape a caller gets from slicing on a date with no estimate."""
    with pytest.raises(ValueError, match='must be a non-empty DataFrame'):
        wrapper_minimise_tracking_error(pd_covar=bad_covar,
                                        benchmark_weights=benchmark(),
                                        constraints=make_constraints())


def test_a_covariance_whose_rows_and_columns_disagree_raises() -> None:
    """Row/column order must match: a transposed or reordered covariance is not detectable later."""
    covar = covar_frame()
    covar.columns = pd.Index(['B', 'A', 'C', 'D'])
    with pytest.raises(ValueError, match='index and columns must match'):
        wrapper_minimise_tracking_error(pd_covar=covar,
                                        benchmark_weights=benchmark(),
                                        constraints=make_constraints())


def test_a_non_finite_covariance_entry_raises() -> None:
    """An infinity survives NaN filtering and would poison the objective."""
    covar = covar_frame()
    covar.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match='pd_covar must be finite'):
        wrapper_minimise_tracking_error(pd_covar=covar,
                                        benchmark_weights=benchmark(),
                                        constraints=make_constraints())


def test_an_asymmetric_covariance_raises() -> None:
    """Asymmetry means the quadratic form is not the risk the caller thinks it is."""
    covar = covar_frame()
    covar.iloc[0, 1] += 0.05
    with pytest.raises(ValueError, match='pd_covar must be symmetric'):
        wrapper_minimise_tracking_error(pd_covar=covar,
                                        benchmark_weights=benchmark(),
                                        constraints=make_constraints())


def test_a_benchmark_with_a_nan_raises_at_the_wrapper() -> None:
    """After NaN filtering the benchmark must still be complete over the surviving assets."""
    bad = benchmark().copy()
    bad['B'] = np.nan
    with pytest.raises(ValueError, match='benchmark_weights must be finite and complete'):
        wrapper_minimise_tracking_error(pd_covar=covar_frame(),
                                        benchmark_weights=bad,
                                        constraints=make_constraints())


def test_incomplete_starting_weights_raise() -> None:
    """``weights_0`` drives turnover terms, so a NaN there silently frees the turnover budget."""
    weights_0 = benchmark().copy()
    weights_0['C'] = np.nan
    with pytest.raises(ValueError, match='weights_0 must be finite and complete'):
        wrapper_minimise_tracking_error(pd_covar=covar_frame(),
                                        benchmark_weights=benchmark(),
                                        constraints=make_constraints(),
                                        weights_0=weights_0)


# --------------------------------------------------------------------------- #
# cvx_minimise_tracking_error
# --------------------------------------------------------------------------- #
def test_the_cvx_layer_solves_a_well_posed_problem() -> None:
    """Called directly with a raw array, the solver returns an accepted outcome."""
    outcome = cvx_minimise_tracking_error(covar=covar_frame().to_numpy(),
                                          constraints=make_constraints())
    assert outcome.accepted


def test_the_cvx_layer_rejects_a_non_square_covariance() -> None:
    """This entry point is public, so it re-checks what the wrapper would have caught."""
    with pytest.raises(ValueError, match='covar must be a square matrix'):
        cvx_minimise_tracking_error(covar=np.ones((3, 4)), constraints=make_constraints())


def test_the_cvx_layer_rejects_a_non_finite_covariance() -> None:
    """A NaN reaching CVXPY produces an unbounded or infeasible status, not a clear error."""
    covar = covar_frame().to_numpy().copy()
    covar[2, 2] = np.nan
    with pytest.raises(ValueError, match='covar must be finite'):
        cvx_minimise_tracking_error(covar=covar, constraints=make_constraints())


def test_the_cvx_layer_requires_benchmark_weights_on_the_constraints() -> None:
    """There is no benchmark argument here: it travels on the constraints or not at all."""
    with pytest.raises(ValueError, match='constraints.benchmark_weights is required'):
        cvx_minimise_tracking_error(covar=covar_frame().to_numpy(),
                                    constraints=make_constraints(benchmark_weights=None))


def test_a_benchmark_of_the_wrong_length_raises() -> None:
    """A benchmark over a different universe would broadcast or truncate silently."""
    short = pd.Series(1.0 / 3.0, index=pd.Index(['A', 'B', 'C']))
    with pytest.raises(ValueError, match='length does not match covariance'):
        cvx_minimise_tracking_error(covar=covar_frame().to_numpy(),
                                    constraints=make_constraints(benchmark_weights=short))


def test_a_solver_failure_becomes_a_rejected_outcome_not_an_exception(monkeypatch) -> None:
    """A SolverError must not escape into the backtest loop.

    The rolling layer calls this once per rebalancing date; an exception there aborts the whole
    backtest, whereas a recorded non-accepted outcome lets the caller fall back for that date.
    """
    def fail(self, **kwargs):
        """Stand in for a solver that errors out."""
        raise cvx.error.SolverError('solver failed')

    monkeypatch.setattr(cvx.Problem, 'solve', fail)
    outcome = cvx_minimise_tracking_error(covar=covar_frame().to_numpy(),
                                          constraints=make_constraints())
    assert not outcome.accepted
    assert outcome.status == 'solver_error'
