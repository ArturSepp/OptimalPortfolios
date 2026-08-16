"""
the three signal-backtest factsheet builders.

``compute_signal_scores`` is covered next door in ``cluster_scoring_test.py``; what is left
here is the layer that turns a score panel into portfolios and a factsheet. These are report
generators, so the assertions are about the wiring rather than about the pictures: which legs
get built, in what order, and whether the caller's parameters actually reach the signal
computation underneath.

That last point is the one worth spending assertions on. Each of these functions takes six or
more span parameters and threads them through ``compute_signal_scores`` by keyword; a
parameter dropped on the way through produces a perfectly good factsheet of the wrong
backtest, and nothing downstream notices. ``cross_backtest_alpha_signals`` is the sharpest
case -- its whole purpose is to vary one span while holding the others fixed, so the legs it
builds are checked against the sweep values by name.

The equal-weight leg is asserted to come *first* in the multi and cross backtests and *second*
in the single-signal one, because those orderings are not the same and both are load-bearing
for how the factsheet labels its benchmark.
"""
# packages
import inspect

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.alphas.backtest_alphas import (
    AlphaSignal,
    CrossBacktestParam,
    backtest_alpha_signals,
    compute_signal_scores,
    cross_backtest_alpha_signals,
    multi_backtest_alpha_signals,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data


@pytest.fixture(autouse=True)
def close_figures():
    """Factsheets open many figures; drop them so the Agg registry does not grow."""
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def universe() -> dict:
    """A short window of the committed monthly universe plus the report inputs."""
    data = load_multiasset_data()
    prices = data.prices.loc['2014':'2019', :]
    group_data = data.group_data
    return dict(prices=prices,
                group_data=group_data,
                group_order=list(pd.unique(group_data)),
                rebalancing_costs=pd.Series(0.0010, index=prices.columns),
                benchmark_prices=prices.iloc[:, :1],
                time_period=qis.TimePeriod('2015-01-01', '2019-12-31'))


def test_classic_parameters_are_appended_to_existing_public_signatures() -> None:
    """Adding classic momentum must not shift any existing positional argument."""
    existing_parameters = {
        compute_signal_scores: [
            'prices', 'alpha_signal', 'group_data', 'benchmark_price', 'returns_freq',
            'mom_long_span', 'mom_short_span', 'beta_span', 'vol_span', 'momentum_span',
        ],
        backtest_alpha_signals: [
            'prices', 'group_data', 'group_order', 'rebalancing_costs', 'benchmark_prices',
            'time_period', 'alpha_signal', 'mom_long_span', 'mom_short_span', 'beta_span',
            'momentum_span', 'rebalancing_freq', 'returns_freq',
        ],
        multi_backtest_alpha_signals: [
            'prices', 'group_data', 'group_order', 'rebalancing_costs', 'benchmark_prices',
            'time_period', 'mom_long_span', 'mom_short_span', 'beta_span', 'momentum_span',
            'rebalancing_freq', 'returns_freq',
        ],
    }
    for function, existing in existing_parameters.items():
        parameters = list(inspect.signature(function).parameters)
        assert parameters == existing + [
            'classic_lookback_periods', 'classic_skip_periods'
        ]


# --------------------------------------------------------------------------- #
# backtest_alpha_signals
# --------------------------------------------------------------------------- #
def test_a_single_signal_backtest_renders_a_factsheet(universe: dict) -> None:
    """The strategy-vs-benchmark factsheet builds for the default momentum signal."""
    figs = backtest_alpha_signals(alpha_signal=AlphaSignal.MOMENTUM, **universe)
    assert len(figs) > 0
    assert all(isinstance(fig, plt.Figure) for fig in figs)


@pytest.mark.parametrize('alpha_signal', list(AlphaSignal))
def test_every_signal_can_be_backtested(universe: dict, alpha_signal: AlphaSignal) -> None:
    """Every registered signal reaches the backtester, composites included.

    The two composite signals forward their smoothing span to ``compute_residual_momentum_alpha``
    as ``long_span``; passing it as ``momentum_span`` used to raise TypeError, so these signals
    never ran at all. Parametrising over the whole enum is what keeps that from recurring.
    """
    figs = backtest_alpha_signals(alpha_signal=alpha_signal, **universe)
    assert len(figs) > 0


def test_the_single_signal_backtest_names_its_legs_signal_then_benchmark(universe: dict,
                                                                         monkeypatch) -> None:
    """The signal leg is built first and the equal-weight benchmark second."""
    tickers = []
    original = qis.backtest_model_portfolio

    def recording(**kwargs):
        """Record each leg's ticker and delegate to the real backtester."""
        tickers.append(kwargs['ticker'])
        return original(**kwargs)

    monkeypatch.setattr(qis, 'backtest_model_portfolio', recording)
    backtest_alpha_signals(alpha_signal=AlphaSignal.LOW_BETA, **universe)
    assert tickers == ['LowBeta', 'Equal Weight']


# --------------------------------------------------------------------------- #
# multi_backtest_alpha_signals
# --------------------------------------------------------------------------- #
def test_the_multi_backtest_builds_one_leg_per_signal_plus_equal_weight(universe: dict,
                                                                        monkeypatch) -> None:
    """Every AlphaSignal becomes a leg, with equal weight built first."""
    tickers = []
    original = qis.backtest_model_portfolio

    def recording(**kwargs):
        """Record each leg's ticker and delegate to the real backtester."""
        tickers.append(kwargs['ticker'])
        return original(**kwargs)

    monkeypatch.setattr(qis, 'backtest_model_portfolio', recording)
    figs = multi_backtest_alpha_signals(**universe)
    assert tickers == ['Equal Weight'] + [signal.value for signal in AlphaSignal]
    assert len(figs) > 0


# --------------------------------------------------------------------------- #
# cross_backtest_alpha_signals
# --------------------------------------------------------------------------- #
SPAN_VALUES = [3, 6, 12, 18, 24, 36, 60]


def record_legs(monkeypatch) -> list:
    """Patch the backtester to record leg tickers, and return the list it fills."""
    tickers = []
    original = qis.backtest_model_portfolio

    def recording(**kwargs):
        """Record each leg's ticker and delegate to the real backtester."""
        tickers.append(kwargs['ticker'])
        return original(**kwargs)

    monkeypatch.setattr(qis, 'backtest_model_portfolio', recording)
    return tickers


@pytest.mark.parametrize('param, label', [
    (CrossBacktestParam.MOM_SPAN, 'long_span'),
    (CrossBacktestParam.BETA_SPAN, 'beta_span'),
    (CrossBacktestParam.MOM_BETA_SPAN, 'spans'),
    (CrossBacktestParam.RESIDUAL_MOM_SPAN, 'res_mom_span'),
])
def test_each_sweep_labels_its_legs_with_the_swept_values(universe: dict, monkeypatch,
                                                          param: CrossBacktestParam,
                                                          label: str) -> None:
    """The sweep builds one leg per span value, named for the parameter it varies.

    This is the assertion that catches a sweep wired to the wrong keyword: the legs would
    still be built and still be labelled, but every one of them would hold the same portfolio.
    """
    tickers = record_legs(monkeypatch)
    figs = cross_backtest_alpha_signals(cross_backtest_param=param, **universe)
    assert tickers == ['Equal Weight'] + [f"{label}={span:0.0f}" for span in SPAN_VALUES]
    assert len(figs) > 0


def test_a_sweep_actually_varies_the_portfolios(universe: dict, monkeypatch) -> None:
    """The swept legs must hold *different* portfolios, not just carry different labels.

    Labels are built from the sweep values directly, so they would look right even if the span
    never reached ``compute_signal_scores``. The weights are what actually prove it did.
    """
    weights = {}
    original = qis.backtest_model_portfolio

    def recording(**kwargs):
        """Record each leg's target weights and delegate to the real backtester."""
        weights[kwargs['ticker']] = kwargs['weights']
        return original(**kwargs)

    monkeypatch.setattr(qis, 'backtest_model_portfolio', recording)
    cross_backtest_alpha_signals(cross_backtest_param=CrossBacktestParam.MOM_SPAN, **universe)

    swept = [weights[f"long_span={span:0.0f}"] for span in SPAN_VALUES]
    for faster, slower in zip(swept, swept[1:]):
        assert not faster.equals(slower)


def test_an_unknown_sweep_parameter_raises(universe: dict) -> None:
    """The sweep dispatch is exhaustive over the enum and refuses anything else."""
    with pytest.raises(NotImplementedError):
        cross_backtest_alpha_signals(cross_backtest_param='MOM_SPAN', **universe)
