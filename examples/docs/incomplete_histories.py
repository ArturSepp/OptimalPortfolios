"""Canonical script of docs/incomplete_histories.md.

The page's six Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: an exact currency ledger of units and cash in fractions, explicit drift
arithmetic and three hand-written EWMA updates. The script runs offline after ``pip install
optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.incomplete_histories

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd

GAP_DAYS = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
LIQUID_PRICES = [100.0, 110.0, 120.0, 130.0]
GAPPED_PRICES = [100.0, None, 110.0, 120.0]
TARGET_WEIGHTS = [0.60, 0.40]


def currency_ledger() -> dict:
    """Units, NAV and cash of the three holdings paths, from exact fractions and one trade."""
    first_liquid, first_gapped = Fraction(3, 5), Fraction(2, 5)
    priced_nav = first_liquid * 110
    new_liquid = priced_nav * Fraction(3, 5) / 110
    cash = (first_liquid - new_liquid) * 110
    return {
        'held_gap': {
            'units': [[first_liquid, first_gapped]] * 4,
            'nav': [100, priced_nav, first_liquid * 120 + first_gapped * 110,
                    first_liquid * 130 + first_gapped * 120],
            'cash': [0] * 4,
        },
        'traded_gap': {
            'units': [[first_liquid, first_gapped]] + [[new_liquid, 0]] * 3,
            'nav': [100, priced_nav, new_liquid * 120 + cash, new_liquid * 130 + cash],
            'cash': [0, cash, cash, cash],
        },
        'late_entry': {
            'units': [[first_liquid, 0]] * 4,
            'nav': [100, first_liquid * 110 + 40, first_liquid * 120 + 40,
                    first_liquid * 130 + 40],
            'cash': [40] * 4,
        },
    }


def holdings_paths() -> dict:
    """Run the page's three qis backtests from the constants, for the exhibit."""
    import warnings

    import qis

    days = pd.DatetimeIndex(GAP_DAYS)
    gap_prices = pd.DataFrame({'Liquid': LIQUID_PRICES,
                               'Gapped': [np.nan if p is None else p for p in GAPPED_PRICES]},
                              index=days)
    opening = pd.DataFrame([TARGET_WEIGHTS], index=days[:1], columns=gap_prices.columns)
    retrade = pd.DataFrame([TARGET_WEIGHTS] * 2, index=days[:2], columns=gap_prices.columns)
    late_prices = gap_prices.assign(Gapped=[np.nan, 100.0, 110.0, 120.0])
    arguments = {'initial_nav': 100.0, 'rebalancing_costs': None, 'weight_implementation_lag': 0}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return {
            'held_gap': qis.backtest_model_portfolio(prices=gap_prices, weights=opening,
                                                     **arguments),
            'traded_gap': qis.backtest_model_portfolio(prices=gap_prices, weights=retrade,
                                                       **arguments),
            'late_entry': qis.backtest_model_portfolio(prices=late_prices, weights=opening,
                                                       **arguments),
        }


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import pandas as pd

    decision_dates = pd.to_datetime(["2024-03-28", "2024-06-28"])
    eligibility = pd.DataFrame(
        {"Liquid": [1.0, 1.0], "Late Starter": [0.0, 1.0]},
        index=decision_dates,
    )
    can_rebalance = pd.DataFrame(
        {"Liquid": [1.0, 1.0], "Locked Fund": [0.0, 0.0]},
        index=decision_dates,
    )

    # Two separate universes, each a complete binary panel on the same decision dates.
    assert eligibility.columns.tolist() == ["Liquid", "Late Starter"]
    assert can_rebalance.columns.tolist() == ["Liquid", "Locked Fund"]
    pd.testing.assert_index_equal(eligibility.index, can_rebalance.index)
    assert eligibility.to_numpy().tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert can_rebalance.to_numpy().tolist() == [[1.0, 0.0], [1.0, 0.0]]

    import numpy as np
    import optimalportfolios as op

    assets = ["Liquid", "Locked Fund"]
    spec = op.Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
        weights_0=pd.Series([0.60, 0.40], index=assets),
    )
    aligned = spec.update_with_valid_tickers(
        valid_tickers=assets,
        rebalancing_indicators=can_rebalance.iloc[0],
    )

    # The stored 40% baseline pins both fund bounds; the liquid box and the input are unchanged.
    np.testing.assert_array_equal(aligned.min_weights, [0.0, 0.4])
    np.testing.assert_array_equal(aligned.max_weights, [1.0, 0.4])
    np.testing.assert_array_equal(spec.min_weights, [0.0, 0.0])
    np.testing.assert_array_equal(spec.max_weights, [1.0, 1.0])
    # A missing box side stays missing, so one side alone is not an equality pin.
    for missing, other in (("min_weights", "max_weights"), ("max_weights", "min_weights")):
        one_sided = replace(spec, **{missing: None}).update_with_valid_tickers(
            valid_tickers=assets, rebalancing_indicators=can_rebalance.iloc[0])
        assert getattr(one_sided, missing) is None
        assert getattr(one_sided, other)["Locked Fund"] == 0.4
    # A supplied baseline overrides the stored one; with no baseline there is nothing to pin.
    supplied = spec.update_with_valid_tickers(
        valid_tickers=assets, rebalancing_indicators=can_rebalance.iloc[0],
        weights_0=pd.Series([0.7, 0.3], index=assets))
    assert supplied.min_weights["Locked Fund"] == supplied.max_weights["Locked Fund"] == 0.3
    cold = replace(spec, weights_0=None).update_with_valid_tickers(
        valid_tickers=assets, rebalancing_indicators=can_rebalance.iloc[0])
    assert cold.min_weights["Locked Fund"] == 0.0 and cold.max_weights["Locked Fund"] == 1.0

    drift_dates = pd.to_datetime(["2024-03-28", "2024-06-28"])
    drift_prices = pd.DataFrame(
        {"Liquid": [100.0, 120.0], "Locked Fund": [np.nan, np.nan]},
        index=drift_dates,
    )
    drifted = op.apply_drift_to_weights_0(
        weights_0=spec.weights_0,
        prices=drift_prices,
        prev_date=drift_dates[0],
        date=drift_dates[1],
    )

    # Values 72 and 40 make a NAV of 112: weights 0.642857 and 0.357143, not 60% and 40%.
    np.testing.assert_allclose(drifted, [float(Fraction(72, 112)), float(Fraction(40, 112))],
                               atol=1e-14)
    assert drifted.round(6).tolist() == [0.642857, 0.357143]

    import warnings
    import qis

    days = pd.date_range("2024-01-02", periods=4, freq="B")
    gap_prices = pd.DataFrame(
        {"Liquid": [100.0, 110.0, 120.0, 130.0],
         "Gapped": [100.0, np.nan, 110.0, 120.0]},
        index=days,
    )
    opening_targets = pd.DataFrame([[0.60, 0.40]], index=days[:1], columns=gap_prices.columns)
    retrade_targets = pd.DataFrame(
        [[0.60, 0.40], [0.60, 0.40]], index=days[:2], columns=gap_prices.columns,
    )
    late_prices = gap_prices.assign(Gapped=[np.nan, 100.0, 110.0, 120.0])

    # Retain the warnings for inspection; these paths intentionally contain missing prices.
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", UserWarning)
        held_gap = qis.backtest_model_portfolio(
            prices=gap_prices, weights=opening_targets, initial_nav=100.0,
            rebalancing_costs=None, weight_implementation_lag=0,
        )
        traded_gap = qis.backtest_model_portfolio(
            prices=gap_prices, weights=retrade_targets, initial_nav=100.0,
            rebalancing_costs=None, weight_implementation_lag=0,
        )
        late_entry = qis.backtest_model_portfolio(
            prices=late_prices, weights=opening_targets, initial_nav=100.0,
            rebalancing_costs=None, weight_implementation_lag=0,
        )
    warning_messages = [str(item.message) for item in captured_warnings]

    # NAV, units and cash of each path against the exact ledger, with no costs charged.
    ledger = currency_ledger()
    paths = {"held_gap": held_gap, "traded_gap": traded_gap, "late_entry": late_entry}
    for name, portfolio in paths.items():
        expected = ledger[name]
        np.testing.assert_allclose(portfolio.nav, np.array(expected["nav"], float), atol=1e-12)
        np.testing.assert_allclose(portfolio.units, np.array(expected["units"], float),
                                   atol=1e-12)
        marked = (portfolio.units * portfolio.prices).sum(axis=1)
        np.testing.assert_allclose(portfolio.nav - marked, np.array(expected["cash"], float),
                                   atol=1e-12)
        np.testing.assert_allclose(portfolio.realized_costs, 0, atol=0)
    # The page's table, and the units and cash its paragraph quotes.
    table = np.array([ledger[name]["nav"] for name in paths], float).T
    np.testing.assert_allclose(table, [[100.00, 100.00, 100.00], [66.00, 66.00, 106.00],
                                       [116.00, 69.60, 112.00], [126.00, 73.20, 118.00]],
                               atol=0.005)
    assert float(ledger["traded_gap"]["units"][1][0]) == 0.36
    assert float(ledger["traded_gap"]["cash"][1]) == 26.4
    # Both interior gaps and both missing execution quotes are reported.
    assert sum("inside the reported history" in m for m in warning_messages) == 2
    assert sum("have no price on their traded date" in m for m in warning_messages) == 2
    for name, portfolio in holdings_paths().items():
        pd.testing.assert_series_equal(portfolio.nav, paths[name].nav, check_freq=False)

    names = ["Liquid", "Zero", "Negative", "Missing", "Warmup"]
    covariance = pd.DataFrame(
        np.diag([0.04, 0.0, -0.01, np.nan, 1e-12]), index=names, columns=names,
    )
    filtered, _ = op.filter_covar_and_vectors_for_nans(covariance)
    floored, _ = op.filter_covar_and_vectors_for_nans(covariance, variance_floor=1e-6)
    eligible, _ = op.filter_covar_and_vectors_for_nans(
        covariance, inclusion_indicators=pd.Series([1, 0, 0, 0, 0], index=names),
    )

    # Only positive diagonals survive; the floor changes the tiny survivor, not eligibility.
    for survivors in (filtered, floored):
        assert survivors.columns.tolist() == ["Liquid", "Warmup"]
    np.testing.assert_array_equal(np.diag(filtered), [0.04, 1e-12])
    np.testing.assert_array_equal(np.diag(floored), [0.04, 1e-6])
    assert eligible.columns.tolist() == ["Liquid"]
    # The helper alone does not validate finiteness: an infinite entry survives.
    for position in ((0, 0), (0, 1)):
        infinite = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"],
                                columns=["A", "B"])
        infinite.iloc[position] = np.inf
        survivor, _ = op.filter_covar_and_vectors_for_nans(infinite)
        assert survivor.shape == (2, 2) and np.isinf(survivor.iloc[position])

    observations = np.array([[0.01, 0.02], [np.nan, 0.03], [0.01, 0.04]])
    covariance_states = qis.compute_ewm_covar_tensor(
        a=observations, span=3, nan_backfill=qis.NanBackfill.ZERO_FILL,
    )

    # Three explicit updates with decay 1/2: a missing update resets the entry to zero.
    scale = Fraction(1, 10000)
    expected_states = np.array([
        [[Fraction(1, 2), 1], [1, 2]],
        [[0, 0], [0, Fraction(11, 2)]],
        [[Fraction(1, 2), 2], [2, Fraction(43, 4)]],
    ], dtype=float) * float(scale)
    np.testing.assert_allclose(covariance_states, expected_states, atol=1e-18)
    assert covariance_states[:, 0, 0].round(8).tolist() == [0.00005, 0.0, 0.00005]
    # A zero return instead of the missing one decays the entry to 0.000025.
    zero_filled = qis.compute_ewm_covar_tensor(
        a=np.nan_to_num(observations), span=3, nan_backfill=qis.NanBackfill.ZERO_FILL)
    assert abs(zero_filled[1, 0, 0] - 0.000025) < 1e-18
    assert_raises(AssertionError, np.testing.assert_allclose, actual=zero_filled,
                  desired=covariance_states)
    print("incomplete_histories: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: NAV of the three missing-price paths and one path's holdings.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = holdings_paths()
    ledger = currency_ledger()
    labels = {'held_gap': 'Hold through the gap', 'traded_gap': 'Rebalance on the gap',
              'late_entry': 'Missing opening price'}
    nav = pd.DataFrame({labels[name]: portfolio.nav for name, portfolio in paths.items()})
    traded = paths['traded_gap']
    composition = pd.DataFrame({
        'Liquid': traded.units['Liquid'] * traded.prices['Liquid'],
        'Gapped': (traded.units['Gapped'] * traded.prices['Gapped']).fillna(0.0),
    })
    composition['Cash'] = traded.nav - composition.sum(axis=1)
    table = pd.concat([nav, composition.add_prefix('rebalance on gap: ')], axis=1)

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = ['#2a78d6', '#eb6834', '#1baf7a']
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    x = np.arange(len(nav))
    ticks = [date.strftime('%d %b') for date in nav.index]
    # The first two paths coincide until 3 January; the dashed line keeps both visible.
    for colour, (label, series) in zip(colours, nav.items()):
        dashed = label == labels['traded_gap']
        left.plot(x, series, color=colour, linewidth=2, marker='o', markersize=5,
                  linestyle='--' if dashed else '-', zorder=3 if dashed else 2)
        left.text(x[-1] + 0.08, series.iloc[-1], label, color=ink, fontsize=9, va='center')
    left.axvspan(0.75, 1.25, color=grid, alpha=0.6, linewidth=0)
    left.text(1.0, 58, 'gapped price\nmissing', color=ink, fontsize=9, ha='center', va='bottom')
    left.set_title('NAV of the three paths', loc='left', color=ink)
    left.set_ylim(50, 135)
    left.set_xlim(-0.2, 4.3)
    bottom = np.zeros(len(composition))
    # Shades of the rebalance path's orange: these are its holdings, not the other paths.
    for colour, column in zip(['#b54a1f', '#f4a582', '#b9b8b3'], composition.columns):
        right.bar(x, composition[column], 0.6, bottom=bottom, color=colour, label=column,
                  edgecolor=surface, linewidth=1.5)
        bottom += composition[column].to_numpy()
    right.set_title('Rebalance on the gap: holdings value', loc='left', color=ink)
    right.legend(frameon=False, loc='upper left', fontsize=10, labelcolor=ink, ncol=3)
    right.set_ylim(0, 135)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_xticks(x, ticks)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    checks = {
        'nav_matches_ledger': bool(all(
            np.allclose(paths[name].nav, np.array(ledger[name]['nav'], float), atol=1e-12)
            for name in paths)),
        'gapped_units_cleared_on_rebalance': bool(traded.units['Gapped'].iloc[1:].eq(0).all()),
        'cash_left_after_clearing': bool(np.isclose(composition['Cash'].iloc[1], 26.4)),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
