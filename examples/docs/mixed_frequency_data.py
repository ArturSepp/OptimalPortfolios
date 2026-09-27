"""Canonical script of docs/mixed_frequency_data.md.

The page's six Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: endpoint price ratios, native per-cadence regressions and an explicit EWMA
weighted sum. The script runs offline after ``pip install optimalportfolios`` and needs no
data file or random seed:

    python -m examples.docs.mixed_frequency_data

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

ASSETS = ['Global Equity', 'Government Bonds', 'Private Assets']
FORMATION_DATE = '2023-12-31'
LOOKBACK = {'ME': 12, 'QE': 4}
SKIP = {'ME': 1, 'QE': 1}


def synthetic_prices() -> pd.DataFrame:
    """Return the page's deterministic panel: two monthly series and quarter-end NAVs."""
    dates = pd.date_range('2018-12-31', '2024-12-31', freq='ME')
    t = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame({
        'Global Equity': 100 * np.exp(0.01 * t + 0.02 * np.sin(t / 3)),
        'Government Bonds': 100 * np.exp(0.003 * t + 0.01 * (np.cos(t / 4) - 1)),
        'Private Assets': 100 * np.exp(0.007 * t + 0.015 * np.sin(t / 5)),
    }, index=dates)
    prices.loc[~dates.is_quarter_end, 'Private Assets'] = np.nan
    return prices


def endpoint_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns between the observed endpoints of each column, with a zero first row."""
    observed = prices.dropna()
    returns = np.log(observed / observed.shift())
    returns.iloc[0] = 0.0
    return returns


def classic_reference(prices: pd.DataFrame, asset: str, lookback: int) -> pd.Series:
    """Classic momentum as one endpoint ratio: skip the latest observation, span ``lookback``."""
    observed = prices[asset].dropna()
    return np.log(observed.shift(1) / observed.shift(lookback + 1))


def ewma_factor_covariance(factor_prices: pd.DataFrame, span: int) -> np.ndarray:
    """Annual monthly factor covariance as an explicit EWMA weighted sum of demeaned returns."""
    returns = np.log(factor_prices).diff().iloc[1:]
    adjusted = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(adjusted) - 1, -1, -1)
    return 12 * np.einsum('t,ti,tj->ij', weights, adjusted, adjusted)


def perturbed_after(frame: pd.DataFrame, cutoff: pd.Timestamp, low: float,
                    high: float) -> pd.DataFrame:
    """Scale every row after ``cutoff`` by a factor rising from ``low`` to ``high``."""
    changed = frame.copy()
    later = changed.index > cutoff
    changed.loc[later] *= np.linspace(low, high, later.sum())[:, None]
    return changed


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2018-12-31", "2024-12-31", freq="ME")
    t = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame({
        "Global Equity": 100 * np.exp(0.01*t + 0.02*np.sin(t/3)),
        "Government Bonds": 100 * np.exp(0.003*t + 0.01*(np.cos(t/4)-1)),
        "Private Assets": 100 * np.exp(0.007*t + 0.015*np.sin(t/5)),
    }, index=dates)
    prices.loc[~dates.is_quarter_end, "Private Assets"] = np.nan

    # The panel: 73 month ends, 25 observed quarterly NAVs.
    pd.testing.assert_frame_equal(prices, synthetic_prices(), rtol=1e-15)
    assert len(prices) == 73 and prices["Private Assets"].count() == 25

    import pandas as pd
    import qis
    import optimalportfolios as opt

    return_frequencies = pd.Series({
        "Global Equity": "ME",
        "Government Bonds": "ME",
        "Private Assets": "QE",
    })
    returns_by_frequency = qis.compute_asset_returns_dict(
        prices=prices,
        returns_freqs=return_frequencies,
        is_log_returns=True,
        drop_first=False,
        is_first_zero=True,
    )

    # Each bucket holds the log returns between its own observed endpoints, from a first zero.
    for frequency, assets, rows in (("ME", ASSETS[:2], 73), ("QE", ASSETS[2:], 25)):
        bucket = returns_by_frequency[frequency]
        assert len(bucket) == rows and list(bucket.columns) == assets
        assert (bucket.iloc[0] == 0.0).all()
        pd.testing.assert_frame_equal(bucket, endpoint_log_returns(prices[assets]),
                                      check_freq=False, rtol=1e-12, atol=1e-15)

    scores, raw_signal = opt.compute_momentum_alpha(
        prices=prices,
        returns_freq=return_frequencies,
        long_span={"ME": 12, "QE": 4},
        vol_span={"ME": 13, "QE": 4},
    )
    ewma_scores, ewma_raw = scores, raw_signal

    scores, raw_signal = opt.compute_classic_momentum_alpha(
        prices=prices,
        returns_freq=return_frequencies,
        lookback_periods={"ME": 12, "QE": 4},
        skip_periods={"ME": 1, "QE": 1},
    )
    classic_scores, classic_raw = scores, raw_signal

    # Classic momentum is one endpoint ratio per asset, skipping one native period.
    for asset, lookback in zip(ASSETS, (12, 12, 4)):
        expected = classic_reference(prices, asset, lookback).reindex(classic_raw.index).ffill()
        pd.testing.assert_series_equal(classic_raw[asset], expected, check_freq=False,
                                       rtol=1e-11, atol=1e-14)
    # The page's table, and its first row from independently dated windows.
    table = classic_raw.loc["2023-12-31":"2024-03-31"]
    np.testing.assert_allclose(table, [[0.133758, 0.023399, 0.064028],
                                       [0.144017, 0.019965, 0.064028],
                                       [0.151632, 0.017527, 0.064028],
                                       [0.155765, 0.016237, 0.078566]], rtol=0, atol=0.5e-6)
    monthly_window = np.log(prices.loc["2023-11-30"].iloc[:2] / prices.loc["2022-11-30"].iloc[:2])
    quarterly_window = np.log(prices.at[pd.Timestamp("2023-09-30"), "Private Assets"]
                              / prices.at[pd.Timestamp("2022-09-30"), "Private Assets"])
    np.testing.assert_allclose(table.iloc[0], [*monthly_window, quarterly_window], atol=0.5e-6)
    # Scores are standardised within each cadence: the lone quarterly asset has none, and its
    # raw signal is carried between quarter ends.
    for raw, cadence_scores in ((classic_raw, classic_scores), (ewma_raw, ewma_scores)):
        assert list(raw.columns) == ASSETS and cadence_scores["Private Assets"].isna().all()
        carried = raw.loc["2023-12-31":"2024-02-29", "Private Assets"]
        assert carried.notna().all() and carried.nunique() == 1
        assert raw.at[pd.Timestamp("2024-03-31"), "Private Assets"] != carried.iloc[0]
        monthly = raw.iloc[:, :2].clip(-5, 5)
        z_scores = monthly.sub(monthly.mean(axis=1), axis=0).div(monthly.std(axis=1, ddof=0),
                                                                axis=0)
        np.testing.assert_allclose(cadence_scores.iloc[:, :2], z_scores, rtol=1e-10, atol=1e-13,
                                   equal_nan=True)

    metadata = pd.DataFrame({
        "name": prices.columns,
        "asset_class": ["Equity", "Bonds", "Private"],
        "currency": "USD",
    }, index=prices.columns)
    universe = opt.UniverseData(prices=prices, metadata=metadata)
    arithmetic_by_frequency = universe.get_asset_returns_dict(
        returns_freqs=return_frequencies,
    )
    log_by_frequency = universe.get_asset_returns_dict(
        returns_freqs=return_frequencies, is_log_returns=True,
    )

    # The wrapper returns arithmetic returns by default and the same log panels on request.
    for frequency, bucket in returns_by_frequency.items():
        pd.testing.assert_frame_equal(log_by_frequency[frequency], bucket)
        np.testing.assert_allclose(arithmetic_by_frequency[frequency], np.expm1(bucket),
                                   rtol=1e-12, atol=1e-15)

    factor_prices = prices[["Global Equity", "Government Bonds"]].rename(columns={
        "Global Equity": "Market", "Government Bonds": "Rates",
    })
    factor_estimator = opt.FactorCovarEstimator(
        lasso_model=opt.LassoModel(
            model_type=opt.LassoModelType.LASSO,
            reg_lambda=1e-5,
            span_freq_dict={"ME": 24, "QE": 8},
            warmup_period=8,
            demean=True,
            solver="CLARABEL",
        ),
        factor_returns_freq="ME",
        factor_covar_span=24,
        rebalancing_freq="QE",
    )
    cutoff = pd.Timestamp("2023-12-31")
    factor_data = factor_estimator.fit_current_factor_covars(
        risk_factor_prices=factor_prices.loc[:cutoff],
        asset_returns_dict={
            freq: returns.loc[:cutoff] for freq, returns in returns_by_frequency.items()
        },
        assets=prices.columns,
        estimation_date=cutoff,
    )
    annual_covar = factor_data.y_covar
    rolling_data = factor_estimator.fit_rolling_factor_covars(
        risk_factor_prices=factor_prices,
        asset_returns_dict=returns_by_frequency,
        assets=prices.columns,
        time_period=qis.TimePeriod("2023-06-30", "2023-12-31"),
    )
    rolling_covars = rolling_data.get_y_covars()

    from factorlasso import VarianceColumns

    # Each bucket is fitted at its own cadence; residual variances are annualised by 12 or 4.
    for frequency, annualisation, span in (("ME", 12, 24), ("QE", 4, 8)):
        bucket = returns_by_frequency[frequency].loc[:cutoff]
        factor_returns = np.log(factor_prices.loc[bucket.index]).diff()
        model = opt.LassoModel(model_type=opt.LassoModelType.LASSO, reg_lambda=1e-5, span=span,
                               warmup_period=8, demean=True, solver="CLARABEL")
        model.fit(x=factor_returns, y=bucket)
        np.testing.assert_allclose(factor_data.y_betas.loc[bucket.columns],
                                   model.estimated_betas, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            factor_data.y_variances.loc[bucket.columns, VarianceColumns.RESIDUAL_VARS.value],
            annualisation * model.estimation_result_.ss_res, rtol=1e-7, atol=1e-12)
    # The factor covariance uses the monthly span 24, annualised by 12, and assembles with the
    # full residual diagonal; the three assets keep their order.
    factor_covar = ewma_factor_covariance(factor_prices.loc[:cutoff], span=24)
    np.testing.assert_allclose(factor_data.x_covar, factor_covar, rtol=1e-10, atol=1e-14)
    betas = factor_data.y_betas.to_numpy()
    residual = factor_data.y_variances[VarianceColumns.RESIDUAL_VARS.value].to_numpy()
    np.testing.assert_allclose(annual_covar, betas @ factor_covar @ betas.T + np.diag(residual),
                               rtol=1e-10, atol=1e-14)
    assert list(annual_covar.columns) == ASSETS
    # Three quarterly keys; the current fit equals the last rolling one, and a monthly schedule
    # adds dates without changing the quarter-end estimates.
    quarter_ends = list(pd.to_datetime(["2023-06-30", "2023-09-30", "2023-12-31"]))
    assert list(rolling_covars) == quarter_ends
    np.testing.assert_allclose(annual_covar, rolling_covars[cutoff], atol=1e-12)

    def fresh_estimator(**overrides) -> opt.FactorCovarEstimator:
        """The page's estimator with a new LASSO model, so no fitted state is shared."""
        model = opt.LassoModel(**factor_estimator.lasso_model.get_params())
        return replace(factor_estimator, lasso_model=model, **overrides)

    monthly_covars = fresh_estimator(rebalancing_freq="ME").fit_rolling_factor_covars(
        risk_factor_prices=factor_prices, asset_returns_dict=returns_by_frequency,
        assets=prices.columns, time_period=qis.TimePeriod("2023-06-30", "2023-12-31"),
    ).get_y_covars()
    assert len(monthly_covars) == 7
    for date in quarter_ends:
        np.testing.assert_allclose(monthly_covars[date], rolling_covars[date], atol=1e-12)

    # Later prices change no earlier signal, score or rolling covariance.
    later_prices = perturbed_after(prices, cutoff, 1.2, 2.8)
    for function, arguments, raw, cadence_scores in (
            (opt.compute_classic_momentum_alpha,
             {"lookback_periods": {"ME": 12, "QE": 4}, "skip_periods": {"ME": 1, "QE": 1}},
             classic_raw, classic_scores),
            (opt.compute_momentum_alpha,
             {"long_span": {"ME": 12, "QE": 4}, "vol_span": {"ME": 13, "QE": 4}},
             ewma_raw, ewma_scores)):
        changed_scores, changed_raw = function(later_prices, returns_freq=return_frequencies,
                                               **arguments)
        pd.testing.assert_frame_equal(changed_scores.loc[:cutoff], cadence_scores.loc[:cutoff])
        pd.testing.assert_frame_equal(changed_raw.loc[:cutoff], raw.loc[:cutoff])
    later_buckets = {}
    for frequency, bucket in returns_by_frequency.items():
        changed = bucket.copy()
        changed.loc[changed.index > cutoff] += 0.3
        later_buckets[frequency] = changed
    changed_covars = fresh_estimator().fit_rolling_factor_covars(
        risk_factor_prices=perturbed_after(factor_prices, cutoff, 1.2, 2.5),
        asset_returns_dict=later_buckets, assets=prices.columns,
        time_period=qis.TimePeriod("2023-06-30", "2023-12-31"),
    ).get_y_covars()
    for date, covar in rolling_covars.items():
        pd.testing.assert_frame_equal(changed_covars[date], covar)

    # Failure modes. The QIS helper omits an unmapped asset while the signal wrapper fails, and
    # a horizon mapping must cover every cadence.
    incomplete = return_frequencies.drop("Private Assets")
    assert set(qis.compute_asset_returns_dict(prices, returns_freqs=incomplete,
                                              is_log_returns=True)) == {"ME"}
    for arguments, error in (({"returns_freq": incomplete}, KeyError),
                             ({"returns_freq": return_frequencies,
                               "lookback_periods": {"ME": 12}}, ValueError)):
        try:
            opt.compute_classic_momentum_alpha(prices, **arguments)
        except error:
            pass
        else:
            raise AssertionError(f"expected {error.__name__} for {arguments}")
    # A missing quarter-end NAV is forward filled into a stale zero return.
    stale = prices.copy()
    stale.loc[cutoff, "Private Assets"] = np.nan
    quarterly = qis.compute_asset_returns_dict(stale, returns_freqs=return_frequencies,
                                               is_log_returns=True)["QE"]
    assert quarterly.at[cutoff, "Private Assets"] == 0.0
    assert returns_by_frequency["QE"].at[cutoff, "Private Assets"] != 0.0
    print("mixed_frequency_data: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the observations each asset's momentum uses, and the carry.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import optimalportfolios as opt

    prices = synthetic_prices()
    frequencies = pd.Series({'Global Equity': 'ME', 'Government Bonds': 'ME',
                             'Private Assets': 'QE'})
    _, raw = opt.compute_classic_momentum_alpha(prices=prices, returns_freq=frequencies,
                                                lookback_periods=LOOKBACK, skip_periods=SKIP)
    formation = pd.Timestamp(FORMATION_DATE)
    windows = {}
    for asset in ASSETS:
        observed = prices[asset].dropna().loc[:formation].index
        cadence = frequencies[asset]
        end = observed[-1 - SKIP[cadence]]
        start = observed[-1 - SKIP[cadence] - LOOKBACK[cadence]]
        windows[asset] = (start, end, observed[-1])
    table = pd.DataFrame({asset: {'window_start': start, 'window_end': end,
                                  'raw_at_formation': raw.at[formation, asset]}
                          for asset, (start, end, _) in windows.items()}).T

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(ASSETS, ['#2a78d6', '#eb6834', '#1baf7a']))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.25, 1.0]})
    shown = prices.loc['2022-06-30':'2024-03-31']
    for row, asset in enumerate(reversed(ASSETS)):
        start, end, last = windows[asset]
        dates = shown[asset].dropna().index
        left.scatter(dates, [row] * len(dates), s=14, color='#b9b8b3', zorder=2)
        inside = dates[(dates >= start) & (dates <= end)]
        left.plot([start, end], [row, row], color=colours[asset], linewidth=6,
                  solid_capstyle='round', alpha=0.35, zorder=1)
        left.scatter(inside, [row] * len(inside), s=22, color=colours[asset], zorder=3)
        left.plot([end, last], [row, row], color=muted, linewidth=1.2, linestyle=':', zorder=1)
        cadence = 'monthly' if frequencies[asset] == 'ME' else 'quarterly'
        left.text(start, row + 0.22, f'{LOOKBACK[frequencies[asset]]} {cadence} returns',
                  color=ink, fontsize=9, va='bottom')
    left.axvline(formation, color=muted, linestyle='--', linewidth=1.0)
    left.text(formation - pd.Timedelta(days=12), 2.62, 'formation\n31 Dec 2023', color=ink,
              fontsize=9, ha='right', va='top')
    left.set_yticks(range(len(ASSETS)), list(reversed(ASSETS)))
    left.set_ylim(-0.5, 2.7)
    left.set_title('Observations in each momentum window', loc='left', color=ink)
    left.tick_params(axis='x', labelrotation=0)
    left.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(3, 9)))
    left.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %y'))

    # Each value holds from its formation date to the next one; the last is held one month.
    path_shown = raw.loc['2023-06-30':'2024-06-30']
    held_until = path_shown.index.append(pd.DatetimeIndex([pd.Timestamp('2024-07-31')]))
    for asset in ASSETS:
        values = np.append(path_shown[asset].to_numpy(), path_shown[asset].iloc[-1])
        right.plot(held_until, values, color=colours[asset], linewidth=2, drawstyle='steps-post')
        right.text(held_until[-1], values[-1], f' {asset}', color=ink, fontsize=9, va='center')
    right.axvline(formation, color=muted, linestyle='--', linewidth=1.0)
    right.set_title('Raw classic momentum (log return)', loc='left', color=ink)
    right.set_xlim(held_until[0], held_until[-1] + pd.Timedelta(days=150))
    right.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(3, 9)))
    right.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %y'))
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='x' if axis is left else 'y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    private = raw.loc['2023-12-31':'2024-02-29', 'Private Assets']
    checks = {
        'windows_match_endpoint_ratios': bool(all(
            np.isclose(raw.at[formation, asset],
                       np.log(prices.at[end, asset] / prices.at[start, asset]))
            for asset, (start, end, _) in windows.items())),
        'monthly_window_ends_one_month_before_formation': bool(
            windows['Global Equity'][1] == pd.Timestamp('2023-11-30')),
        'quarterly_window_ends_one_quarter_before_formation': bool(
            windows['Private Assets'][1] == pd.Timestamp('2023-09-30')),
        'private_signal_carried_until_march': bool(
            private.nunique() == 1 and raw.at[pd.Timestamp('2024-03-31'), 'Private Assets']
            != private.iloc[0]),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
