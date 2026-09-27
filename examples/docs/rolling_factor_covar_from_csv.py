"""Canonical script of docs/rolling_factor_covar_from_csv.md.

The page's eight Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: reference-currency wealth rebuilt from endpoint prices, spots and contracted
forwards; covariances summed term by term; an explicit EWMA weighted sum; refits of perturbed
and defective copies of the bundle; and the ``load`` command in a fresh process that may neither
import ``yfinance`` nor open a socket. Sockets are blocked throughout, and the Yahoo fetch block
runs on synthetic closes in place of the download.

The page imports the repository example
``examples/covar_estimation/rolling_factor_covar_from_csv.py``, so the script runs from a source
checkout with the core install. It writes only to a temporary directory that it removes:

    python -m examples.docs.rolling_factor_covar_from_csv
"""
import contextlib
import hashlib
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.covar_estimation import rolling_factor_covar_from_csv as csv_example

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = frozenset({
    'futures_risk_factors.csv', 'fx_hedging_data_fx_spots.csv',
    'fx_hedging_data_domestic_rates.csv', 'asset_prices.csv', 'asset_metadata.csv',
    'risk_model_settings.csv',
})
FACTORS = ['Equity', 'Rates']
ASSETS = ['Growth', 'Income', 'Domestic']
SNAPSHOT_DATES = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31']
FUTURE_CUTOFF = '2020-12-31'  # inputs after this date are perturbed; earlier snapshots must hold
# The page's Yahoo calibration, verbatim.
YAHOO_SETTINGS = """\
setting,value
reference_ccy,CHF
is_log_returns,True
is_excess_returns,False
factor_returns_freq,ME
factor_names,Equity|Rates|Credit|Commodities|Fx
factor_covar_span,36
rebalancing_freq,YE
lasso_model_type,HIERARCHICAL_CLUSTER_GROUP_LASSO
reg_lambda,1e-05
beta_span,36
warmup_period,36
demean,True
solver,CLARABEL
estimation_start,2019-12-31
estimation_end,2025-12-31
"""
# The page's Yahoo request: seven USD assets, four factor proxies, the CHF quote and ^IRX.
FETCH_START, FETCH_END = '2007-12-31', '2026-01-01'
HEDGED_ASSETS = ['IEF', 'HYG']
YAHOO_ASSETS = ['QQQ', 'EFA', 'EEM', 'IEF', 'HYG', 'VNQ', 'GSG']
FACTOR_PROXIES = {'SPY': 'Equity', 'TLT': 'Rates', 'LQD': 'Credit', 'GLD': 'Commodities'}
FETCH_TICKERS = YAHOO_ASSETS + list(FACTOR_PROXIES) + ['CHF=X', '^IRX']
# Synthetic download defects: GSG starts late, one business day is missing, QQQ has one gap.
LATE_START, MISSING_DAY, QQQ_GAP = 20, 400, 700
BUSINESS_DAYS_PER_YEAR = 252  # the QIS year of the daily FX carry
# The load command's acceptance checks that the page quotes.
RECONSTRUCTION_RTOL, RECONSTRUCTION_ATOL, MIN_EIGENVALUE = 1e-12, 1e-14, -1e-10
# One defect per documented loader check, and the message the loader must give.
DEFECTS = {
    'missing_file': 'incomplete', 'factor_order': 'ordered factor_names',
    'duplicate_dates': 'sorted and unique', 'duplicate_spots': 'sorted and unique',
    'unsorted_factors': 'sorted and unique', 'nonpositive_nav': 'positive NAVs',
    'nonpositive_price': 'positive prices', 'nonpositive_spot': 'positive spots',
    'nonfinite_asset': 'finite numeric', 'missing_metadata': 'same assets',
    'hedge_bounds': 'hedge_ratio', 'bad_frequency': 'return_frequency',
    'missing_currency': 'currencies', 'bad_boolean': 'boolean',
    'simple_returns': 'log factor returns', 'missing_setting': 'missing',
    'future_end': 'estimation_end', 'start_after_end': 'estimation_start',
}
# Run the real load command with yfinance imports and socket connections denied.
LOAD_GUARD = r"""
import runpy, sys
def audit(event, args):
    if event == 'import' and args[0].split('.')[0] == 'yfinance':
        raise AssertionError('CSV-only load imported yfinance')
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('CSV-only load attempted network access')
sys.addaudithook(audit)
sys.path.insert(0, sys.argv[1])
sys.argv = ['rolling_factor_covar_from_csv', 'load', '--data-dir', sys.argv[2]]
runpy.run_module('examples.covar_estimation.rolling_factor_covar_from_csv', run_name='__main__')
"""
OFFLINE = AssertionError('The documented walkthrough must remain offline')


@contextlib.contextmanager
def offline_scratch():
    """Deny sockets and the Yahoo download, and keep ``tempfile`` output in a removed directory.

    Used as the decorator of ``main``, so the page's blocks stay at function level. The first
    block's ``mkdtemp`` then creates its bundle inside the directory removed on exit.
    """
    with (tempfile.TemporaryDirectory(prefix='op-factor-csv-docs-') as scratch,
          patch.object(tempfile, 'tempdir', scratch),
          patch.object(socket, 'create_connection', side_effect=OFFLINE),
          patch.object(socket.socket, 'connect', side_effect=OFFLINE),
          patch.object(csv_example, '_download_close', side_effect=OFFLINE)):
        yield


@contextlib.contextmanager
def working_directory(path: Path):
    """Change the working directory for the duration of the block."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def file_hashes(directory: Path) -> dict:
    """SHA-256 of every file in ``directory``, by file name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).iterdir()}


def copy_bundle(source: Path) -> Path:
    """Copy the six bundle files into a new temporary directory and return it."""
    destination = Path(tempfile.mkdtemp(prefix='op-factor-csv-copy-'))
    for name in BUNDLE:
        shutil.copyfile(Path(source) / name, destination / name)
    return destination


def rewrite_csv(path: Path, change, dtype=None) -> None:
    """Apply ``change`` to one CSV frame, keeping its index and header."""
    change(pd.read_csv(path, index_col=0, dtype=dtype)).to_csv(path)


def with_settings(frame: pd.DataFrame, **values: str) -> pd.DataFrame:
    """Return a copy of a settings frame with the given values, adding rows as needed."""
    changed = frame.copy()
    for setting, value in values.items():
        changed.loc[setting, 'value'] = value
    return changed


def break_bundle(directory: Path, defect: str) -> None:
    """Break one part of the six-file contract in a copied bundle."""
    settings = {
        'bad_boolean': {'demean': 'maybe'}, 'simple_returns': {'is_log_returns': 'False'},
        'future_end': {'estimation_end': '2023-12-31'},
        'start_after_end': {'estimation_start': '2023-06-30'},
    }
    frames = {
        'factor_order': ('futures_risk_factors.csv', lambda f: f.iloc[:, ::-1]),
        'duplicate_dates': ('asset_prices.csv', lambda f: pd.concat([f, f.tail(1)])),
        'duplicate_spots': ('fx_hedging_data_fx_spots.csv', lambda f: pd.concat([f, f.tail(1)])),
        'unsorted_factors': ('futures_risk_factors.csv', lambda f: f.iloc[::-1]),
        'nonpositive_nav': ('futures_risk_factors.csv', lambda f: f.assign(Equity=0.0)),
        'nonpositive_price': ('asset_prices.csv', lambda f: f.assign(Income=-1.0)),
        'nonpositive_spot': ('fx_hedging_data_fx_spots.csv', lambda f: f.assign(CHF=0.0)),
        'nonfinite_asset': ('asset_prices.csv', lambda f: f.assign(Growth=np.inf)),
        'missing_metadata': ('asset_metadata.csv', lambda f: f.iloc[1:]),
        'hedge_bounds': ('asset_metadata.csv', lambda f: f.assign(hedge_ratio=1.1)),
        'bad_frequency': ('asset_metadata.csv', lambda f: f.assign(return_frequency='INVALID')),
        'missing_currency': ('asset_metadata.csv', lambda f: f.assign(currency='EUR')),
    }
    if defect == 'missing_file':
        (directory / 'asset_prices.csv').unlink()
    elif defect == 'missing_setting':
        rewrite_csv(directory / 'risk_model_settings.csv', lambda f: f.drop(index='solver'),
                    dtype=str)
    elif defect in settings:
        rewrite_csv(directory / 'risk_model_settings.csv',
                    lambda f: with_settings(f, **settings[defect]), dtype=str)
    else:
        name, change = frames[defect]
        rewrite_csv(directory / name, change)


def expect_failure(call, errors, pattern: str) -> None:
    """Require ``call()`` to raise one of ``errors`` with a message matching ``pattern``."""
    try:
        call()
    except errors as error:
        assert re.search(pattern, str(error)), f'{pattern!r} does not match {error}'
    else:
        raise AssertionError(f'expected an error matching {pattern!r}')


def hedged_log_returns(prices: pd.Series, spot: pd.Series, local_rate: pd.Series,
                       reference_rate: pd.Series, hedge: float) -> pd.Series:
    """Monthly log return of reference-currency wealth: endpoint prices and spots, and a forward
    on ``hedge`` of opening principal contracted at the period start."""
    price_gross, spot_gross = prices / prices.shift(), spot / spot.shift()
    forward_ratio = (1 + reference_rate.shift() / 12) / (1 + local_rate.shift() / 12)
    return np.log(price_gross * spot_gross + hedge * (forward_ratio - spot_gross))


def ewma_factor_covariance(factor_prices: pd.DataFrame, span: int) -> np.ndarray:
    """Annual monthly factor covariance as an explicit EWMA weighted sum of demeaned returns."""
    returns = np.log(factor_prices).diff().iloc[1:]
    adjusted = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(adjusted) - 1, -1, -1)
    return 12 * np.einsum('t,ti,tj->ij', weights, adjusted, adjusted)


def synthetic_closes(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Offline stand-in for the Yahoo download: deterministic closes on business days.

    GSG starts ``LATE_START`` days late, the ``MISSING_DAY`` row is absent and QQQ has no close
    at ``QQQ_GAP``, so the fetch must drop incomplete rows and fill its business-day grid.
    """
    dates = pd.bdate_range(start, end, inclusive='left')
    step = np.arange(len(dates), dtype=float)
    closes = pd.DataFrame({
        ticker: 100 * np.exp(0.0002 * step + 0.03 * np.sin(step / (40 + 7 * k)))
        for k, ticker in enumerate(tickers)
    }, index=dates)
    closes['CHF=X'] = 0.95 * np.exp(0.05 * np.sin(step / 90))
    closes['^IRX'] = 2.0 + 1.5 * np.sin(step / 300)
    closes.iloc[:LATE_START, closes.columns.get_loc('GSG')] = np.nan
    closes.iloc[QQQ_GAP, closes.columns.get_loc('QQQ')] = np.nan
    return closes.drop(index=dates[MISSING_DAY])


@offline_scratch()
def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    from dataclasses import replace
    from pathlib import Path
    from tempfile import mkdtemp

    import numpy as np
    import pandas as pd
    import qis

    from examples.covar_estimation.rolling_factor_covar_from_csv import RiskModelSettings

    data_dir = Path(mkdtemp(prefix="op-factor-csv-"))
    dates = pd.date_range("2016-12-31", "2022-12-31", freq="ME")
    step = np.arange(len(dates), dtype=float)
    factor_steps = np.column_stack((
        0.004 + 0.025 * np.sin(0.7 * step),
        0.002 + 0.012 * np.cos(0.4 * step),
    ))
    native_steps = np.column_stack((
        factor_steps[:, 0] + 0.004 * np.cos(1.3 * step),
        factor_steps[:, 1] + 0.003 * np.sin(1.1 * step),
        0.5 * factor_steps.sum(axis=1) + 0.002 * np.cos(1.7 * step),
    ))
    factor_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(factor_steps, axis=0)),
        index=dates, columns=["Equity", "Rates"],
    )
    asset_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(native_steps, axis=0)),
        index=dates, columns=["Growth", "Income", "Domestic"],
    )
    fx_spots = pd.DataFrame(
        {"USD": 1.0, "CHF": np.exp(0.03 * np.sin(0.3 * step))}, index=dates,
    )
    domestic_rates = pd.DataFrame(
        {"USD": 0.03 + 0.005 * np.sin(0.2 * step), "CHF": 0.01}, index=dates,
    )
    metadata = pd.DataFrame(
        {"currency": ["USD", "USD", "CHF"], "hedge_ratio": [0.0, 1.0, 0.0],
         "return_frequency": ["ME", "ME", "ME"]},
        index=pd.Index(asset_prices.columns, name="asset"),
    )
    settings = replace(
        RiskModelSettings.yahoo_demo(), factor_names=tuple(factor_prices.columns),
        estimation_end=pd.Timestamp("2022-12-31"),
    )
    qis.save_df_to_csv(factor_prices, file_name="futures_risk_factors", local_path=str(data_dir))
    qis.save_df_dict_to_csv(
        {"fx_spots": fx_spots, "domestic_rates": domestic_rates},
        file_name="fx_hedging_data", local_path=str(data_dir),
    )
    qis.save_df_to_csv(asset_prices, file_name="asset_prices", local_path=str(data_dir))
    qis.save_df_to_csv(metadata, file_name="asset_metadata", local_path=str(data_dir))
    qis.save_df_to_csv(
        settings.to_frame(), file_name="risk_model_settings", local_path=str(data_dir),
    )
    print(data_dir)

    # Six files inside the removed scratch directory: 73 month ends, two factors, three assets.
    assert Path(tempfile.gettempdir()) in data_dir.parents
    bundle_hashes = file_hashes(data_dir)
    assert set(bundle_hashes) == BUNDLE and len(BUNDLE) == 6
    assert len(dates) == 73 and dates.is_month_end.all()
    assert list(factor_prices.columns) == FACTORS and list(asset_prices.columns) == ASSETS
    # The page's 15 Yahoo settings are the example's defaults, in its CSV form; the bundle
    # changes only the factor names and the last snapshot date.
    yahoo = RiskModelSettings.yahoo_demo()
    documented = pd.read_csv(StringIO(YAHOO_SETTINGS), index_col=0, dtype=str)
    assert len(documented) == 15 and RiskModelSettings.from_frame(documented) == yahoo
    assert yahoo.to_frame().to_csv(lineterminator='\n') == YAHOO_SETTINGS
    assert settings == replace(yahoo, factor_names=tuple(FACTORS),
                               estimation_end=pd.Timestamp('2022-12-31'))
    written = {'factor_prices': factor_prices, 'asset_prices': asset_prices,
               'metadata': metadata, 'fx_spots': fx_spots, 'domestic_rates': domestic_rates,
               'settings': settings}

    from pathlib import Path
    import qis

    from examples.covar_estimation.rolling_factor_covar_from_csv import (
        load_inputs_from_csv,
    )

    inputs = load_inputs_from_csv(data_dir)

    assert isinstance(inputs.factors_data, qis.FactorsData)
    assert isinstance(inputs.fx_rates_data, qis.FxRatesData)
    factor_prices = inputs.factors_data.get_prices()
    asset_prices = inputs.asset_prices
    metadata = inputs.asset_metadata
    settings = inputs.settings

    # Every file round-trips: prices, FX, metadata and the typed settings load unchanged.
    assert {path.name for path in Path(data_dir).iterdir()} == BUNDLE
    for name, loaded in (('factor_prices', factor_prices), ('asset_prices', asset_prices),
                         ('metadata', metadata), ('fx_spots', inputs.fx_rates_data.fx_spots),
                         ('domestic_rates', inputs.fx_rates_data.domestic_rates)):
        pd.testing.assert_frame_equal(loaded, written[name], check_freq=False,
                                      check_names=False, rtol=1e-13, atol=1e-14)
    assert settings == written['settings']

    asset_returns_dict = inputs.fx_rates_data.compute_fx_adjusted_returns(
        prices=inputs.asset_prices,
        hedge_ratios=metadata["hedge_ratio"],
        local_ccys=metadata["currency"].astype(str),
        reference_ccy=settings.reference_ccy,
        freq=metadata["return_frequency"].astype(str),
        is_log_returns=settings.is_log_returns,
        is_excess_returns=settings.is_excess_returns,
    )

    # One monthly bucket of CHF total log returns: the log of wealth rebuilt from endpoint prices
    # and spots and the forward contracted at the period start. The first return is missing.
    assert set(asset_returns_dict) == {'ME'} and settings.reference_ccy == 'CHF'
    returns = asset_returns_dict['ME']
    for asset in ASSETS:
        currency, hedge = written['metadata'].loc[asset, ['currency', 'hedge_ratio']]
        spot = written['fx_spots'][currency] / written['fx_spots']['CHF']
        expected = hedged_log_returns(written['asset_prices'][asset], spot,
                                      written['domestic_rates'][currency],
                                      written['domestic_rates']['CHF'], hedge)
        np.testing.assert_allclose(returns[asset], expected, rtol=1e-11, atol=1e-14,
                                   equal_nan=True)
        assert np.isnan(returns[asset].iloc[0])
    # Excess returns subtract log(1 + starting reference cash); a varying CHF rate tells the
    # starting rate from the terminal one.
    rates = inputs.fx_rates_data.domestic_rates.copy()
    rates['CHF'] = 0.01 + 0.006 * np.sin(np.arange(len(rates)) * 0.7)
    varying = qis.FxRatesData(fx_spots=inputs.fx_rates_data.fx_spots.copy(),
                              domestic_rates=rates)
    conversion = dict(prices=inputs.asset_prices, hedge_ratios=metadata['hedge_ratio'],
                      local_ccys=metadata['currency'], reference_ccy='CHF',
                      freq=metadata['return_frequency'], is_log_returns=True)
    total = varying.compute_fx_adjusted_returns(**conversion, is_excess_returns=False)['ME']
    excess = varying.compute_fx_adjusted_returns(**conversion, is_excess_returns=True)['ME']
    np.testing.assert_allclose(excess, total.sub(np.log1p(rates['CHF'].shift() / 12), axis=0),
                               rtol=1e-11, atol=1e-14, equal_nan=True)
    assert not np.allclose(excess, total.sub(np.log1p(rates['CHF'] / 12), axis=0),
                           rtol=1e-11, atol=1e-14, equal_nan=True)
    # By default every exact zero return is missing, a genuine flat month included;
    # zero_return_to_nan=False keeps it.
    flat = {**conversion, 'prices': inputs.asset_prices.copy()}
    flat['prices'].iloc[25, 2] = flat['prices'].iloc[24, 2]
    dropped = inputs.fx_rates_data.compute_fx_adjusted_returns(**flat)['ME']
    assert dropped.iloc[24, 2] != 0 and np.isnan(dropped.iloc[25, 2])
    assert np.isfinite(dropped.iloc[26, 2])
    kept = inputs.fx_rates_data.compute_fx_adjusted_returns(**flat, zero_return_to_nan=False)
    assert kept['ME'].iloc[25, 2] == 0.0

    import optimalportfolios as opt
    import qis

    model_type = opt.LassoModelType[settings.lasso_model_type]
    lasso_model = opt.LassoModel(
        model_type=model_type,
        reg_lambda=settings.reg_lambda,
        span=settings.beta_span,
        warmup_period=settings.warmup_period,
        demean=settings.demean,
        solver=settings.solver,
    )
    estimator = opt.FactorCovarEstimator(
        rebalancing_freq=settings.rebalancing_freq,
        lasso_model=lasso_model,
        factor_returns_freq=settings.factor_returns_freq,
        factor_covar_span=settings.factor_covar_span,
        demean=settings.demean,
    )

    rolling = estimator.fit_rolling_factor_covars(
        risk_factor_prices=inputs.factors_data.get_prices(),
        asset_returns_dict=asset_returns_dict,
        assets=inputs.asset_prices.columns,
        time_period=qis.TimePeriod(
            start=settings.estimation_start,
            end=settings.estimation_end,
        ),
    )
    risk_model = opt.build_risk_model(rolling)

    latest_date = rolling.dates[-1]
    latest = rolling.get_latest()

    betas = latest.y_betas
    factor_covar = latest.x_covar
    asset_covars = rolling.get_y_covars()
    r_squared = rolling.get_r2()
    residual_variances = rolling.get_residual_vars()

    # Each dated snapshot holds factor covariance, betas, variances and diagnostics, residual
    # history and HCGL clusters and linkage; the panels collect them by date.
    assert latest_date == pd.Timestamp('2022-12-31') and latest is rolling[latest_date]
    assert betas.shape == (3, 2) and factor_covar.shape == (2, 2)
    residual_column = opt.VarianceColumns.RESIDUAL_VARS.value
    for date, snapshot in rolling.data.items():
        assert all(getattr(snapshot, name) is not None
                   for name in ('residuals', 'clusters', 'linkages'))
        pd.testing.assert_frame_equal(asset_covars[date], snapshot.get_y_covar())
        np.testing.assert_allclose(r_squared.loc[date], snapshot.y_variances['r2'])
        np.testing.assert_allclose(residual_variances.loc[date],
                                   snapshot.y_variances[residual_column])

    import numpy as np
    import optimalportfolios as opt

    snapshot = rolling.get_latest()
    betas = snapshot.y_betas
    factor_covar = snapshot.x_covar.reindex(
        index=betas.columns,
        columns=betas.columns,
    )
    residual_vars = snapshot.y_variances[
        opt.VarianceColumns.RESIDUAL_VARS.value
    ].reindex(betas.index)
    expected = (
        betas.to_numpy()
        @ factor_covar.to_numpy()
        @ betas.to_numpy().T
        + np.diag(residual_vars.to_numpy())
    )
    actual = snapshot.get_y_covar().reindex(
        index=betas.index,
        columns=betas.index,
    )
    np.testing.assert_allclose(
        actual.to_numpy(), expected, rtol=1.0e-12, atol=1.0e-14
    )

    # The page's table: four year-end snapshots of 3 x 2 betas, 2 x 2 factor and 3 x 3 asset
    # covariances. Each covariance is the explicit sum over labelled factor pairs plus the
    # residual variance, finite, with no eigenvalue below -1e-10; the factor covariance is the
    # monthly EWMA covariance with span 36 of demeaned log returns, times 12. The factor part
    # alone is singular, and the smallest eigenvalue is at least the smallest residual variance.
    pd.testing.assert_index_equal(rolling.dates, pd.DatetimeIndex(SNAPSHOT_DATES),
                                  check_names=False)
    assert asset_prices.shape == (73, 3) and factor_prices.shape == (73, 2)
    for date, snapshot in rolling.data.items():
        betas, factor = snapshot.y_betas, snapshot.x_covar
        assert betas.shape == (3, 2) and factor.shape == (2, 2)
        assert snapshot.estimation_date == date
        residual = snapshot.y_variances[residual_column]
        expected = pd.DataFrame(0.0, index=betas.index, columns=betas.index)
        for a in betas.index:
            for b in betas.index:
                expected.at[a, b] = sum(betas.at[a, f] * factor.at[f, g] * betas.at[b, g]
                                        for f in factor.index for g in factor.columns)
                if a == b:
                    expected.at[a, b] += residual[a]
        actual = snapshot.get_y_covar()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert np.isfinite(actual.to_numpy()).all()
        eigenvalues = np.linalg.eigvalsh(actual.to_numpy())
        assert eigenvalues.min() >= MIN_EIGENVALUE
        np.testing.assert_allclose(factor, ewma_factor_covariance(factor_prices.loc[:date], 36),
                                   rtol=1e-10, atol=1e-14)
        loadings = betas.to_numpy()
        factor_part = np.linalg.eigvalsh(
            loadings @ factor.loc[betas.columns, betas.columns].to_numpy() @ loadings.T)
        assert abs(factor_part.min()) < 1e-12 * factor_part.max()
        assert residual.min() > 0 and eigenvalues.min() >= residual.min() - 1e-15
    # The stored residuals are 12 times (asset log return minus fitted factor contribution),
    # with no alpha subtracted although the in-sample alpha is not zero.
    snapshot = rolling.get_latest()
    factor_returns = np.log(factor_prices / factor_prices.shift())
    expected = 12 * (returns - factor_returns @ snapshot.y_betas.T)
    np.testing.assert_allclose(snapshot.residuals.loc[expected.index[2:]], expected.iloc[2:],
                               rtol=1e-11, atol=1e-14)
    assert snapshot.y_variances['insample_alpha'].abs().max() > 1e-5
    assert snapshot.y_variances['r2'].between(0, 1).all()
    # The factor covariance ignores the estimator's demean field; the LASSO model uses its own.
    undemeaned = replace(estimator, demean=False,
                         lasso_model=opt.LassoModel(**lasso_model.get_params()))
    current = undemeaned.fit_current_factor_covars(
        risk_factor_prices=factor_prices.loc[:latest_date],
        asset_returns_dict={'ME': returns.loc[:latest_date]},
        assets=asset_prices.columns, estimation_date=latest_date)
    pd.testing.assert_frame_equal(current.x_covar, snapshot.x_covar)
    # The load command's check accepts a reconstruction error within rtol 1e-12 and atol 1e-14
    # and rejects one twice that size, any non-finite entry and an eigenvalue below -1e-10.
    one = opt.RollingFactorCovarData(data={latest_date: snapshot})
    covar = snapshot.get_y_covar()
    bound = RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * abs(covar.iat[0, 0])
    verify = csv_example._verify_rolling_decomposition
    for size, entry, error, message in ((0.5 * bound, (0, 0), None, ''),
                                        (2.0 * bound, (0, 0), AssertionError, 'tolerance'),
                                        (np.nan, (0, 1), ValueError, 'non-finite')):
        tampered = covar.copy()
        tampered.iat[entry] += size
        with patch.object(type(snapshot), 'get_y_covar', return_value=tampered):
            if error is None:
                verify(one)
            else:
                expect_failure(lambda: verify(one), error, message)
    for shift, accepted in ((0.5 * MIN_EIGENVALUE, True), (2.0 * MIN_EIGENVALUE, False)):
        # Three assets on two factors leave one zero eigenvalue for the shift to move.
        negative = replace(snapshot, y_variances=snapshot.y_variances.assign(
            **{residual_column: shift}))
        shifted = opt.RollingFactorCovarData(data={latest_date: negative})
        if accepted:
            verify(shifted)
        else:
            expect_failure(lambda: verify(shifted), ValueError, 'positive semidefinite')
    manual_rolling = rolling

    # The live fetch runs in a scratch working directory, with the download replaced by
    # deterministic closes.
    workdir = Path(mkdtemp(prefix="op-yahoo-fetch-"))
    with (patch.object(csv_example, '_download_close', side_effect=synthetic_closes) as download,
          working_directory(workdir)):
        from pathlib import Path

        from examples.covar_estimation.rolling_factor_covar_from_csv import (
            fetch_and_save_yahoo_csvs,
        )

        fetch_and_save_yahoo_csvs(Path("path/to/risk_model_inputs"))

    # One request from 2007-12-31 to 2026-01-01 (exclusive); the six files load. Rows missing
    # any series are dropped and the business-day grid is forward filled.
    download.assert_called_once_with(tickers=FETCH_TICKERS, start=FETCH_START, end=FETCH_END)
    fetched = workdir / 'path' / 'to' / 'risk_model_inputs'
    assert set(file_hashes(fetched)) == BUNDLE
    yahoo_inputs = load_inputs_from_csv(fetched)
    served = synthetic_closes(FETCH_TICKERS, FETCH_START, FETCH_END)
    business_days = pd.bdate_range(FETCH_START, FETCH_END, inclusive='left')
    grid = business_days[LATE_START:]
    closes = served.reindex(grid).ffill()
    yahoo_prices = yahoo_inputs.asset_prices
    pd.testing.assert_index_equal(yahoo_prices.index, grid, check_names=False, exact=False)
    assert business_days[MISSING_DAY] not in served.index
    for day, columns in ((MISSING_DAY, YAHOO_ASSETS), (QQQ_GAP, ['QQQ'])):
        pd.testing.assert_series_equal(yahoo_prices.loc[business_days[day], columns],
                                       yahoo_prices.loc[business_days[day - 1], columns],
                                       check_names=False)
    observed = served.loc[grid[0]:].dropna()
    np.testing.assert_allclose(yahoo_prices.loc[observed.index], observed[YAHOO_ASSETS],
                               rtol=1e-13)
    # USD per CHF inverts Yahoo's CHF per USD; the USD rate is ^IRX / 100 and the illustrative
    # CHF rate one percentage point lower. Five USD assets are unhedged, IEF and HYG hedged.
    spots, rates = yahoo_inputs.fx_rates_data.fx_spots, yahoo_inputs.fx_rates_data.domestic_rates
    assert (spots['USD'] == 1.0).all()
    np.testing.assert_allclose(spots['CHF'] * closes['CHF=X'], 1.0, rtol=1e-13)
    np.testing.assert_allclose(rates['USD'], closes['^IRX'] / 100, rtol=1e-13)
    np.testing.assert_allclose(rates['USD'] - rates['CHF'], 0.01, atol=1e-15)
    assert (rates['CHF'] < 0).any()  # negative rates load
    yahoo_metadata = yahoo_inputs.asset_metadata
    assert list(yahoo_metadata.index) == YAHOO_ASSETS
    assert (yahoo_metadata['currency'] == 'USD').all()
    assert (yahoo_metadata['return_frequency'] == 'ME').all()
    assert yahoo_metadata['hedge_ratio'].to_dict() == {
        asset: float(asset in HEDGED_ASSETS) for asset in YAHOO_ASSETS}
    assert yahoo_inputs.settings == yahoo
    # Each proxy becomes a CHF factor NAV with a monthly hedge of opening USD principal, and Fx
    # is the daily USD spot-and-carry NAV; both are checked by their monthly log returns.
    navs = yahoo_inputs.factors_data.get_prices()
    assert list(navs.columns) == list(yahoo.factor_names)
    month_ends = closes.resample('ME').last()
    usd_rate, chf_rate = month_ends['^IRX'] / 100, month_ends['^IRX'] / 100 - 0.01
    nav_returns = np.log(navs / navs.shift()).iloc[1:]
    for proxy, factor in FACTOR_PROXIES.items():
        expected = hedged_log_returns(month_ends[proxy], month_ends['CHF=X'], usd_rate,
                                      chf_rate, hedge=1.0)
        np.testing.assert_allclose(nav_returns[factor], expected.loc[nav_returns.index],
                                   rtol=1e-10, atol=1e-14)
    daily_usd, daily_chf = closes['^IRX'] / 100, closes['^IRX'] / 100 - 0.01
    carry = ((1 + daily_usd / BUSINESS_DAYS_PER_YEAR)
             / (1 + daily_chf / BUSINESS_DAYS_PER_YEAR) - 1).shift()
    fx_nav = (1 + (closes['CHF=X'].pct_change() + carry).fillna(0.0)).cumprod()
    fx_returns = np.log(fx_nav.resample('ME').last()).diff()
    np.testing.assert_allclose(nav_returns['Fx'], fx_returns.loc[nav_returns.index],
                               rtol=1e-10, atol=1e-14)

    from pathlib import Path

    from examples.covar_estimation.rolling_factor_covar_from_csv import (
        fit_rolling_risk_model_from_csv,
    )

    rolling, risk_model = fit_rolling_risk_model_from_csv(
        data_dir
    )

    # The convenience function refits the same snapshots and writes nothing; the adapter's
    # exposures are weighted sums of the dated betas, for unequal weights too.
    weights = pd.Series([0.2, 0.3, 0.5], index=ASSETS)
    for date, snapshot in rolling.data.items():
        pd.testing.assert_frame_equal(snapshot.get_y_covar(), manual_rolling[date].get_y_covar())
        exposures = risk_model.compute_exposures_at_date(weights, date=date)
        pd.testing.assert_series_equal(exposures, snapshot.y_betas.mul(weights, axis=0).sum(),
                                       check_names=False, rtol=1e-12)
    assert file_hashes(Path(data_dir)) == bundle_hashes

    # The load command in a fresh process imports no yfinance, opens no socket and prints the
    # snapshot count and date, the reconstruction error and four tables.
    loaded = subprocess.run([sys.executable, '-c', LOAD_GUARD, str(REPO_ROOT), str(data_dir)],
                            cwd=REPO_ROOT, text=True, capture_output=True, timeout=120)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    for line in ('Rolling snapshots: 4', 'Latest snapshot: 2022-12-31',
                 'Maximum covariance reconstruction error:', 'Latest factor loadings',
                 'Latest annualised factor covariance', 'Latest annualised residual volatilities',
                 'Equal-weight portfolio factor exposures'):
        assert line in loaded.stdout, line
    assert file_hashes(data_dir) == bundle_hashes
    # Without a mode the command fetches and then loads, in <checkout>/tmp/yahoo_factor_risk_model.
    with (patch.object(csv_example, 'fetch_and_save_yahoo_csvs') as fetch,
          patch.object(csv_example, 'fit_rolling_risk_model_from_csv') as load):
        csv_example.main([])
    default = REPO_ROOT / 'tmp' / 'yahoo_factor_risk_model'
    fetch.assert_called_once_with(data_dir=default)
    load.assert_called_once_with(data_dir=default)

    # Later factor, asset, spot and rate inputs change no earlier snapshot.
    perturbed = copy_bundle(data_dir)
    for name in ('futures_risk_factors.csv', 'asset_prices.csv', 'fx_hedging_data_fx_spots.csv',
                 'fx_hedging_data_domestic_rates.csv'):
        frame = pd.read_csv(perturbed / name, index_col=0)
        later = frame.index > FUTURE_CUTOFF
        column = 'CHF' if 'fx_spots' in name else frame.columns[0]
        frame.loc[later, column] *= np.linspace(1.1, 2.0, later.sum())
        frame.to_csv(perturbed / name)
    changed, _ = fit_rolling_risk_model_from_csv(perturbed)
    for date, snapshot in rolling.data.items():
        if date <= pd.Timestamp(FUTURE_CUTOFF):
            pd.testing.assert_frame_equal(snapshot.y_betas, changed[date].y_betas,
                                          rtol=1e-10, atol=1e-13)
            pd.testing.assert_frame_equal(snapshot.get_y_covar(), changed[date].get_y_covar(),
                                          rtol=1e-10, atol=1e-13)
    assert not np.allclose(changed.get_latest().get_y_covar(), rolling.get_latest().get_y_covar())

    # The loader rejects each documented defect with an actionable message.
    for defect, message in DEFECTS.items():
        broken = copy_bundle(data_dir)
        break_bundle(broken, defect)
        expect_failure(lambda: load_inputs_from_csv(broken), (ValueError, FileNotFoundError),
                       message)
    # It repairs or accepts what the page lists: unsorted assets, metadata, spots and rates;
    # a spot gap; a shorter rate history; a USD spot of 2; a fractional count; an unknown model
    # name, rejected only by the estimator; and an extra setting and file.
    repaired = copy_bundle(data_dir)
    rewrite_csv(repaired / 'asset_prices.csv', lambda f: f.iloc[::-1])
    rewrite_csv(repaired / 'asset_metadata.csv', lambda f: f.iloc[::-1])
    rewrite_csv(repaired / 'fx_hedging_data_domestic_rates.csv', lambda f: f.iloc[:-12][::-1])
    spots = pd.read_csv(repaired / 'fx_hedging_data_fx_spots.csv', index_col=0)
    spots.iloc[10, 1] = np.nan
    spots['USD'] = 2.0
    spots.iloc[::-1].to_csv(repaired / 'fx_hedging_data_fx_spots.csv')
    rewrite_csv(repaired / 'risk_model_settings.csv',
                lambda f: with_settings(f, beta_span='36.9', lasso_model_type='UNKNOWN',
                                        extra_setting='accepted'), dtype=str)
    (repaired / 'delivery-note.txt').write_text('An additional file is permitted.')
    accepted = load_inputs_from_csv(repaired)
    assert accepted.settings.beta_span == 36
    assert accepted.asset_prices.index.is_monotonic_increasing
    assert accepted.asset_metadata.index.equals(accepted.asset_prices.columns)
    accepted_spots = accepted.fx_rates_data.fx_spots
    assert accepted_spots.index.is_monotonic_increasing
    assert accepted_spots.iloc[10, 1] == spots.iloc[9, 1]
    assert (accepted_spots['USD'] == 2.0).all()
    accepted_rates = accepted.fx_rates_data.domestic_rates
    assert accepted_rates.index[-1] == pd.Timestamp('2022-12-31')
    pd.testing.assert_series_equal(accepted_rates.iloc[-1], accepted_rates.iloc[-13],
                                   check_names=False)
    expect_failure(lambda: fit_rolling_risk_model_from_csv(repaired), ValueError, 'UNKNOWN')
    print("rolling_factor_covar_from_csv: all page statements verified.")


if __name__ == '__main__':
    main()
