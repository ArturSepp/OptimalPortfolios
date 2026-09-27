"""Canonical script of docs/alphas_module_readme.md.

The page's fifteen Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: the page's filter formulas as explicit geometric sums, an EWMA regression written
as loops, endpoint price ratios, residuals rebuilt from dated loadings, and clipped population
or sample z-scores written out by hand. Later prices, benchmark, carry and cluster labels are
perturbed to show that no earlier signal or score moves. Sockets are blocked while the blocks
run. The script runs offline after ``pip install optimalportfolios`` and needs no data file or
random seed:

    python -m examples.docs.alphas_module_readme

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import contextlib
from dataclasses import fields, replace
import inspect
import socket
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

# Formation dates up to this cutoff must not move when later inputs change.
CUTOFF = '2023-06-30'
BETA_FAMILIES = ['low_beta', 'residual_momentum', 'residual_reversal']
FAMILIES = ['momentum', 'classic_momentum', *BETA_FAMILIES, 'ra_carry']
# The page's signal matrix: every paired constructor exported by the signals subpackage.
SIGNAL_MATRIX = [f'compute_{family}{suffix}_alpha' for family in FAMILIES
                 for suffix in ('', '_cluster')] + ['compute_managers_alpha']
# The page's AlphasData Fields table, in the dataclass order.
ALPHAS_DATA_FIELDS = [
    'alpha_scores', 'momentum', 'momentum_score', 'momentum_cluster', 'momentum_cluster_score',
    'beta', 'beta_score', 'beta_cluster', 'beta_cluster_score', 'managers_alphas',
    'managers_scores', 'residual_momentum', 'residual_momentum_score',
    'residual_momentum_cluster', 'residual_momentum_cluster_score', 'clusters',
]
# The page's scoring table: raw value, standard score and cluster score of assets A to F.
PROBE_TABLE = [[-4, -1.517929, -1.161895], [-2, -0.889821, -0.387298],
               [0, -0.261712, 0.387298], [2, 0.366397, 1.161895],
               [4, 0.994505, 0.617213], [8, 1.308560, 1.543033]]
# The exhibit: annual log drift of each synthetic asset, its cluster, and the classic
# momentum settings; the drift is then exactly the twelve-month signal at every formation date.
PANEL_START = '2023-10-31'
FORMATION_DATE = '2024-12-31'
TRENDS = {'Equity A': 0.22, 'Equity B': 0.16, 'Equity C': 0.12, 'Equity D': 0.08,
          'Equity E': 0.02, 'Bond A': 0.05, 'Bond B': 0.03, 'Bond C': 0.0, 'Bond D': -0.03}
CLUSTERS = {'Equity A': 'Equity-like', 'Equity B': 'Equity-like', 'Equity C': 'Equity-like',
            'Equity D': 'Equity-like', 'Equity E': 'Equity-like', 'Bond A': 'Bond-like',
            'Bond B': 'Bond-like', 'Bond C': 'Bond-like', 'Bond D': 'Bond-like'}
LOOKBACK = 12
SKIP = 1
MIN_CLUSTER_SIZE = 3
OFFLINE = AssertionError('The alpha walkthrough must remain offline')


@contextlib.contextmanager
def offline():
    """Deny socket connections; used as the decorator of ``main``."""
    with (patch.object(socket, 'create_connection', side_effect=OFFLINE),
          patch.object(socket.socket, 'connect', side_effect=OFFLINE)):
        yield


def log_returns(prices):
    """Log returns between adjacent rows of a price panel or series, with a zero first row."""
    returns = np.log(prices / prices.shift())
    returns.iloc[0] = 0.0
    return returns


def geometric_sums(values: np.ndarray, span: float) -> np.ndarray:
    """Rows of the sum over j up to t of lambda^(t-j) values_j, for lambda = 1 - 2/(span+1)."""
    decay = 1 - 2 / (span + 1)
    age = np.subtract.outer(np.arange(len(values)), np.arange(len(values)))
    return np.where(age >= 0, decay ** np.maximum(age, 0), 0.0) @ values


def seeded_ewma(values: np.ndarray, span: float) -> np.ndarray:
    """EWMA whose state starts at the first row: lambda^t x_0 plus (1-lambda) sums of the rest."""
    decay = 1 - 2 / (span + 1)
    later = np.array(values, dtype=float)
    later[0] = 0.0
    start = np.multiply.outer(decay ** np.arange(len(values)), np.asarray(values)[0])
    return start + (1 - decay) * geometric_sums(later, span)


def filter_reference(values: np.ndarray, long_span: int, short_span=None) -> np.ndarray:
    """Long minus short geometric sums, scaled to unit variance for unit white noise."""
    long_decay = 1 - 2 / (long_span + 1)
    if short_span is None:
        return np.sqrt(1 - long_decay ** 2) * geometric_sums(values, long_span)
    short_decay = 1 - 2 / (short_span + 1)
    variance = (1 / (1 - long_decay ** 2) + 1 / (1 - short_decay ** 2)
                - 2 / (1 - long_decay * short_decay))
    return (geometric_sums(values, long_span) - geometric_sums(values, short_span)) / np.sqrt(
        variance)


def ewma_beta_reference(benchmark_returns: pd.Series, returns: pd.DataFrame, span: int,
                        demean: bool = True) -> pd.DataFrame:
    """EWMA beta by explicit loops: means start at each column's first return, moments at zero."""
    decay = 1 - 2 / (span + 1)

    def centred(column: np.ndarray) -> np.ndarray:
        """Subtract a running mean that starts at the column's first finite value."""
        out = np.full(len(column), np.nan)
        mean = np.nan
        for t, value in enumerate(column):
            if np.isfinite(value):
                mean = value if np.isnan(mean) else decay * mean + (1 - decay) * value
                out[t] = value - mean if demean else value
        return out

    x = centred(benchmark_returns.to_numpy())
    betas = np.full(returns.shape, np.nan)
    for i, column in enumerate(returns.to_numpy().T):
        y = centred(column)
        cross = second = 0.0
        for t in range(len(x)):
            if np.isfinite(x[t]) and np.isfinite(y[t]):
                cross = decay * cross + (1 - decay) * x[t] * y[t]
            if np.isfinite(x[t]):
                second = decay * second + (1 - decay) * x[t] ** 2
            if t > span and second > 0:  # the warm-up masks rows 0 to beta_span
                betas[t, i] = cross / second
    betas[betas == 0.0] = np.nan  # the raw-beta path reports an exact zero as missing
    return pd.DataFrame(betas, index=returns.index, columns=returns.columns)


def clipped_population_score(raw: pd.DataFrame) -> pd.DataFrame:
    """Clip at plus or minus 5, then centre and scale each row with population moments."""
    clipped = raw.clip(-5.0, 5.0)
    return clipped.sub(clipped.mean(axis=1), axis=0).div(clipped.std(axis=1, ddof=0), axis=0)


def sample_score(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Centre and scale ``values`` with the sample (ddof=1) moments of ``reference``."""
    return (values - reference.mean()) / reference.std(ddof=1)


def cluster_score_reference(raw: pd.DataFrame, clusters: dict, min_size: int) -> pd.DataFrame:
    """Row by row: within-cluster sample z-scores above ``min_size``, assigned-universe below."""
    dates = sorted(clusters)
    scores = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for date, row in raw.iterrows():
        known = [d for d in dates if d <= date]
        if not known:
            continue
        labels = clusters[known[-1]]
        assigned = row[labels.index]
        for label in labels.unique():
            members = labels.index[labels == label]
            reference = row[members] if len(members) > min_size else assigned
            scores.loc[date, members] = sample_score(row[members], reference)
    return scores


def assert_raises(error: type, function, *args, **kwargs) -> None:
    """Fail unless ``function(*args, **kwargs)`` raises ``error``."""
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def without_benchmark(function, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Raw signal with no benchmark; the all-missing first row's mean warning is silenced."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        return function(prices, **kwargs)[1]


def family_arguments(family: str, benchmark: pd.Series, carry: pd.DataFrame) -> dict:
    """The monthly arguments of one family: a benchmark or a carry panel where it takes one."""
    arguments = {'returns_freq': 'ME'}
    if family not in ('classic_momentum', 'ra_carry'):
        arguments['benchmark_price'] = benchmark
    if family == 'ra_carry':
        arguments['carry'] = carry
    return arguments


def exhibit_prices() -> pd.DataFrame:
    """Month-end prices whose log drift per year is each asset's entry in ``TRENDS``."""
    dates = pd.date_range(PANEL_START, FORMATION_DATE, freq='ME')
    months = np.arange(len(dates), dtype=float)
    return pd.DataFrame({asset: 100 * np.exp(trend * months / 12)
                         for asset, trend in TRENDS.items()}, index=dates)


def exhibit_scores() -> pd.DataFrame:
    """Classic momentum of the exhibit universe, scored across the panel and within clusters."""
    import optimalportfolios.alphas.signals as signals

    prices = exhibit_prices()
    labels = pd.Series(CLUSTERS)
    clusters = {prices.index[0]: labels}
    cross_section, raw = signals.compute_classic_momentum_alpha(
        prices, returns_freq='ME', lookback_periods=LOOKBACK, skip_periods=SKIP)
    within, cluster_raw = signals.compute_classic_momentum_cluster_alpha(
        prices, rolling_clusters=clusters, returns_freq='ME', lookback_periods=LOOKBACK,
        skip_periods=SKIP, min_cluster_size=MIN_CLUSTER_SIZE)
    formation = pd.Timestamp(FORMATION_DATE)
    table = pd.DataFrame({
        'cluster': labels,
        'raw_signal': raw.loc[formation],
        'cluster_raw_signal': cluster_raw.loc[formation],
        'cross_section_score': cross_section.loc[formation],
        'within_cluster_score': within.loc[formation],
    })
    table['cross_section_rank'] = table['cross_section_score'].rank(ascending=False)
    table['within_cluster_rank'] = table['within_cluster_score'].rank(ascending=False)
    return table


@offline()
def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd
    import qis
    import optimalportfolios as opt
    import optimalportfolios.alphas as alphas
    import optimalportfolios.alphas.signals as signals

    dates = pd.date_range("2016-12-31", "2024-12-31", freq="ME")
    step = np.arange(len(dates), dtype=float)
    factor_changes = np.column_stack((
        0.005 + 0.025 * np.sin(0.7 * step),
        0.002 + 0.012 * np.cos(0.4 * step),
    ))
    factor_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(factor_changes, axis=0)),
        index=dates, columns=["Growth", "Rates"],
    )
    loadings = np.array([
        [1.2, 0.1], [0.9, 0.3], [0.7, 0.2], [0.5, 0.4],
        [0.2, 1.1], [0.3, 0.8], [0.4, 0.6], [0.6, 0.5],
    ])
    specific_changes = (
        0.003 * np.sin(step[:, None] * np.arange(0.9, 1.7, 0.1)[None, :])
        + np.arange(8)[None, :] * 0.0002
    )
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(factor_changes @ loadings.T + specific_changes, axis=0)),
        index=dates, columns=list("ABCDEFGH"),
    )
    asset_prices = prices
    benchmark = factor_prices["Growth"]
    carry = pd.DataFrame(
        np.broadcast_to(np.linspace(0.02, 0.05, 8), prices.shape),
        index=dates, columns=prices.columns,
    )

    # Eight assets and two factors on 97 month ends, 2016-12-31 to 2024-12-31; the benchmark
    # is the Growth factor and carry is an annual decimal from 2% to 5%.
    assert len(prices) == 97 and prices.shape[1] == 8 and asset_prices is prices
    assert (prices.index[0], prices.index[-1]) == (pd.Timestamp("2016-12-31"),
                                                   pd.Timestamp("2024-12-31"))
    assert list(factor_prices.columns) == ["Growth", "Rates"] and benchmark.name == "Growth"
    assert np.isfinite(prices.to_numpy()).all() and (prices.to_numpy() > 0).all()
    assert (carry.to_numpy() == np.linspace(0.02, 0.05, 8)).all() and carry.shape == (97, 8)
    # Each one-period log return is the two-factor exposure plus a small specific drift.
    np.testing.assert_allclose(log_returns(prices).iloc[1:],
                               (factor_changes @ loadings.T + specific_changes)[1:],
                               rtol=0, atol=1e-13)

    from optimalportfolios.alphas import compute_momentum_alpha

    score, raw = compute_momentum_alpha(
        prices=prices,
        benchmark_price=benchmark,
        returns_freq='ME',
        long_span=12,
    )
    mom_score, raw_momentum = score, raw

    # Defaults: long span 12, no short leg, volatility span 13, no mean adjustment.
    defaults = inspect.signature(signals.compute_momentum_alpha).parameters
    assert (defaults['long_span'].default, defaults['short_span'].default,
            defaults['vol_span'].default) == (12, None, 13)
    assert defaults['mean_adj_type'].default == qis.MeanAdjType.NONE
    # The warm-up masks the first 13 rows (the missing first return and 12 more).
    assert raw_momentum.iloc[:13].isna().all().all()
    assert np.isfinite(raw_momentum.iloc[13:].to_numpy()).all()
    # The page's filter: benchmark-relative log returns over a contemporaneous EWMA volatility
    # whose variance starts at the first observed square, then sqrt(1 - lambda^2) times the
    # geometric sum.
    relative = log_returns(prices).sub(log_returns(benchmark), axis=0).to_numpy()
    variance = np.zeros(relative.shape)
    variance[1:] = seeded_ewma(relative[1:] ** 2, span=13)
    normalised = np.divide(relative, np.sqrt(variance), out=np.zeros_like(relative),
                           where=variance > 0)
    np.testing.assert_allclose(raw_momentum.iloc[13:], filter_reference(normalised, 12)[13:],
                               rtol=1e-11, atol=1e-12)
    # No benchmark means no subtraction; a short leg subtracts a second geometric sum rather
    # than skipping recent returns; vol_span=None disables the normalisation.
    own = log_returns(prices).to_numpy()
    for short_span in (None, 3):
        _, unbenchmarked = signals.compute_momentum_alpha(
            prices, benchmark_price=None, vol_span=None, short_span=short_span)
        np.testing.assert_allclose(unbenchmarked.iloc[13:],
                                   filter_reference(own, 12, short_span)[13:],
                                   rtol=1e-11, atol=1e-12)

    from optimalportfolios.alphas import compute_low_beta_alpha

    score, raw_beta = compute_low_beta_alpha(
        prices=prices,
        benchmark_price=benchmark,
        returns_freq='ME',
        beta_span=12,
    )
    beta_score = score

    # Defaults: beta span 12 with EWMA means.
    defaults = inspect.signature(signals.compute_low_beta_alpha).parameters
    assert defaults['beta_span'].default == 12
    assert defaults['mean_adj_type'].default == qis.MeanAdjType.EWMA
    # The raw beta is an EWMA regression whose means start at each column's first return.
    asset_returns = np.log(prices / prices.shift())
    benchmark_returns = np.log(benchmark / benchmark.shift())
    pd.testing.assert_frame_equal(raw_beta, ewma_beta_reference(benchmark_returns, asset_returns,
                                                                span=12),
                                  check_freq=False, check_names=False, rtol=1e-10, atol=1e-12)
    # A late starter's mean starts at its own first return after inception, and later prices
    # leave its earlier betas unchanged. That first return is centred to zero, so its beta is
    # an exact zero, reported missing; a zero seed would have given a finite beta there.
    late = prices.copy()
    late.loc[:"2019-11-30", "H"] = np.nan
    _, late_beta = signals.compute_low_beta_alpha(late, benchmark_price=benchmark)
    late_reference = ewma_beta_reference(benchmark_returns, np.log(late / late.shift()), span=12)
    pd.testing.assert_frame_equal(late_beta, late_reference, check_freq=False, check_names=False,
                                  rtol=1e-10, atol=1e-12)
    assert late_beta.loc[:"2020-01-31", "H"].isna().all()
    assert late_beta.loc["2020-02-29":, "H"].notna().all()
    moved = late.copy()
    moved.loc[moved.index > pd.Timestamp(CUTOFF), "H"] *= 1.5
    _, moved_beta = signals.compute_low_beta_alpha(moved, benchmark_price=benchmark)
    pd.testing.assert_frame_equal(moved_beta.loc[:CUTOFF], late_beta.loc[:CUTOFF])
    # NONE regresses through the origin: a different estimator on the same data.
    _, origin_beta = signals.compute_low_beta_alpha(prices, benchmark_price=benchmark,
                                                    mean_adj_type=qis.MeanAdjType.NONE)
    pd.testing.assert_frame_equal(origin_beta, ewma_beta_reference(
        benchmark_returns, asset_returns, span=12, demean=False), check_freq=False,
        check_names=False, rtol=1e-10, atol=1e-12)
    assert not np.allclose(origin_beta.iloc[13:], raw_beta.iloc[13:])
    # Known betas are recovered from proportional returns; the score reverses the order, and
    # clipping is a scoring step that leaves a raw beta above 5 intact.
    true_betas = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0])
    proportional = pd.DataFrame((benchmark.to_numpy()[:, None] / benchmark.iloc[0]) ** true_betas,
                                index=dates, columns=prices.columns)
    known_score, known_beta = signals.compute_low_beta_alpha(proportional,
                                                             benchmark_price=benchmark)
    np.testing.assert_allclose(known_beta.iloc[13:],
                               np.broadcast_to(true_betas, known_beta.iloc[13:].shape),
                               rtol=1e-10, atol=1e-11)
    assert (known_score.iloc[-1].diff().iloc[1:] < 0).all() and known_beta.iloc[-1, -1] > 5
    # An exact zero loading is reported as missing, not as a zero beta.
    _, flat_beta = signals.compute_low_beta_alpha(prices.assign(H=100.0),
                                                  benchmark_price=benchmark)
    assert flat_beta["H"].isna().all() and flat_beta["A"].iloc[13:].notna().all()
    # Without a benchmark, the fit uses the equal-weight mean log return of the assets passed.
    mean_benchmark = pd.Series(np.exp(asset_returns.mean(axis=1).fillna(0.0).cumsum()),
                               index=dates)
    implicit_beta = without_benchmark(signals.compute_low_beta_alpha, prices)
    _, explicit_beta = signals.compute_low_beta_alpha(prices, benchmark_price=mean_benchmark)
    np.testing.assert_allclose(implicit_beta, explicit_beta, rtol=1e-9, atol=1e-12,
                               equal_nan=True)

    from optimalportfolios.alphas import compute_residual_momentum_alpha

    score, raw_residual = compute_residual_momentum_alpha(
        prices=prices,
        benchmark_price=benchmark,
        returns_freq='ME',
        beta_span=12,
        long_span=12,
        vol_span=13,
    )
    res_score = score

    # Defaults: beta and long spans 12, volatility span 13, no short leg.
    defaults = inspect.signature(signals.compute_residual_momentum_alpha).parameters
    assert (defaults['beta_span'].default, defaults['long_span'].default,
            defaults['vol_span'].default, defaults['short_span'].default) == (12, 12, 13, None)
    # The residual uses the preceding observation's beta; a one-period filter without
    # normalisation exposes it directly.
    residual_arguments = dict(prices=prices, benchmark_price=benchmark, beta_span=12,
                              long_span=1, vol_span=None)
    one_period_score, one_period = signals.compute_residual_momentum_alpha(**residual_arguments)
    # The first residual needs the beta of row 13, so it is row 14; each filter then masks its
    # own warm-up after that row: one row for the one-period filter, 12 for the default.
    lagged_beta = ewma_beta_reference(benchmark_returns, asset_returns, span=12).shift(1)
    expected_residual = log_returns(prices) - lagged_beta.mul(log_returns(benchmark), axis=0)
    assert expected_residual.iloc[:14].isna().all().all()
    assert one_period.iloc[:15].isna().all().all() and raw_residual.iloc[:26].isna().all().all()
    np.testing.assert_allclose(one_period.iloc[15:], expected_residual.iloc[15:], rtol=1e-10,
                               atol=1e-12)
    # With the default settings, the residual passes through the same volatility filter, whose
    # variance starts at the first residual's square.
    residual = expected_residual.fillna(0.0).to_numpy()
    variance = np.zeros(residual.shape)
    variance[14:] = seeded_ewma(residual[14:] ** 2, span=13)
    normalised = np.divide(residual, np.sqrt(variance), out=np.zeros_like(residual),
                           where=variance > 0)
    np.testing.assert_allclose(raw_residual.iloc[26:], filter_reference(normalised, 12)[26:],
                               rtol=1e-10, atol=1e-12)
    # Without a benchmark, the residual regression also uses the equal-weight mean log return.
    np.testing.assert_allclose(
        without_benchmark(signals.compute_residual_momentum_alpha, prices),
        signals.compute_residual_momentum_alpha(prices, benchmark_price=mean_benchmark)[1],
        rtol=1e-9, atol=1e-12, equal_nan=True)

    classic_score, raw_classic = signals.compute_classic_momentum_alpha(
        prices, returns_freq="ME", lookback_periods=12, skip_periods=1,
    )
    reversal_score, raw_reversal = signals.compute_residual_reversal_alpha(
        prices, benchmark_price=benchmark, returns_freq="ME", long_span=1,
    )
    carry_score, raw_carry = signals.compute_ra_carry_alpha(
        prices, carry=carry, returns_freq="ME", vol_span=13,
    )
    legacy_carry_score = alphas.compute_ra_carry_alphas(
        prices, carry=carry, returns_freq="ME", vol_span=13,
    )

    # Classic momentum: exactly twelve log returns, skipping the latest, as one price ratio;
    # the returns helper gives the same raw sum and nothing else.
    defaults = inspect.signature(signals.compute_classic_momentum_alpha).parameters
    assert (defaults['lookback_periods'].default, defaults['skip_periods'].default) == (12, 1)
    np.testing.assert_allclose(raw_classic.iloc[13:],
                               np.log(prices.shift(1) / prices.shift(13)).iloc[13:],
                               rtol=1e-11, atol=1e-14)
    assert raw_classic.iloc[:13].isna().all().all()
    direct = signals.compute_classic_momentum_from_returns(
        np.log(prices / prices.shift()), lookback_periods=12, skip_periods=1)
    pd.testing.assert_frame_equal(direct, raw_classic, check_freq=False, rtol=1e-11,
                                  atol=1e-14)
    # Reversal: default long span 1; with every setting equal it is the negated residual
    # momentum, score and raw, but the default outputs are not negatives of each other.
    defaults = inspect.signature(signals.compute_residual_reversal_alpha).parameters
    assert defaults['long_span'].default == 1 and defaults['beta_span'].default == 12
    reversed_score, reversed_raw = signals.compute_residual_reversal_alpha(**residual_arguments)
    np.testing.assert_allclose(reversed_raw, -one_period, equal_nan=True, atol=1e-13)
    np.testing.assert_allclose(reversed_score, -one_period_score, equal_nan=True, atol=1e-12)
    assert not np.allclose(raw_reversal.iloc[30:], -raw_residual.iloc[30:])
    # Carry: yield over annualised EWMA volatility whose variance starts at the first square;
    # vol_span=None keeps the normalisation with the qis decay 0.94.
    defaults = inspect.signature(signals.compute_ra_carry_alpha).parameters
    assert (defaults['returns_freq'].default, defaults['vol_span'].default) == ("W-WED", 13)
    default_decay_carry = signals.compute_ra_carry_alpha(prices, carry=carry, returns_freq="ME",
                                                         vol_span=None)[1]
    for ra_carry, span in ((raw_carry, 13), (default_decay_carry, 2 / (1 - 0.94) - 1)):
        annual_vol = np.sqrt(12 * seeded_ewma(own[1:] ** 2, span=span))
        np.testing.assert_allclose(ra_carry.iloc[1:], carry.to_numpy()[1:] / annual_vol,
                                   rtol=1e-11, atol=1e-12)
    # The legacy helper returns only the global score; the singular pair is not exported by
    # the parent package.
    pd.testing.assert_frame_equal(legacy_carry_score, carry_score)
    assert not hasattr(alphas, "compute_ra_carry_alpha") and hasattr(alphas,
                                                                     "compute_ra_carry_alphas")
    # All six standard scores clip at 5 and use population moments; low beta reverses the sign.
    raws = {"momentum": raw_momentum, "classic_momentum": raw_classic, "low_beta": raw_beta,
            "residual_momentum": raw_residual, "residual_reversal": raw_reversal,
            "ra_carry": raw_carry}
    standard_scores = {"momentum": mom_score, "classic_momentum": classic_score,
                       "low_beta": beta_score, "residual_momentum": res_score,
                       "residual_reversal": reversal_score, "ra_carry": carry_score}
    for family, raw_signal in raws.items():
        expected = clipped_population_score(-raw_signal if family == "low_beta" else raw_signal)
        np.testing.assert_allclose(standard_scores[family], expected, rtol=1e-11, atol=1e-12,
                                   equal_nan=True)
        assert raw_signal.shape == (97, 8)
    # Clipping precedes standardisation, so a score is bounded by neither 1 nor 5; a singleton
    # or zero-dispersion group has no score.
    assert qis.df_to_cross_sectional_score(pd.DataFrame([[0.0] * 29 + [5.0]])).iloc[0, -1] > 5
    for degenerate in ([[1.0]], [[2.0, 2.0, 2.0]]):
        assert qis.df_to_cross_sectional_score(pd.DataFrame(degenerate)).isna().all().all()
    # The signal matrix: 13 paired constructors, each returning (score, raw) with a monthly
    # default cadence except carry.
    exported = {name for name, value in vars(signals).items()
                if name.startswith("compute_") and name.endswith("_alpha") and callable(value)}
    assert exported == set(SIGNAL_MATRIX) and len(exported) == 13
    for name in SIGNAL_MATRIX:
        cadence = inspect.signature(getattr(signals, name)).parameters["returns_freq"].default
        assert cadence == ("W-WED" if "carry" in name else "ME")

    factor_estimator = opt.FactorCovarEstimator(
        rebalancing_freq="YE", factor_returns_freq="ME", factor_covar_span=36,
        lasso_model=opt.LassoModel(
            model_type=opt.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1.0e-5, span=36, warmup_period=36, demean=True, solver="CLARABEL",
        ),
    )
    rolling_data = factor_estimator.fit_rolling_factor_covars(
        risk_factor_prices=factor_prices,
        asset_returns_dict={
            "ME": qis.to_returns(prices, freq="ME", is_log_returns=True, drop_first=True)
        },
        assets=prices.columns,
        time_period=qis.TimePeriod("2022-12-31", "2024-12-31"),
    )
    taa_covar_data = rolling_data
    asset_prices = prices.loc["2022-11-30":]

    # Three year-end fits with dated loadings and clusters; the manager panel keeps the
    # November 2022 endpoint before the first loading date.
    year_ends = list(pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"]))
    assert list(rolling_data.dates) == year_ends and list(rolling_data.get_y_betas()) == year_ends
    assert asset_prices.index[0] == pd.Timestamp("2022-11-30") and len(asset_prices) == 26

    from optimalportfolios.alphas import compute_managers_alpha

    score, raw_alpha = compute_managers_alpha(
        prices=asset_prices,
        risk_factor_prices=factor_prices,
        estimated_betas=rolling_data.get_y_betas(),
        returns_freq='ME',
        alpha_span=12,
    )
    mgr_score = score

    # Residuals use the latest loadings dated at or before the preceding return date and the
    # factor returns over the asset's own periods; manager returns begin in January 2023.
    beta_history = rolling_data.get_y_betas()
    manager_returns = np.log(asset_prices / asset_prices.shift())
    factor_returns = np.log(factor_prices / factor_prices.shift())
    residuals = {}
    for previous, current in zip(asset_prices.index[1:-1], asset_prices.index[2:]):
        eligible = [date for date in beta_history if date <= previous]
        if eligible:
            loading = beta_history[max(eligible)].loc[asset_prices.columns, factor_prices.columns]
            residuals[current] = manager_returns.loc[current] - loading.mul(
                factor_returns.loc[current], axis=1).sum(axis=1)
    residuals = pd.DataFrame.from_dict(residuals, orient="index")
    assert residuals.index[0] == pd.Timestamp("2023-01-31") == raw_alpha.index[0]
    # Annual scaling by 12 and EWMA smoothing with span 12; the score divides by the population
    # standard deviation without centring, so a common mean remains.
    expected_alpha = 12 * seeded_ewma(residuals.to_numpy(), span=12)
    pd.testing.assert_index_equal(raw_alpha.index, residuals.index)
    np.testing.assert_allclose(raw_alpha, expected_alpha, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mgr_score, expected_alpha / np.std(expected_alpha, axis=1,
                                                                  keepdims=True),
                               rtol=1e-10, atol=1e-11)
    assert np.abs(mgr_score.mean(axis=1)).max() > 0.1
    _, period_alpha = signals.compute_managers_alpha(asset_prices, factor_prices, beta_history,
                                                     returns_freq="ME", alpha_span=12,
                                                     annualise=False)
    np.testing.assert_allclose(period_alpha * 12, raw_alpha, rtol=1e-10, atol=1e-12)
    defaults = inspect.signature(signals.compute_managers_alpha).parameters
    assert defaults["alpha_span"].default == 12 and defaults["annualise"].default is True
    assert "group_data" not in defaults and not hasattr(signals, "compute_managers_cluster_alpha")
    # A snapshot dated at a return endpoint cannot residualise that period's return.
    extreme = {date: loading.copy() for date, loading in beta_history.items()}
    extreme[max(extreme)] *= 100
    extreme_score, extreme_alpha = signals.compute_managers_alpha(asset_prices, factor_prices,
                                                                  extreme)
    pd.testing.assert_frame_equal(extreme_alpha, raw_alpha)
    pd.testing.assert_frame_equal(extreme_score, mgr_score)

    from optimalportfolios.alphas.signals import extract_rolling_clusters

    rolling_clusters = extract_rolling_clusters(
        rolling_covar_data=taa_covar_data,
        assets=prices.columns.tolist(),
    )
    # Dict[pd.Timestamp, pd.Series]  →  {date: pd.Series(ticker → cluster_id)}

    # The labels are the ones each HCGL fit stored, one date per fit, with a cadence prefix.
    assert list(rolling_clusters) == year_ends
    for date, labels in rolling_clusters.items():
        pd.testing.assert_series_equal(labels, rolling_data[date].clusters.reindex(prices.columns),
                                       check_names=False)
        assert labels.astype(str).str.startswith("ME:").all()

    from optimalportfolios.alphas.signals.utils import score_within_clusters

    cluster_score = score_within_clusters(
        raw_signal=raw_momentum,        # T × N DataFrame
        rolling_clusters=rolling_clusters,
    )

    # Zero before the first assignment, even where the raw signal is missing; afterwards
    # within-cluster sample statistics above three members and assigned-universe statistics
    # otherwise.
    assert (cluster_score.loc[:"2022-11-30"] == 0.0).all().all()
    assert raw_momentum.loc[:"2017-12-31"].isna().all().all()
    np.testing.assert_allclose(cluster_score, cluster_score_reference(raw_momentum,
                                                                      rolling_clusters, 3),
                               rtol=1e-12, atol=1e-12)
    # Aligning labels through time keeps memberships and scores.
    aligned, _ = alphas.align_rolling_clusters(rolling_clusters)
    for date, labels in rolling_clusters.items():
        pairs = pd.crosstab(labels, aligned[date]).gt(0)
        assert pairs.sum(axis=0).eq(1).all() and pairs.sum(axis=1).eq(1).all()
    pd.testing.assert_frame_equal(score_within_clusters(raw_momentum, aligned), cluster_score)

    from optimalportfolios.alphas import compute_momentum_cluster_alpha

    score, raw = compute_momentum_cluster_alpha(
        prices=prices,
        benchmark_price=benchmark,
        rolling_clusters=rolling_clusters,
        returns_freq='ME',
        long_span=12,
    )
    mom_cluster_score, raw_momentum_cluster = score, raw
    beta_cluster_score, raw_beta_cluster = signals.compute_low_beta_cluster_alpha(
        prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
    )
    res_cluster_score, raw_residual_cluster = signals.compute_residual_momentum_cluster_alpha(
        prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
    )
    classic_cluster_score, raw_classic_cluster = signals.compute_classic_momentum_cluster_alpha(
        prices, rolling_clusters=rolling_clusters,
    )
    reversal_cluster_score, raw_reversal_cluster = signals.compute_residual_reversal_cluster_alpha(
        prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
    )
    carry_cluster_score, raw_carry_cluster = signals.compute_ra_carry_cluster_alpha(
        prices, carry=carry, returns_freq="ME", rolling_clusters=rolling_clusters,
    )

    # With one cadence and an explicit benchmark, each cluster variant keeps its standard raw
    # signal and scores it within the clusters (low beta after negation).
    cluster_raws = {"momentum": raw_momentum_cluster, "classic_momentum": raw_classic_cluster,
                    "low_beta": raw_beta_cluster, "residual_momentum": raw_residual_cluster,
                    "residual_reversal": raw_reversal_cluster, "ra_carry": raw_carry_cluster}
    cluster_scores = {"momentum": mom_cluster_score, "classic_momentum": classic_cluster_score,
                      "low_beta": beta_cluster_score, "residual_momentum": res_cluster_score,
                      "residual_reversal": reversal_cluster_score,
                      "ra_carry": carry_cluster_score}
    for family, raw_signal in cluster_raws.items():
        np.testing.assert_allclose(raw_signal, raws[family], rtol=1e-11, atol=1e-12,
                                   equal_nan=True)
        oriented = -raws[family] if family == "low_beta" else raws[family]
        np.testing.assert_allclose(cluster_scores[family],
                                   cluster_score_reference(oriented, rolling_clusters, 3),
                                   rtol=1e-12, atol=1e-12)
        assert (inspect.signature(getattr(signals, f"compute_{family}_cluster_alpha"))
                .parameters["min_cluster_size"].default == 3)
    pd.testing.assert_frame_equal(mom_cluster_score, cluster_score)
    # Point in time: later prices, benchmark, carry and cluster labels leave every earlier raw
    # signal and score unchanged, for all six families, standard and cluster, and for the
    # beta-based families under both EWMA and NONE mean adjustment.
    cutoff = pd.Timestamp(CUTOFF)
    later = prices.index > cutoff
    later_prices = prices.copy()
    later_prices.loc[later, "A"] *= np.linspace(1.1, 2.0, later.sum())
    later_benchmark = benchmark.copy()
    later_benchmark.loc[later] *= np.linspace(1.03, 1.3, later.sum())
    later_carry = carry.copy()
    later_carry.loc[later, "A"] *= 2
    later_clusters = {**rolling_clusters,
                      pd.Timestamp("2024-01-31"): pd.Series("future", index=prices.columns)}
    for family in FAMILIES:
        adjustments = [None, qis.MeanAdjType.NONE] if family in BETA_FAMILIES else [None]
        for cluster in (False, True):
            for adjustment in adjustments:
                function = getattr(signals, f"compute_{family}{'_cluster' if cluster else ''}"
                                            "_alpha")
                before = family_arguments(family, benchmark, carry)
                after = family_arguments(family, later_benchmark, later_carry)
                if cluster:
                    before["rolling_clusters"], after["rolling_clusters"] = (rolling_clusters,
                                                                             later_clusters)
                if adjustment is not None:
                    before["mean_adj_type"] = after["mean_adj_type"] = adjustment
                original, changed = function(prices, **before), function(later_prices, **after)
                for kept, moved in zip(original, changed):
                    np.testing.assert_allclose(kept.loc[:cutoff], moved.loc[:cutoff],
                                               rtol=1e-11, atol=1e-12, equal_nan=True)
                assert not np.allclose(original[1].iloc[-6:], changed[1].iloc[-6:],
                                       equal_nan=True)

    probe_date = pd.Timestamp("2024-12-31")
    raw_probe = pd.DataFrame(
        [[-4.0, -2.0, 0.0, 2.0, 4.0, 8.0]], index=[probe_date], columns=list("ABCDEF"),
    )
    probe_clusters = {
        probe_date: pd.Series(["Large"] * 4 + ["Small"] * 2, index=raw_probe.columns),
    }
    standard_probe = qis.df_to_cross_sectional_score(raw_probe)
    cluster_probe = signals.score_within_clusters(raw_probe, probe_clusters, min_cluster_size=3)
    score_comparison = pd.DataFrame({
        "Raw": raw_probe.loc[probe_date],
        "Standard": standard_probe.loc[probe_date],
        "Cluster": cluster_probe.loc[probe_date],
    })
    print(score_comparison.round(6))

    # The page's table: the standard score clips F to 5 and uses population moments; A to D
    # use their own sample moments and the two-member cluster E, F those of all six.
    values = raw_probe.iloc[0]
    standard = (values.clip(-5, 5) - values.clip(-5, 5).mean()) / values.clip(-5, 5).std(ddof=0)
    within = pd.concat([sample_score(values.iloc[:4], values.iloc[:4]),
                        sample_score(values.iloc[4:], values)])
    expected = np.column_stack((values, standard, within))
    np.testing.assert_allclose(PROBE_TABLE, expected, rtol=0, atol=0.5e-6)
    np.testing.assert_allclose(score_comparison, expected, rtol=1e-12, atol=1e-12)
    assert score_comparison.at["F", "Standard"] > 1 and score_comparison.at["F", "Cluster"] != 0
    # The threshold is inclusive: at four, the four-member cluster also uses all six, and at
    # the default three, so do two three-member clusters.
    universe = sample_score(values, values)
    np.testing.assert_allclose(signals.score_within_clusters(raw_probe, probe_clusters, 4)
                               .iloc[0], universe, rtol=1e-12)
    threes = {probe_date: pd.Series(["Low"] * 3 + ["High"] * 3, index=raw_probe.columns)}
    np.testing.assert_allclose(signals.score_within_clusters(raw_probe, threes).iloc[0],
                               universe, rtol=1e-12)
    # One cluster uses the assigned universe; degenerate values score zero; an empty mapping
    # falls back to the clipped population score; before the first assignment every score is
    # zero, even for a missing raw value; an unassigned asset scores zero while the others use
    # the assigned five; a missing assigned value stays missing.
    single = {probe_date: pd.Series("All", index=raw_probe.columns)}
    np.testing.assert_allclose(signals.score_within_clusters(raw_probe, single).iloc[0],
                               universe, rtol=1e-12)
    for labels in (single, probe_clusters):
        flat = signals.score_within_clusters(raw_probe * 0 + 1.0, labels)
        assert (flat == 0.0).all().all()
    lone = {probe_date: pd.Series(["Large"], index=["A"])}
    assert (signals.score_within_clusters(raw_probe, lone) == 0.0).all().all()
    np.testing.assert_allclose(signals.score_within_clusters(raw_probe, {}),
                               clipped_population_score(raw_probe), rtol=1e-12)
    missing_probe = raw_probe.copy()
    missing_probe.iloc[0, 0] = np.nan
    next_day = probe_date + pd.Timedelta(days=1)
    before_first = signals.score_within_clusters(missing_probe, {next_day: single[probe_date]})
    assert (before_first == 0.0).all().all()
    incomplete = probe_clusters[probe_date].drop(index="F")
    unassigned = signals.score_within_clusters(raw_probe, {probe_date: incomplete})
    assert unassigned.at[probe_date, "F"] == 0.0
    assert abs(unassigned.at[probe_date, "E"] - sample_score(values["E"], values.iloc[:5])) < 1e-12
    assert np.isnan(signals.score_within_clusters(missing_probe, probe_clusters).iloc[0, 0])

    mixed_prices = prices.copy()
    mixed_prices.loc[~mixed_prices.index.is_quarter_end, ["G", "H"]] = np.nan
    return_frequencies = pd.Series("ME", index=prices.columns)
    return_frequencies.loc[["G", "H"]] = "QE"
    mixed_score, mixed_raw = signals.compute_classic_momentum_alpha(
        mixed_prices, returns_freq=return_frequencies,
        lookback_periods={"ME": 12, "QE": 4}, skip_periods={"ME": 1, "QE": 1},
    )

    # Each bucket skips one native period: one month for A to F, one quarter for G and H,
    # whose values are carried between quarter ends; G and H are scored only against each other.
    for assets, lookback in ((list("ABCDEF"), 12), (list("GH"), 4)):
        observed = mixed_prices[assets].dropna()
        expected = np.log(observed.shift(1) / observed.shift(lookback + 1))
        np.testing.assert_allclose(mixed_raw.loc[observed.index, assets].iloc[lookback + 1:],
                                   expected.iloc[lookback + 1:], rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(mixed_score[assets].dropna(),
                                   clipped_population_score(mixed_raw[assets]).dropna(),
                                   rtol=1e-11, atol=1e-12)
    assert (mixed_raw.loc["2024-09-30":"2024-11-30", ["G", "H"]].nunique() == 1).all()

    group_data = pd.Series(["Group 1"] * 4 + ["Group 2"] * 4, index=prices.columns)
    group_score, group_raw = signals.compute_momentum_alpha(
        prices, benchmark_price=benchmark, returns_freq="ME", group_data=group_data,
    )

    # Groups change the comparison set only: the raw momentum is unchanged and each group is
    # scored with its own clipped population moments.
    pd.testing.assert_frame_equal(group_raw, raw_momentum)
    for assets in (list("ABCD"), list("EFGH")):
        np.testing.assert_allclose(group_score[assets],
                                   clipped_population_score(raw_momentum[assets]),
                                   rtol=1e-11, atol=1e-12, equal_nan=True)
    # Without a benchmark, the single-cadence path fits all assets together, but the
    # mixed-frequency fixed-group path fits each group, so its implicit benchmark changes.
    low_beta = signals.compute_low_beta_alpha
    pd.testing.assert_frame_equal(without_benchmark(low_beta, prices, group_data=group_data),
                                  implicit_beta)
    per_group_beta = without_benchmark(
        low_beta, prices, returns_freq=pd.Series("ME", index=prices.columns),
        group_data=group_data)
    for assets in (list("ABCD"), list("EFGH")):
        np.testing.assert_allclose(per_group_beta[assets],
                                   without_benchmark(low_beta, prices[assets]), rtol=1e-12,
                                   equal_nan=True)
    assert not np.allclose(per_group_beta.iloc[13:], implicit_beta.iloc[13:])

    # An illustrative application-level blend; the container does not choose this rule.
    combined_scores = 0.5 * mom_score + 0.5 * beta_score
    cluster_assignments = pd.DataFrame.from_dict(rolling_clusters, orient="index")
    cluster_assignments = cluster_assignments.reindex(prices.index, method="ffill")
    from optimalportfolios.alphas import AlphasData

    data = AlphasData(
        alpha_scores=combined_scores,                           # (T × N) — input to optimiser
        momentum_score=mom_score,                               # fixed-group component scores
        momentum_cluster_score=mom_cluster_score,               # cluster component scores
        beta_score=beta_score,
        beta_cluster_score=beta_cluster_score,
        managers_scores=mgr_score,
        residual_momentum_score=res_score,
        residual_momentum_cluster_score=res_cluster_score,
        momentum=raw_momentum,                                  # raw signals
        momentum_cluster=raw_momentum_cluster,
        beta=raw_beta,
        beta_cluster=raw_beta_cluster,
        managers_alphas=raw_alpha,
        residual_momentum=raw_residual,
        residual_momentum_cluster=raw_residual_cluster,
        clusters=cluster_assignments,                           # T × N cluster IDs
    )

    # snapshot at a single date (all available components)
    snapshot = data.get_alphas_snapshot(date=pd.Timestamp('2024-12-31'))

    # export to dict (only non-None fields, safe for Excel)
    output = data.to_dict()

    # The container keeps the supplied blend without a CDF or range; eight rows and 16
    # populated columns; to_dict returns the 16 populated fields, which are the page's table.
    pd.testing.assert_frame_equal(data.alpha_scores, 0.5 * mom_score + 0.5 * beta_score)
    assert snapshot.shape == (8, 16) and list(snapshot.index) == list(prices.columns)
    assert [field.name for field in fields(AlphasData)] == ALPHAS_DATA_FIELDS
    assert list(output) == ALPHAS_DATA_FIELDS and output["alpha_scores"] is data.alpha_scores
    assert cluster_assignments.loc["2022-11-30"].isna().all()
    assert (cluster_assignments.loc["2024-06-30"] == rolling_clusters[year_ends[1]]).all()
    # The date must exist in alpha_scores; a component missing that date contributes its
    # last row, even a later one.
    late_component = pd.DataFrame(17.0, index=[pd.Timestamp("2025-01-31")],
                                  columns=prices.columns)
    misaligned = replace(data, momentum_score=late_component)
    assert (misaligned.get_alphas_snapshot(pd.Timestamp("2024-06-30"))["Momentum Score"]
            == 17.0).all()
    assert_raises(KeyError, data.get_alphas_snapshot, pd.Timestamp("2000-01-31"))

    profiles = alphas.backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores={"Momentum": mom_score, "Low beta": beta_score},
        quantile=0.25, rebalancing_freq="QE",
        time_period=qis.TimePeriod("2021-12-31", "2024-12-31"),
    )
    component_panels = alphas.signal_diagnostics_panel(data)
    diagnostics = alphas.run_signal_diagnostics(
        asset_returns_dict={
            "ME": qis.to_returns(prices, freq="ME", is_log_returns=True, drop_first=True)
        },
        signal=mom_score, horizons=(1, 3), is_log_returns=True,
    )
    mean_dates = pd.date_range("2023-03-31", "2024-12-31", freq="QE")
    annual_log_means = alphas.estimate_rolling_ewma_means(
        prices, rebalancing_dates=list(mean_dates), returns_freq="ME", span=12, annualize=True,
    )

    # Two signal strategies and the equal-weight benchmark, without costs; eight populated
    # score panels; two horizons; eight dates by eight assets of annual EWMA log means.
    assert [leg.ticker for leg in profiles.portfolio_datas] == ["Momentum", "Low beta",
                                                                "Equal Weight"]
    for leg in profiles.portfolio_datas:
        assert np.allclose(leg.realized_costs, 0.0)
    score_fields = [name for name in ALPHAS_DATA_FIELDS if name.endswith(("score", "scores"))]
    assert sorted(component_panels) == sorted(score_fields) and len(component_panels) == 8
    assert len(diagnostics.horizon_labels) == 2
    returns = log_returns(prices).iloc[1:]
    all_means = pd.DataFrame(12 * seeded_ewma(returns.to_numpy(), span=12),
                             index=returns.index, columns=returns.columns)
    pd.testing.assert_frame_equal(annual_log_means, all_means.loc[mean_dates], check_freq=False,
                                  rtol=1e-11, atol=1e-12)
    assert annual_log_means.shape == (8, 8)
    defaults = inspect.signature(alphas.estimate_rolling_ewma_means).parameters
    assert (defaults["returns_freq"].default, defaults["span"].default,
            defaults["annualize"].default) == ("W-WED", 52, True)
    # A date between observations takes the latest estimate; a date before the sample is missing.
    between = alphas.estimate_rolling_ewma_means(
        prices, rebalancing_dates=[pd.Timestamp("2016-06-30"), pd.Timestamp("2024-11-15")],
        returns_freq="ME", span=12)
    assert between.iloc[0].isna().all()
    np.testing.assert_allclose(between.iloc[1], all_means.loc["2024-10-31"], rtol=1e-11)
    # Ranking: ceil(quantile x assets), ties by column order, and a mask of non-missing
    # values only, so an infinite score with a negative price is still selected.
    ties = pd.DataFrame([[3.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]], index=[probe_date],
                        columns=prices.columns)
    top = alphas.compute_top_quantile_equal_weights(ties, prices.loc[ties.index], quantile=0.125)
    np.testing.assert_array_equal(top.iloc[0], [1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(alphas.compute_top_quantile_equal_weights(
        ties, prices.loc[ties.index], quantile=0.3).iloc[0], [1 / 3] * 3 + [0] * 5)
    ties.iloc[0, -1] = np.inf
    negative = prices.loc[ties.index].copy()
    negative.iloc[0, -1] = -1.0
    assert alphas.compute_top_quantile_equal_weights(ties, negative,
                                                     quantile=0.125).iloc[0, -1] == 1.0

    # The figure and the Insight: each asset's classic momentum is its annual drift; both
    # clusters exceed three members, so within-cluster scores have mean zero and sample
    # standard deviation one in each. Across all nine, the equity-like mean is positive and
    # every bond-like score negative; the strongest bond-like asset moves from fifth (-0.29) to
    # second (1.07), and the weakest equity-like asset from seventh to ninth.
    table = exhibit_scores()
    np.testing.assert_allclose(table["raw_signal"], pd.Series(TRENDS)[table.index], atol=1e-12)
    for cluster, members in table.groupby("cluster"):
        assert len(members) > MIN_CLUSTER_SIZE
        within = members["within_cluster_score"]
        np.testing.assert_allclose(within, sample_score(members["raw_signal"],
                                                        members["raw_signal"]), atol=1e-12)
        assert abs(within.mean()) < 1e-12 and abs(within.std(ddof=1) - 1) < 1e-12
    cross_section = table.groupby("cluster")["cross_section_score"]
    assert cross_section.mean()["Equity-like"] > 0 and cross_section.max()["Bond-like"] < 0
    bond, equity = table.loc["Bond A"], table.loc["Equity E"]
    assert (bond["cross_section_rank"], bond["within_cluster_rank"]) == (5, 2)
    assert (round(bond["cross_section_score"], 2), round(bond["within_cluster_score"], 2)) == (
        -0.29, 1.07)
    assert (equity["cross_section_rank"], equity["within_cluster_rank"]) == (7, 9)
    assert (table["cross_section_rank"] < 5).sum() == 4 and table.loc[
        table["cross_section_rank"] < 5, "cluster"].eq("Equity-like").all()
    print("alphas_module_readme: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: cross-sectional against within-cluster scores of one signal.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prices = exhibit_prices()
    table = exhibit_scores()
    formation = pd.Timestamp(FORMATION_DATE)
    table = table.assign(cluster=pd.Categorical(table['cluster'],
                                                ['Equity-like', 'Bond-like'], ordered=True))
    table = table.sort_values(['cluster', 'raw_signal'], ascending=[True, False])

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = {'Equity-like': '#2a78d6', 'Bond-like': '#eb6834'}
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      sharey=True)
    rows = np.arange(len(table))[::-1]
    bar_colours = [colours[cluster] for cluster in table['cluster']]
    panels = ((left, 'cross_section_score', 'cross_section_rank',
               'Scored across all nine assets'),
              (right, 'within_cluster_score', 'within_cluster_rank',
               'Scored within each cluster'))
    for axis, column, rank, title in panels:
        axis.barh(rows, table[column], height=0.68, color=bar_colours)
        # The rank among all nine assets, in a column at the left edge of the panel.
        for row, position in zip(rows, table[rank]):
            axis.text(-2.3, row, f'#{position:.0f}', color=muted, fontsize=9, va='center')
        for cluster, members in table.groupby('cluster', observed=True):
            span = rows[table['cluster'].to_numpy() == cluster]
            centre = members[column].mean()
            axis.plot([centre, centre], [span.min() - 0.45, span.max() + 0.45], color=ink,
                      linewidth=1.2, linestyle='--')
        axis.axvline(0.0, color=muted, linewidth=0.8)
        axis.set_title(title, loc='left', color=ink)
        axis.set_xlim(-2.35, 2.1)
        axis.set_xticks([-2, -1, 0, 1, 2])
        axis.set_facecolor(surface)
        axis.grid(axis='x', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis='y', length=0)
    left.set_yticks(rows, table.index)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in colours.values()]
    handles.append(plt.Line2D([], [], color=ink, linewidth=1.2, linestyle='--'))
    right.legend(handles, [*colours, 'cluster mean'], frameon=False, loc='lower right',
                 fontsize=9, labelcolor=ink)
    fig.supxlabel('Score of twelve-month classic momentum at 31 December 2024; '
                  '#n is the rank among the nine assets', color=muted, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    raw = table['raw_signal']
    endpoint = np.log(prices.shift(SKIP) / prices.shift(SKIP + LOOKBACK)).loc[formation]
    cross = (raw - raw.mean()) / raw.std(ddof=0)
    groups = table.groupby('cluster', observed=True)['within_cluster_score']
    checks = {
        'raw_signal_is_the_endpoint_log_return': bool(
            np.allclose(raw, endpoint[raw.index], atol=1e-12)
            and np.allclose(raw, pd.Series(TRENDS)[raw.index], atol=1e-12)),
        'cluster_constructor_keeps_the_raw_signal': bool(
            np.allclose(table['cluster_raw_signal'], raw, rtol=0, atol=0)),
        'cross_section_is_a_population_z_score': bool(
            np.allclose(table['cross_section_score'], cross, atol=1e-12)),
        'within_cluster_mean_is_zero': bool((groups.mean().abs() < 1e-12).all()),
        'within_cluster_sample_std_is_one': bool(
            (groups.std(ddof=1).sub(1.0).abs() < 1e-12).all()),
        'strongest_bond_rises_from_fifth_to_second': bool(
            table.loc['Bond A', 'cross_section_rank'] == 5
            and table.loc['Bond A', 'within_cluster_rank'] == 2),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
