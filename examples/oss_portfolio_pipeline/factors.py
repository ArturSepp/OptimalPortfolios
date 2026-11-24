"""Declare excess-return baskets in FM and execute its risk targets on actual month-ends."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import qis
from factormodels import analytics as fm

from examples.oss_portfolio_pipeline.config import PipelineConfig
from examples.oss_portfolio_pipeline.data import validate_prices


class MacroFactor(fm.RiskFactors):
    """Ordered public macro-factor identifiers, independent of any proprietary catalogue."""

    ENERGY = 'Energy'
    AGRICULTURE = 'Agriculture'
    BASE_METALS = 'Base metals'
    PRECIOUS_METALS = 'Precious metals'
    DM_EQUITY = 'DM equity'
    EM_EQUITY = 'EM vs DM equity'
    INFLATION = 'Nominal vs TIPS'
    RATES = 'Global government bonds'
    USD_DM = 'USD vs DM'
    USD_EM = 'EM FX stress vs DM'
    CREDIT = 'HY vs IG credit'


@dataclass
class FactorPanel:
    """Full-history excess NAVs, daily log returns and actual dated exposure targets."""

    navs: pd.DataFrame
    log_returns: pd.DataFrame
    target_weights: dict[str, pd.DataFrame]


def factor_model(config: PipelineConfig) -> fm.FactorModelSpec:
    """Declare zero-net BIL-financed baskets and the requested annual volatility targets."""
    commodities = fm.FactorFamily('Commodities', 0.15)
    equities = fm.FactorFamily('Equities', 0.15)
    rates = fm.FactorFamily('Rates', 0.075)
    fx = fm.FactorFamily('FX', 0.075)
    credit = fm.FactorFamily('Credit', 0.05)
    definitions = (
        (MacroFactor.ENERGY, commodities, ('DBE',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.AGRICULTURE, commodities, ('DBA',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.BASE_METALS, commodities, ('DBB',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.PRECIOUS_METALS, commodities, ('DBP',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.DM_EQUITY, equities, ('URTH',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.EM_EQUITY, equities, ('EEM',), (1.,), ('URTH',), (1.,)),
        (MacroFactor.INFLATION, rates, ('GOVT',), (1.,), ('TIP',), (1.,)),
        (MacroFactor.RATES, rates, ('GOVT', 'IGOV'), (.5, .5), ('BIL',), (1.,)),
        (MacroFactor.USD_DM, fx, ('UUP',), (1.,), ('BIL',), (1.,)),
        (MacroFactor.USD_EM, fx, ('BIL',), (2.,), ('CEW', 'UUP'), (1., 1.)),
        (MacroFactor.CREDIT, credit, ('HYG',), (1.,), ('LQD',), (1.,)),
    )
    baskets = {
        member: fm.Factor(
            name=member.value, family=family,
            long_side_tickers=longs, long_side_weights=long_weights,
            short_side_tickers=shorts, short_side_weights=short_weights,
            rebalancing_freq='ME', vol_span=config.vol_span,
            vol_warmup_period=config.vol_warmup_period, require_all_constituents=True,
        )
        for member, family, longs, long_weights, shorts, short_weights in definitions
    }
    return fm.FactorModelSpec(
        name='Yahoo macro 11 excess', factors=MacroFactor,
        baskets=baskets, file_name='factor_excess_navs')


def month_end_dates(index: pd.DatetimeIndex, end_exclusive: str) -> pd.DatetimeIndex:
    """Map completed calendar month-ends backward to the supplied US trading sessions."""
    cutoff = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    anchors = pd.date_range(index[0], cutoff, freq='ME')
    positions = index.get_indexer(anchors, method='pad')
    dates = index[positions[positions >= 0]]
    if len(dates) and not (dates.to_period('M') == anchors[positions >= 0].to_period('M')).all():
        raise ValueError('No price in a completed calendar month')
    if dates.has_duplicates:
        raise ValueError('Month-end anchors map to duplicate price dates')
    return dates


def build_factors(prices: pd.DataFrame, config: PipelineConfig) -> FactorPanel:
    """Use FM risk analytics and QIS holdings accounting to build monthly excess factors.

    Explicit BIL legs finance the baskets before volatility targeting. The NAVs already
    represent excess P&L; cash must not be subtracted from their returns a second time.
    """
    spec = factor_model(config)
    prices = validate_prices(prices, tuple(spec.required_tickers()))
    prices = prices.loc[(prices.index >= config.start) & (prices.index < config.end)]
    if prices.empty:
        raise ValueError('No prices in the construction interval')
    dates = month_end_dates(prices.index, config.end)
    navs, weights = {}, {}
    for member in spec.factors:
        factor = spec.basket(member)
        legs = fm.select_factor_prices(factor, prices)
        # fm.compute_factor_nav labels weights at calendar ME; QIS then maps holidays
        # forward. Use FM's public weekly risk targets to implement actual session ME.
        weekly = fm.compute_volatility_targeted_portfolio(
            prices=legs, strategic_weights=np.asarray(factor.signed_weights()),
            returns_freq='W-WED', rebalancing_freq='W-WED',
            portfolio_vol_target=factor.resolved_vol_target,
            vol_span=factor.vol_span, vol_warmup_period=factor.vol_warmup_period)
        target = weekly.reindex(dates, method='ffill').fillna(0.0)
        if not target.abs().sum(axis=1).gt(0).any():
            raise ValueError(f'{factor.name}: insufficient nonzero volatility history')
        if prices.index[0] not in target.index:
            opening = pd.DataFrame(0.0, index=prices.index[:1], columns=legs.columns)
            target = pd.concat([opening, target])
        portfolio = qis.backtest_model_portfolio(
            prices=legs, weights=target, ticker=factor.name)
        nav = portfolio.get_portfolio_nav()
        if not np.isfinite(nav).all() or nav.le(0).any():
            raise ValueError(f'{factor.name}: nonpositive or nonfinite excess NAV')
        navs[factor.name] = nav
        weights[factor.name] = target
    panel = pd.DataFrame(navs)
    returns = qis.to_returns(panel, is_log_returns=True, ffill_nans=False)
    return FactorPanel(navs=panel, log_returns=returns, target_weights=weights)
