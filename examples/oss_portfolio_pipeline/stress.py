"""QIS stress report for the last optimised target book and its dated FM/FL model."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis

from examples.oss_portfolio_pipeline.backtest import PortfolioBacktest
from examples.oss_portfolio_pipeline.covariance import RollingCovariances
from examples.oss_portfolio_pipeline.data import external_directory
from examples.oss_portfolio_pipeline.factors import MacroFactor, month_end_dates
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs


FACTOR_LABELS = dict(zip(
    (factor.value for factor in MacroFactor),
    ('Energy', 'Agri', 'BaseMet', 'PrecMet', 'DM Eq', 'EM/DM Eq', 'Nom/TIPS',
     'GovtBond', 'USD/DM', 'EMFX/DM', 'HY/IG'),
))


@dataclass
class StressStudy:
    """Dated target book, explicit scenarios and detached QIS results ready for reporting."""

    portfolio: qis.InstrumentPortfolio
    scenarios: qis.StressScenarios
    history: pd.DataFrame
    result: qis.PortfolioStressResult
    report_config: qis.StressReportConfig
    model_date: pd.Timestamp
    settings: dict


def macro_groups() -> dict[str, qis.FactorGroupSpec]:
    """Group the eleven unchanged factor coordinates for QIS display and contributions."""
    groups = {
        'Commodities': tuple(MacroFactor)[:4],
        'Equities': (MacroFactor.DM_EQUITY, MacroFactor.EM_EQUITY),
        'Rates': (MacroFactor.INFLATION, MacroFactor.RATES),
        'FX': (MacroFactor.USD_DM, MacroFactor.USD_EM),
        'Credit': (MacroFactor.CREDIT,),
    }
    return {name: qis.FactorGroupSpec(name, tuple(f.value for f in members), label=name)
            for name, members in groups.items()}


def stress_requests() -> tuple[qis.StressScenarios, dict[str, qis.StressScenarios]]:
    """Declare illustrative simple changes of the vol-targeted excess factor NAVs.

    A bond-factor shock is not a yield shock. Relative equity, inflation and credit
    coordinates are their constructed long/short baskets, not raw asset indices.
    NaN leaves an anchor unspecified; an explicit zero would constrain it to zero.
    """
    f = MacroFactor
    commodities = {factor.value: -.15 for factor in tuple(f)[:4]}
    requests = {
        'DM equity -20%': {f.DM_EQUITY.value: -.20},
        'DM equity +20%': {f.DM_EQUITY.value: .20},
        'Global bonds -7.5%': {f.RATES.value: -.075},
        'HY/IG credit -5%': {f.CREDIT.value: -.05},
        'USD/DM +7.5%': {f.USD_DM.value: .075},
        'EM/DM equity -15%': {f.EM_EQUITY.value: -.15},
        'EM FX stress +7.5%': {f.USD_EM.value: .075},
        'Nominal/TIPS +7.5%': {f.INFLATION.value: .075},
        'Commodities -15% each': commodities,
        'Joint risk-off': {f.DM_EQUITY.value: -.20, f.CREDIT.value: -.05,
                           f.RATES.value: .075, f.USD_DM.value: .075, **commodities},
    }
    scenarios = qis.StressScenarios(pd.DataFrame.from_dict(requests, orient='index'),
                                    convention=qis.ShockConvention.SIMPLE)
    grids = {}
    for factor, width in ((f.DM_EQUITY, .30), (f.RATES, .15), (f.CREDIT, .10),
                           (f.USD_DM, .15), (f.ENERGY, .30), (f.EM_EQUITY, .30)):
        axis = pd.Index(np.linspace(-width, width, 41), name='Factor NAV simple change')
        grids[factor.value] = qis.StressScenarios(
            pd.DataFrame({factor.value: axis}, index=axis),
            mode=qis.ScenarioMode.CONDITIONAL, convention=qis.ShockConvention.SIMPLE)
    return scenarios, grids


def build_stress_study(inputs: PipelineInputs, risk: RollingCovariances,
                       full: PortfolioBacktest, notional: float = 1_000_000.) -> StressStudy:
    """Stress the last accepted target allocation on a normalized USD investment notional.

    Use the exact model assigned to that decision, never a later available snapshot.
    FM exports the full selected asset covariance plus an explicit diagonal-residual
    reference for QIS factor decomposition/bands. Total covariance risk is preserved.
    Marks and units describe a hypothetical target book, before costs or subsequent drift.
    """
    if isinstance(notional, bool) or not np.isfinite(notional) or notional <= 0:
        raise ValueError('Stress notional must be finite and positive')
    if full.executed.targets.empty:
        raise ValueError('Stress reporting needs an accepted portfolio decision')
    date = full.executed.targets.index[-1]
    targets = full.executed.targets.loc[date].reindex(inputs.universe.index)
    if (not np.isfinite(targets).all() or targets.lt(0).any()
            or not np.isclose(targets.sum(), 1., rtol=0., atol=1e-8)):
        raise ValueError('Stress targets must be finite, long-only and fully invested')
    matched = risk.schedule.index[risk.schedule.observation_date.eq(date)]
    if len(matched) != 1:
        raise ValueError('Last target must have exactly one matching FM snapshot')
    model_date = matched[0]
    snapshot = risk.model.snapshots[model_date]
    if pd.Timestamp(snapshot.provenance['factor_observation_date']) > date:
        raise ValueError('Stress model contains a future factor observation')
    if not snapshot.asset_covariance.equals(full.risk_model.covar[date]):
        raise ValueError('Stress covariance differs from the portfolio decision model')
    native = snapshot.to_risk_model(diagonal_reference=True)
    # Relabel the completed calendar ME to its observed trading session, retaining all matrices.
    model = replace(native, covar={date: native.covar[model_date]},
                    factor_loadings={date: native.factor_loadings[model_date]},
                    factor_covar={date: native.factor_covar[model_date]},
                    residual_vars={date: native.residual_vars[model_date]},
                    factor_groups=macro_groups())
    prices = inputs.prices.loc[date].reindex(targets.index)
    if not np.isfinite(prices).all() or prices.le(0).any():
        raise ValueError('Stress snapshot needs positive adjusted prices at the decision close')
    audit = pd.DataFrame({'target_weight': targets, 'adjusted_price': prices,
                          'target_mtm_usd': notional * targets})
    audit['adjusted_price_units'] = audit.target_mtm_usd / prices
    audit['represented_holding'] = targets.gt(0)
    underlyings, holdings = {}, []
    for ticker, row in audit.loc[audit.represented_holding].iterrows():
        underlyings[ticker] = qis.Underlying(ticker, row.adjusted_price, 'USD', ticker,
                                            qis.ResponseBasis.REFERENCE)
        holdings.append(qis.PortfolioHolding(
            ticker, ticker, row.target_mtm_usd,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, ticker, row.adjusted_price_units),)))
    provenance = {
        ticker: qis.ResponseProvenance(
            qis.HistorySource.OBSERVED, 'USD', qis.ReturnBasis.EXCESS,
            'ME log changes of Yahoo adjusted ETF price / adjusted BIL price',
            assumptions=('Instantaneous shocks hold the BIL numeraire fixed.',),
            limitations=('Adjusted-price units are synthetic, not executable share counts.',))
        for ticker in targets.index}
    portfolio = qis.InstrumentPortfolio(
        tuple(holdings), underlyings, model, date, date, 'USD', float(notional),
        'Illustrative fully invested target notional', response_provenance=provenance)
    # Actual session labels preserve a completed weekend ME without inventing future data.
    factor_prices = inputs.factor_navs.loc[:date]
    sessions = month_end_dates(factor_prices.index, str((model_date + pd.Timedelta(days=1)).date()))
    history = qis.to_returns(factor_prices.loc[sessions], is_log_returns=True,
                              ffill_nans=False).iloc[1:]
    scenarios, grids = stress_requests()
    result = qis.run_portfolio_stress_test(
        portfolio, scenarios, historical_factor_log_returns=history, factor_grids=grids,
        config=qis.StressTestConfig(historical_count=10, horizon_years=1/12,
                                    confidence=.95, include_conditional_comparison=True))
    zero = targets * 0.
    tre = model.compute_tre_at_date(inputs.benchmark_weights, targets, date)
    reference_vol = result.risk.annual_factor_model_vol
    summary = pd.DataFrame({'Value': {
        'Target / observed risk date': str(date.date()),
        'FM calendar snapshot': str(model_date.date()),
        'Illustrative USD notional': f'{notional:,.0f}',
        'Full-covariance annual volatility': f'{result.risk.annual_total_vol:.4%}',
        'Diagonal-reference annual volatility': f'{reference_vol:.4%}',
        'Full-covariance target tracking error': f'{tre:.4%}',
        'Fixed benchmark annual volatility':
            f'{model.compute_tre_at_date(zero, inputs.benchmark_weights, date):.4%}',
        'Historical vectors':
            f'{len(history)} completed months through {history.index[-1]:%Y-%m-%d}',
    }})
    diagnostics = dict(result.report_diagnostics)
    diagnostics.update({
        'OP target snapshot': audit, 'Production risk audit': summary,
        'Production asset covariance': snapshot.asset_covariance,
        'Production full residual covariance': snapshot.residual_covariance,
        'Historical factor vectors': history,
        'Historical date mapping': pd.DataFrame({
            'calendar_month_end': history.index.to_period('M').to_timestamp('M')},
            index=history.index),
        'Requested simple factor shocks': scenarios.anchors,
    })
    result = replace(result, report_diagnostics=diagnostics)
    notes = (
        f'Latest optimised target book: {date:%d %b %Y}; hypothetical USD {notional:,.0f}.',
        'Factor shocks are excess-NAV changes, not yield, spot-FX or credit-spread shocks.',
        'Full covariance is preserved. Factor risk decomposition and bands use diag(residual R).',
        f'Full annual vol {result.risk.annual_total_vol:.2%}; '
        f'diagonal reference {reference_vol:.2%}; '
        f'target TE {tre:.2%}.',
        'Historical replay uses current loadings and targets; it is not a realised backtest.',
        'Instantaneous BIL-relative shocks exclude carry, transaction costs and rebalancing.',
    )
    report_config = qis.StressReportConfig(
        title='OSS optimal ETF portfolio', model_name='Macro', factor_labels=FACTOR_LABELS,
        model_label='FM/FL; diagonal reference', notes=notes,
        response_diagnostics=snapshot.fit_statistics[['r2']],
        cluster_memberships=snapshot.clusters_by_frequency or {},
        cluster_linkages=snapshot.linkages_by_frequency or {},
        cluster_cutoffs=snapshot.cutoffs_by_frequency or {},
        selected_grids=tuple(grids), write_workbook=False, write_previews=True,
        appendix_table=summary, appendix_title='Production risk and snapshot audit',
        appendix_subtitle='Authoritative ROSAA covariance versus the QIS diagonal reference',
        appendix_notes=notes)
    return StressStudy(portfolio, scenarios, history, result, report_config, model_date,
                       asdict(full.settings))


def save_stress_report(study: StressStudy, output_dir: Path | str) -> qis.StressReportArtifacts:
    """Write QIS PDF, complete CSV tables and provenance in a fresh dated stress directory."""
    output = external_directory(output_dir)
    sources = {name: output / name for name in ('backtest_manifest.json',
                                               'covariance_manifest.json')}
    if not all(path.is_file() for path in sources.values()):
        raise ValueError('Save the covariance and baseline backtest before the stress report')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = output / 'stress' / str(study.portfolio.risk_date.date()) / stamp
    # QIS hides tick marks; their reserved padding separates near-zero bar annotations from IDs.
    with plt.rc_context({'ytick.major.size': 40, 'ytick.minor.size': 40}):
        artifacts = qis.generate_portfolio_stress_report(
            study.result, destination, study.report_config)
    manifest = {
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in sources.items()},
        'qis_manifest_sha256': hashlib.sha256(artifacts.manifest_path.read_bytes()).hexdigest(),
        'model_date': str(study.model_date.date()),
        'decision_date': str(study.portfolio.risk_date.date()),
        'portfolio': 'last accepted optimal target allocation; hypothetical funded book',
        'notional_usd': study.portfolio.reporting_denominator,
        'settings': study.settings,
        'factor_display_labels': dict(study.report_config.factor_labels),
        'risk_contract': 'exact selected asset covariance; diagonal-residual F/D reference',
        'historical_contract': 'completed factor returns at observed session dates, no future rows',
        'versions': {name: version(name) for name in ('qis', 'factormodels', 'factorlasso',
                                                    'optimalportfolios')},
    }
    (destination / 'pipeline_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return artifacts
