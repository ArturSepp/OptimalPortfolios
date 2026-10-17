"""Provider-neutral CMA inference reports from completed, immutable GLS results.

Provider adapters own source selection, cash, alpha and scenario construction.
This module only presents those results using the established QIS report primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import qis

from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticResult
from optimalportfolios.matf.cma_uncertainty import CmaConditionalGlsResult
from optimalportfolios.matf.cma_identification import CmaIdentificationResult
from optimalportfolios.matf.gls_diagnostic import (
    GlsLoadingStabilityResult, GlsResolutionResult, GlsUncertaintyScenario,
)
from optimalportfolios.reports.cma_diagnostics import (
    _INK, _LEFT, _MUTED, _TEAL, _number, _section, _table, _text, _value, _write,
)


@dataclass(frozen=True)
class CmaInferenceView:
    """One explicitly labelled, already computed provider scenario."""

    key: str
    label: str
    result: CmaDiagnosticResult
    horizon: str
    cash_rate: float
    alpha_weight: float
    conditional_intervals: CmaConditionalGlsResult | None = None
    identification: CmaIdentificationResult | None = None
    resolution: GlsResolutionResult | None = None
    uncertainty_scenarios: tuple[GlsUncertaintyScenario, ...] = ()
    loading_stability: GlsLoadingStabilityResult | None = None


@dataclass(frozen=True)
class CmaInferenceReport:
    """Presentation context kept separate from the numerical diagnostic contract."""

    title: str
    model_date: str
    currency: str
    forecast_vintage: str
    headline_key: str
    views: tuple[CmaInferenceView, ...]
    source: str
    methodology: str
    qualifications: str
    asset_labels: Mapping[str, str] = field(default_factory=dict)
    provider_contrasts: tuple[GlsUncertaintyScenario, ...] = ()
    extension_notes: tuple[str, ...] = ()
    loading_stability_contrasts: tuple[GlsLoadingStabilityResult, ...] = ()


def _validate(report: CmaInferenceReport) -> tuple[CmaInferenceView, list[str]]:
    """Reject ambiguous comparisons instead of silently dropping scenarios or factors."""
    if not 1 <= len(report.views) <= 6:
        raise ValueError('A compact inference report requires one to six views.')
    keys = [view.key for view in report.views]
    if len(set(keys)) != len(keys) or report.headline_key not in keys:
        raise ValueError('View keys must be unique and include the headline key.')
    headline = next(view for view in report.views if view.key == report.headline_key)
    factors = list(headline.result.factors.index)
    with_intervals = [v.conditional_intervals is not None for v in report.views]
    if any(with_intervals) and not all(with_intervals):
        raise ValueError('Supply conditional intervals for every view or none.')
    with_identification = [v.identification is not None for v in report.views]
    if any(with_identification) and (not all(with_identification) or not all(with_intervals)):
        raise ValueError('Supply identification and conditional intervals for every view or none.')
    with_resolution = [v.resolution is not None for v in report.views]
    if any(with_resolution) and (not all(with_resolution) or not all(with_intervals)):
        raise ValueError('Supply resolution and conditional intervals for every view or none.')
    if (any(v.uncertainty_scenarios or v.loading_stability is not None for v in report.views)
            or report.provider_contrasts or report.loading_stability_contrasts
            ) and not all(with_resolution):
        raise ValueError('Uncertainty extensions require explicit resolution results.')
    for view in report.views:
        if not np.isfinite([view.cash_rate, view.alpha_weight]).all():
            raise ValueError('Cash and alpha weight must be finite.')
        if view.result.summary.get('accounting_status') != 'pass':
            raise ValueError(f'Unreconciled diagnostic result: {view.key}')
        if view.result.factors.index.has_duplicates:
            raise ValueError('Factor labels must be unique.')
        ci = view.conditional_intervals
        if ci is not None:
            if (not ci.factors.index.equals(view.result.factors.index)
                    or ci.summary['method'] != 'conditional_gls_student_t'
                    or ci.summary['scope'] != 'pointwise'
                    or ci.summary['confidence_level'] !=
                    headline.conditional_intervals.summary['confidence_level']
                    or any(ci.summary[key] != view.result.summary[key]
                           for key in ('n_assets', 'n_factors', 'rank'))
                    or not np.isclose(ci.summary['weighted_residual_sse'],
                                      view.result.summary['residual_sharpe2'], atol=1e-12)):
                raise ValueError('Conditional intervals must match their diagnostic view.')
            identified = view.result.factors.status.isin(['identified', 'full_rank'])
            if not np.allclose(ci.factors.loc[identified, 'estimate'],
                               view.result.factors.loc[identified, 'implied_premium'],
                               atol=1e-12, rtol=1e-10):
                raise ValueError('Conditional intervals must match their diagnostic view.')
        identification = view.identification
        if identification is not None:
            if (not identification.factors.index.equals(ci.factors.index)
                    or identification.summary['method'] != 'conditional_gls_identification'
                    or any(identification.summary[key] != ci.summary[key]
                           for key in ('n_assets', 'n_factors', 'rank',
                                       'confidence_level', 'status'))
                    or any(identification.summary[key] != headline.identification.summary[key]
                           for key in ('precision_target', 'equivalence_margin', 'factor_pair'))
                    or not np.allclose(identification.factors[['estimate', 'lower', 'upper']],
                                       ci.factors[['estimate', 'lower', 'upper']], equal_nan=True,
                                       atol=1e-12, rtol=1e-10)):
                raise ValueError('Identification must match its view and declared settings.')
        _validate_extensions(view)
        for factor, row in view.result.factors.iterrows():
            if factor not in factors:
                factors.append(factor)
            if factor in headline.result.factors.index:
                reference = headline.result.factors.at[factor, 'calibrated_premium']
                if not np.isclose(row.calibrated_premium, reference, atol=1e-12, rtol=0):
                    raise ValueError('Views must share the calibrated factor basis.')
    if len(factors) > 12:
        raise ValueError('A compact inference report supports up to twelve factors.')
    for contrast in report.provider_contrasts:
        _validate_scenario(contrast, headline, contrast=True)
    for contrast in report.loading_stability_contrasts:
        _validate_loading_result(contrast, headline, contrast=True)
    return headline, factors


def _validate_scenario(scenario, view, *, contrast=False):
    """Keep alternative covariance assumptions separate from the default CI contract."""
    if not isinstance(scenario, GlsUncertaintyScenario):
        raise ValueError('Uncertainty scenarios require a typed GLS result.')
    info = scenario.summary
    names = view.result.factors.index
    required = {'method', 'target', 'units', 'assumptions', 'calibration_status',
                'reference_provenance', 'interval_status', 'confidence_level',
                'distribution', 'rank', 'n_factors'}
    if (not required.issubset(info) or not scenario.factors.index.equals(names)
            or not scenario.covariance.index.equals(names)
            or not scenario.covariance.columns.equals(names)
            or info['units'] != 'annual_decimal'
            or info['rank'] != view.result.summary['rank']
            or info['n_factors'] != view.result.summary['n_factors']
            or not info['target'] or not info['assumptions']
            or info['method'] == 'conditional_gls_student_t'):
        raise ValueError('Uncertainty scenarios must declare matching factors, units '
                         'and assumptions.')
    if contrast:
        if info['method'] != 'paired_provider_contrast' or not info.get('comparison_label'):
            raise ValueError('Provider contrasts must declare a paired comparison.')
    else:
        if info['method'] == 'paired_provider_contrast' or not np.allclose(
                scenario.factors.estimate, view.conditional_intervals.factors.estimate,
                equal_nan=True, atol=1e-12, rtol=1e-10):
            raise ValueError('Uncertainty scenarios must preserve their view point estimates.')


def _validate_extensions(view):
    """Reject results from another estimator, scenario, or stability convention before rendering."""
    result = view.resolution
    if result is not None:
        if (not isinstance(result, GlsResolutionResult)
                or not result.factors.index.equals(view.result.factors.index)
                or not result.assets.index.equals(view.result.assets.index)
                or result.summary.get('method') != 'gls_resolution'
                or any(result.summary.get(key) != view.result.summary[key]
                       for key in ('rank', 'n_assets', 'n_factors'))):
            raise ValueError('Resolution must match its diagnostic view.')
        ci = view.conditional_intervals.factors
        if (not np.allclose(result.factors.ci_half_width, (ci.upper-ci.lower)/2,
                            equal_nan=True, atol=1e-12, rtol=1e-10)
                or not np.allclose(result.factors.gross_mimicking_weight,
                                   view.result.factors.gross_gls_multiplier,
                                   equal_nan=True, atol=1e-12, rtol=1e-10)):
            raise ValueError('Resolution must use the unchanged conditional intervals '
                             'and geometry.')
        if 'normalized_ci_half_width' in result.factors and view.identification is not None:
            normalized = view.identification.factors.get('normalized_ci_half_width')
            if normalized is not None and not np.allclose(
                    normalized, result.factors.normalized_ci_half_width,
                    equal_nan=True, atol=1e-12, rtol=1e-10):
                raise ValueError('Resolution must use the unchanged conditional intervals.')
        if not ci.index.equals(result.factors.index):
            raise ValueError('Resolution must use the unchanged conditional intervals.')
    for scenario in view.uncertainty_scenarios:
        _validate_scenario(scenario, view)
    if view.loading_stability is not None:
        _validate_loading_result(view.loading_stability, view)


def _validate_loading_result(stability, view, *, contrast=False):
    """Bind ordinary and paired-difference stability to their explicitly stated baseline."""
    method = 'paired_loading_stability_contrast' if contrast else 'paired_loading_stability'
    if (not isinstance(stability, GlsLoadingStabilityResult)
            or not stability.bands.index.equals(view.result.factors.index)
            or stability.summary.get('method') != method
            or stability.summary.get('calibrated_ci') is not False
            or stability.summary.get('confidence_level') !=
            view.conditional_intervals.summary['confidence_level']
            or not stability.summary.get('baseline_provenance')
            or not stability.summary.get('assumptions')
            or (contrast and not stability.summary.get('comparison_label'))):
        raise ValueError('Loading stability must declare a matching baseline and band convention.')


def _premium(view: CmaInferenceView, factor: str) -> str:
    """Keep omitted and nonidentified factor coordinates visibly unavailable."""
    if factor not in view.result.factors.index:
        return 'N/A'
    row = view.result.factors.loc[factor]
    if row.get('status') not in ('identified', 'full_rank'):
        return 'N/A'
    return _value(row, 'implied_premium', 'return')


def _label(report: CmaInferenceReport, asset: str, width: int = 32) -> str:
    """Use provider-supplied concise labels, with full identities retained in data exports."""
    return _text(report.asset_labels.get(asset, asset), width, 1)


def _premium_interval(view: CmaInferenceView, factor: str) -> str:
    """Append a labelled interval only when that coordinate has supported inference."""
    premium = _premium(view, factor)
    if view.conditional_intervals is None or premium == 'N/A':
        return premium
    row = view.conditional_intervals.factors.loc[factor]
    if row.status != 'conditional_working_model':
        return premium + '\n[CI N/A]'
    return premium + f'\n[{100*row.lower:.2f}, {100*row.upper:.2f}]'


def _page(report: CmaInferenceReport, view: CmaInferenceView,
          section: str, number: int) -> Figure:
    """Create the common provider page without substituting a source or valuation date."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    _write(fig, _LEFT, .974, 'CMA INFERENCE  /  PROVIDER ANALYSIS', 8, _TEAL, weight='bold')
    _write(fig, _LEFT, .942, _text(report.title, 57, 2), 18, weight='bold')
    _write(fig, _LEFT, .882,
           f'{report.currency} | Model {report.model_date} | Forecast vintage: '
           + report.forecast_vintage, 8.1, _MUTED)
    _write(fig, _LEFT, .856,
           f'Headline: {view.label.replace(chr(10), " ")} | Horizon {view.horizon} | '
           f'Cash {_number(view.cash_rate, "return")} | Alpha w = {view.alpha_weight:g}',
           8.1, _MUTED)
    _write(fig, _LEFT, .032, section.upper(), 7.3, _MUTED)
    _write(fig, .945, .032, f'{number} / 3', 7.3, _MUTED, ha='right')
    return fig


def _comparison_page(report: CmaInferenceReport, headline: CmaInferenceView,
                     factors: list[str]) -> Figure:
    """Compare explicitly labelled views, retaining unavailable reduced-universe factors."""
    fig = _page(report, headline, 'Scenario comparison', 1)
    has_ci = headline.conditional_intervals is not None
    level = (100*headline.conditional_intervals.summary['confidence_level'] if has_ci else None)
    note = (f'Annual excess returns: estimate and [conditional {level:g}% CI], in percent.'
            if has_ci else
            'Annual excess returns. Alpha w scales the already admitted native CMA alpha once.')
    _section(fig, .817, 'GLS-implied factor premia', note)
    rows = []
    for factor in factors:
        calibrated = next(v.result.factors.at[factor, 'calibrated_premium']
                          for v in report.views if factor in v.result.factors.index)
        rows.append([factor, _number(calibrated, 'return')]
                    + [_premium_interval(view, factor) for view in report.views])
    width = .72 / len(report.views)
    _table(fig, rows, ['Factor', 'Our MATF'] + [v.label for v in report.views],
           .425 if has_ci else .478, .338 if has_ci else .285,
           [.18, .10] + [width] * len(report.views), 7.0 if has_ci else 7.5)
    _write(fig, _LEFT, .414 if has_ci else .467,
           'Pointwise Student-t CI; fixed B, D, cash and alpha. Full assumptions on page 3. '
           'N/A = unavailable.' if has_ci else
           'N/A means absent or nonidentified; it is not a zero premium. '
           'Provider forecasts are inferred through our model.', 7.2, _MUTED)
    _section(fig, .382 if has_ci else .423, 'Scenario diagnostics',
             'Universe exclusions retain the other estimated exposures, residual risks and alpha.')
    rows = [[view.label.replace('\n', ' '), view.horizon,
             _number(view.cash_rate, 'return'), f'{view.alpha_weight:g}',
             str(view.result.summary['n_assets']),
             f"{view.result.summary['rank']}/{view.result.summary['n_factors']}",
             *([str(view.conditional_intervals.summary['residual_df'])] if has_ci else []),
             _number(view.result.summary.get('condition_number')),
             _number(view.result.summary.get('residual_sharpe2'))] for view in report.views]
    _table(fig, rows, ['View', 'Horizon', 'Cash', 'Alpha w', 'Assets', 'Rank',
                       *(['CI df'] if has_ci else []),
                       'Condition', 'Residual S2'],
           .211 if has_ci else .218, .117 if has_ci else .152,
           [.21, .12, .09, .08, .08, .09, .07, .13, .13] if has_ci else
           [.25, .13, .10, .09, .09, .09, .12, .13], 7.1 if has_ci else 7.4)
    _section(fig, .198 if has_ci else .205, 'Method and interpretation')
    _write(fig, _LEFT, .172 if has_ci else .179, _text(report.methodology, 119, 4), 7.7, _INK)
    _write(fig, _LEFT, .111, _text(report.qualifications, 122, 3), 7.3, _MUTED)
    _write(fig, _LEFT, .059, _text(report.source, 125, 2), 7.0, _MUTED)
    return fig


def _contribution_rows(report: CmaInferenceReport, view: CmaInferenceView) -> list[list[str]]:
    """Rank signed asset contributions without changing their basis or numerical values."""
    rows = []
    result = view.result
    for factor, factor_row in result.factors.iterrows():
        if _premium(view, factor) == 'N/A':
            rows.append([factor, 'N/A', 'N/A', 'Nonidentified', 'Nonidentified'])
            continue
        values = result.factor_contributions[factor].dropna()
        top = values[values > 0].sort_values(ascending=False, kind='stable').head(2)
        bottom = values[values < 0].sort_values(kind='stable').head(2)
        cells = ['\n'.join(f'{_label(report, asset, 30)}  {_number(value, "bp")}'
                           for asset, value in tail.items()) or '-'
                 for tail in (top, bottom)]
        rows.append([factor, _premium(view, factor), _value(factor_row, 'revision', 'bp'),
                     *cells])
    return rows


def _carry_factor(view: CmaInferenceView) -> str:
    """Select the current blend or a correctly labelled historical carry coordinate."""
    return next((name for name in ('Carry FX', 'Carry FX futures', 'FX Carry', 'Carry G10')
                 if name in view.result.factors.index), 'Carry G10')


def _contributors_page(report: CmaInferenceReport, headline: CmaInferenceView) -> Figure:
    """Show every headline factor and both signs of the remaining asset forecast residual."""
    fig = _page(report, headline, 'Headline attribution', 2)
    _section(fig, .817, 'Top two / bottom two contributors for every MATF factor',
             'Asset contributions to GLS minus calibrated premia, in annual basis points.')
    _table(fig, _contribution_rows(report, headline),
           ['Factor', 'Implied\npremium', 'Revision\nbp', 'Top two: asset / bp',
            'Bottom two: asset / bp'],
           .363, .400, [.15, .10, .09, .33, .33], 7.4)
    _write(fig, _LEFT, .348,
           'Contribution = forecast gap x GLS multiplier. Gap = provider total return - cash '
           '- w x admitted alpha\n- calibrated systematic return. All assets, plus any saved '
           'coordinate offset, reconcile to the premium revision.', 7.4, _MUTED)
    carry = _carry_factor(headline)
    if headline.identification is not None and carry in headline.result.factors.index:
        _section(fig, .285, carry + ': forecast gaps and offsetting portfolio weights',
                 'Top two positive / bottom two negative contributions; the weights are not betas.')
        rows = _carry_influence_rows(report, headline)
        _table(fig, rows, ['Asset category', 'Gap bp', 'GLS weight', 'Contribution bp',
                           '+25bp shift bp'], .11, .123, [.40, .14, .15, .16, .15], 7.6)
        _write(fig, _LEFT, .089,
               'All scenario contributions, residual tails, forecast sensitivities and deletion '
               'results remain in the CSV exports.\n'
               'GLS weights describe the fitted factor-mimicking portfolio, not asset betas.',
               7.2, _MUTED)
        return fig
    _section(fig, .285, 'Largest positive / negative asset residuals',
             'Residual = alpha-adjusted provider excess return minus the GLS fitted return.')
    assets = headline.result.assets
    tails = [assets[assets.residual > 0].sort_values('residual', ascending=False).head(3),
             assets[assets.residual < 0].sort_values('residual').head(3)]
    rows = [[sign, _label(report, asset, 45), _value(row, 'residual', 'bp'),
             _value(row, 'residual_vol', 'risk'), _value(row, 'residual_share', 'percent')]
            for sign, tail in zip(('+', '-'), tails) for asset, row in tail.iterrows()]
    _table(fig, rows, ['Tail', 'Asset category', 'Residual bp', 'Residual vol',
                       'Residual S2 share'],
           .076, .157, [.07, .45, .15, .15, .18], 7.6)
    _write(fig, _LEFT, .059,
           'All scenario contributions and residuals are supplied as CSV tables. '
           'Short asset labels are expanded in asset_labels.csv.', 7.1, _MUTED)
    return fig


def _carry_influence_rows(report, headline):
    """Reuse exact contribution and shock rows for the carry interpretation table."""
    sensitivity = headline.result.factor_sensitivity
    carry = _carry_factor(headline)
    selected = (sensitivity[sensitivity.factor.eq(carry)]
                if not sensitivity.empty else sensitivity)
    if selected.empty:
        return []
    ordered = selected.sort_values('premium_revision_contribution', ascending=False)
    tails = (ordered[ordered.premium_revision_contribution > 0].head(2),
             ordered[ordered.premium_revision_contribution < 0].tail(2).iloc[::-1])
    return [[_label(report, row.asset, 37), _value(row, 'forecast_gap', 'bp'),
             _value(row, 'gls_multiplier'), _value(row, 'premium_revision_contribution', 'bp'),
             _value(row, 'premium_change_25bp', 'bp')]
            for tail in tails for _, row in tail.iterrows()]


def _contrast_interval(view, key):
    """Format annual percentages or dimensionless normalized premia without confusing units."""
    table = view.identification.contrasts
    if key not in table.index:
        return 'N/A'
    row = table.loc[key]
    if not np.isfinite(row.estimate):
        return 'N/A'
    scale = 100 if row.units == 'annual_excess_decimal' else 1
    estimate = f'{scale*row.estimate:.2f}'
    if row.status != 'conditional_working_model':
        return estimate+'\n[CI N/A]'
    return estimate+f'\n[{scale*row.lower:.2f}, {scale*row.upper:.2f}]'


def _identification_page(report, headline):
    """Keep individual sensitivity, pair contrasts, equivalence and assumptions on one page."""
    carry = _carry_factor(headline)
    single = carry in ('Carry FX', 'Carry FX futures', 'FX Carry')
    fig = _page(report, headline, f'Precision and {carry} sensitivity' if single else
                'Precision, carry contrasts and sensitivity', 3)
    ident, ci = headline.identification, headline.conditional_intervals
    target = ident.summary['precision_target']
    margin = ident.summary['equivalence_margin']
    target_note = ('No precision threshold.' if target is None else
                   f'W = CI half-width exceeds {100*target:g} pp; OK = within target.')
    _section(fig, .817, 'Individual identification and unchanged sensitivity ranges', target_note)
    rows = []
    for factor, row in headline.result.factors.iterrows():
        info = ident.factors.loc[factor]
        status = {'weak_for_target': 'W', 'within_target': 'OK',
                  'not_assessed': '-', 'unavailable': 'N/A'}[info.precision_status]
        rows.append([factor, _value(info, 'variance_inflation'),
                     _value(info, 'ci_half_width', 'return'), status,
                     _value(row, 'gross_gls_multiplier'), _value(row, 'max_abs_25bp_shift', 'bp'),
                     _value(row, 'loo_low', 'return'), _value(row, 'loo_high', 'return')])
    _table(fig, rows, ['Factor', 'VIF', 'CI half\nwidth pp', 'Flag', 'Gross\nweight',
                       'Max shock\nbp', 'LOO low\n%', 'LOO high\n%'],
           .548, .215, [.19, .09, .12, .07, .12, .14, .135, .135], 7.1)
    _write(fig, _LEFT, .537, 'VIF includes all other factors. Max shock = one +25 bp forecast '
           'change. LOO ranges are sensitivity, not CIs.', 7.0, _MUTED)
    if single:
        return _single_carry_precision(fig, report, headline)
    first, second = ident.summary['factor_pair']
    pair_title = ('Common carry and G10 - EM spread' if (first, second) ==
                  ('Carry G10', 'Carry EM') else 'Common and spread')
    _section(fig, .504, pair_title+': estimates and conditional CIs',
             f'Common = ({first} + {second})/2; spread = {first} - {second}. '
             'Normalized = premium / factor volatility.')
    rows = [[v.label.replace('\n', ' ')] + [_contrast_interval(v, key) for key in
             ('common', 'spread', 'normalized_common', 'normalized_spread')] for v in report.views]
    _table(fig, rows, ['View', 'Common\nannual %', 'Spread\nannual %',
                       'Common\nnormalized', 'Spread\nnormalized'],
           .309, .142, [.22, .195, .195, .195, .195], 7.0)
    _section(fig, .288, 'Pair identification after controlling for all other MATF factors')
    rows = []
    for view in report.views:
        pair = view.identification.pair
        direction = ('N/A' if pair.get('direction_status') != 'unique' else
                     f"{pair['weak_first']:+.2f}, {pair['weak_second']:+.2f}")
        equivalence = {'equivalent': 'Yes', 'not_established': 'Unproven',
                       'not_assessed': '-', 'unavailable': 'N/A'}[pair['equivalence_status']]
        rows.append([view.label.replace('\n', ' '), _number(pair.get('partial_loading_cosine')),
                     _number(pair.get('mutual_variance_inflation')),
                     _number(pair.get('information_min'), 'scientific'),
                     _number(pair.get('information_ratio')), direction,
                     _number(pair.get('equality_p')), equivalence])
    _table(fig, rows, ['View', 'Partial\ncosine', 'Pair\nVIF', 'Min info', 'Min/max\ninfo',
                       'Weak dir\nfirst, second', 'Equal SR\np-value', 'Equivalent?'],
           .143, .118, [.20, .10, .08, .13, .10, .16, .11, .12], 6.8)
    settings = ('No equivalence threshold.' if margin is None else
                f'Equivalence: normalized spread within +/-{margin:g}, '
                f"TOST at {100*(1-ci.summary['confidence_level']):g}%.")
    _write(fig, _LEFT, .133, settings+' Non-rejection of equality does not establish equivalence.',
           6.8, _MUTED)
    info = ci.summary
    _write(fig, _LEFT, .111,
           f"Pointwise {100*info['confidence_level']:g}% Student-t CIs: y = B lambda + e, "
           'e ~ N(0, tau^2 D); no intercept. B, diagonal D, cash, alpha and\n'
           'factor volatilities fixed. Independent forecast errors proportional to D are a working '
           'assumption; D itself is return risk.\n'
           f"tau^2 = residual S2 / (N-K); headline df={info['residual_df']}, "
           f"tau^2={_number(info['scale_squared'], 'scientific')}. Each view uses its own scale.\n"
           'Excludes fitted-model/input uncertainty; no simultaneous coverage. No factor merge or '
           'forecast change. Weak directions use\n'
           'volatility-normalized premia; small information means large geometric variance. '
           'Rank/df/zero-scale failures give CI N/A.\n'
           'Adviser envelopes and LOO remain sensitivities. Thresholds are diagnostic settings; '
           'full information and TOST intervals are in CSV.', 6.7, _INK)
    return fig


def _single_carry_precision(fig, report, headline):
    """Show the refitted single-carry model without inventing absent pair contrasts."""
    carry = _carry_factor(headline)
    blend = carry == 'Carry FX'
    _section(fig, .504, carry + ' across alpha and horizon views',
             ('Daily 50/50 G10/EM blend; separate carry regressors are absent from this fit.'
              if blend else 'Existing currency-basket factor; separate G10 and EM factors '
              'are absent from this fit.'))
    rows = []
    has_bands = all(v.result.provenance.get('adviser_box_bounds') for v in report.views)
    for view in report.views:
        if carry not in view.identification.factors.index:
            rows.append([view.label.replace('\n', ' ')] + ['N/A'] * (7 if has_bands else 5))
            continue
        info = view.identification.factors.loc[carry]
        vol = info.get('factor_volatility', np.nan)
        normalized = 'N/A'
        if np.isfinite(vol) and vol > 0 and np.isfinite(info.estimate):
            normalized = f'{info.estimate/vol:.2f}\n'
            normalized += (f'[{info.lower/vol:.2f}, {info.upper/vol:.2f}]'
                           if info.status == 'conditional_working_model' else '[CI N/A]')
        row = [view.label.replace('\n', ' '), _premium_interval(view, carry),
               normalized, _value(info, 'ci_half_width', 'return'),
               _value(info, 'variance_inflation'),
               {'within_target': 'OK', 'weak_for_target': 'Weak',
                'unavailable': 'N/A', 'not_assessed': '-'}[info.precision_status]]
        if has_bands:
            for box in ('advisor_p25_p75_box', 'advisor_min_max_box'):
                bounds = next(b for b in view.result.provenance['adviser_box_bounds']
                              if b['factor'] == carry and b['box'] == box
                              and b['cash_treatment'] == 'fixed_cash')
                row.append(f"[{100*bounds['lower']:.2f},\n{100*bounds['upper']:.2f}]")
        rows.append(row)
    level = 100*headline.conditional_intervals.summary['confidence_level']
    _table(fig, rows, ['View', f'Annual %\n{level:g}% CI', f'Normalized\n{level:g}% CI',
                       'Half-width\npp', 'VIF', 'Precision']
           + (['Adviser IQR\nbox %', 'Adviser full\nbox %'] if has_bands else []),
           .305, .145,
           [.16, .17, .16, .10, .08, .09, .12, .12] if has_bands else
           [.22, .21, .21, .14, .11, .11],
           6.5 if has_bands else 7.2)
    _section(fig, .28, 'Interpretation of the replacement model')
    definition = (
        'Carry FX is a daily 50/50 portfolio of the constructed Carry G10 '
        'and Carry EM factor returns.\n'
        'There is no second volatility target. The premium prior and scenario shocks '
        'average the two legs.\n'
        'Carry FX futures is the retired currency basket and is not a regressor in this model.\n\n'
        if blend else
        f'{carry} is the existing long AUD/NZD/CAD versus short JPY/CHF strategy, '
        'renamed from Carry.\n'
        'Its construction and 5% volatility target are unchanged. '
        'It is not a 50/50 G10/EM blend.\n\n')
    _write(fig, _LEFT, .25,
           definition +
           f'The historical model is refitted with {len(headline.result.factors)} factors. '
           'Loadings, residual risk and native '
           'admitted alpha are reestimated.\n'
           'Other factor definitions and the one-HAC-standard-error exposure-floor methodology '
           'are retained.\n\n'
           'A narrower CI is conditional on this replacement model; it does not establish that '
           'the omitted carry risks are redundant.\n'
           'G10-EM common/spread tests require a separate G10/EM fit '
           'and are not defined in this model.', 7.6, _INK)
    info = headline.conditional_intervals.summary
    _write(fig, _LEFT, .112,
           f"Pointwise {100*info['confidence_level']:g}% Student-t CIs: y = B lambda + e, "
           'e ~ N(0, tau^2 D), no intercept. Fixed B, diagonal D, cash and alpha.\n'
           'Normalized premia also condition on factor volatilities. Independent forecast errors '
           'proportional to D are a working assumption.\n'
           f"tau^2 = residual S2 / (N-K); headline df={info['residual_df']}. "
           'No allowance for estimated loadings or model selection; no simultaneous coverage.\n'
           'Precision flags use the annual half-width target stated above. '
           'LOO ranges are sensitivities.\n'
           + ('Adviser boxes use approximate geometric-to-arithmetic conversion, '
              'fixed cash and alpha; '
              'not CIs or factor quantiles. Shared-cash boxes are in CSV.' if has_bands else
              'No adviser forecast bands are supplied for this universe.'), 6.9, _MUTED)
    return fig


def _stability_page(report: CmaInferenceReport, headline: CmaInferenceView) -> Figure:
    """Present existing sensitivity and leave-one-out results without inventing uncertainty."""
    if headline.identification is not None:
        return _identification_page(report, headline)
    fig = _page(report, headline, 'Identification and influence', 3)
    _section(fig, .817, 'Factor identification and leave-one-out ranges',
             'A +25 bp shock changes one forecast; cash, alpha and the risk model stay fixed.')
    factors = headline.result.factors
    rows = [[factor, _value(row, 'weighted_loading_overlap', 'percent'),
             _value(row, 'gross_gls_multiplier'), _value(row, 'max_abs_25bp_shift', 'bp'),
             _value(row, 'loo_low', 'return'), _value(row, 'loo_high', 'return'),
             _label(report, row.get('loo_worst_asset', 'N/A'), 29)]
            for factor, row in factors.iterrows()]
    _table(fig, rows, ['Factor', 'Loading\noverlap', 'Gross GLS\nmultiplier',
                       'Max |25bp|\nshift bp', 'LOO low', 'LOO high', 'Worst excluded asset'],
           .471, .292, [.16, .10, .11, .11, .10, .10, .32], 7.2)
    _write(fig, _LEFT, .457,
           'Overlap: exposure explained by other whitened factors. Gross multiplier: total '
           'absolute forecast sensitivity.\nLOO omits one asset and reprojects retained '
           'forecasts. These ranges are not confidence intervals.', 7.3, _MUTED)
    _section(fig, .396, 'Carry G10: separating forecast gaps from GLS multipliers',
             'Top two positive and bottom two negative contributions in the headline view.')
    sensitivity = headline.result.factor_sensitivity
    selected = (sensitivity[sensitivity.factor.eq('Carry G10')]
                if not sensitivity.empty else sensitivity)
    rows = []
    if not selected.empty:
        ordered = selected.sort_values('premium_revision_contribution', ascending=False)
        tails = (ordered[ordered.premium_revision_contribution > 0].head(2),
                 ordered[ordered.premium_revision_contribution < 0].tail(2).iloc[::-1])
        rows = [[_label(report, row.asset, 37), _value(row, 'forecast_gap', 'bp'),
                 _value(row, 'gls_multiplier'),
                 _value(row, 'premium_revision_contribution', 'bp'),
                 _value(row, 'premium_change_25bp', 'bp')]
                for tail in tails for _, row in tail.iterrows()]
    _table(fig, rows, ['Asset category', 'Gap bp', 'GLS multiplier', 'Contribution bp',
                       '+25bp shift bp'], .225, .119, [.40, .14, .15, .16, .15], 7.6)
    ci = headline.conditional_intervals
    if ci is not None:
        _section(fig, .203, 'Conditional GLS intervals: assumptions')
        info = ci.summary
        assumptions = (
            f"Pointwise {100*info['confidence_level']:g}% Student-t intervals. Working model: "
            'y = B lambda + e; e ~ N(0, tau^2 D).\n'
            'B, diagonal D, cash and admitted alpha are fixed. Cross-asset forecast errors '
            'are independent,\nwith variance proportional to D. D is return risk; this '
            'forecast-error shape is an added assumption.\n'
            "tau^2 = residual S2 / (N-K); covariance(lambda) = tau^2 (B' D^-1 B)^-1.\n"
            f"Headline: df = {info['residual_df']}; tau^2 = "
            f"{_number(info['scale_squared'], 'scientific')}; "
            f"t critical = {_number(info['critical_value'])}. "
            'Each scenario uses its own residual scale.\n'
            'Excludes exposure, residual-risk, cash, alpha and factor-definition uncertainty. '
            'No simultaneous coverage.\n'
            'Not adviser dispersion or realized-return bands. Rank/df/scale failures give CI N/A; '
            'see CSV status.')
        _write(fig, _LEFT, .176, assumptions, 7.0, _INK)
        _write(fig, _LEFT, .076,
               'Alpha changes forecast gaps; fixed-model GLS multipliers stay unchanged. '
               'LOO reprojects the retained universe.\n'
               'Existing leave-one-out ranges and adviser envelopes remain sensitivity measures.',
               7.0, _MUTED)
        return fig
    _section(fig, .203, 'Reading the result')
    unavailable = factors.get('loo_unavailable_count')
    excluded = int(unavailable.max()) if unavailable is not None else None
    deletion_note = ('Deletion identification detail unavailable.' if excluded is None else
                     f'Maximum nonidentified deletions for a factor: {excluded}; '
                     'these are omitted from its range.')
    _write(fig, _LEFT, .175, _text(
        'Alpha changes forecast gaps, not fixed-model GLS multipliers. Deletion influence '
        'can be large even when a current contribution is small: deleting an asset changes '
        'the inversion for all retained categories. ' + deletion_note, 119, 5), 7.8, _INK)
    _write(fig, _LEFT, .089, _text(report.qualifications, 122, 3), 7.3, _MUTED)
    return fig


def _band(row, lower='lower', upper='upper'):
    """Format a supplied annual interval or sensitivity range without computing one."""
    if not np.isfinite([row.get(lower, np.nan), row.get(upper, np.nan)]).all():
        return 'N/A'
    return f'[{100*row[lower]:.2f}, {100*row[upper]:.2f}]'


def _precision_flag(value):
    """Distinguish unavailable precision from a failed economic precision target."""
    if value is None or pd.isna(value):
        return 'N/A'
    return 'OK' if value else 'Weak'


def _resolution_page(report, headline, factors):
    """Show the unchanged conditional estimates beside scale-free estimator geometry."""
    fig = _page(report, headline, 'Premia and precision', 1)
    level = 100*headline.conditional_intervals.summary['confidence_level']
    _section(fig, .817, 'GLS premia: current conditional intervals remain the default',
             f'Annual excess return %, [pointwise {level:g}% Student-t CI]. '
             'Each view retains its own residual scale; no scenario replaces these CIs.')
    rows = []
    for factor in factors:
        calibrated = next(v.result.factors.at[factor, 'calibrated_premium']
                          for v in report.views if factor in v.result.factors.index)
        rows.append([factor, _number(calibrated, 'return')]
                    + [_premium_interval(view, factor) for view in report.views])
    _table(fig, rows, ['Factor', 'Our MATF'] + [v.label for v in report.views],
           .422, .338, [.18, .10] + [.72/len(report.views)]*len(report.views), 6.9)
    descriptions = [f"{v.label.replace(chr(10), ' ')}: N={v.result.summary['n_assets']}, "
                    f"rank={v.result.summary['rank']}/{v.result.summary['n_factors']}, "
                    f"df={v.conditional_intervals.summary['residual_df']}" for v in report.views]
    _write(fig, _LEFT, .409, _text(' | '.join(descriptions), 126, 3), 6.8, _MUTED)
    settings = headline.resolution.summary
    annual = settings['annual_precision_target']
    normalized = settings['normalized_precision_target']
    _section(fig, .354, 'Headline geometry and precision',
             f'Annual target +/-{100*annual:g} pp; normalized target '
             f'+/-{normalized:g} premium/vol. '
             'Diagnostic settings, not constraints.')
    rows = []
    for factor, info in headline.resolution.factors.iterrows():
        ci = headline.conditional_intervals.factors.loc[factor]
        rows.append([factor, _value(info, 'normalized_resolution_per_unit_scale'),
                     _value(info, 'normalized_ci_half_width'),
                     _number((ci.upper-ci.lower)/2, 'return'),
                     _precision_flag(info.annual_precision_met),
                     _precision_flag(info.normalized_precision_met),
                     _value(info, 'gross_mimicking_weight')])
    _table(fig, rows, ['Factor', 'Resolution\nper unit scale', 'CI half-width\npremium/vol',
                       'CI half\nwidth pp', 'Annual\nprecision', 'Normalized\nprecision',
                       'Gross\nweight'],
           .104, .197, [.19, .16, .16, .14, .11, .13, .11], 6.6)
    _write(fig, _LEFT, .09,
           'Resolution measures geometric standard error per unit forecast-error scale, in '
           'premium/vol units. Lower is better.\n'
           'Gross weight is the absolute factor-mimicking exposure. Rank failure means '
           'individual premia are not identified. Full assumptions on page 3.', 6.8, _MUTED)
    _write(fig, _LEFT, .056, _text(report.source, 127, 2), 6.6, _MUTED)
    return fig


def _visibility_rows(report, headline):
    """Select low-visibility and high-influence sleeves from completed numerical tables."""
    assets = headline.resolution.assets
    sensitivity = headline.result.factor_sensitivity
    by_asset = {}
    if not sensitivity.empty:
        ordered = sensitivity.assign(_absolute=sensitivity.premium_change_25bp.abs())
        ordered = ordered.sort_values('_absolute', ascending=False, kind='stable')
        by_asset = {asset: group.iloc[0] for asset, group in ordered.groupby('asset', sort=False)}
    low = assets.residual_visibility.sort_values(kind='stable').head(3).index.tolist()
    high = list(by_asset)[:3]
    selected = list(dict.fromkeys(low+high))
    rows = []
    for asset in selected:
        row = assets.loc[asset]
        influence = by_asset.get(asset)
        rows.append([_label(report, asset, 34),
                     'Low visibility' if asset in low else 'High influence',
                     _value(row, 'fitted_leverage'), _value(row, 'residual_visibility'),
                     str(influence.factor) if influence is not None else 'N/A',
                     _value(influence, 'premium_change_25bp', 'bp')
                     if influence is not None else 'N/A'])
    return rows


def _geometry_contributors_page(report, headline):
    """Retain all factor revision tails while exposing what residual checks can observe."""
    fig = _page(report, headline, 'Contributions and residual visibility', 2)
    _section(fig, .817, 'Top two / bottom two benchmark-revision contributors',
             'Every MATF factor. Contributions explain GLS minus calibrated premia; '
             'they do not estimate premium bias.')
    _table(fig, _contribution_rows(report, headline),
           ['Factor', 'Implied\npremium', 'Revision\nbp', 'Top two: asset / bp',
            'Bottom two: asset / bp'], .363, .400, [.15, .10, .09, .33, .33], 7.2)
    _write(fig, _LEFT, .350,
           'Gap = provider total return - cash - w x admitted alpha '
           '- calibrated systematic return. '
           'Contribution = gap x GLS weight.\n'
           'All contributions plus any saved coordinate offset reconcile to the revision. '
           'Weights are factor-mimicking holdings, not betas.', 7.1, _MUTED)
    _write(fig, _LEFT, .316, _text('Source qualification: '+report.qualifications, 132, 2),
           6.5, _MUTED)
    diagonal = headline.resolution.summary['visibility_basis'] == 'asset_variance_whitened'
    if diagonal:
        _section(fig, .292, 'Lowest residual visibility and largest forecast influence',
                 'Three lowest visibility sleeves and three largest absolute single-sleeve '
                 '+25 bp effects; duplicates appear once.')
        _table(fig, _visibility_rows(report, headline),
               ['Asset / sleeve', 'Selection', 'Fitted\nleverage', 'Residual\nvisibility',
                'Most affected\nfactor', '+25bp effect\nbp'],
               .110, .130, [.28, .17, .12, .13, .17, .13], 6.7)
        interpretation = ('Visibility = 1 - fitted leverage. Low visibility means a sleeve shock '
                          'is mostly absorbed into premia and is hard to detect\n'
                          'in residual lack-of-fit. Visibility sums to N - rank. A pure '
                          'factor-span shock changes premia without changing the residual score.')
    else:
        _section(fig, .292, 'Residual visibility uses ordered whitened coordinates')
        interpretation = _text(headline.resolution.summary['asset_visibility_interpretation'],
                               120, 5)
        _write(fig, _LEFT, .252, interpretation, 8.0, _INK)
        interpretation = ('Dense residual covariance mixes assets during whitening. '
                          'No individual asset visibility ranking is displayed.\n'
                          'The full ordered-coordinate geometry and original-coordinate '
                          'leverage remain in the sidecar tables.')
    _write(fig, _LEFT, .090, interpretation, 6.9, _MUTED)
    _write(fig, _LEFT, .055,
           'Residual alpha and premium influence answer different questions. '
           'Complete contribution, visibility and sensitivity tables are supplied '
           'alongside the PDF.',
           6.7, _MUTED)
    return fig


def _scenario_label(scenario):
    """Prefer a declared scenario label while preserving its actual inference method."""
    return str(scenario.summary.get('label') or scenario.summary.get('scenario_label')
               or scenario.summary['method']).replace('_', ' ')


def _scenario_description(scenario):
    """Print the distribution and reference assumptions that distinguish scenario ranges."""
    info = scenario.summary
    law = {'none': 'covariance only, no distributional interval',
           'normal': 'Gaussian covariance scenario',
           'reference_student_t': 'Student-t reference scenario'}.get(
               info['distribution'], info['distribution'].replace('_', ' '))
    if info['distribution'] == 'reference_student_t':
        law += (f" (reference df={info['reference_degrees_of_freedom']:g}; "
                'independent scale assumed)')
    status = ('adviser dispersion, not a CI' if info['method'] == 'adviser_disagreement' or
              info['calibration_status'] == 'adviser_disagreement_scenario_not_confidence_interval'
              else 'assumed scenario, not empirically calibrated')
    reference = info.get('reference_provenance')
    source = ' Reference: '+_text(reference, 55, 1)+'.' if reference else ''
    return (f"{_scenario_label(scenario)}: {law}; {status}; "
            f"target: {info['target'].replace('_', ' ')}."+source)


def _forecast_box_note(headline):
    """Retain the existing carry forecast-box endpoints beside the new uncertainty panels."""
    carry = _carry_factor(headline)
    boxes = headline.result.provenance.get('adviser_box_bounds', ())
    selected = {box['box']: box for box in boxes
                if box['factor'] == carry and box['cash_treatment'] == 'fixed_cash'}
    parts = []
    for key, label in (('advisor_p25_p75_box', 'IQR'), ('advisor_min_max_box', 'full')):
        if key in selected:
            box = selected[key]
            parts.append(f'{label} [{100*box["lower"]:.2f}, {100*box["upper"]:.2f}]%')
    return ((carry+' unchanged adviser sensitivity boxes: '+', '.join(parts)+
             '. Fixed cash/alpha; approximate geometric-to-arithmetic conversion. '
             'All views and factors are in sidecars.') if parts else '')


def _extension_stability_page(report, headline):
    """Distinguish conditional inference, refit stability and declared covariance scenarios."""
    fig = _page(report, headline, 'Stability and uncertainty scenarios', 3)
    stability = headline.loading_stability
    scenarios = headline.uncertainty_scenarios[:2]
    _section(fig, .817, 'Loading stability and unchanged forecast sensitivity',
             'Annual %. LOO and adviser boxes are deterministic sensitivity ranges; '
             'paired-refit percentiles are stability bands, not calibrated CIs.')
    rows = []
    for factor, row in headline.result.factors.iterrows():
        band = stability.bands.loc[factor] if stability is not None else None
        rows.append([factor, _band(row, 'loo_low', 'loo_high'),
                     _band(band) if band is not None else 'Not supplied',
                     _value(band, 'median_shift', 'bp') if band is not None else 'N/A']
                    + [_band(s.factors.loc[factor], 'ci_lower', 'ci_upper') for s in scenarios])
    count = len(scenarios)
    level = 100*headline.conditional_intervals.summary['confidence_level']
    _table(fig, rows, ['Factor', 'LOO range\n%', f'Paired loading\n{level:g}% band %',
                       'Median shift\nfrom baseline bp']
           + [_text(_scenario_label(s), 15, 2)
              +f"\n{100*s.summary['confidence_level']:g}% range %" for s in scenarios],
           .493, .270, [.17, .16, .22, .17] + [.28/count]*count if count else
           [.23, .22, .29, .26], 6.7 if count else 7.1)
    if stability is None:
        note = ('Paired loading/alpha draws not supplied for this view; no loading-uncertainty '
                'band is inferred from the conditional CI.')
    else:
        info = stability.summary
        solver_note = (f"solver failures={info['solver_failure_count']}"
                       if info.get('solver_status_provided', True)
                       else 'solver status not archived')
        note = (f"Finite full-rank draws: {info['n_available']}/{info['n_draws']} "
                f"({100*info['availability_rate']:.1f}%); "
                f"rank losses={info['rank_failure_count']}, "
                f'{solver_note}, '
                f"nonfinite={info['nonfinite_count']}. "
                f"Alpha policy: {info['alpha_policy'].replace('_', ' ')}.")
    _write(fig, _LEFT, .480, _text(note, 132, 2), 6.7, _MUTED)
    _section(fig, .438, 'Optional uncertainty assumptions and paired comparisons')
    details = []
    for scenario in scenarios:
        details.append(_scenario_description(scenario))
    if not scenarios:
        details.append('No alternative forecast-error covariance or '
                       'external reference scale supplied.')
    if len(headline.uncertainty_scenarios) > len(scenarios):
        details.append(f'Showing {len(scenarios)} of {len(headline.uncertainty_scenarios)} '
                       'declared scenarios; all scenarios are retained in the sidecars.')
    _write(fig, _LEFT, .413, _text(' '.join(details), 133, 4), 6.7, _INK)
    contrast_rows, contrast_notes = [], []
    n_covariance = 1 if report.loading_stability_contrasts else 2
    for contrast in report.provider_contrasts[:n_covariance]:
        largest = contrast.factors.estimate.abs().sort_values(ascending=False).head(2).index
        for factor in largest:
            row = contrast.factors.loc[factor]
            label = _text(contrast.summary['comparison_label']+' (covariance)', 39, 1)
            contrast_rows.append([label, factor,
                                  _value(row, 'estimate', 'bp'),
                                  _band(row, 'ci_lower', 'ci_upper')])
    n_loading = 1 if report.provider_contrasts else 2
    for contrast in report.loading_stability_contrasts[:n_loading]:
        largest = contrast.bands.baseline.abs().sort_values(ascending=False).head(2).index
        info = contrast.summary
        for factor in largest:
            row = contrast.bands.loc[factor]
            contrast_rows.append([_text(info['comparison_label']+' (loading)', 39, 1), factor,
                                  _value(row, 'baseline', 'bp'), _band(row)])
        contrast_notes.append(f"{_text(info['comparison_label'], 25, 1)}: "
                              f"{info['n_available']}/{info['n_draws']} "
                              f"paired draws, rank losses={info['rank_failure_count']}; "
                              f"{info['alpha_policy'].replace('_', ' ')}.")
    if contrast_rows:
        _table(fig, contrast_rows,
               ['Paired comparison / method', 'Largest two absolute gaps',
                'Gap bp', 'Scenario / band %'],
               .266, .075, [.34, .29, .13, .24], 6.7)
        contrast_note = ('Covariance gaps use projected means; loading gaps use matching '
                         'bootstrap baselines. Two largest absolute gaps per selected comparison; '
                         'all comparisons are in sidecars. '
                         +' '.join(contrast_notes))
    else:
        contrast_note = ('No paired provider contrast supplied. Independence or a consensus '
                         'reference is not inferred for this universe.')
    _write(fig, _LEFT, .255, _text(contrast_note, 137, 3), 6.7, _MUTED)
    _section(fig, .214, 'Assumptions and interpretation')
    level = 100*headline.conditional_intervals.summary['confidence_level']
    assumptions = (
        f'Default pointwise {level:g}% Student-t CIs: fixed B, diagonal D, cash and alpha; '
        'centered Gaussian forecast errors with covariance tau^2 D. '
        'tau^2 = residual S2 / (N - rank). D measures return risk; this forecast-error shape '
        'is an added working assumption. No simultaneous coverage or '
        'loading/model-selection uncertainty. '
        'The estimand is the provider projected mean, including any factor-span subjective view.\n'
        'Paired loading bands condition on the stated baseline and finite full-rank draws. '
        'Adviser disagreement describes cross-adviser dispersion, not consensus-mean estimation '
        'error or forecast accuracy. Forecast boxes and LOO are sensitivities, not CIs.')
    _write(fig, _LEFT, .188, _text(assumptions, 137, 7), 6.6, _INK)
    notes = list(report.extension_notes)
    box_note = _forecast_box_note(headline)
    if box_note:
        notes.insert(0, box_note)
    if stability is not None:
        notes.insert(0, 'Bootstrap baseline (separate from native point estimate): '
                     +str(stability.summary['baseline_provenance']))
    elif report.loading_stability_contrasts:
        notes.insert(0, 'Paired contrast baseline: '
                     +str(report.loading_stability_contrasts[0].summary['baseline_provenance']))
    _write(fig, _LEFT, .090, _text(' '.join(notes) or report.qualifications, 134, 3), 6.5, _MUTED)
    return fig


def _build_inference_pages(report: CmaInferenceReport) -> list[Figure]:
    """Compose three bounded pages and clean partial figures after a rendering error."""
    headline, factors = _validate(report)
    previous = set(plt.get_fignums())
    try:
        if headline.resolution is not None:
            figures = [_resolution_page(report, headline, factors),
                       _geometry_contributors_page(report, headline),
                       _extension_stability_page(report, headline)]
        else:
            figures = [_comparison_page(report, headline, factors),
                       _contributors_page(report, headline), _stability_page(report, headline)]
        return figures
    except Exception:
        for number in set(plt.get_fignums()) - previous:
            plt.close(number)
        raise


def validate_cma_inference_report(report: CmaInferenceReport) -> None:
    """Validate completed report bindings without constructing figures or writing files.

    Provider adapters can use this before exporting sidecars when PDF rendering is
    deferred. It applies exactly the same validation as :func:`render_cma_inference`.
    """
    _validate(report)


def render_cma_inference(report: CmaInferenceReport, output_path: str | Path) -> Path:
    """Atomically render three provider-analysis pages; never fit or mutate numerical inputs.

    Up to six precomputed views and twelve factors share one calibrated basis.
    Missing/nonidentified factor coordinates are displayed as N/A. The chosen headline
    supplies attribution and stability pages; complete scenario tables remain caller-owned.
    Explicit resolution results select the extended three-page layout. Alternative
    covariance scenarios and paired loading bands remain separate from default CIs;
    no uncertainty scale, scenario selection policy or refit is estimated here.
    """
    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != '.pdf':
        raise ValueError('output_path must have the .pdf extension.')
    _validate(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    figures = []
    temporary = None
    try:
        with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                             'ps.fonttype': 42, 'text.usetex': False}):
            figures = _build_inference_pages(report)
            with NamedTemporaryFile(prefix='.cma-inference-', suffix='.pdf',
                                    dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            qis.save_figs_to_pdf(figures, temporary.stem, local_path=str(temporary.parent),
                                 orientation='portrait', add_current_date=False)
            temporary.replace(target)
    finally:
        for fig in figures:
            plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
