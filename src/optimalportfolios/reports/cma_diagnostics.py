"""Compact CMA audit pages rendered from completed numerical diagnostic results.

The renderer only selects, formats and aggregates display rows. GLS, policy decisions and
scenario construction belong to ``optimalportfolios.matf.cma_diagnostics``. Complete result
tables and provenance are persisted by the caller beside this intentionally bounded report.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import qis

from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticResult


_INK = '#183047'
_MUTED = '#556576'
_TEAL = '#176b70'
_AMBER = '#986313'
_RED = '#a33b33'
_PALE = '#edf3f5'
_LEFT = 0.055
_WIDTH = 0.890
_ASSET_ROWS = 8
_FACTOR_ROWS = 12


def _text(value: Any, width: int = 80, lines: int = 2) -> str:
    """Normalize whitespace and bound labels without interpreting financial names as math."""
    if value is None:
        return 'Not supplied'
    value = ' '.join(str(value).replace('\u2011', '-').split())
    parts = wrap(value, width=width, break_long_words=True, break_on_hyphens=False)
    if len(parts) > lines:
        parts = parts[:lines]
        parts[-1] = parts[-1].rstrip(' .')[:max(1, width - 3)] + '...'
    return '\n'.join(parts)


def _number(value: Any, style: str = 'score') -> str:
    """Format a supplied scalar while keeping unavailable values visibly unavailable."""
    if value is None or not np.isfinite(float(value)):
        return 'N/A'
    value = float(value)
    if style == 'percent':
        return f'{100.0 * value:.1f}%' if abs(value) < 1000 else f'{100 * value:.1e}%'
    if style in ('return', 'risk'):
        if abs(value) >= 1000:
            return f'{100 * value:.1e}%'
        decimals = 3 if style == 'risk' and abs(value) < 0.01 else 2
        return f'{100.0 * value:.{decimals}f}%'
    if style == 'bp':
        return f'{10000.0 * value:+.0f}' if abs(value) < 100 else f'{10000 * value:+.1e}'
    if style == 'scientific':
        return f'{value:.1e}'
    if style == 'integer':
        return str(int(value))
    return f'{value:.3f}' if abs(value) < 10000 else f'{value:.2e}'


def _value(row: pd.Series, key: str, style: str = 'score') -> str:
    """Format an optional column of an already calculated diagnostic table."""
    return _number(row.get(key), style)


def _stability_cells(row: pd.Series) -> list[str]:
    """Display deterministic forecast sensitivity and identified deletion endpoints."""
    return [_value(row, 'max_abs_25bp_shift', 'bp'),
            _value(row, 'loo_low', 'return') + '\n' + _value(row, 'loo_high', 'return')]


def _name(frame: pd.DataFrame, key: Any, width: int = 34) -> str:
    """Prefer the supplied display name while keeping the row identity as a fallback."""
    value = frame.at[key, 'name'] if 'name' in frame else key
    if pd.isna(value):
        value = key
    return _text(value, width, 2)


def _write(fig: Figure, x: float, y: float, value: Any, size: float = 9.0,
           color: str = _INK, **kwargs: Any) -> None:
    """Place safe, consistently styled page text in figure coordinates."""
    fig.text(x, y, str(value), fontsize=size, color=color, va='top',
             fontfamily='DejaVu Sans', parse_math=False, **kwargs)


def _page(result: CmaDiagnosticResult, title: str, section: str) -> Figure:
    """Create an A4 page with a repeatable identity strip and no implicit date changes."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    _write(fig, _LEFT, 0.978, 'CAPITAL MARKET ASSUMPTIONS  /  DIAGNOSTIC REPORT', 8, _TEAL,
           weight='bold')
    _write(fig, _LEFT, 0.947, _text(title, 56, 2), 18, weight='bold')
    meta = result.provenance
    identity = ' | '.join(str(meta.get(key) or 'Not supplied') for key in
                          ('universe', 'as_of', 'reference_ccy'))
    _write(fig, _LEFT, 0.879, _text(identity, 106, 2), 8.3, _MUTED)
    basis_fields = ['model_name']
    if meta.get('factor_model'):
        basis_fields.append('factor_model')
    basis_fields.append('forecast_basis')
    basis = ' | '.join(str(meta.get(key) or 'Not supplied') for key in basis_fields)
    if meta.get('cash_rate') is not None:
        basis = 'Reference cash ' + _number(meta['cash_rate'], 'return') + ' | ' + basis
    _write(fig, _LEFT, 0.849, _text(basis, 108, 1), 8.0, _MUTED)
    _write(fig, _LEFT, 0.035, section.upper(), 7.5, _MUTED)
    return fig


def _cards(fig: Figure, entries: list[tuple[str, str]], y: float = 0.749) -> None:
    """Draw a bounded row of metric cards using precomputed values."""
    gap = 0.014
    width = (_WIDTH - gap * (len(entries) - 1)) / len(entries)
    for i, (label, value) in enumerate(entries):
        x = _LEFT + i * (width + gap)
        fig.patches.append(Rectangle((x, y), width, 0.081, transform=fig.transFigure,
                                    facecolor=_PALE, edgecolor='none', zorder=-1))
        _write(fig, x + 0.012, y + 0.071, label, 8.1, _MUTED)
        _write(fig, x + 0.012, y + 0.041, value, 20, _TEAL, weight='bold')


def _section(fig: Figure, y: float, title: str, note: str | None = None) -> None:
    """Place a section title and its optional bounded explanatory line."""
    _write(fig, _LEFT, y, title, 10.3, weight='bold')
    if note:
        _write(fig, _LEFT, y - 0.022, _text(note, 119, 2), 7.5, _MUTED)


def _table(fig: Figure, rows: list[list[str]], columns: list[str],
           bottom: float, height: float, widths: list[float],
           fontsize: float = 8.0) -> None:
    """Compose the established QIS table primitive into a fixed page panel."""
    if not rows:
        _write(fig, _LEFT, bottom + height - 0.02, 'No applicable rows.', 9, _MUTED)
        return
    line_count = max(value.count('\n') + 1 for value in columns)
    line_count += sum(max(value.count('\n') + 1 for value in row) for row in rows)
    natural_height = line_count * (fontsize * 1.45 + 3) / (fig.get_figheight() * 72)
    table_height = min(height, natural_height)
    ax = fig.add_axes([_LEFT, bottom + height - table_height, _WIDTH, table_height])
    qis.plot_df_table(pd.DataFrame(rows, columns=columns), add_index_as_column=False,
                      col_widths=widths, fontsize=fontsize, header_color=_INK,
                      row_colors=('#f1f5f6', 'white'), edge_color='white',
                      left_aligned_first_col=True, ax=ax)
    table = ax.tables[0]
    row_lines = {row: max(cell.get_text().get_text().count('\n') + 1
                          for (r, _), cell in table.get_celld().items() if r == row)
                 for row in range(len(rows) + 1)}
    for (row, column), cell in table.get_celld().items():
        cell.PAD = 0.06
        cell.set_height(row_lines[row])
        cell.get_text().set_parse_math(False)
        cell.get_text().set_fontfamily('DejaVu Sans')
        cell.get_text().set_linespacing(1.1)
        if row == 0:
            cell.get_text().set_weight('bold')
        if column > 0:
            cell.get_text().set_ha('center' if row == 0 else 'right')
        else:
            cell.get_text().set_ha('center' if row == 0 else 'left')


def _status_label(value: Any) -> str:
    """Shorten common statuses without presenting unavailable or nonunique values as passes."""
    labels = {'full_rank': 'Unique', 'rank_deficient': 'Nonunique',
              'saturated': 'Nonunique', 'not_identified': 'Nonunique',
              'identified': 'Unique', 'unavailable': 'N/A', 'not_applicable': 'N/A'}
    return _text(labels.get(str(value), str(value).replace('_', ' ')), 17, 2)


def _rank_note(summary: dict[str, Any]) -> tuple[str, str]:
    """Translate numerical identification into a concise report qualification."""
    state = summary.get('identification_status')
    counts = 'Assets {} | Factors {} | Rank {} | Residual dimensions {}'.format(
        *(summary.get(key, 'N/A') for key in
          ('n_assets', 'n_factors', 'rank', 'residual_dimension')))
    if state == 'saturated':
        return counts + '\nSaturated universe: zero residual has no diagnostic power.', _AMBER
    if state == 'rank_deficient':
        return (counts + '\nFactor premia are nonunique; fitted asset returns remain identified.',
                _AMBER)
    return counts, _MUTED


def _forecast_page(result: CmaDiagnosticResult, title: str) -> Figure:
    """Show forecast accounting, factor premia and bounded residual contributors."""
    fig = _page(result, title, '01 / Forecast consistency')
    s = result.summary
    _cards(fig, [('Total squared Sharpe', _number(s.get('total_sharpe2'))),
                 ('Implied systematic', _number(s.get('implied_systematic_sharpe2'))),
                 ('Full residual', _number(s.get('residual_sharpe2'))),
                 ('Factor access / FPIR', _number(s.get('fpir'), 'percent'))])
    note, color = _rank_note(s)
    exclusions = result.provenance.get('excluded_assets')
    if exclusions is not None and len(exclusions):
        first, *rest = note.split('\n')
        note = '\n'.join([first + f' | Excluded {len(exclusions)}', *rest])
    note += ('\nWhitened condition number ' + _number(s.get('condition_number'), 'scientific')
             + ' | Relative rank cutoff ' + _number(s.get('relative_cutoff'), 'scientific'))
    _write(fig, _LEFT, 0.732, note, 8.3, color)
    factors = result.factors
    rows = [[_text(key, 36, 2), _value(row, 'calibrated_premium', 'bp'),
             _value(row, 'implied_premium', 'bp'), _value(row, 'revision', 'bp'),
             _status_label(row.get('status', s.get('identification_status', 'unavailable')))]
            for key, row in factors.iloc[:_FACTOR_ROWS].iterrows()]
    while sum(max(value.count('\n') + 1 for value in row) for row in rows) > 18:
        rows.pop()
    _section(fig, 0.682, 'Factor premia',
             f'Annual excess premia in basis points. Showing {len(rows)} '
             f'of {len(factors)} factors; factor order is preserved.')
    columns = ['Factor', 'Calibrated\nbp', 'GLS implied\nbp', 'Revision\nbp', 'Identification']
    widths = [0.38, 0.15, 0.15, 0.14, 0.18]
    stability = 'max_abs_25bp_shift' in factors
    if stability:
        rows = [row + _stability_cells(factors.iloc[i]) for i, row in enumerate(rows)]
        columns += ['Max |25bp|\nshift bp', 'LOO premia\nlow / high']
        widths = [.27, .11, .11, .10, .12, .12, .17]
    _table(fig, rows, columns, 0.390, 0.242, widths, 7.4 if stability else 8.)
    assets = result.assets
    ordered = assets.sort_values('residual_score', ascending=False, kind='stable')
    selected = ordered.iloc[:_ASSET_ROWS]
    rows = [[_name(assets, key, 30), _value(row, 'excess_cma', 'return'),
             _value(row, 'fitted_excess_cma', 'return'), _value(row, 'residual', 'bp'),
             _value(row, 'residual_vol', 'risk'), _value(row, 'residual_score'),
             _value(row, 'residual_share', 'percent')] for key, row in selected.iterrows()]
    remaining = ordered.iloc[_ASSET_ROWS:]
    if not remaining.empty:
        rows.append([f'Remaining {len(remaining)} assets', '-', '-', '-', '-',
                     _number(remaining['residual_score'].sum(min_count=1)),
                     _number(remaining['residual_share'].sum(min_count=1), 'percent')])
    _section(fig, 0.366, 'Largest residual contributions',
             'Signed gaps identify the view; squared-Sharpe scores measure its residual risk. '
             'Complete rows are in assets.csv.')
    _table(fig, rows, ['Asset', 'Excess\nCMA', 'GLS\nfitted', 'Residual\nbp',
                       'Residual\nvol', 'Score', 'Share'], 0.078, 0.232,
           [0.32, 0.115, 0.115, 0.115, 0.115, 0.105, 0.115], 7.9)
    note = ('All Sharpe measures are unconstrained, model-implied excess opportunities.\n'
            'Small residual risks can magnify forecast adjustments.')
    if stability:
        note += ('\n25bp = one forecast shock; LOO = omit one asset, fixed risk/alpha. '
                 'Rank-loss deletions are excluded.')
    _write(fig, _LEFT, 0.075 if stability else 0.062, note, 7.3, _MUTED)
    return fig


def _asset_policy(checks: pd.DataFrame, asset: Any) -> str:
    """Summarize supplied policy decisions for an asset without reevaluating limits."""
    if checks.empty:
        return 'No policy'
    applicable = checks.loc[checks['asset'] == asset, 'status']
    if applicable.empty:
        return 'No checks'
    if (applicable == 'breach').any():
        return f'{int((applicable == "breach").sum())} breach(es)'
    if (applicable == 'unavailable').any():
        return 'Unavailable'
    if (applicable == 'pass').any():
        return 'Pass'
    return 'Not applicable'


def _admission_page(result: CmaDiagnosticResult, title: str) -> Figure:
    """Separate raw admission, off-span admission, policy checks and quality evidence."""
    fig = _page(result, title, '02 / Admission and model quality')
    s = result.summary
    _cards(fig, [('Raw admission score', _number(s.get('admission_score'))),
                 ('Projected admission', _number(s.get('admission_residual_score'))),
                 ('Full forecast residual', _number(s.get('residual_sharpe2')))])
    policy = s.get('policy_name') or 'No admission policy supplied'
    budget = 'Budget {} | Used/budget {} | Admission in factor span {}'.format(
        _number(s.get('admission_budget')),
        _number(s.get('admission_budget_ratio'), 'percent'),
        _number(s.get('admission_in_span_share'), 'percent'))
    _write(fig, _LEFT, 0.732, budget, 8.6, _TEAL)
    _write(fig, _LEFT, 0.715,
           'Calibrated systematic score ' + _number(s.get('calibrated_systematic_sharpe2'))
           + ' | Direct-factor ceiling ' + _number(s.get('factor_ceiling2')), 8.0, _MUTED)
    assets = result.assets
    checks = result.policy_checks
    breaches = checks.loc[checks['status'] == 'breach'] if not checks.empty else checks
    if not checks.empty:
        for check, label, style in [('cap1_positive_ir', 'IR cap', 'score'),
                                     ('cap2_admission_share', 'share cap', 'percent')]:
            limits = checks.loc[checks['check'] == check, 'limit'].dropna().unique()
            if len(limits) == 1:
                policy += f' | {label} {_number(limits[0], style)}'
    _section(fig, 0.698, 'Historical admission review', _text(policy, 113, 1))
    if 'admitted_alpha' in assets and assets['admitted_alpha'].notna().any():
        selection = assets.loc[assets['admitted_alpha'].fillna(0).ne(0)].copy()
        selection['_breach'] = selection.index.isin(breaches.get('asset', []))
        selection['_size'] = selection['admitted_alpha'].abs()
        selection = selection.sort_values(['_breach', '_size'], ascending=False, kind='stable')
        rows = [[_name(assets, key, 32), _value(row, 'admitted_alpha', 'bp'),
                 _value(row, 'admission_ir'), _value(row, 'admission_share', 'percent'),
                 _value(row, 'r_squared', 'percent'), _asset_policy(checks, key)]
                for key, row in selection.iloc[:_ASSET_ROWS].iterrows()]
        if rows:
            _table(fig, rows, ['Asset', 'Admitted\nbp', 'Admitted\nIR', 'Admission\nshare',
                               'Model\nR-squared', 'Policy checks'], 0.438, 0.214,
                   [0.33, 0.125, 0.12, 0.13, 0.13, 0.165], 8.0)
        else:
            _write(fig, _LEFT, 0.610, 'All supplied admissions are zero.', 10, _MUTED)
        hidden = max(0, len(selection) - _ASSET_ROWS)
        detail = (f'{len(breaches)} breached checks in the complete policy table. '
                  f'{hidden} additional nonzero admissions are in assets.csv.')
    else:
        _write(fig, _LEFT, 0.610, 'Admission construction data are not available.', 10, _MUTED)
        detail = 'The GLS audit remains available; admission cannot be inferred from its residual.'
    _write(fig, _LEFT, 0.418, _text(detail, 117, 2), 7.7, _MUTED)
    _section(fig, 0.382, 'Recorded adjustment channels',
             'Residual attribution is additive and signed. Individual channel scores do not add.')
    channels = result.channels
    channel_labels = {'discretionary_add_on': 'Discretionary add-on',
                      'admitted_alpha': 'Admitted historical alpha',
                      'regional_adjustment': 'Regional adjustment',
                      'fx_uip_shift': 'FX / UIP shift'}
    rows = [[_text(channel_labels.get(key, str(key).replace('_', ' ')), 45, 2),
             _value(row, 'raw_score'), _value(row, 'projected_score'),
             _value(row, 'residual_attribution')] for key, row in channels.iloc[:5].iterrows()]
    while sum(max(value.count('\n') + 1 for value in row) for row in rows) > 8:
        rows.pop()
    if len(channels) > len(rows):
        remainder = channels.iloc[len(rows):]
        rows.append([f'Remaining {len(remainder)} channels', '-', '-',
                     _number(remainder['residual_attribution'].sum(min_count=1))])
    _table(fig, rows, ['Channel', 'Raw\nscore', 'Projected\nscore', 'Full-residual\nattribution'],
           0.206, 0.126, [0.47, 0.17, 0.17, 0.19], 7.9)
    validation = result.validation
    _section(fig, 0.184, 'Numerical integrity and availability')
    ordered = validation.copy()
    if not ordered.empty:
        ordered['_failure'] = ordered['status'].eq('fail')
        ordered = ordered.sort_values('_failure', ascending=False, kind='stable')
    rows = [[_text(row.get('check'), 54, 1), _value(row, 'error', 'scientific'),
             _value(row, 'tolerance', 'scientific'), _status_label(row.get('status'))]
            for _, row in ordered.iloc[:4].iterrows()]
    _table(fig, rows, ['Check', 'Error', 'Tolerance', 'Status'], 0.073, 0.084,
           [0.53, 0.16, 0.16, 0.15], 7.4)
    failed = int(validation['status'].eq('fail').sum()) if not validation.empty else 0
    _write(fig, _LEFT, 0.062,
           f'{failed} failed numerical checks; {len(validation)} checks recorded. '
           'Full validation and exclusions are in the report bundle.\n'
           'Caps govern historical admission only; passing them does not govern other adjustments.',
           7.1, _RED if failed else _MUTED)
    return fig


def _scenario_page(result: CmaDiagnosticResult, title: str) -> Figure:
    """Show supplied scenario CMAs and baseline admissions without constructing new shocks."""
    fig = _page(result, title, '03 / Supplied scenarios')
    scenarios = result.scenarios
    selected = list(scenarios.columns[:3])
    _write(fig, _LEFT, 0.809, 'Scenario CMAs on the supplied universe', 12, weight='bold')
    _write(fig, _LEFT, 0.782,
           _text(f'Annual excess returns, showing {len(selected)} of {len(scenarios.columns)} '
                 'scenarios. Each panel lists its six lowest asset forecasts. '
                 'Scenario horizon and treatment of adjustments follow the supplied inputs.',
                 116, 3), 8.2, _MUTED)
    metadata = result.provenance.get('scenario_metadata') or {}
    for i, scenario in enumerate(selected):
        top = 0.714 - i * 0.211
        _section(fig, top, _text(scenario, 86, 1))
        descriptor = metadata.get(str(scenario), {}) if isinstance(metadata, dict) else {}
        if isinstance(descriptor, dict):
            descriptor = '; '.join(f'{key}: {value}' for key, value in descriptor.items())
        _write(fig, _LEFT, top - 0.022,
               _text(descriptor or 'Scenario convention: see saved input metadata.', 117, 1),
               7.5, _MUTED)
        chosen = scenarios[scenario].sort_values(kind='stable').iloc[:6]
        rows = []
        for key, value in chosen.items():
            row = result.assets.loc[key]
            rows.append([_name(result.assets, key, 43), _value(row, 'excess_cma', 'return'),
                         _number(value, 'return'), _value(row, 'admitted_alpha', 'return')])
        _table(fig, rows, ['Asset', 'Base\nexcess CMA', 'Scenario\nexcess CMA',
                           'Baseline\nadmission'], top - 0.198, 0.151,
               [0.43, 0.19, 0.19, 0.19], 8.0)
    _write(fig, _LEFT, 0.068,
           'Baseline admission is shown for context; it does not assert survival under stress.\n'
           'These are asset-level forecasts. No portfolio weights or benchmark floor are inferred.',
           7.6, _MUTED)
    return fig


def _residual_tails(assets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select up to five positive and negative signed gaps, preserving input ties."""
    ordered = assets.sort_values('residual', ascending=False, kind='stable')
    top = ordered.loc[ordered.residual > 0].head(5)
    bottom = assets.loc[assets.residual < 0].sort_values('residual', kind='stable').head(5)
    return top, bottom


def _alpha_comparison_page(result: CmaDiagnosticResult, zero: CmaDiagnosticResult,
                           title: str) -> Figure:
    """Compare already computed forecast accounting and implied factor premia."""
    fig = _page(result, title, 'Alpha views / Comparison')
    _section(fig, .815, 'Same universe and risk model; two admission views',
             'Zero view removes only specified admitted alpha. Other CMA adjustments remain.')
    rows = [[label, _number(zero.summary.get(key), style),
             _number(result.summary.get(key), style)] for key, label, style in (
        ('total_sharpe2', 'Total squared Sharpe', 'score'),
        ('implied_systematic_sharpe2', 'Implied systematic squared Sharpe', 'score'),
        ('residual_sharpe2', 'Residual squared Sharpe', 'score'),
        ('admission_score', 'Raw admission score', 'score'),
        ('fpir', 'Calibrated factor access / FPIR', 'percent'))]
    _table(fig, rows, ['Diagnostic', 'Zero admitted alpha', 'As specified'], .632, .136,
           [.50, .25, .25], 8.1)
    note, color = _rank_note(result.summary)
    _write(fig, _LEFT, .613, note, 8, color)
    _section(fig, .570, 'Calibrated and GLS-implied factor premia',
             'Annual excess premia in basis points. Alpha effect = specified minus zero view.')
    rows = [[_text(key, 31, 2), _value(row, 'calibrated_premium', 'bp'),
             _value(zero.factors.loc[key], 'implied_premium', 'bp'),
             _value(row, 'implied_premium', 'bp'),
             _number(row.implied_premium-zero.factors.loc[key, 'implied_premium'], 'bp')]
            for key, row in result.factors.iloc[:_FACTOR_ROWS].iterrows()]
    columns = ['Factor', 'Calibrated', 'GLS\nzero alpha', 'GLS\nas specified', 'Alpha effect\nbp']
    widths = [.36, .16, .16, .16, .16]
    stability = 'max_abs_25bp_shift' in result.factors
    if stability:
        rows = [row + _stability_cells(result.factors.iloc[i]) for i, row in enumerate(rows)]
        columns += ['Max |25bp|\nshift bp', 'LOO specified\nlow / high']
        widths = [.24, .12, .12, .12, .12, .12, .16]
    _table(fig, rows, columns, .215, .304, widths, 7.4 if stability else 7.9)
    omitted = max(0, len(result.factors)-_FACTOR_ROWS)
    s = result.summary
    _write(fig, _LEFT, .198,
           f'{omitted} additional factors in factors.csv. Whitened condition number '
           + _number(s.get('condition_number'), 'scientific') + '\n'
           'Admission budget ' + _number(s.get('admission_budget'))
           + ' | Specified usage ' + _number(s.get('admission_budget_ratio'), 'percent')
           + ' | Projected admission ' + _number(s.get('admission_residual_score')),
           8, _MUTED)
    for y, label, view in [(.144, 'Zero alpha', zero), (.120, 'As specified', result)]:
        failures = int(view.validation.status.eq('fail').sum())
        _write(fig, _LEFT, y,
               f'{label}: {failures} failed numerical checks / {len(view.validation)} recorded.',
               8, _RED if failures else _TEAL)
    note = ('GLS-implied premia are diagnostic coordinates, not a change to calibrated CMAs.\n'
            'All scores are unconstrained model-implied excess opportunities.\n')
    note += ('25bp shocks and leave-one-out (LOO) ranges hold risk/alpha fixed; '
             'rank-loss cases are excluded.\n'
             'Complete gap x multiplier and deletion tables are saved for both views.'
             if stability else
             'Complete assets, channels, scenarios and policy checks are saved for both views.')
    _write(fig, _LEFT, .087 if stability else .077, note, 7.5, _MUTED)
    return fig


def _factor_contributor_rows(result: CmaDiagnosticResult) -> list[list[str]]:
    """Format all factors with two positive and two negative asset revision contributions."""
    rows = []
    stability = 'max_abs_25bp_shift' in result.factors
    for factor, row in result.factors.iterrows():
        values = result.factor_contributions[factor].dropna()
        top = values.loc[values > 0].sort_values(ascending=False, kind='stable').head(2)
        bottom = values.loc[values < 0].sort_values(kind='stable').head(2)
        cells = []
        for selected in (top, bottom):
            width = 15 if stability else 22
            cells.extend([_text(_name(result.assets, key, width), width, 1) + '\n'
                          + _number(value, 'bp') for key, value in selected.items()])
            cells.extend(['-']*(2-len(selected)))
        rows.append([_text(factor, 18, 2), _value(row, 'revision', 'bp'), *cells,
                     *(_stability_cells(row) if stability else [])])
    return rows


def _compact_alpha_contributors_page(result: CmaDiagnosticResult, title: str,
                                     label: str) -> Figure:
    """Keep asset residual tails and every MATF factor attribution on one bounded page."""
    fig = _page(result, title, 'Alpha views / ' + label)
    _write(fig, _LEFT, .817, label, 13, weight='bold')
    s = result.summary
    _cards(fig, [('Total squared Sharpe', _number(s.get('total_sharpe2'))),
                 ('Residual S2', _number(s.get('residual_sharpe2'))),
                 ('Raw admission score', _number(s.get('admission_score')))], y=.716)
    _section(fig, .693, 'Top 5 / Bottom 5: signed GLS residuals',
             '+ ranks the largest positive gaps; - ranks the most negative. '
             'Residual = excess CMA minus GLS fit.')
    rows = []
    for sign, selected in zip(('+', '-'), _residual_tails(result.assets)):
        for rank, (key, row) in enumerate(selected.iterrows(), start=1):
            rows.append([f'{sign}{rank}', _text(_name(result.assets, key, 26), 26, 1),
                         _value(row, 'admitted_alpha', 'bp'), _value(row, 'excess_cma', 'return'),
                         _value(row, 'fitted_excess_cma', 'return'), _value(row, 'residual', 'bp'),
                         _value(row, 'residual_vol', 'risk'), _value(row, 'residual_score'),
                         _value(row, 'residual_share', 'percent')])
    _table(fig, rows, ['Tail\nrank', 'Asset', 'Alpha\nbp', 'Excess\nCMA', 'GLS\nfitted',
                       'Residual\nbp', 'Residual\nvol', 'S2\ncontrib.', 'S2\nshare'],
           .444, .198, [.07, .25, .095, .10, .10, .10, .105, .09, .09], 7.1)
    _section(fig, .431, 'MATF factor revision contributors',
             'Asset / signed annual bp. Revision = GLS minus calibrated premium. '
             'Top and bottom contributions need not sum to the revision.')
    columns = ['Factor', 'Revision\nbp', 'Top 1\nasset / bp', 'Top 2\nasset / bp',
               'Bottom 1\nasset / bp', 'Bottom 2\nasset / bp']
    widths = [.135, .075, .1975, .1975, .1975, .1975]
    stability = 'max_abs_25bp_shift' in result.factors
    if stability:
        columns += ['Max |25bp|\nshift bp', 'LOO premia\nlow / high']
        widths = [.14, .07, .145, .145, .145, .145, .085, .125]
    _table(fig, _factor_contributor_rows(result), columns,
           .083 if stability else .075, .292 if stability else .300, widths,
           6.6 if stability else 7.1)
    nonunique = s.get('rank', 0) < s.get('n_factors', len(result.factors))
    note = ('Nonunique factor coordinates; asset sums plus the saved coordinate offset equal '
            'the revision.' if nonunique else
            'All asset contributions sum to each revision; complete rows are in '
            'factor_contributions.csv.')
    if stability:
        lost = int(result.factors.loo_unavailable_count.max())
        note += (f'\nLOO excludes {lost} nonidentified deletions. '
                 'Gap x GLS multiplier and 25bp shocks: factor_sensitivity.csv.')
    note = '\n'.join(_text(line, 124, 1) for line in note.splitlines())
    _write(fig, _LEFT, .069 if stability else .059, note, 7.1,
           _AMBER if nonunique else _MUTED)
    return fig


def _alpha_contributors_page(result: CmaDiagnosticResult, title: str, label: str) -> Figure:
    """Show disjoint signed residual tails and their squared-Sharpe contributions."""
    if not getattr(result, 'factor_contributions', pd.DataFrame()).empty:
        return _compact_alpha_contributors_page(result, title, label)
    fig = _page(result, title, 'Alpha views / ' + label)
    _write(fig, _LEFT, .817, label, 13, weight='bold')
    s = result.summary
    _cards(fig, [('Total squared Sharpe', _number(s.get('total_sharpe2'))),
                 ('Residual S2', _number(s.get('residual_sharpe2'))),
                 ('Raw admission score', _number(s.get('admission_score')))], y=.716)
    _write(fig, _LEFT, .703,
           'GLS residual = excess CMA minus fitted CMA. Ranking uses the signed gap in bp.',
           8, _MUTED)
    top, bottom = _residual_tails(result.assets)
    for y, name, selected, description in [
            (.675, 'Top 5', top, 'Largest positive GLS residuals; forecast above factor fit.'),
            (.409, 'Bottom 5', bottom, 'Most negative GLS residuals; forecast below factor fit.')]:
        _section(fig, y, name, description)
        rows = [[_name(result.assets, key, 29), _value(row, 'admitted_alpha', 'bp'),
                 _value(row, 'excess_cma', 'return'), _value(row, 'fitted_excess_cma', 'return'),
                 _value(row, 'residual', 'bp'), _value(row, 'residual_vol', 'risk'),
                 _value(row, 'residual_score'), _value(row, 'residual_share', 'percent')]
                for key, row in selected.iterrows()]
        _table(fig, rows, ['Asset', 'Alpha\nbp', 'Excess\nCMA', 'GLS\nfitted', 'Residual\nbp',
                           'Residual\nvol', 'S2\ncontrib.', 'S2\nshare'], y-.253, .201,
               [.32, .095, .10, .10, .10, .105, .09, .09], 7.5)
    shown = len(top)+len(bottom)
    _write(fig, _LEFT, .139,
           f'Showing {shown} of {len(result.assets)} assets; '
           'fewer rows when a sign has fewer than five.\n'
           'S2 contribution = squared residual / residual variance; '
           'both signs contribute positively.',
           7.4, _MUTED)
    checks = result.policy_checks
    breaches = int(checks.status.eq('breach').sum()) if not checks.empty else 0
    policy = s.get('policy_name') or 'No admission policy supplied'
    _write(fig, _LEFT, .101,
           _text(f'{policy} | {breaches} breached checks. Caps govern admitted alpha only.', 119, 1)
           + '\n' + f'{len(result.scenarios.columns)} supplied scenarios retained in this view; '
           'scenario shocks are held fixed.\n'
           'Small residual risks can magnify forecast adjustments. Full details are in the bundle.',
           7.3, _MUTED)
    return fig


def _build_pages(result: CmaDiagnosticResult, pages: int,
                 title: str | None = None, *, zero_alpha_result=None) -> list[Figure]:
    """Build bounded report pages while leaving every supplied table and scalar unchanged."""
    if pages not in (2, 3):
        raise ValueError('CMA diagnostic reports support two or three pages.')
    title = title or result.provenance.get('title') or 'CMA diagnostic report'
    figures: list[Figure] = []
    previous_figures = set(plt.get_fignums())
    try:
        if zero_alpha_result is not None:
            if not result.assets.index.equals(zero_alpha_result.assets.index):
                raise ValueError('Alpha views must use the same asset universe and order')
            if not result.factors.index.equals(zero_alpha_result.factors.index):
                raise ValueError('Alpha views must use the same factor universe and order')
            if pages == 3:
                figures.append(_alpha_comparison_page(result, zero_alpha_result, title))
            figures.append(_alpha_contributors_page(
                zero_alpha_result, title, 'Zero admitted alpha'))
            figures.append(_alpha_contributors_page(result, title, 'Admitted alpha as specified'))
        else:
            figures.append(_forecast_page(result, title))
            figures.append(_admission_page(result, title))
            if pages == 3 and not result.scenarios.empty:
                figures.append(_scenario_page(result, title))
        for i, fig in enumerate(figures, start=1):
            _write(fig, 0.945, 0.035, f'{i} / {len(figures)}', 7.5, _MUTED, ha='right')
        return figures
    except Exception:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
        raise


def render_cma_diagnostics(result: CmaDiagnosticResult, output_path: str | Path,
                           *, pages: int = 3, title: str | None = None,
                           zero_alpha_result: CmaDiagnosticResult | None = None) -> Path:
    """Render a completed CMA audit to a readable, atomically replaced two/three-page PDF.

    Args:
        result: Completed numerical result; all calculation and availability decisions are
            supplied by its owner. Tables are never modified by rendering.
        output_path: Destination ending in ``.pdf``. The parent directory is created if needed.
        pages: Two pages, or up to three when supplied scenario data are available.
        title: Optional display title; otherwise use result provenance and then the default.
        zero_alpha_result: Optional precomputed counterfactual. Produces a contributor page
            for each alpha view and, with three pages, a factor-premium comparison page.

    Returns:
        Absolute path of the completed PDF. Failure leaves any previous output intact.
    """
    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != '.pdf':
        raise ValueError('output_path must have the .pdf extension.')
    if pages not in (2, 3):
        raise ValueError('CMA diagnostic reports support two or three pages.')
    target.parent.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []
    temporary: Path | None = None
    try:
        with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                             'ps.fonttype': 42, 'text.usetex': False}):
            figures = _build_pages(result, pages, title, zero_alpha_result=zero_alpha_result)
            with NamedTemporaryFile(prefix='.cma-report-', suffix='.pdf',
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
