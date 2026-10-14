"""Acceptance checks for complete, bounded CMA diagnostic PDF rendering."""

from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from optimalportfolios.reports.cma_diagnostics import _build_pages, render_cma_diagnostics


def _result(n_assets: int = 18, scenarios: int = 0,
            state: str = 'full_rank', long_names: bool = False) -> SimpleNamespace:
    """Supply completed synthetic tables without invoking any estimator or diagnostic kernel."""
    ids = pd.Index([f'asset-{i}' for i in range(n_assets)], name='asset')
    names = [f'Asset {i}' for i in range(n_assets)]
    factor_names = [f'Factor {i}' for i in range(12)]
    if long_names:
        names = [f'Asset {i}: exceptionally long institutional market name ' * 3
                 for i in range(n_assets)]
        factor_names = [f'Factor {i}: long tradable portfolio strategy title ' * 3
                        for i in range(12)]
    assets = pd.DataFrame({
        'name': names, 'excess_cma': np.linspace(0.025, 0.11, n_assets),
        'fitted_excess_cma': np.linspace(0.02, 0.075, n_assets),
        'residual': np.linspace(-0.025, 0.032, n_assets), 'residual_vol': 0.073,
        'standardized_residual': 0.32, 'residual_score': np.arange(n_assets) / 1000,
        'residual_share': np.arange(n_assets) / max(1, np.arange(n_assets).sum()),
        'admitted_alpha': np.linspace(-0.01, 0.024, n_assets), 'admission_ir': 0.4,
        'admission_share': 0.3, 'projected_admission': 0.005, 'r_squared': 0.6,
        'observations': 120,
    }, index=ids)
    factors = pd.DataFrame({'calibrated_premium': 0.035, 'implied_premium': 0.041,
                            'revision': 0.006, 'identification_scale': 1.0,
                            'status': state}, index=factor_names)
    channels = pd.DataFrame({'raw_score': [0.14, 1.0], 'projected_score': [0.10, 0.75],
                             'residual_attribution': [-0.02, 0.78]},
                            index=['Regional adjustment', 'Admitted historical alpha'])
    policy = pd.DataFrame([
        {'asset': key, 'check': 'cap1_positive_ir', 'value': 0.7,
         'limit': 0.5, 'status': 'breach'} for key in ids
    ] + [
        {'asset': key, 'check': 'cap2_admission_share', 'value': 0.3,
         'limit': 0.5, 'status': 'pass'} for key in ids
    ] + [{'asset': '__universe__', 'check': 'cap3_admission_budget',
          'value': 1.0, 'limit': 0.2, 'status': 'breach'}])
    scenario_frame = pd.DataFrame({f'Scenario {i}': assets['excess_cma'] - i * 0.02
                                   for i in range(scenarios)}, index=ids)
    validation = pd.DataFrame({'check': ['Return reconstruction', 'GLS orthogonality',
                                         'Sharpe identity', 'Covariance reconstruction'],
                                'error': [1e-15] * 4, 'tolerance': [1e-9] * 4,
                                'status': ['pass'] * 4})
    summary = dict(n_assets=n_assets, n_factors=12, rank=min(12, n_assets),
                   residual_dimension=max(n_assets - 12, 0), identification_status=state,
                   condition_number=10.0, rank_cutoff=1e-12, relative_cutoff=1e-14,
                   total_sharpe2=1.19, implied_systematic_sharpe2=0.48, residual_sharpe2=0.71,
                   calibrated_systematic_sharpe2=0.4, factor_ceiling2=0.97, fpir=0.42,
                   admission_score=1.08, admission_residual_score=0.76,
                   admission_in_span_share=0.30, admission_budget=0.20,
                   admission_budget_ratio=5.4, policy_name='Paper illustrative (kappa = 0.5)',
                   top1_residual_share=0.3, top5_residual_share=0.8)
    provenance = dict(title='Q3 2026 CMA diagnostics', universe='Synthetic acceptance universe',
                      as_of='2026-09-30', reference_ccy='USD', cash_rate=0.0418,
                      forecast_basis='Optimiser-facing shifted excess', model_name='MATF test',
                      scenario_metadata={'Scenario 0': {'horizon': '5 years',
                                                        'overlay': 'linear annualization'}})
    return SimpleNamespace(summary=summary, assets=assets, factors=factors,
                           channels=channels, policy_checks=policy, scenarios=scenario_frame,
                           validation=validation, provenance=provenance)


def _all_text(figures: list) -> str:
    """Collect rendered labels and cells for content acceptance assertions."""
    strings = []
    for fig in figures:
        strings.extend(text.get_text() for text in fig.texts)
        for ax in fig.axes:
            for table in ax.tables:
                strings.extend(cell.get_text().get_text() for cell in table.get_celld().values())
    return '\n'.join(strings)


@pytest.fixture(autouse=True)
def close_report_figures():
    """Isolate test plots; individual checks create caller figures to preserve."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.mark.parametrize(('requested', 'scenarios', 'expected'), [(3, 0, 2), (3, 4, 3), (2, 4, 2)])
def test_pdf_page_count_and_immutable_inputs(tmp_path: Path, requested: int,
                                           scenarios: int, expected: int) -> None:
    """Page limits are enforced, and rendering leaves every diagnostic input intact."""
    result = _result(scenarios=scenarios)
    before = deepcopy(result)
    plt.figure()
    figures_before = set(plt.get_fignums())
    output = render_cma_diagnostics(result, tmp_path / 'report.pdf', pages=requested)
    content = output.read_bytes()
    assert content.startswith(b'%PDF-')
    assert len(re.findall(rb'/Type\s*/Page\b', content)) == expected
    assert output == (tmp_path / 'report.pdf').resolve()
    for name in ('assets', 'factors', 'channels', 'policy_checks', 'scenarios', 'validation'):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(before, name))
    assert result.summary == before.summary
    assert result.provenance == before.provenance
    assert set(plt.get_fignums()) == figures_before
    assert not list(tmp_path.glob('.cma-report-*'))


def test_large_universe_shows_omitted_counts_and_policy_breaches() -> None:
    """Large universes retain a readable bounded table and disclose hidden detail."""
    result = _result(181, scenarios=4)
    figures = _build_pages(result, 3)
    text = _all_text(figures)
    assert 'Remaining 173 assets' in text
    assert '173 additional nonzero admissions' in text
    assert '182 breached checks' in text
    assert 'showing 3 of 4 scenarios' in text
    assert 'no diagnostic power' not in text
    assert 'Baseline admission is shown for context' in text
    for fig in figures:
        for ax in fig.axes:
            for table in ax.tables:
                assert max(row for row, _ in table.get_celld()) <= 12


@pytest.mark.parametrize('state', ['rank_deficient', 'saturated'])
def test_nonidentification_and_unavailability_are_visible(state: str) -> None:
    """A zero residual or absent admission data cannot look like an economic approval."""
    result = _result(1, state=state)
    result.assets = result.assets.drop(columns=['admitted_alpha', 'r_squared'])
    result.policy_checks = result.policy_checks.iloc[:0]
    result.summary.update(policy_name=None, admission_score=None, admission_budget=None,
                          admission_budget_ratio=None, admission_residual_score=None,
                          admission_in_span_share=None, fpir=None)
    result.provenance['excluded_assets'] = ['Cash']
    result.provenance['factor_model'] = 'matf_custom_ig_hy'
    figures = _build_pages(result, 3)
    text = _all_text(figures)
    assert len(figures) == 2
    assert 'Admission construction data are not available.' in text
    assert 'N/A' in text
    assert 'No admission policy supplied' in text
    assert 'Reference cash 4.18%' in text
    assert 'matf_custom_ig_hy' in text
    assert 'Excluded 1' in text
    if state == 'saturated':
        assert 'zero residual has no diagnostic power' in text
    else:
        assert 'Factor premia are nonunique' in text


def test_long_labels_fit_inside_table_cells() -> None:
    """Long institutional labels remain within both horizontal and vertical cell bounds."""
    figures = _build_pages(_result(181, scenarios=3, long_names=True), 3)
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for table in ax.tables:
                for cell in table.get_celld().values():
                    text = cell.get_text().get_window_extent(renderer)
                    bounds = cell.get_window_extent(renderer)
                    assert text.width <= bounds.width + 1.0, cell.get_text().get_text()
                    assert text.height <= bounds.height + 1.0, cell.get_text().get_text()


def test_small_universe_does_not_stretch_one_row_into_a_full_panel() -> None:
    """A one-asset report retains normal row heights instead of oversized header blocks."""
    figures = _build_pages(_result(1, state='saturated'), 2)
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for table in ax.tables:
                for cell in table.get_celld().values():
                    assert cell.get_window_extent(renderer).height < 65.0


def test_small_residual_risk_and_nonadmission_adjustments_are_visible() -> None:
    """Admission-policy passes cannot hide a discretionary return over tiny residual risk."""
    result = _result()
    result.assets.loc['asset-17', 'residual_vol'] = 0.000773
    result.channels.loc['discretionary_add_on'] = [676.89, 675.0, 674.0]
    result.summary['residual_sharpe2'] = 676.89
    text = _all_text(_build_pages(result, 2))
    assert '0.077%' in text
    assert '676.890' in text
    assert 'Discretionary add-on' in text
    assert 'Caps govern historical admission only' in text
    assert 'Small residual risks can magnify forecast adjustments.' in text
    assert 'share cap 50.0%' in text


@pytest.mark.parametrize('paired', [False, True])
def test_stability_explanation_clears_page_footer(paired):
    """Additional stability definitions must not collide with the page identity."""
    result = _result()
    result.factors['max_abs_25bp_shift'] = .0272
    result.factors['loo_low'] = -.0048
    result.factors['loo_high'] = .0645
    result.factors['loo_unavailable_count'] = 1
    kwargs = {'zero_alpha_result': deepcopy(result)} if paired else {}
    fig = _build_pages(result, 3 if paired else 2, **kwargs)[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    note = next(t for t in fig.texts if ('25bp shocks' if paired else '25bp =') in t.get_text())
    footer = next(t for t in fig.texts if t.get_position()[1] == .035)
    assert note.get_window_extent(renderer).y0 > footer.get_window_extent(renderer).y1 + 3


def test_failed_write_preserves_existing_pdf_and_closes_figures(tmp_path: Path,
                                                              monkeypatch) -> None:
    """A renderer failure cannot overwrite a previously completed production report."""
    import optimalportfolios.reports.cma_diagnostics as report

    target = tmp_path / 'report.pdf'
    target.write_bytes(b'previous completed report')
    plt.figure()
    figures_before = set(plt.get_fignums())

    def fail_save(*args, **kwargs):
        """Emulate a partial temporary write followed by a rendering failure."""
        candidate = Path(kwargs['local_path']) / f'{args[1]}.pdf'
        candidate.write_bytes(b'partial')
        raise OSError('Synthetic writer failure')

    monkeypatch.setattr(report.qis, 'save_figs_to_pdf', fail_save)
    with pytest.raises(OSError, match='Synthetic writer failure'):
        render_cma_diagnostics(_result(), target)
    assert target.read_bytes() == b'previous completed report'
    assert not list(tmp_path.glob('.cma-report-*'))
    assert set(plt.get_fignums()) == figures_before


@pytest.mark.parametrize(('name', 'pages'), [('report.txt', 3), ('report.pdf', 1)])
def test_invalid_output_contract_fails_before_writing(tmp_path: Path, name: str,
                                                    pages: int) -> None:
    """Reject unsupported outputs before creating an incomplete artifact."""
    with pytest.raises(ValueError):
        render_cma_diagnostics(_result(), tmp_path / name, pages=pages)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('pages', [2, 3])
def test_alpha_views_show_both_rankings_and_keep_old_results(tmp_path, pages):
    """Each view displays signed residual tails with their nonnegative S-squared impact."""
    result = _result(18, scenarios=2)
    zero = deepcopy(result)
    zero.assets['admitted_alpha'] = 0.
    zero.assets['residual'] = -result.assets.residual
    original = deepcopy(result)
    figures = _build_pages(result, pages, zero_alpha_result=zero)
    assert len(figures) == pages
    text = _all_text(figures)
    assert 'Zero admitted alpha' in text
    assert 'Admitted alpha as specified' in text
    assert text.count('Top 5') == 2 and text.count('Bottom 5') == 2
    assert 'Residual S2' in text and 'GLS residual' in text
    assert 'Asset 0' in text and 'Asset 17' in text
    pd.testing.assert_frame_equal(result.assets, original.assets)
    path = render_cma_diagnostics(result, tmp_path/'paired.pdf', pages=pages,
                                 zero_alpha_result=zero)
    assert len(re.findall(rb'/Type\s*/Page\b', path.read_bytes())) == pages


def test_residual_tails_are_disjoint_stable_and_sign_aware():
    """Sparse tails do not pad with opposite signs or duplicate a small universe."""
    from optimalportfolios.reports.cma_diagnostics import _residual_tails

    result = _result(4)
    result.assets['residual'] = [0., .03, -.01, .03]
    top, bottom = _residual_tails(result.assets)
    assert top.index.tolist() == ['asset-1', 'asset-3']
    assert bottom.index.tolist() == ['asset-2']


def test_alpha_views_long_names_and_universe_mismatch():
    """Comparison tables bound long labels and reject incompatible universe axes."""
    result = _result(181, long_names=True)
    zero = deepcopy(result)
    figures = _build_pages(result, 3, zero_alpha_result=zero)
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for table in ax.tables:
                for cell in table.get_celld().values():
                    text = cell.get_text().get_window_extent(renderer)
                    bounds = cell.get_window_extent(renderer)
                    assert text.width <= bounds.width + 1.0, cell.get_text().get_text()
                    assert text.height <= bounds.height + 1.0, cell.get_text().get_text()
    zero.assets = zero.assets.iloc[::-1]
    with pytest.raises(ValueError, match='same asset'):
        _build_pages(result, 3, zero_alpha_result=zero)


@pytest.mark.parametrize('stability', [False, True])
def test_compact_factor_contributors_cover_every_factor_and_signed_tails(stability):
    """Both views show all twelve factor rows, with stable positive/negative contributor ranks."""
    from optimalportfolios.reports.cma_diagnostics import _factor_contributor_rows

    result = _result(18, long_names=True)
    result.factor_contributions = pd.DataFrame(
        np.tile(np.linspace(-.02, .03, 18)[:, None], (1, 12)),
        index=result.assets.index, columns=result.factors.index)
    if stability:
        result.factors['max_abs_25bp_shift'] = .0272
        result.factors['loo_low'] = -.0048
        result.factors['loo_high'] = .0645
        result.factors['loo_unavailable_count'] = 1
    rows = _factor_contributor_rows(result)
    assert len(rows) == 12
    assert 'Asset 17' in rows[0][2] and '+300' in rows[0][2]
    assert 'Asset 16' in rows[0][3]
    assert 'Asset 0' in rows[0][4] and '-200' in rows[0][4]
    assert 'Asset 1' in rows[0][5]
    figures = _build_pages(result, 3, zero_alpha_result=deepcopy(result))
    assert len(figures) == 3
    text = _all_text(figures)
    assert text.count('MATF factor revision contributors') == 2
    assert text.count('Top 5') == 2 and text.count('Bottom 5') == 2
    if stability:
        assert 'Max |25bp|' in text and 'LOO premia' in text
        assert '-0.48%' in text and '6.45%' in text
        assert '1 nonidentified deletions' in text
    else:
        assert 'Max |25bp|' not in text and 'factor_sensitivity.csv' not in text
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for table in ax.tables:
                for cell in table.get_celld().values():
                    bounds = cell.get_window_extent(renderer)
                    extent = cell.get_text().get_window_extent(renderer)
                    assert extent.width <= bounds.width+1, cell.get_text().get_text()
                    assert extent.height <= bounds.height+1, cell.get_text().get_text()
