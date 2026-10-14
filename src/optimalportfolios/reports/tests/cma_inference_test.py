"""Provider comparisons preserve identities, unavailable premia and frozen results."""

from copy import deepcopy
from dataclasses import replace
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticInputs, compute_cma_diagnostics
from optimalportfolios.matf.cma_uncertainty import compute_conditional_gls_intervals
from optimalportfolios.matf.cma_identification import compute_cma_identification
from optimalportfolios.reports import cma_inference as report_module
from optimalportfolios.reports.cma_inference import (
    CmaInferenceReport, CmaInferenceView, _build_inference_pages, _premium, render_cma_inference,
)


def _report(with_intervals=False, with_identification=False):
    """Provide a reconciled model and a smaller universe with an absent factor."""
    betas = pd.DataFrame([[1., 0.], [0., 1.], [.5, .5], [1., 1.5]],
                         index=['Stock', 'Private fund', 'Mixed fund', 'Other fund'],
                         columns=['Equity', 'Private Equity'])
    inputs = CmaDiagnosticInputs(
        betas, pd.Series(.0004, index=betas.index),
        pd.Series([.04, .015, .025, .06], index=betas.index),
        factor_covariance=pd.DataFrame(np.eye(2)*.04, index=betas.columns, columns=betas.columns),
        factor_premia=pd.Series([.03, .025], index=betas.columns))
    full = compute_cma_diagnostics(inputs)
    reduced_inputs = replace(
        inputs, betas=betas[['Equity']], factor_covariance=inputs.factor_covariance.loc[
            ['Equity'], ['Equity']], factor_premia=inputs.factor_premia[['Equity']])
    reduced = compute_cma_diagnostics(reduced_inputs)
    views = (CmaInferenceView('full', 'Full\nw = 1', full, '10-15y', .031, 1.),
             CmaInferenceView('reduced', 'No proxies', reduced, '10-15y', .031, 1.))
    if with_intervals:
        views = tuple(replace(view, conditional_intervals=compute_conditional_gls_intervals(data))
                      for view, data in zip(views, (inputs, reduced_inputs)))
    if with_identification:
        # A custom pair exercises actual contrasts while the reduced view lacks one member.
        views = tuple(replace(view, identification=compute_cma_identification(
            data, factor_pair=('Equity', 'Private Equity'), precision_target=.01,
            equivalence_margin=.1)) for view, data in zip(views, (inputs, reduced_inputs)))
    return CmaInferenceReport('Synthetic provider', '2026-09-30', 'USD', '2025-09-30',
                              'full', views, 'Synthetic input', 'Same frozen estimator',
                              'Different forecast and model dates')


def test_missing_and_nonidentified_coordinates_never_become_zero():
    """A reduced provider universe cannot make missing Private Equity look investable."""
    report = _report()
    assert _premium(report.views[1], 'Private Equity') == 'N/A'
    result = deepcopy(report.views[0].result)
    result.factors.at['Private Equity', 'status'] = 'not_identified'
    assert _premium(replace(report.views[0], result=result), 'Private Equity') == 'N/A'
    # Presentation indexes by labels, not by the column position in another view.
    shuffled = replace(result, factors=result.factors.iloc[::-1])
    assert _premium(replace(report.views[0], result=shuffled), 'Equity') == _premium(
        report.views[0], 'Equity')


@pytest.mark.parametrize('with_bands', [False, True])
@pytest.mark.parametrize('carry', ['FX Carry', 'Carry FX futures', 'Carry FX'])
def test_fx_carry_report_does_not_present_absent_pair_tests(with_bands, carry):
    """A replacement model shows its own CIs without implying a 50/50 blend or pair test."""
    b = pd.DataFrame([[1., .2], [.1, 1.], [.5, .5], [.8, .7]],
                     index=list('ABCD'), columns=['Equity', carry])
    data = CmaDiagnosticInputs(b, pd.Series(.01, index=b.index),
        pd.Series([.04, .02, .03, .038], index=b.index),
        factor_covariance=pd.DataFrame(np.eye(2)*.01, index=b.columns, columns=b.columns),
        factor_premia=pd.Series([.03, .01], index=b.columns))
    view = CmaInferenceView('full', 'w = 1', compute_cma_diagnostics(data), '10y', .03, 1.,
                           compute_conditional_gls_intervals(data),
                           compute_cma_identification(data, precision_target=.01))
    if with_bands:
        bounds = [dict(factor=carry, box=box, cash_treatment='fixed_cash',
                       lower=-.12, upper=.21)
                  for box in ('advisor_p25_p75_box', 'advisor_min_max_box')]
        view = replace(view, result=replace(view.result,
                       provenance={'adviser_box_bounds': bounds}))
    report = replace(_report(), views=(view,))
    figures = _build_inference_pages(report)
    try:
        text = '\n'.join(t.get_text() for fig in figures for t in fig.texts)
        assert len(figures) == 3
        assert carry + ': forecast gaps' in text
        if carry == 'Carry FX':
            assert 'daily 50/50 portfolio' in text and 'no second volatility target' in text
            assert 'not a 50/50 G10/EM blend' not in text
        else:
            assert 'not a 50/50 G10/EM blend' in text
        assert 'TOST' not in text and 'No factor merge' not in text
        if with_bands:
            cells = '\n'.join(cell.get_text().get_text() for fig in figures for ax in fig.axes
                              for table in ax.tables for cell in table.get_celld().values())
            assert '[-12.00,\n21.00]' in cells and 'Adviser IQR\nbox %' in cells
            assert 'not CIs or factor quantiles' in text
    finally:
        for fig in figures:
            plt.close(fig)


def test_pdf_retains_source_dates_and_inputs(tmp_path):
    """The common layout preserves vintage distinctions and all caller-owned tables."""
    report = _report()
    before = deepcopy(report)
    figures = _build_inference_pages(report)
    text = '\n'.join(t.get_text() for fig in figures for t in fig.texts)
    assert '2025-09-30' in text and '2026-09-30' in text and '10-15y' in text
    assert len(figures) == 3
    for fig in figures:
        plt.close(fig)
    target = render_cma_inference(report, tmp_path/'report.pdf')
    assert len(re.findall(rb'/Type\s*/Page\b', target.read_bytes())) == 3
    for actual, original in zip(report.views, before.views):
        for field in actual.result.__dataclass_fields__:
            left, right = getattr(actual.result, field), getattr(original.result, field)
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right, check_exact=True)
            else:
                assert left == right
    assert not plt.get_fignums()


@pytest.mark.parametrize('defect', ['covariance', 'missing_factor', 'zero_scale'])
def test_single_carry_unavailable_inputs_are_reported_as_na(defect):
    """Reduced and unsupported carry views must render N/A instead of crashing or printing nan."""
    report = _report(with_intervals=True, with_identification=True)
    source = report.views[0]
    betas = pd.DataFrame([[1., .2], [.1, 1.], [.5, .5], [.8, .7]],
                         index=list('ABCD'), columns=['Equity', 'Carry FX'])
    inputs = CmaDiagnosticInputs(
        betas, pd.Series(.01, index=betas.index),
        pd.Series([.04, .02, .03, .038], index=betas.index),
        factor_covariance=pd.DataFrame(np.eye(2)*.01, index=betas.columns,
                                       columns=betas.columns),
        factor_premia=pd.Series([.03, .01], index=betas.columns))
    if defect == 'covariance':
        inputs = replace(inputs, factor_covariance=None)
    elif defect == 'zero_scale':
        inputs = replace(inputs, excess_cma=betas @ np.array([.03, .01]))
    view = replace(source, result=compute_cma_diagnostics(inputs),
                   conditional_intervals=compute_conditional_gls_intervals(inputs),
                   identification=compute_cma_identification(inputs))
    views = (view,)
    if defect == 'missing_factor':
        reduced = replace(inputs, betas=betas[['Equity']],
                          factor_covariance=inputs.factor_covariance.loc[['Equity'], ['Equity']],
                          factor_premia=inputs.factor_premia[['Equity']])
        views += (replace(view, key='reduced', result=compute_cma_diagnostics(reduced),
                          conditional_intervals=compute_conditional_gls_intervals(reduced),
                          identification=compute_cma_identification(reduced)),)
    figures = _build_inference_pages(replace(report, views=views))
    try:
        cells = '\n'.join(cell.get_text().get_text() for fig in figures for ax in fig.axes
                          for table in ax.tables for cell in table.get_celld().values())
        assert 'N/A' in cells
        assert 'nan' not in cells.lower()
    finally:
        for fig in figures:
            plt.close(fig)


def test_identification_contrasts_preserve_sensitivity_and_three_pages():
    """Every scenario has correctly scaled contrast CIs and unavailable cases stay explicit."""
    report = _report(with_intervals=True, with_identification=True)
    figures = _build_inference_pages(report)
    try:
        text = '\n'.join(t.get_text() for fig in figures for t in fig.texts)
        cells = '\n'.join(cell.get_text().get_text() for fig in figures for ax in fig.axes
                          for table in ax.tables for cell in table.get_celld().values())
        assert len(figures) == 3
        assert 'No factor merge' in text and 'TOST at 5%' in text
        assert 'Common = (Equity + Private Equity)/2' in text
        assert 'LOO low' in cells and 'Pair\nVIF' in cells
        row = report.views[0].identification.contrasts.loc['common']
        assert f'[{100*row.lower:.2f}, {100*row.upper:.2f}]' in cells
        assert report_module._contrast_interval(report.views[1], 'common') == 'N/A'
        assert 'Full assumptions on page 3' in text
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize('defect', ['partial', 'estimate', 'threshold'])
def test_identification_must_match_view_and_thresholds(defect):
    """A previous scenario or different diagnostic settings cannot be silently attached."""
    report = _report(with_intervals=True, with_identification=True)
    views = list(report.views)
    ident = deepcopy(views[1].identification)
    if defect == 'partial':
        ident = None
    elif defect == 'estimate':
        ident.factors['estimate'] += .01
    else:
        ident.summary['precision_target'] = .02
    views[1] = replace(views[1], identification=ident)
    with pytest.raises(ValueError, match='[Ii]dentification'):
        _build_inference_pages(replace(report, views=tuple(views)))


def test_conditional_intervals_are_labelled_and_keep_sensitivity_tables():
    """Every case has intervals while assumptions, factor omissions and LOO remain explicit."""
    report = _report(with_intervals=True)
    figures = _build_inference_pages(report)
    try:
        text = '\n'.join(t.get_text() for fig in figures for t in fig.texts)
        cells = '\n'.join(cell.get_text().get_text() for fig in figures for ax in fig.axes
                          for table in ax.tables for cell in table.get_celld().values())
        assert 'conditional 95% CI' in text
        assert 'fixed' in text and 'independent' in text and 'No simultaneous coverage' in text
        assert 'tau^2 = residual S2 / (N-K)' in text
        assert 'CI df' in cells and 'LOO low' in cells and 'LOO high' in cells
        ci = report.views[0].conditional_intervals.factors.loc['Equity']
        assert f'[{100*ci.lower:.2f}, {100*ci.upper:.2f}]' in cells
        assert report_module._premium_interval(report.views[1], 'Private Equity') == 'N/A'
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize('defect', ['partial', 'estimate', 'confidence'])
def test_mismatched_conditional_intervals_are_rejected(tmp_path, defect):
    """A report must not attach a different scenario's uncertainty to its point estimates."""
    report = _report(with_intervals=True)
    views = list(report.views)
    ci = deepcopy(views[1].conditional_intervals)
    if defect == 'partial':
        ci = None
    elif defect == 'estimate':
        ci.factors['estimate'] += .01
    else:
        ci.summary['confidence_level'] = .9
    views[1] = replace(views[1], conditional_intervals=ci)
    target = tmp_path/'must-not-exist.pdf'
    with pytest.raises(ValueError, match='[Cc]onditional intervals'):
        render_cma_inference(replace(report, views=tuple(views)), target)
    assert not target.exists()


@pytest.mark.parametrize('defect', ['basis', 'keys', 'accounting', 'too_many'])
def test_ambiguous_comparisons_fail_before_writing(tmp_path, defect):
    """Mixed calibrations, ambiguous identities and failed accounting cannot be published."""
    report = _report()
    views = list(report.views)
    if defect == 'basis':
        bad = deepcopy(views[1].result)
        bad.factors.at['Equity', 'calibrated_premium'] += .001
        views[1] = replace(views[1], result=bad)
    elif defect == 'keys':
        views[1] = replace(views[1], key='full')
    elif defect == 'accounting':
        bad = deepcopy(views[1].result)
        bad.summary['accounting_status'] = 'fail'
        views[1] = replace(views[1], result=bad)
    else:
        views = [replace(views[0], key=str(i)) for i in range(7)]
    target = tmp_path/'must-not-exist.pdf'
    with pytest.raises(ValueError):
        render_cma_inference(replace(report, views=tuple(views)), target)
    assert not target.exists()


def test_failed_export_preserves_previous_report_and_closes_figures(tmp_path, monkeypatch):
    """An export error cannot replace a reviewed PDF with a partial new file."""
    target = tmp_path/'reviewed.pdf'
    target.write_bytes(b'previous report')
    def fail(*args, **kwargs):
        """Simulate a failed PDF backend."""
        raise OSError('export failed')
    monkeypatch.setattr(report_module.qis, 'save_figs_to_pdf', fail)
    with pytest.raises(OSError, match='export failed'):
        render_cma_inference(_report(), target)
    assert target.read_bytes() == b'previous report'
    assert not list(tmp_path.glob('.cma-inference-*'))
    assert not plt.get_fignums()
