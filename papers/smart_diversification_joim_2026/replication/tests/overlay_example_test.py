"""Public, offline checks for the synthetic 60/40 overlay example only."""
from dataclasses import FrozenInstanceError, replace
import socket

import numpy as np
import pandas as pd
import pytest

from papers.smart_diversification_joim_2026 import run_overlay_example as ex


@pytest.mark.parametrize('local, prefix, figures', [
    (ex.Locals.RUN_SINGLE_OPTIMISATION, 'single_optimisation',
     ('smart_diversification', 'regime_contributions', 'overlay_weights',
      'overlay_allocation_frontier')),
    (ex.Locals.RUN_COVERAGE_FLOOR_FRONTIER, 'coverage_floor_frontier',
     ('coverage_floor_frontier', 'coverage_floor_weights', 'overlay_allocation_frontier')),
])
def test_example_executes_offline_and_honors_capital_and_coverage(
        tmp_path, monkeypatch, local, prefix, figures):
    """Generate every public artifact offline and check the floor from the actual returns."""
    def no_network(*args, **kwargs):
        """Reject network access by any transitive example dependency."""
        raise AssertionError('The public example must be self-contained.')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(ex, 'get_output_path', lambda: str(tmp_path))
    solver = ex.op.cvx_maximize_portfolio_sharpe
    solves = []

    def counted_solver(*args, **kwargs):
        """Ensure drawing the single-case view does not solve a frontier behind the scenes."""
        solves.append(1)
        return solver(*args, **kwargs)

    monkeypatch.setattr(ex.op, 'cvx_maximize_portfolio_sharpe', counted_solver)
    result = ex.run_local(local)
    if local == ex.Locals.RUN_SINGLE_OPTIMISATION:
        assert len(solves) == 1
    output = tmp_path / 'joim-overlay-example' / local.name.lower()
    weights = result.weights.drop(columns='Benchmark')
    panel = pd.read_csv(output / 'monthly_excess_returns.csv', index_col=0)
    assert len(panel) == 360
    np.testing.assert_allclose(weights.loc[ex.CORE], 1.0, atol=1e-6)
    np.testing.assert_allclose(weights.drop(index=ex.CORE).sum(), 1.0, atol=1e-6)
    # Independent sample-mask reference: no premium-table or mixture-covariance call.
    bear = panel[ex.CORE] <= panel[ex.CORE].quantile(0.16)
    benchmark_loss = panel.loc[bear, ex.CORE].sum()
    contributions = (panel.loc[bear] @ weights).sum()
    coverage = 1.0 - contributions / benchmark_loss
    if 'No floor' in coverage:
        assert coverage['No floor'] < 0.40
    assert coverage['theta=40%'] == pytest.approx(0.40, abs=1e-6)
    comparison = pd.read_csv(output / f'{prefix}_statistics.csv', index_col=0)
    np.testing.assert_allclose(comparison.loc[weights.columns, 'bear_loss_coverage'], coverage)
    for name in figures:
        assert (output / f'{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_optimizer_beats_feasible_grid_and_rejects_infeasible_floor():
    """Compare OP's objective with an independently evaluated simplex grid."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    inputs = ex.estimate_inputs(panel)
    weights = ex.solve_overlay(inputs, 1.0, 0.40).to_numpy()
    assert weights[-1] == pytest.approx(0.16426596, abs=1e-6)
    cov = inputs.covariance.to_numpy()
    mu = inputs.expected_returns.to_numpy()
    coefficients = inputs.bear_contributions.to_numpy()
    floor = 0.60 * coefficients[0]
    best_grid = -np.inf
    for i in range(21):
        for j in range(21 - i):
            for k in range(21 - i - j):
                trial = np.array([1.0, i / 20, j / 20, k / 20, (20 - i - j - k) / 20])
                if coefficients @ trial >= floor:
                    best_grid = max(best_grid, mu @ trial / np.sqrt(trial @ cov @ trial))
    assert np.isfinite(best_grid)
    assert mu @ weights / np.sqrt(weights @ cov @ weights) >= best_grid - 1e-6
    with pytest.raises(ValueError, match='infeasible'):
        ex.solve_overlay(inputs, 0.01, 1.0)


def test_estimation_inventory_is_frozen_and_full_precision(tmp_path, monkeypatch):
    """Estimation supplies seven fund inputs and shared moments without running a solver."""
    def no_solver(*args, **kwargs):
        """Estimation must stop before allocation."""
        raise AssertionError('ESTIMATE_INPUTS must not optimize')

    monkeypatch.setattr(ex.op, 'cvx_maximize_portfolio_sharpe', no_solver)
    monkeypatch.setattr(ex, 'get_output_path', lambda: str(tmp_path))
    inputs = ex.run_local(ex.Locals.ESTIMATE_INPUTS)
    output = tmp_path / 'joim-overlay-example' / 'estimate_inputs'
    assert isinstance(inputs, ex.OverlayInputs)
    with pytest.raises(FrozenInstanceError):
        inputs.covariance = pd.DataFrame()
    sheet = inputs.input_sheet
    assert list(sheet.index) == [ex.CORE, 'Trend', 'Equity L/S', 'Market neutral', 'Tail hedge']
    assert set(('sharpe', 'ann_vol', 'bear_sharpe', 'beta_bear', 'beta_normal',
                'beta_bull', 'idio_vol')).issubset(sheet.columns)
    np.testing.assert_allclose(sheet.loc[ex.CORE, ['beta_bear', 'beta_normal', 'beta_bull']], 1.)
    assert sheet.loc[ex.CORE, 'idio_vol'] == 0.
    np.testing.assert_allclose(inputs.expected_returns, ex.AF * inputs.returns.mean())
    np.testing.assert_allclose(inputs.bear_contributions, sheet.ann_vol * sheet.bear_sharpe)
    assert inputs.regime_moments.probability.sum() == pytest.approx(1.)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / 'allocator_input_sheet.csv', index_col=0), sheet,
        check_exact=False, rtol=1e-12, atol=1e-15)
    assert not list(output.glob('*weights*'))


def test_single_case_reuses_inputs_and_solves_once(monkeypatch):
    """The single optimisation consumes an estimated snapshot without re-estimation."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    inputs = ex.estimate_inputs(panel)
    original = inputs.covariance.copy(deep=True)
    solver = ex.op.cvx_maximize_portfolio_sharpe
    calls = []

    def counted_solver(*args, **kwargs):
        """Count OP solves without replacing their numerical work."""
        calls.append(1)
        return solver(*args, **kwargs)

    def no_estimation(*args, **kwargs):
        """Do not regenerate inputs when the caller supplies them."""
        raise AssertionError('Inputs must be reused')

    monkeypatch.setattr(ex.op, 'cvx_maximize_portfolio_sharpe', counted_solver)
    monkeypatch.setattr(ex, 'estimate_inputs', no_estimation)
    result = ex.run_single_optimisation(inputs, coverage=0.40)
    assert len(calls) == 1
    assert result.weights.columns.tolist() == ['Benchmark', 'theta=40%']
    assert result.weights.loc['Tail hedge', 'theta=40%'] == pytest.approx(0.16426596, abs=1e-6)
    assert result.table.loc['Realized coverage theta', 'theta=40%'] == pytest.approx(40., abs=1e-4)
    pd.testing.assert_frame_equal(inputs.covariance, original)


def test_frontier_has_table4_shape_and_independent_numerical_checks():
    """Check each column against raw returns and independent concentration arithmetic."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    inputs = ex.estimate_inputs(panel)
    result = ex.run_coverage_floor_frontier(inputs, theta_grid=[0., 0.4, 0.6], budget=0.8)
    assert result.table.columns.tolist() == [
        'Benchmark', 'Equal weight', 'No floor', 'theta=0%', 'theta=40%', 'theta=60%']
    funds = panel.columns.drop(ex.CORE)
    np.testing.assert_allclose(result.weights.loc[ex.CORE], 1., atol=1e-6)
    np.testing.assert_allclose(result.weights.loc[funds].sum().iloc[1:], 0.8, atol=1e-6)
    np.testing.assert_allclose(result.table.loc[funds].sum().iloc[1:], 100., atol=1e-5)
    assert np.isnan(result.table.loc['Effective N', 'Benchmark'])
    assert result.table.loc['Effective N', 'Equal weight'] == pytest.approx(4.)
    bear = panel[ex.CORE] <= panel[ex.CORE].quantile(0.16)
    for name in result.weights:
        weights = result.weights[name]
        stack = panel @ weights
        coverage = 1. - stack.loc[bear].sum() / panel.loc[bear, ex.CORE].sum()
        assert result.table.loc['Realized coverage theta', name] == pytest.approx(100. * coverage)
        assert result.table.loc['Volatility sigma_P', name] == pytest.approx(
            100. * np.sqrt(ex.AF) * stack.std(ddof=1))
        assert result.table.loc['Realized SR_Total', name] == pytest.approx(
            np.sqrt(ex.AF) * stack.mean() / stack.std(ddof=1))
        assert result.table.loc['Bear-regime contribution p.a.', name] == pytest.approx(
            100. * ex.AF * stack.loc[bear].sum() / len(stack))
        if name != 'Benchmark':
            shares = weights.loc[funds].clip(lower=0.)
            shares /= shares.sum()
            assert result.table.loc['Effective N', name] == pytest.approx(1. / (shares @ shares))
        floor = result.coverage_floors[name]
        if pd.notna(floor):
            assert coverage >= floor - 1e-6
            standalone = ex.solve_overlay(inputs, budget=0.8, coverage=floor)
            np.testing.assert_allclose(weights, standalone, atol=1e-8)
    buckets = ['Trend bucket', 'L/S bucket', 'Long-vol bucket']
    np.testing.assert_allclose(result.table.loc[buckets].sum(), result.table.loc[funds].sum())


@pytest.mark.parametrize('grid', [[], [0.4, 0.4], [np.nan], [-0.1], [1.1]])
def test_frontier_rejects_invalid_grids(grid):
    """Do not silently omit, duplicate or relabel a requested policy."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    with pytest.raises(ValueError, match='grid'):
        ex.run_coverage_floor_frontier(ex.estimate_inputs(panel), theta_grid=grid)


def test_frontier_rejects_unattainable_floor():
    """An infeasible frontier column must not be replaced with a fallback portfolio."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    with pytest.raises(ValueError, match='theta=100%.*infeasible'):
        ex.run_coverage_floor_frontier(ex.estimate_inputs(panel), theta_grid=[1.], budget=0.01)


def test_figure4_uses_stacks_at_the_requested_budget_and_reuses_solved_statistics(tmp_path):
    """Compare the plotted fund points with raw core-plus-overlay returns at a non-unit budget."""
    params = ex.ExampleParams(overlay_budget=0.8)
    panel, _ = ex.create_example_data(params)
    inputs = ex.estimate_inputs(panel)
    result = ex.run_coverage_floor_frontier(inputs, theta_grid=[0.6, 0.0, 0.4], budget=0.8)
    ex.create_allocation_frontier_illustration(inputs, result, tmp_path, params)
    points = pd.read_csv(tmp_path / 'overlay_allocation_points.csv', index_col=0)
    frontier = pd.read_csv(tmp_path / 'overlay_allocation_frontier.csv', index_col=0)
    bear = panel[ex.CORE] <= panel[ex.CORE].quantile(0.16)
    for fund in panel.columns.drop(ex.CORE):
        stack = panel[ex.CORE] + 0.8 * panel[fund]
        scale = np.sqrt(ex.AF) / stack.std(ddof=1)
        assert points.loc[fund, 'sharpe'] == pytest.approx(scale * stack.mean())
        assert points.loc[fund, 'bear_sharpe'] == pytest.approx(
            scale * stack.loc[bear].sum() / len(stack))
    np.testing.assert_array_equal(frontier.theta, [0.6, 0.0, 0.4])
    pd.testing.assert_frame_equal(
        frontier.drop(columns='theta'), result.statistics.loc[frontier.index],
        check_exact=False, rtol=1e-12, atol=1e-15)
    for name in ['Benchmark', 'Equal weight', 'No floor', 'theta=40%']:
        np.testing.assert_allclose(points.loc[name], result.statistics.loc[name],
                                   rtol=1e-12, atol=1e-15)


def test_input_snapshot_copies_frames_and_rejects_misalignment():
    """A frozen input binding also owns its data and checks the optimizer's asset order."""
    panel, _ = ex.create_example_data(ex.ExampleParams())
    inputs = ex.estimate_inputs(panel)
    copied = replace(inputs)
    copied.statistics.iloc[0, 0] = 999.
    assert inputs.statistics.iloc[0, 0] != 999.
    copied.covariance.iloc[0, 0] = 999.
    assert inputs.covariance.iloc[0, 0] != 999.
    with pytest.raises(ValueError, match='aligned'):
        replace(inputs, covariance=inputs.covariance.iloc[::-1, ::-1])
    panel.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match='complete finite'):
        ex.estimate_inputs(panel)
