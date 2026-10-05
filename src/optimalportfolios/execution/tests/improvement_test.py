"""Independent scoring references and audited support-search regressions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from optimalportfolios.execution import schema as s
from optimalportfolios.execution.improvement import (
    ExecutionScoreMethod, ExecutionSearchConfig, diagnose_execution_corridors,
    improve_ranked_execution, score_execution_candidates,
)
from optimalportfolios.execution.types import ExecutionRankingConfig, ResolvedExecutionProblem
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints, GroupLowerUpperConstraints


def _problem(max_trades=1):
    """Make correlated offsetting moves and a smaller independent alternative."""
    index = pd.Index(['A', 'B', 'C', 'Cash'])
    current = pd.Series([0.3, 0.3, 0.2, 0.2], index=index)
    model = pd.Series([0.4, 0.2, 0.21, 0.19], index=index)
    cash = pd.Series([False, False, False, True], index=index)
    target = pd.DataFrame({
        s.CURRENT_WEIGHT: current, s.BASE_WEIGHT: current, s.RAW_MODEL_WEIGHT: model,
        s.EFFECTIVE_MODEL_WEIGHT: model, s.DESIRED_REWEIGHT: (model-current).where(~cash, 0),
        s.SETTLEMENT_CASH: cash, s.MANDATORY_TRADE: False, s.MANDATORY_FUNDING: 0.,
        s.RULE4_TRADE: ~cash, s.TRADE_CANDIDATE: ~cash, s.MATERIAL_TRADE: ~cash,
        s.REBALANCE_CADENCE_ELIGIBLE: True, s.POLICY_MIN_WEIGHT: 0., s.POLICY_MAX_WEIGHT: 1.,
    })
    cov = np.array([[1., .99, 0, 0], [.99, 1., 0, 0], [0, 0, .005, 0], [0, 0, 0, .0001]])
    return ResolvedExecutionProblem(
        target=target, covariance=pd.DataFrame(cov, index=index, columns=index),
        alphas=pd.Series(0., index=index), asset_classes=pd.Series('All', index=index),
        constraints=Constraints(min_weights=pd.Series(0., index=index),
                                max_weights=pd.Series(1., index=index)),
        ranking_config=ExecutionRankingConfig({'All': 1.}, max_trades=max_trades),
        optimiser_config=OptimiserConfig(solver='CLARABEL'),
    )


def test_partial_score_matches_independent_bounded_minimisation():
    """A full move overshoots while a small interior move reduces covariance risk."""
    p = _problem()
    full = score_execution_candidates(p, ExecutionScoreMethod.FULL_RISK)
    partial = score_execution_candidates(p, ExecutionScoreMethod.PARTIAL_RISK)
    assert full.index[full[s.SELECTED_TRADE]].tolist() == ['C']
    assert partial.index[partial[s.SELECTED_TRADE]].tolist() == ['A']
    active = (p.target[s.BASE_WEIGHT]-p.target[s.RAW_MODEL_WEIGHT]).to_numpy()
    cov = p.covariance.to_numpy()
    # Independent full-portfolio oracle for the selection kernel, not reporting analytics.
    before = float(active @ cov @ active)
    for i, name in enumerate(p.target.index[:-1]):
        direction = np.zeros(4)
        direction[i], direction[-1] = 1., -1.
        delta = p.target.loc[name, s.DESIRED_REWEIGHT]

        def objective(x):
            """Rebuild a complete funded active vector for an independent reference."""
            moved = active+x*direction
            return float(moved @ cov @ moved)

        oracle = minimize_scalar(objective, bounds=(min(0., delta), max(0., delta)),
                                 method='bounded', options={'xatol': 1e-14})
        assert partial.loc[name, 'scored_displacement'] == pytest.approx(oracle.x, abs=1e-8)
        assert partial.loc[name, s.TRADE_SCORE] == pytest.approx(before-oracle.fun, abs=1e-12)


def test_search_exchange_improves_risk_without_adding_tickets():
    """Joint projection replaces the misleading full-move selection with useful A."""
    p = _problem()
    result = improve_ranked_execution(p, ExecutionScoreMethod.FULL_RISK,
                                     ExecutionSearchConfig(max_removal_trials=0,
                                                           max_exchange_trials=10))
    assert result.result.accepted and result.result.compliant
    assert result.summary['final_tickets'] <= result.summary['initial_tickets']
    assert result.summary['final_te_bp'] < result.summary['initial_te_bp']
    assert result.attempts['retained'].any()
    assert result.result.trade_table.loc['A', s.SELECTED_TRADE]


def test_compression_uses_total_risk_allowance_and_preserves_input():
    """Removing a tiny optional move saves a ticket within the initial TE allowance."""
    p = _problem(max_trades=3)
    p.target.loc['C', [s.RAW_MODEL_WEIGHT, s.EFFECTIVE_MODEL_WEIGHT]] = .200001
    p.target.loc['Cash', [s.RAW_MODEL_WEIGHT, s.EFFECTIVE_MODEL_WEIGHT]] = .199999
    p.target.loc['C', s.DESIRED_REWEIGHT] = .000001
    before = p.target.copy(deep=True)
    result = improve_ranked_execution(p, ExecutionScoreMethod.LEGACY,
                                     ExecutionSearchConfig(max_removal_trials=20,
                                                           max_exchange_trials=0,
                                                           te_allowance_bp=1.))
    assert result.summary['final_tickets'] < result.summary['initial_tickets']
    assert result.summary['final_te_bp'] <= result.summary['initial_te_bp']+1.
    pd.testing.assert_frame_equal(p.target, before)


def test_full_corridor_diagnostic_keeps_scope_and_does_not_relax():
    """Opening every eligible coordinate cannot repair a corridor floor above a cap."""
    p = _problem()
    loads = pd.DataFrame({'AB': [1., 1., 0., 0.]}, index=p.target.index)
    groups = GroupLowerUpperConstraints(group_loadings=loads,
        group_min_allocation=pd.Series({'AB': 0.}), group_max_allocation=pd.Series({'AB': .4}))
    p = replace(p, constraints=p.constraints.copy(group_lower_upper_constraints=groups))
    result = diagnose_execution_corridors(p)
    assert result.status == 'interval_infeasible'
    assert result.bridges.loc['AB', 'bridge_max'] > 0
    assert result.result is None


@pytest.mark.parametrize('kwargs', [{'max_removal_trials': -1}, {'max_exchange_trials': True},
                                  {'te_allowance_bp': -1}, {'te_allowance_bp': np.nan},
                                  {'exchange_candidate_limit': 0}])
def test_search_configuration_rejects_invalid_budgets(kwargs):
    """Bad search controls fail before numerical work."""
    with pytest.raises(ValueError):
        ExecutionSearchConfig(**kwargs)


def test_legacy_identity_and_no_alpha_effective_class_weights():
    """The opt-in entry preserves legacy tables and removes only alpha in A1."""
    from optimalportfolios.execution.ranking import score_execution_trades

    p = replace(_problem(), alphas=pd.Series([4., 3., 2., 1.], index=_problem().target.index),
                ranking_config=ExecutionRankingConfig({'All': 2.}, max_trades=1),
                asset_class_tre_weights=pd.Series({'All': 3.}))
    legacy = score_execution_trades(p.target, p.alphas, p.covariance, p.asset_classes,
                                    p.ranking_config, p.asset_class_tre_weights)
    pd.testing.assert_frame_equal(score_execution_candidates(p), legacy)
    no_alpha = score_execution_candidates(p, ExecutionScoreMethod.NO_ALPHA)
    np.testing.assert_allclose(no_alpha[s.TRADE_SCORE], -6*legacy[s.MARGINAL_TRE])


@pytest.mark.parametrize('settings', [{'minimum_trade_score': 0.},
                                      {'sequential_greedy_selection': True}])
def test_research_scores_reject_incompatible_saved_controls(settings):
    """A cutoff in old score units cannot silently filter a new scoring method."""
    p = replace(_problem(), ranking_config=ExecutionRankingConfig({'All': 1.}, **settings))
    with pytest.raises(ValueError, match='research scores require'):
        score_execution_candidates(p, ExecutionScoreMethod.PARTIAL_RISK)


def test_sequential_scores_against_complete_funded_vectors():
    """Rebuild each virtual portfolio independently to verify all selection steps."""
    p = _problem(max_trades=None)
    table = score_execution_candidates(p, ExecutionScoreMethod.SEQUENTIAL_FULL_RISK)
    active = (p.target[s.BASE_WEIGHT]-p.target[s.RAW_MODEL_WEIGHT]).to_numpy(copy=True)
    cov = p.covariance.to_numpy()
    remaining = list(range(3))
    for step in range(3):
        before = active @ cov @ active
        gains = {}
        for i in remaining:
            moved = active.copy()
            delta = p.target.iloc[i][s.DESIRED_REWEIGHT]
            moved[i] += delta
            moved[-1] -= delta
            gains[i] = before-moved @ cov @ moved
        winner = min(remaining, key=lambda i: (-gains[i],
                     -gains[i]/abs(p.target.iloc[i][s.DESIRED_REWEIGHT]), i))
        name = p.target.index[winner]
        assert table.at[name, s.SEQUENTIAL_SELECTION_ORDER] == step+1
        assert table.at[name, s.SELECTION_SCORE] == pytest.approx(gains[winner], abs=1e-12)
        active[winner] += p.target.at[name, s.DESIRED_REWEIGHT]
        active[-1] -= p.target.at[name, s.DESIRED_REWEIGHT]
        remaining.remove(winner)


def test_partial_scores_submaterial_rescue_rows_and_rejects_invalid_domain(monkeypatch):
    """Rescue-only rows get valid scores; a nonzero-only optional interval is rejected."""
    from optimalportfolios.execution import improvement as module

    p = _problem()
    p.target.loc['A', [s.MATERIAL_TRADE, s.TRADE_CANDIDATE]] = False
    scored = score_execution_candidates(p, ExecutionScoreMethod.PARTIAL_RISK)
    assert scored.loc['A', s.TRADE_SCORE] > 0
    assert not scored.loc['A', s.SELECTED_TRADE]

    def exclude_zero(*args):
        """Emulate an unsupported optional corridor without changing eligibility."""
        return p.target[s.BASE_WEIGHT]+.01, p.target[s.BASE_WEIGHT]+.1

    monkeypatch.setattr(module, '_build_selected_execution_bound_series', exclude_zero)
    with pytest.raises(ValueError, match='intervals must contain zero'):
        score_execution_candidates(p, ExecutionScoreMethod.PARTIAL_RISK)


def test_zero_curvature_and_invalid_directional_variance():
    """A flat funded direction chooses zero, while indefinite curvature is rejected."""
    from optimalportfolios.execution.improvement import _risk_gains

    gains, moves = _risk_gains(np.ones((2, 2)), np.array([.1, -.1]), 1,
                              np.array([.1, 0.]), np.array([-.1, 0.]), np.array([.1, 0.]))
    np.testing.assert_array_equal(gains, 0.)
    np.testing.assert_array_equal(moves, 0.)
    with pytest.raises(ValueError, match='funded-trade variance'):
        _risk_gains(np.array([[1., 2.], [2., 1.]]), np.zeros(2), 1, np.zeros(2))


def test_total_te_budget_is_not_accumulated_per_removal():
    """Two individually cheap deletions together exceed the single initial allowance."""
    p = _problem(max_trades=3)
    p.covariance.iloc[:, :] = np.diag([.01, .01, .01, 1e-8])
    moves = pd.Series([.0008, -.0008, .0008, -.0008], index=p.target.index)
    p.target[s.RAW_MODEL_WEIGHT] = p.target[s.CURRENT_WEIGHT]+moves
    p.target[s.EFFECTIVE_MODEL_WEIGHT] = p.target[s.RAW_MODEL_WEIGHT]
    p.target[s.DESIRED_REWEIGHT] = moves.where(~p.target[s.SETTLEMENT_CASH], 0.)
    result = improve_ranked_execution(p, config=ExecutionSearchConfig(
        max_removal_trials=20, max_exchange_trials=0, te_allowance_bp=.8))
    assert result.attempts['retained'].sum() == 1
    assert result.summary['final_tickets'] == 2
    assert result.summary['final_te_bp'] <= result.summary['initial_te_bp']+.8


def test_disabled_search_preserves_weights_and_is_not_budget_exhaustion():
    """Zero budgets are disabled stages, and must not be reported as truncated work."""
    from optimalportfolios.execution.api import solve_ranked_execution

    p = _problem()
    expected = solve_ranked_execution(p)
    result = improve_ranked_execution(p, config=ExecutionSearchConfig(0, 0))
    pd.testing.assert_series_equal(result.result.weights, expected.weights)
    assert result.attempts.empty
    assert not result.summary['removal_budget_exhausted']
    assert not result.summary['exchange_budget_exhausted']


@pytest.mark.parametrize('status,diagnostic', [('infeasible', 'solver_reported_infeasible'),
                                             ('solver_error', 'unresolved_or_rejected')])
def test_failed_baseline_and_diagnostic_statuses(monkeypatch, status, diagnostic):
    """Unaccepted returns remain explicit failures and launch no improvement trials."""
    from optimalportfolios.execution import improvement as module

    failed = SimpleNamespace(accepted=False, compliant=False,
                             outcome=SimpleNamespace(status=status, reason='synthetic failure'))
    monkeypatch.setattr(module, '_solve', lambda *a, **k: failed)
    result = improve_ranked_execution(_problem())
    assert result.summary == {'stop_reason': 'baseline_not_accepted'}
    assert result.attempts.empty
    assert diagnose_execution_corridors(_problem()).status == diagnostic


def test_continuous_diagnostic_and_protected_rows():
    """Opening rescue candidates cannot remove mandatory rows or admit cadence pins."""
    p = _problem(max_trades=3)
    p.target.loc['A', s.MANDATORY_TRADE] = True
    p.target.loc[['A', 'B'], s.TRADE_CANDIDATE] = False
    p.target.loc['B', s.REBALANCE_CADENCE_ELIGIBLE] = False
    diagnostic = diagnose_execution_corridors(p)
    assert diagnostic.status == 'accepted_continuous'
    result = improve_ranked_execution(p, config=ExecutionSearchConfig(10, 10, 10000.))
    assert not result.attempts['removed'].isin(['A', 'B', 'Cash']).any()
    assert not result.attempts['added'].isin(['A', 'B', 'Cash']).any()


def test_exchange_budget_and_failed_trial_rollback(monkeypatch):
    """Structural or rejected exchange trials count once and retain the incumbent."""
    from optimalportfolios.execution import improvement as module

    p = _problem()
    baseline = module._solve(p, score_execution_candidates(p), rescue=True)
    calls = []

    def fail_trial(problem, table, rescue):
        """Return a real audited initial portfolio, then a structural failed edit."""
        calls.append(rescue)
        if rescue:
            return baseline
        raise module.ExecutionSolverInfeasibility('synthetic trial conflict')

    monkeypatch.setattr(module, '_solve', fail_trial)
    result = improve_ranked_execution(p, config=ExecutionSearchConfig(0, 1))
    assert calls == [True, False]
    assert result.summary['exchange_budget_exhausted']
    assert len(result.attempts) == 1
    assert result.attempts.iloc[0]['status'].startswith('interval_infeasible')
    assert not result.attempts['retained'].any()
    pd.testing.assert_series_equal(result.result.weights, baseline.weights)


def test_search_rejects_relaxed_incumbent_and_missing_risk(monkeypatch):
    """Unsupported incumbent domains and unavailable risk cannot yield an improvement."""
    from optimalportfolios.execution import improvement as module

    p = _problem()
    baseline = module._solve(p, score_execution_candidates(p), rescue=True)
    baseline.trade_table[module.RELAXED_CORRIDOR_SOLVE] = True
    monkeypatch.setattr(module, '_solve', lambda *a, **k: baseline)
    with pytest.raises(NotImplementedError, match='strict-corridor'):
        improve_ranked_execution(p)
    baseline.trade_table[module.RELAXED_CORRIDOR_SOLVE] = False
    monkeypatch.setattr(module, 'build_risk_model', lambda *a: SimpleNamespace(
        compute_tre_at_date=lambda *args: np.nan))
    with pytest.raises(ValueError, match='tracking error is unavailable'):
        improve_ranked_execution(p)
    with pytest.raises(ValueError, match='strictly positive'):
        ExecutionSearchConfig(min_exchange_improvement_bp=0.)


@pytest.mark.parametrize('matrix,active,delta,message', [
    (-np.eye(2), np.zeros(2), .1, 'funded-trade'),
    (np.array([[1., 2.], [2., 3.]]), np.array([1., 0.]), 2., 'incremental active'),
])
def test_vectorized_legacy_variance_guards(matrix, active, delta, message):
    """The optimized one-shot kernel retains the scalar reference's rejection guards."""
    from optimalportfolios.execution.ranking import score_execution_trades

    p = _problem()
    target = p.target.loc[['A', 'Cash']].copy()
    target[s.BASE_WEIGHT] = target[s.RAW_MODEL_WEIGHT]+active
    target[s.DESIRED_REWEIGHT] = [delta, 0.]
    covariance = pd.DataFrame(matrix, index=target.index, columns=target.index)
    with pytest.raises(ValueError, match=message):
        score_execution_trades(target, p.alphas, covariance, p.asset_classes, p.ranking_config)
