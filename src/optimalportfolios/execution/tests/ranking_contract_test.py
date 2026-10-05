"""Independent funded-trade references and resolved-input validation."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import (
    ExecutionRankingConfig, ResolvedExecutionProblem, resolve_effective_max_trades,
    score_execution_trades, solve_ranked_execution,
)
from optimalportfolios.execution import schema as s
from optimalportfolios.execution.ranking import _incremental_funded_tre
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints, GroupLowerUpperConstraints


def make_problem(**kwargs) -> ResolvedExecutionProblem:
    """Build a redistributable four-asset decision with positive-definite risk."""
    index = pd.Index(['A', 'B', 'C', 'Cash'])
    current = pd.Series([0.30, 0.25, 0.25, 0.20], index=index)
    model = pd.Series([0.40, 0.15, 0.30, 0.15], index=index)
    cash = pd.Series([False, False, False, True], index=index)
    target = pd.DataFrame({
        s.CURRENT_WEIGHT: current, s.RAW_MODEL_WEIGHT: model,
        s.EFFECTIVE_MODEL_WEIGHT: model, s.BASE_WEIGHT: current,
        s.DESIRED_REWEIGHT: (model - current).where(~cash, 0.0),
        s.MANDATORY_FUNDING: 0.0, s.POLICY_MIN_WEIGHT: 0.0,
        s.POLICY_MAX_WEIGHT: 1.0, s.SETTLEMENT_CASH: cash,
        s.MANDATORY_TRADE: False, s.RULE4_TRADE: ~cash,
        s.REBALANCE_CADENCE_ELIGIBLE: True,
        s.TRADE_CANDIDATE: ~cash, s.MATERIAL_TRADE: ~cash,
    }, index=index)
    factor = np.array([[0.2, 0.0, 0.0, 0.0], [0.1, 0.2, 0.0, 0.0],
                       [-0.1, 0.1, 0.15, 0.0], [0.02, 0.0, 0.0, 0.01]])
    args = dict(
        target=target, covariance=pd.DataFrame(factor @ factor.T, index=index, columns=index),
        alphas=pd.Series([0.02, -0.01, 0.03, 0.001], index=index),
        asset_classes=pd.Series(['Risk', 'Risk', 'Risk', 'Cash'], index=index),
        constraints=Constraints(min_weights=pd.Series(0.0, index=index),
                                max_weights=pd.Series(1.0, index=index)),
        ranking_config=ExecutionRankingConfig({'Risk': 1.0, 'Cash': 0.0}, max_trades=2),
        optimiser_config=OptimiserConfig(solver='CLARABEL'),
    )
    args.update(kwargs)
    return ResolvedExecutionProblem(**args)


def score(problem: ResolvedExecutionProblem) -> pd.DataFrame:
    """Evaluate the public scoring seam on a numerical problem."""
    return score_execution_trades(
        problem.target, problem.alphas, problem.covariance, problem.asset_classes,
        problem.ranking_config, problem.asset_class_tre_weights,
    )


def test_scores_match_independent_funded_portfolios() -> None:
    """Use full before/after portfolios to check the incremental selection kernel."""
    problem = make_problem()
    table = score(problem)
    active = (problem.target[s.BASE_WEIGHT] - problem.target[s.RAW_MODEL_WEIGHT]).to_numpy(
        copy=True)
    covariance = problem.covariance.to_numpy()
    # Independent test oracle: qis.RiskModel does not expose per-candidate selection scores.
    before = np.sqrt(active @ covariance @ active)
    for i, asset in enumerate(problem.target.index[:-1]):
        delta = problem.target.loc[asset, s.DESIRED_REWEIGHT]
        after = active.copy()
        after[i] += delta
        after[-1] -= delta
        tre_delta = np.sqrt(after @ covariance @ after) - before
        alpha_delta = delta * (problem.alphas.loc[asset] - problem.alphas.iloc[-1])
        assert table.loc[asset, s.MARGINAL_TRE] == pytest.approx(tre_delta, abs=1e-14)
        assert table.loc[asset, s.TRADE_SCORE] == pytest.approx(alpha_delta - tre_delta)
    assert not table.loc['Cash', s.SELECTED_TRADE]


@pytest.mark.parametrize('sequential', [False, True])
def test_selection_capacity_cutoff_and_missing_alpha(sequential) -> None:
    """Mandatory tickets consume the allowance only under the total-cap convention."""
    problem = make_problem()
    problem.target.loc['A', s.MANDATORY_TRADE] = True
    problem.target.loc['A', [s.TRADE_CANDIDATE, s.RULE4_TRADE]] = False
    problem.target.loc['A', s.DESIRED_REWEIGHT] = 0.0
    problem.alphas.loc['B'] = np.nan
    config = replace(problem.ranking_config, sequential_greedy_selection=sequential)
    problem = replace(problem, ranking_config=config)
    table = score(problem)
    assert table[s.SELECTED_TRADE].sum() == 1
    assert table.loc['B', s.ALPHA] == 0.0
    exhausted = score(replace(problem, ranking_config=replace(config, max_trades=1)))
    assert not exhausted[s.SELECTED_TRADE].any()
    table = score(replace(problem, ranking_config=replace(
        config, add_mandatory_trades_to_max_trades=True)))
    assert table[s.SELECTED_TRADE].sum() == 2
    table = score(replace(problem, ranking_config=replace(config, minimum_trade_score=100.0)))
    assert not table[s.SELECTED_TRADE].any()
    table = score(replace(problem, ranking_config=replace(config, max_trades=None)))
    assert table[s.SELECTED_TRADE].sum() == 2


def test_sequential_scores_match_explicit_state_updates() -> None:
    """Every recorded sequential score belongs to the updated funded portfolio."""
    problem = make_problem()
    problem = replace(problem, ranking_config=replace(
        problem.ranking_config, sequential_greedy_selection=True, max_trades=5))
    table = score(problem)
    order = table.dropna(subset=[s.SEQUENTIAL_SELECTION_ORDER]).sort_values(
        s.SEQUENTIAL_SELECTION_ORDER).index
    active = (problem.target[s.BASE_WEIGHT] - problem.target[s.RAW_MODEL_WEIGHT]).to_numpy(
        copy=True)
    covariance = problem.covariance.to_numpy()
    for asset in order:
        position = problem.target.index.get_loc(asset)
        delta = problem.target.loc[asset, s.DESIRED_REWEIGHT]
        before = np.sqrt(max(active @ covariance @ active, 0.0))
        active[position] += delta
        active[-1] -= delta
        expected = np.sqrt(max(active @ covariance @ active, 0.0)) - before
        assert table.loc[asset, s.SELECTION_MARGINAL_TRE] == pytest.approx(expected, abs=1e-14)
    pd.testing.assert_frame_equal(table, score(problem))


def test_product_tre_multipliers_and_ties() -> None:
    """Class coefficients multiply the same full covariance score; ties retain source order."""
    problem = make_problem(asset_class_tre_weights=pd.Series({'Risk': 2.0, 'Cash': 1.0}))
    table = score(problem)
    assert table.loc['A', s.TRE_WEIGHT] == 2.0
    np.testing.assert_allclose(
        table[s.TRADE_SCORE],
        table[s.MARGINAL_ALPHA] - table[s.TRE_WEIGHT] * table[s.MARGINAL_TRE])
    problem.alphas[:] = 0.0
    problem = replace(problem, ranking_config=ExecutionRankingConfig({'Risk': 0.0, 'Cash': 0.0}, 2))
    assert score(problem).index[score(problem)[s.SELECTED_TRADE]].tolist() == ['A', 'B']


def test_end_to_end_copies_inputs_and_reaches_unrestricted_model() -> None:
    """A full discretionary selection projects to the raw model and preserves caller state."""
    problem = make_problem()
    problem = replace(problem, ranking_config=replace(problem.ranking_config, max_trades=None))
    before = problem.target.copy(deep=True)
    result = solve_ranked_execution(problem)
    assert result.accepted and result.compliant
    # The model is at corridor endpoints. Assess solver objective accuracy here;
    # same-backend OP/ROSAA parity uses the separate, much tighter weight check.
    active = (result.weights - problem.target[s.RAW_MODEL_WEIGHT]).to_numpy()
    assert active @ problem.covariance.to_numpy() @ active < 1e-8
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    pd.testing.assert_frame_equal(problem.target, before)
    detached = replace(problem)
    problem.target.iloc[0, 0] = 0.123
    assert detached.target.iloc[0, 0] == 0.3
    problem.target.loc['A', s.BASE_WEIGHT] = np.nan
    with pytest.raises(ValueError, match='base_weight must be finite'):
        solve_ranked_execution(problem)


@pytest.mark.parametrize('kwargs', [
    {'tre_weight_by_asset_class': {}}, {'tre_weight_by_asset_class': {'': 1.0}},
    {'tre_weight_by_asset_class': {'Risk': -1.0}},
    {'tre_weight_by_asset_class': {'Risk': np.nan}},
    {'max_trades': 0}, {'max_trades': True}, {'max_trades': 1.5},
    {'minimum_trade_score': np.inf}, {'sequential_greedy_selection': 1},
    {'add_mandatory_trades_to_max_trades': 'yes'},
])
def test_invalid_ranking_config(kwargs) -> None:
    """Invalid controls fail before selection rather than silently changing the cap."""
    args = {'tre_weight_by_asset_class': {'Risk': 1.0}}
    args.update(kwargs)
    with pytest.raises(ValueError):
        ExecutionRankingConfig(**args)


@pytest.mark.parametrize('count', [-1, True, 1.5])
def test_invalid_mandatory_count(count) -> None:
    """Only a nonnegative integer mandatory ticket count is meaningful."""
    with pytest.raises(ValueError, match='mandatory_ticket_count'):
        resolve_effective_max_trades(ExecutionRankingConfig({'Risk': 1.0}), count)


@pytest.mark.parametrize('case', [
    'empty', 'duplicate_assets', 'duplicate_columns', 'missing', 'nan_numeric',
    'nan_flag', 'string_flag', 'two_cash', 'cash_mandatory', 'bad_bounds', 'bad_candidate',
    'duplicate_covar', 'missing_covar', 'nan_covar', 'asymmetric',
    'duplicate_alpha', 'infinite_alpha', 'missing_group', 'unknown_group',
    'duplicate_product', 'invalid_product', 'bad_control', 'duplicate_partition',
    'missing_partition', 'overlap_partition',
])
def test_invalid_resolved_inputs(case) -> None:
    """Malformed aligned inputs fail explicitly before the solver can filter them."""
    problem = make_problem()
    kwargs = {}
    if case == 'empty':
        kwargs['target'] = problem.target.iloc[:0]
    elif case == 'duplicate_assets':
        problem.target.index = ['A', 'A', 'C', 'Cash']
    elif case == 'duplicate_columns':
        kwargs['target'] = pd.concat([problem.target, problem.target[[s.BASE_WEIGHT]]], axis=1)
    elif case == 'missing':
        kwargs['target'] = problem.target.drop(columns=s.BASE_WEIGHT)
    elif case == 'nan_numeric':
        problem.target.loc['A', s.BASE_WEIGHT] = np.nan
    elif case in ('nan_flag', 'string_flag'):
        problem.target[s.MANDATORY_TRADE] = problem.target[s.MANDATORY_TRADE].astype(object)
        problem.target.loc['A', s.MANDATORY_TRADE] = np.nan if case == 'nan_flag' else 'False'
    elif case == 'two_cash':
        problem.target.loc['A', s.SETTLEMENT_CASH] = True
    elif case == 'cash_mandatory':
        problem.target.loc['Cash', s.MANDATORY_TRADE] = True
    elif case == 'bad_bounds':
        problem.target.loc['A', s.POLICY_MIN_WEIGHT] = 2.0
    elif case == 'bad_candidate':
        problem.target.loc['A', s.MANDATORY_TRADE] = True
    elif case == 'duplicate_covar':
        problem.covariance.index = ['A', 'A', 'C', 'Cash']
    elif case == 'missing_covar':
        kwargs['covariance'] = problem.covariance.drop(index='A')
    elif case == 'nan_covar':
        problem.covariance.loc['A', 'A'] = np.nan
    elif case == 'asymmetric':
        problem.covariance.loc['A', 'B'] = 1.0
    elif case == 'duplicate_alpha':
        problem.alphas.index = ['A', 'A', 'C', 'Cash']
    elif case == 'infinite_alpha':
        problem.alphas.loc['A'] = np.inf
    elif case == 'missing_group':
        kwargs['asset_classes'] = problem.asset_classes.drop(index='A')
    elif case == 'unknown_group':
        problem.asset_classes.loc['A'] = 'Unknown'
    elif case == 'duplicate_product':
        kwargs['asset_class_tre_weights'] = pd.Series([1.0, 1.0], index=['Risk', 'Risk'])
    elif case == 'invalid_product':
        kwargs['asset_class_tre_weights'] = pd.Series({'Risk': -1.0})
    elif case == 'bad_control':
        kwargs['expand_for_feasibility'] = 'yes'
    elif case == 'duplicate_partition':
        kwargs['partition_groups'] = ('Risk', 'Risk')
    elif case == 'missing_partition':
        kwargs['partition_groups'] = ('Unknown',)
    elif case == 'overlap_partition':
        kwargs['constraints'] = replace(problem.constraints,
            group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({'Risk': 1.0, 'Cash': 1.0}, index=problem.target.index),
            group_min_allocation=None, group_max_allocation=None))
        kwargs['partition_groups'] = ('Risk', 'Cash')
    with pytest.raises(ValueError):
        replace(problem, **kwargs)


def test_valid_partition_and_reordered_inputs() -> None:
    """The explicit partition and all input arrays survive canonical alignment."""
    problem = make_problem()
    constraints = replace(
        problem.constraints, group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'Risk': [1, 1, 1, 0], 'Cash': [0, 0, 0, 1]},
                                    index=problem.target.index),
        group_min_allocation=None, group_max_allocation=None))
    aligned = replace(problem, covariance=problem.covariance.iloc[::-1, ::-1],
                      alphas=problem.alphas.iloc[::-1], partition_groups=('Risk', 'Cash'),
                      constraints=constraints)
    pd.testing.assert_frame_equal(aligned.covariance, problem.covariance)
    pd.testing.assert_series_equal(aligned.alphas, problem.alphas)


@pytest.mark.parametrize('case', [
    'missing', 'missing_covar', 'nan_covar', 'asymmetric', 'missing_group',
    'unsupported', 'bad_product', 'infinite_alpha', 'cash', 'cash_mandatory',
    'negative_variance',
])
def test_direct_ranking_rejects_invalid_inputs(case) -> None:
    """The standalone scoring entry point preserves its reference failure checks."""
    problem = make_problem()
    kwargs = dict(target=problem.target, alphas=problem.alphas, covariance=problem.covariance,
                  asset_classes=problem.asset_classes, config=problem.ranking_config)
    if case == 'missing':
        kwargs['target'] = problem.target.drop(columns=s.BASE_WEIGHT)
    elif case == 'missing_covar':
        kwargs['covariance'] = problem.covariance.drop(index='A')
    elif case == 'nan_covar':
        problem.covariance.loc['A', 'A'] = np.nan
    elif case == 'asymmetric':
        problem.covariance.loc['A', 'B'] = 1.0
    elif case == 'missing_group':
        kwargs['asset_classes'] = problem.asset_classes.drop(index='A')
    elif case == 'unsupported':
        problem.asset_classes.loc['A'] = 'Unknown'
    elif case == 'bad_product':
        kwargs['asset_class_tre_weights'] = pd.Series({'Risk': -1.0})
    elif case == 'infinite_alpha':
        problem.alphas.loc['A'] = np.inf
    elif case == 'cash':
        problem.target[s.SETTLEMENT_CASH] = False
    elif case == 'cash_mandatory':
        problem.target.loc['Cash', s.MANDATORY_TRADE] = True
    elif case == 'negative_variance':
        kwargs['covariance'] = -problem.covariance
    with pytest.raises(ValueError):
        score_execution_trades(**kwargs)


@pytest.mark.parametrize('covariance,active,delta,message', [
    (-np.eye(2), np.ones(2), 0.1, 'active covariance'),
    (-np.eye(2), np.zeros(2), 0.1, 'funded-trade'),
    (np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, -0.5]), 0.0, None),
    (np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([1.0, 0.0]), 2.0, 'incremental active'),
])
def test_incremental_kernel_variance_guards(covariance, active, delta, message) -> None:
    """Reject materially negative variances while retaining a zero-risk finite trade."""
    if message is None:
        assert _incremental_funded_tre(covariance, active, 0, 1, delta) == 0.0
    else:
        with pytest.raises(ValueError, match=message):
            _incremental_funded_tre(covariance, active, 0, 1, delta)
