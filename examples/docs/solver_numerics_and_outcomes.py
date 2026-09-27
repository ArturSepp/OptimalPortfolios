"""Canonical script of docs/solver_numerics_and_outcomes.md.

The page shows excerpts of ``main``, in the same order; every number and property it states is
asserted after them against a reference computed a different way: eigenvalues from
``numpy.linalg.eigvalsh``, condition numbers from singular values, portfolio variances from a
Cholesky factor, the minimum-variance portfolio from the Moore-Penrose pseudo-inverse, residuals
from the weights and bounds by hand, weight drift from the price ratios, and recording stand-ins
for the configuration fields that reach the solvers. The script runs offline after
``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.solver_numerics_and_outcomes

Several steps deliberately produce rejection logs. ``exhibit`` draws the page's figure;
``tools/docs_analytics/teaching.py`` calls it with the constants below and records their values.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import inspect
import logging

import cvxpy as cvx
import numpy as np
import pandas as pd

import optimalportfolios as op

TICKERS = ['Govt', 'Credit', 'Equity', 'Private A', 'Private B', 'Gold']
# Annual volatilities and correlations. Private A and Private B are one proxy series: they have
# the same volatility, correlation one and the same correlation with every other asset.
VOLS = [0.06, 0.08, 0.16, 0.12, 0.12, 0.15]
CORR = [
    [1.00, 0.30, -0.20, 0.00, 0.00, 0.10],
    [0.30, 1.00, 0.30, 0.20, 0.20, 0.10],
    [-0.20, 0.30, 1.00, 0.40, 0.40, 0.00],
    [0.00, 0.20, 0.40, 1.00, 1.00, 0.00],
    [0.00, 0.20, 0.40, 1.00, 1.00, 0.00],
    [0.10, 0.10, 0.00, 0.00, 0.00, 1.00],
]
FLOOR = 1e-10  # the documented default eigenvalue floor of factorize_covariance
# The exhibit: one minus the correlation of the private pair, and the gap of the left panel.
GAPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
SPECTRUM_GAP = 1e-12
PAIR = ['Private A', 'Private B']
# The page's printed results.
DISTINCT_CONDITION = 12.2
DUPLICATE_CAP = 3.885e8
MIN_VARIANCE_WEIGHTS = [0.6193, 0.1251, 0.0889, 0.0479, 0.0479, 0.0709]


def covariance(vols, corr, tickers) -> pd.DataFrame:
    """Return the covariance matrix with the given volatilities and correlations."""
    vols = np.asarray(vols, dtype=float)
    return pd.DataFrame(np.outer(vols, vols) * np.asarray(corr, dtype=float), index=tickers,
                        columns=tickers)


def with_pair_gap(gap: float) -> np.ndarray:
    """Return the covariance with the correlation of the private pair set to one minus ``gap``."""
    corr = np.array(CORR, dtype=float)
    first, second = TICKERS.index(PAIR[0]), TICKERS.index(PAIR[1])
    corr[first, second] = corr[second, first] = 1.0 - gap
    return covariance(VOLS, corr, TICKERS).to_numpy()


def pseudo_inverse_minimum_variance(covar: np.ndarray) -> np.ndarray:
    """Return the fully invested minimum-variance weights Sigma^+ 1 / (1' Sigma^+ 1)."""
    direction = np.linalg.pinv(covar) @ np.ones(len(covar))
    return direction / direction.sum()


def drifted(weights: pd.Series, prices: pd.DataFrame, start, end) -> pd.Series:
    """Drift ``weights`` from ``start`` to ``end`` with simple price returns, renormalised."""
    growth = prices.loc[end] / prices.loc[start]
    return weights * growth / float((weights * growth).sum())


def assert_raises(error: type, function, *args, **kwargs) -> str:
    """Fail unless ``function(*args, **kwargs)`` raises ``error``; return its message."""
    try:
        function(*args, **kwargs)
    except error as raised:
        return str(raised)
    raise AssertionError(f'expected {error.__name__}')


@contextmanager
def replaced(target, **attributes):
    """Replace attributes of a module or class inside the block and restore them afterwards."""
    saved = {name: getattr(target, name) for name in attributes}
    for name, value in attributes.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(target, name, value)


class PayloadRecorder(logging.Handler):
    """Collect ``(level, payload)`` for log records that carry a given structured attribute."""

    def __init__(self, attribute: str) -> None:
        """Record the payloads stored under ``attribute`` at every level."""
        super().__init__(level=logging.DEBUG)
        self.attribute = attribute
        self.items = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record's level and payload when it carries the attribute."""
        payload = getattr(record, self.attribute, None)
        if payload is not None:
            self.items.append((record.levelno, payload))


@contextmanager
def captured(logger_name: str, attribute: str):
    """Yield the ``(level, payload)`` list of one logger's structured records in the block."""
    logger = logging.getLogger(logger_name)
    recorder = PayloadRecorder(attribute)
    level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)
    try:
        yield recorder.items
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(level)


def solve_arguments(config, covar: pd.DataFrame) -> list:
    """Return the keyword arguments that a minimum-variance wrapper passes to ``problem.solve``."""
    calls = []
    original = cvx.Problem.solve

    def recording(problem, *args, **kwargs):
        """Record the call and solve quietly."""
        calls.append(dict(kwargs))
        return original(problem, *args, **{**kwargs, 'verbose': False})

    with replaced(cvx.Problem, solve=recording):
        op.wrapper_quadratic_optimisation(covar, op.Constraints(is_long_only=True),
                                          optimiser_config=config)
    return calls


def slsqp_display(config, covar: pd.DataFrame) -> bool:
    """Return the ``disp`` option that the maximum-diversification wrapper gives SLSQP."""
    module = importlib.import_module('optimalportfolios.optimization.general.max_diversification')
    options = []
    original = module.minimize

    def recording(*args, **kwargs):
        """Record SLSQP's options and minimise quietly."""
        options.append(dict(kwargs['options']))
        return original(*args, **{**kwargs, 'options': {**kwargs['options'], 'disp': False}})

    with replaced(module, minimize=recording):
        op.wrapper_maximise_diversification(covar, op.Constraints(is_long_only=True),
                                            optimiser_config=config)
    return options[0]['disp']


def alpha_over_tre_hooks(config, covar: pd.DataFrame, constraints) -> tuple:
    """Count the input-contract and failure-diagnosis calls of one alpha-over-TE wrapper call."""
    module = importlib.import_module('optimalportfolios.optimization.taa.maximise_alpha_over_tre')
    contracts, diagnoses = [], []

    def contract(*args, **kwargs):
        """Record the pre-solve input contract."""
        contracts.append(kwargs)

    def diagnosis(status, *args, **kwargs):
        """Record the post-rejection diagnosis and its solver status."""
        diagnoses.append(status)

    with replaced(module, validate_solver_inputs=contract, diagnose_solver_failure=diagnosis):
        benchmark = pd.Series(1.0 / len(covar), index=covar.index)
        op.wrapper_maximise_alpha_over_tre(covar, pd.Series(0.01, index=covar.index), benchmark,
                                           constraints, optimiser_config=config)
    return contracts, diagnoses


def spectrum_and_conditioning() -> tuple:
    """Return the exhibit's spectrum table (raw and floored) and conditioning table by gap."""
    near = with_pair_gap(SPECTRUM_GAP)
    stabilized = op.factorize_covariance(near)
    raw_spectrum = np.linalg.eigvalsh(near)[::-1]
    floored_spectrum = np.linalg.eigvalsh(stabilized.covar)[::-1]
    spectrum = pd.DataFrame({'raw': raw_spectrum, 'floored': floored_spectrum},
                            index=pd.RangeIndex(1, len(TICKERS) + 1, name='rank'))
    rows = []
    for gap in GAPS:
        matrix = with_pair_gap(gap)
        factorization = op.factorize_covariance(matrix)
        eigenvalues = np.linalg.eigvalsh(matrix)
        rows.append({'gap': gap, 'raw_min_eigenvalue': eigenvalues[0],
                     'max_eigenvalue': eigenvalues[-1],
                     'raw': factorization.raw_condition_number,
                     'floored': factorization.stabilized_condition_number})
    return spectrum, pd.DataFrame(rows).set_index('gap')


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    # The documented floor is the default of factorize_covariance.
    signature = inspect.signature(op.factorize_covariance).parameters
    assert signature['eigenvalue_floor'].default == FLOOR
    assert signature['negative_eigenvalue_tolerance'].default == 1e-10

    covar = covariance(VOLS, CORR, TICKERS)
    distinct = covar.drop(index='Private B', columns='Private B')
    factorization = op.factorize_covariance(distinct.to_numpy())
    factor = factorization.factor
    print(factorization.n_eigenvalues_floored, round(factorization.raw_condition_number, 1))

    # Proposition 1 without a floor: B B' reproduces the matrix, B'B is the diagonal of its
    # eigenvalues, and every portfolio variance equals the one from a Cholesky factor.
    matrix = distinct.to_numpy()
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert factorization.n_eigenvalues_floored == 0 and eigenvalues.min() > 100 * FLOOR
    assert factor.shape == (5, 5)
    assert np.abs(factor @ factor.T - matrix).max() < 1e-16
    assert np.abs(factorization.covar - matrix).max() < 1e-16
    np.testing.assert_allclose(factor.T @ factor, np.diag(eigenvalues), rtol=0.0, atol=1e-16)
    cholesky = np.linalg.cholesky(matrix)
    for w in [*np.eye(5), np.full(5, 0.2), np.linspace(-1.0, 2.0, 5)]:
        np.testing.assert_allclose(np.sum((factor.T @ w) ** 2), np.sum((cholesky.T @ w) ** 2),
                                   rtol=1e-12)
    # The raw condition number is the ratio of the extreme singular values.
    singular = np.linalg.svd(matrix, compute_uv=False)
    np.testing.assert_allclose(factorization.raw_condition_number, singular[0] / singular[-1],
                               rtol=1e-10)
    assert round(factorization.raw_condition_number, 1) == DISTINCT_CONDITION
    assert factorization.stabilized_condition_number == factorization.raw_condition_number
    assert factorization.max_eigenvalue_adjustment == 0.0

    floored = op.factorize_covariance(covar.to_numpy())
    print(floored.n_eigenvalues_floored, floored.stabilized_min_eigenvalue,
          f'{floored.stabilized_condition_number:.3e}')

    # The duplicate pair makes one eigenvalue zero up to rounding; it is raised to the floor,
    # the others and all eigenvectors are unchanged, and the change is FLOOR v v' along the null
    # direction v = (e_A - e_B) / sqrt(2).
    raw_eigenvalues = np.linalg.eigvalsh(covar.to_numpy())
    new_eigenvalues = np.linalg.eigvalsh(floored.covar)
    assert abs(raw_eigenvalues[0]) < 1e-16 and raw_eigenvalues[1] > 1e-3
    assert floored.n_eigenvalues_floored == 1 and floored.stabilized_min_eigenvalue == FLOOR
    np.testing.assert_allclose(new_eigenvalues[0], FLOOR, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(new_eigenvalues[1:], raw_eigenvalues[1:], rtol=1e-12)
    pair = [TICKERS.index(name) for name in PAIR]
    direction = np.zeros(len(TICKERS))
    direction[pair] = [2 ** -0.5, -(2 ** -0.5)]
    np.testing.assert_allclose(floored.covar - covar.to_numpy(),
                               FLOOR * np.outer(direction, direction), rtol=0.0, atol=5e-17)
    assert abs(floored.max_eigenvalue_adjustment - FLOOR) < 1e-16
    assert np.abs(floored.factor @ floored.factor.T - floored.covar).max() < 1e-16
    # Its spectral distance from the input is the largest adjustment, at most floor + tolerance.
    distance = np.linalg.norm(floored.covar - covar.to_numpy(), ord=2)
    assert abs(distance - floored.max_eigenvalue_adjustment) < 1e-16 and distance <= 2 * FLOOR
    # Raw conditioning is infinite or of order 1e16, by the sign of a rounding error; the
    # floored matrix is conditioned at the largest eigenvalue over the floor.
    assert floored.raw_condition_number > 1e14
    np.testing.assert_allclose(floored.stabilized_condition_number, raw_eigenvalues[-1] / FLOOR,
                               rtol=1e-12)
    assert float(f'{floored.stabilized_condition_number:.3e}') == DUPLICATE_CAP
    # A portfolio with no weight along v keeps its variance; one along it gains FLOOR / 2.
    equal = np.full(len(TICKERS), 1.0 / len(TICKERS))
    only_a = np.eye(len(TICKERS))[pair[0]]
    assert abs(equal @ floored.covar @ equal - equal @ covar.to_numpy() @ equal) < 5e-17
    np.testing.assert_allclose(only_a @ floored.covar @ only_a - only_a @ covar.to_numpy() @ only_a,
                               FLOOR / 2, rtol=1e-6)

    null = np.zeros(len(TICKERS))
    null[[TICKERS.index('Private A'), TICKERS.index('Private B')]] = [2 ** -0.5, -(2 ** -0.5)]
    residue = op.factorize_covariance(covar.to_numpy() - 5e-12 * np.outer(null, null))
    try:
        op.factorize_covariance(covar.to_numpy() - 1e-6 * np.outer(null, null))
        refused = ''
    except ValueError as error:
        refused = str(error)
    print(f'{residue.raw_min_eigenvalue:.1e}', residue.stabilized_min_eigenvalue)
    print(refused)

    # Residue down to -1e-10 (scaled by the largest eigenvalue when above one) is floored, and
    # the result equals the floored duplicate matrix; anything more negative is refused.
    assert np.array_equal(null, direction)
    assert abs(residue.raw_min_eigenvalue + 5e-12) < 1e-16
    assert residue.stabilized_min_eigenvalue == FLOOR and residue.n_eigenvalues_floored == 1
    assert residue.raw_condition_number == float('inf')
    np.testing.assert_allclose(residue.covar, floored.covar, rtol=0.0, atol=5e-17)
    assert refused == ('covar is materially indefinite: minimum eigenvalue -1e-06 is below '
                       '-1e-10')
    assert raw_eigenvalues[-1] < 1.0  # so the tolerance is 1e-10 itself
    # Units matter: in basis points squared (times 1e8) the largest eigenvalue exceeds one, the
    # tolerance becomes 1e-10 times it, 3.9e-4, and the same residue, now -5e-4, is refused.
    bp_message = assert_raises(ValueError, op.factorize_covariance,
                               1e8 * (covar.to_numpy() - 5e-12 * np.outer(null, null)))
    assert bp_message == ('covar is materially indefinite: minimum eigenvalue -0.0005 is below '
                          '-0.000388549')
    assert f'{1e8 * raw_eigenvalues[-1] * 1e-10:.1e}' == '3.9e-04'
    assert round(raw_eigenvalues[-1], 3) == 0.039
    assert f'{1e8 * raw_eigenvalues[-1]:.1e}' == '3.9e+06'
    edge = op.factorize_covariance(covar.to_numpy() - 0.9e-10 * np.outer(null, null))
    assert edge.n_eigenvalues_floored == 1
    assert_raises(ValueError, op.factorize_covariance,
                  covar.to_numpy() - 1.1e-10 * np.outer(null, null))
    # A non-finite entry is refused too.
    not_finite = covar.to_numpy().copy()
    not_finite[0, 1] = not_finite[1, 0] = np.nan
    assert 'finite' in assert_raises(ValueError, op.factorize_covariance, not_finite)
    # The figure: five eigenvalues between 0.002 and 0.04 are unchanged, the sixth, 1e-14, is
    # raised to the floor; the raw condition number grows as 1 / gap to 3e13 while the floored
    # one follows it until the floor binds, then stays at the largest eigenvalue over the floor,
    # 3.9e8; a condition number near 1e8 is left exactly as it is.
    spectrum, conditioning = spectrum_and_conditioning()
    assert spectrum['raw'].iloc[:5].between(0.002, 0.04).all()
    assert f"{spectrum['raw'].iloc[5]:.0e}" == '1e-14'
    np.testing.assert_allclose(spectrum['floored'].iloc[5], FLOOR, rtol=1e-6)
    np.testing.assert_allclose(spectrum['floored'].iloc[:5], spectrum['raw'].iloc[:5], rtol=1e-10)
    assert f"{conditioning['raw'].iloc[-1]:.0e}" == '3e+13'
    assert f"{conditioning['floored'].iloc[-1]:.1e}" == '3.9e+08'
    near_1e8 = conditioning.loc[1e-8]
    assert f"{near_1e8['raw']:.0e}" == '3e+08' and near_1e8['floored'] == near_1e8['raw']
    slopes = np.diff(np.log10(conditioning['raw'].to_numpy()))
    np.testing.assert_allclose(slopes[1:], 1.0, atol=0.01)

    panel = covar.reindex(index=TICKERS + ['Cash', 'New fund'],
                          columns=TICKERS + ['Cash', 'New fund'], fill_value=0.0)
    panel.loc['New fund', 'New fund'] = np.nan
    alphas = pd.Series([0.1, 0.2, 0.3, 0.2, 0.2, np.nan, 0.0, 0.4], index=panel.index)
    kept, vectors = op.filter_covar_and_vectors_for_nans(
        panel, vectors={'alphas': alphas}, variance_floor=0.07 ** 2, drop_non_finite_vectors=True)
    print(kept.index.tolist())
    print(np.sqrt(np.diag(kept)).round(2).tolist())

    # Cash (zero variance), New fund (missing variance) and Gold (missing alpha) are dropped;
    # the alphas follow; only Govt's variance is below 0.07^2 and is raised; the off-diagonal
    # entries are unchanged, so Govt's correlations fall.
    survivors = ['Govt', 'Credit', 'Equity', 'Private A', 'Private B']
    assert kept.index.tolist() == survivors and kept.columns.tolist() == survivors
    assert vectors['alphas'].index.tolist() == survivors
    assert vectors['alphas'].tolist() == [0.1, 0.2, 0.3, 0.2, 0.2]
    assert np.sqrt(np.diag(kept)).round(2).tolist() == [0.07, 0.08, 0.16, 0.12, 0.12]
    subset = covar.loc[survivors, survivors].to_numpy()
    off_diagonal = ~np.eye(len(survivors), dtype=bool)
    assert np.array_equal(kept.to_numpy()[off_diagonal], subset[off_diagonal])
    assert kept.iloc[0, 0] == 0.07 ** 2 and np.array_equal(np.diag(kept)[1:], np.diag(subset)[1:])
    # Raising a diagonal adds a positive semi-definite matrix: no eigenvalue falls. The
    # duplicate's zero eigenvalue survives, because its direction has no weight on Govt.
    before, after = np.linalg.eigvalsh(subset), np.linalg.eigvalsh(kept.to_numpy())
    assert (after >= before - 1e-16).all() and abs(after[0]) < 1e-15
    # Without a floor or vector check the default drops only non-positive or missing variances.
    default_kept, _ = op.filter_covar_and_vectors_for_nans(panel)
    assert default_kept.index.tolist() == TICKERS
    assert np.array_equal(default_kept.to_numpy(), covar.to_numpy())
    # A missing off-diagonal entry is not a variance: filtering keeps the asset and the default
    # factorisation of the solver refuses the matrix.
    gap_panel = covar.copy()
    gap_panel.loc['Govt', 'Credit'] = gap_panel.loc['Credit', 'Govt'] = np.nan
    assert op.filter_covar_and_vectors_for_nans(gap_panel)[0].index.tolist() == TICKERS
    assert_raises(ValueError, op.wrapper_quadratic_optimisation, gap_panel,
                  op.Constraints(is_long_only=True))
    # Only the risk-budgeting wrapper passes a variance floor, of 0.001^2.
    for name in ('general.quadratic', 'general.max_sharpe', 'general.max_diversification',
                 'general.minimum_tracking_error', 'saa.min_variance_target_return',
                 'saa.max_return_target_vol', 'taa.maximise_alpha_over_tre',
                 'taa.maximise_alpha_with_target_yield', 'risk_allocation.risk_budgeting'):
        module = importlib.import_module(f'optimalportfolios.optimization.{name}')
        source = inspect.getsource(module)
        floors = source.count('variance_floor=')
        assert floors == (1 if name == 'risk_allocation.risk_budgeting' else 0)
        # Only the quadratic, maximum-Sharpe and two tactical wrappers drop non-finite vectors.
        drops = 'drop_non_finite_vectors=' in source
        assert drops == (name in ('general.quadratic', 'general.max_sharpe',
                                  'taa.maximise_alpha_over_tre',
                                  'taa.maximise_alpha_with_target_yield'))
    assert 'variance_floor=0.001**2' in inspect.getsource(
        importlib.import_module('optimalportfolios.optimization.risk_allocation.risk_budgeting'))

    config = op.OptimiserConfig(apply_total_to_good_ratio=False)
    weights, outcome = op.wrapper_quadratic_optimisation(
        covar, op.Constraints(is_long_only=True), optimiser_config=config, context='duplicate')
    print(weights.round(4).tolist())
    print(outcome.accepted, outcome.status, outcome.covar_factorization.n_eigenvalues_floored)

    # Proposition 2: the floored solve returns the pseudo-inverse portfolio, which splits the
    # duplicate equally, is interior, and has the minimum variance of the raw matrix.
    reference = pseudo_inverse_minimum_variance(covar.to_numpy())
    assert reference.min() > 0.04 and abs(reference[pair[0]] - reference[pair[1]]) < 1e-12
    np.testing.assert_allclose(weights, reference, rtol=0.0, atol=2e-6)
    assert weights.round(4).tolist() == MIN_VARIANCE_WEIGHTS
    assert abs(weights['Private A'] - weights['Private B']) < 1e-7
    raw = covar.to_numpy()
    np.testing.assert_allclose(weights @ raw @ weights, reference @ raw @ reference, rtol=1e-8)
    ones = np.ones(len(TICKERS))
    np.testing.assert_allclose(reference @ raw @ reference, 1 / (ones @ np.linalg.pinv(raw) @ ones),
                               rtol=1e-10)
    # The proof's step: 1 is orthogonal to v, so the floored inverse maps 1 to Sigma^+ 1.
    assert abs(direction @ ones) < 1e-15
    np.testing.assert_allclose(np.linalg.solve(floored.covar, ones), np.linalg.pinv(raw) @ ones,
                               rtol=1e-6)
    # Insight: both proxies hold 4.79%.
    assert round(100 * weights['Private A'], 2) == round(100 * weights['Private B'], 2) == 4.79
    assert outcome.accepted and outcome.status == 'optimal' and outcome.compliant
    assert outcome.solver == config.solver == 'CLARABEL' and outcome.context == 'duplicate'
    assert np.array_equal(outcome.covar_factorization.covar, floored.covar)
    assert outcome.fallback_source is None and outcome.reason == ''
    # The outcome's arrays refer to the filtered universe, the Series to the original one.
    padded = covar.reindex(index=TICKERS + ['Cash'], columns=TICKERS + ['Cash'], fill_value=0.0)
    padded_weights, padded_outcome = op.wrapper_quadratic_optimisation(
        padded, op.Constraints(is_long_only=True), optimiser_config=config)
    assert padded_weights.shape == (7,) and padded_weights['Cash'] == 0.0
    assert padded_outcome.weights.shape == (6,)
    assert padded_outcome.constraints.min_exposure == 1.0
    # Each CVXPY solver factorises once when factorize_covar is set, and every wrapper that takes
    # a labelled covariance filters it first.
    for name in ('general.quadratic', 'general.max_sharpe', 'general.minimum_tracking_error',
                 'saa.min_variance_target_return', 'saa.max_return_target_vol',
                 'taa.maximise_alpha_over_tre', 'taa.maximise_alpha_with_target_yield'):
        module = importlib.import_module(f'optimalportfolios.optimization.{name}')
        assert 'factorize_covariance(raw_covar) if factorize_covar else None' in (
            inspect.getsource(module))
        assert 'filter_covar_and_vectors_for_nans(' in inspect.getsource(module)
    for name in ('general.max_diversification', 'risk_allocation.risk_budgeting'):
        module = importlib.import_module(f'optimalportfolios.optimization.{name}')
        assert 'filter_covar_and_vectors_for_nans(' in inspect.getsource(module)
        assert 'factorize_covariance' not in inspect.getsource(module)
    # The six objects of the page are public, and the factorisation container is frozen and
    # checks its own reconstruction.
    for name in ('CovarianceFactorization', 'factorize_covariance', 'OptimizationOutcome',
                 'ConstraintResidual', 'evaluate_constraint_residuals',
                 'filter_covar_and_vectors_for_nans'):
        assert hasattr(op, name)
    assert op.CovarianceFactorization.__dataclass_params__.frozen
    assert 'reconstruct' in assert_raises(ValueError, op.CovarianceFactorization,
                                          covar=matrix, factor=2.0 * factor)
    # CVXPY's CLARABEL statuses, and a raised SolverError recorded as solver_error.
    from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
    assert CLARABEL.STATUS_MAP['AlmostSolved'] == 'optimal_inaccurate'
    assert CLARABEL.STATUS_MAP['AlmostPrimalInfeasible'] == 'infeasible_inaccurate'
    assert CLARABEL.STATUS_MAP['NumericalError'] == 'solver_error'

    def failing(problem, *args, **kwargs):
        """Raise the error CVXPY raises for a solver failure."""
        raise cvx.error.SolverError('numerical error')

    with replaced(cvx.Problem, solve=failing):
        _, failed = op.wrapper_quadratic_optimisation(distinct, op.Constraints(is_long_only=True),
                                                      optimiser_config=config)
    assert failed.status == 'solver_error' and not failed.accepted
    # The SciPy and risk-budgeting validators share the budget and box check and the fallback.
    from optimalportfolios.optimization import solver_diagnostics
    for validator in (solver_diagnostics.validate_scipy_solution,
                      solver_diagnostics.validate_rb_solution,
                      solver_diagnostics.validate_solution):
        source = inspect.getsource(validator)
        assert '_validate_weight_vector(' in source and '_compute_fallback(' in source
    # The legacy quad_form path, factorize_covar=False, stores no factorisation; here it returns
    # the same portfolio.
    legacy_weights, legacy = op.wrapper_quadratic_optimisation(
        covar, op.Constraints(is_long_only=True),
        optimiser_config=replace(config, factorize_covar=False))
    assert legacy.accepted and legacy.covar_factorization is None
    np.testing.assert_allclose(legacy_weights, weights, rtol=0.0, atol=2e-6)
    # solver and verbose reach problem.solve; verbose also sets SLSQP's disp in the SciPy paths.
    loud = replace(config, verbose=True)
    assert solve_arguments(loud, distinct) == [{'verbose': True, 'solver': 'CLARABEL'}]
    assert solve_arguments(config, distinct) == [{'verbose': False, 'solver': 'CLARABEL'}]
    assert slsqp_display(loud, distinct) is True and slsqp_display(config, distinct) is False

    print(outcome.residuals_frame()[['constraint_type', 'actual', 'violation', 'tolerance',
                                     'passed']].round(6))
    capped = op.Constraints(is_long_only=True, max_weights=pd.Series(0.5, index=TICKERS))
    audit = op.evaluate_constraint_residuals(weights.to_numpy(), capped, covar=covar.to_numpy())
    breaches = [(r.name, round(r.violation, 4)) for r in audit if r.hard and not r.passed]
    print(outcome.compliant, breaches)

    # The outcome audits exposure at 1e-4 and long-only at 1e-6; the independent audit of the
    # same weights against 50% caps finds only Govt, over by its weight minus 0.5.
    frame = outcome.residuals_frame()
    assert frame['constraint_type'].tolist() == ['exposure', 'long_only']
    assert frame['tolerance'].tolist() == [1e-4, 1e-6] and frame['passed'].all()
    np.testing.assert_allclose(frame['actual'], [weights.sum(), weights.min()], rtol=0.0,
                               atol=1e-12)
    assert frame['violation'].max() < 1e-12
    assert list(frame.columns) == ['constraint_type', 'name', 'actual', 'lower', 'upper',
                                   'violation', 'tolerance', 'hard', 'passed']
    assert all(isinstance(r, op.ConstraintResidual) for r in audit)
    assert breaches == [('Govt', round(weights['Govt'] - 0.5, 4))] == [('Govt', 0.1193)]
    cap_rows = [r for r in audit if r.constraint_type == 'instrument_weight']
    assert len(cap_rows) == len(TICKERS)
    for r in cap_rows:
        assert r.upper == 0.5 and r.tolerance == 1e-6
        assert r.violation == max(0.0, weights[r.name] - 0.5)
    # In the utility form the volatility, turnover and tracking-error rows are soft: they report
    # their violation and always pass; the group rows stay hard.
    equal_weights = pd.Series(1.0 / len(TICKERS), index=TICKERS)
    utility = op.Constraints(
        is_long_only=True, max_target_portfolio_vol_an=0.01, weights_0=equal_weights,
        turnover_constraint=0.01, benchmark_weights=equal_weights,
        tracking_err_vol_constraint=0.001,
        constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    soft = {r.constraint_type: r for r in op.evaluate_constraint_residuals(
        weights.to_numpy(), utility, covar=covar.to_numpy())}
    for kind in ('portfolio_volatility', 'turnover', 'tracking_error'):
        assert not soft[kind].hard and soft[kind].passed and soft[kind].violation > 0.0
    assert soft['exposure'].hard and soft['exposure'].tolerance == 1e-4
    np.testing.assert_allclose(soft['portfolio_volatility'].actual,
                               np.sqrt(weights @ raw @ weights), rtol=1e-12)
    np.testing.assert_allclose(soft['turnover'].actual,
                               np.abs(weights - equal_weights).sum(), rtol=1e-12)

    from optimalportfolios.optimization.solver_diagnostics import validate_solution
    blown_up = validate_solution(1.5e6 * weights.to_numpy(), 'optimal', outcome.constraints,
                                 n=len(TICKERS), covar_factorization=outcome.covar_factorization)
    imprecise = validate_solution(weights.to_numpy(), 'optimal_inaccurate', outcome.constraints,
                                  n=len(TICKERS), covar_factorization=outcome.covar_factorization)
    print(blown_up.accepted, blown_up.fallback_source, blown_up.reason)
    print(imprecise.accepted, imprecise.status)

    # A vector the solver calls optimal is rejected on its budget; without weights_0 or a
    # benchmark the fallback is zeros, and the zeros fail the budget residual.
    assert not blown_up.accepted and blown_up.fallback_source == 'zeros'
    assert blown_up.reason == 'budget violated: sum(w)=1.5e+06 vs target 1 (atol=0.0001)'
    assert np.array_equal(blown_up.weights, np.zeros(len(TICKERS))) and not blown_up.compliant
    assert imprecise.accepted and imprecise.status == 'optimal_inaccurate'
    np.testing.assert_array_equal(imprecise.weights, weights.to_numpy())

    def verdict(vector, status='optimal', **kwargs) -> bool:
        """Accept or reject ``vector`` under the minimum-variance outcome's constraints."""
        return validate_solution(vector, status, outcome.constraints, n=len(TICKERS),
                                 covar_factorization=outcome.covar_factorization,
                                 **kwargs).accepted

    base = weights.to_numpy()
    shifted = base.copy()
    # The acceptance table: status, budget at 1e-4, long-only at 1e-6, inaccurate on request.
    assert verdict(None) is False
    for status in ('infeasible', 'infeasible_inaccurate', 'unbounded', 'solver_error',
                   'user_limit', None):
        assert verdict(base, status) is False
    assert verdict(base, 'optimal_inaccurate', accept_inaccurate=False) is False
    assert verdict(base * (1 + 5e-5)) and not verdict(base * (1 + 2e-4))
    shifted[0] += 5e-7
    shifted[3] -= 5e-7 + base[3]  # Private A at -5e-7, the budget kept
    shifted[4] += base[3]
    assert verdict(shifted)
    shifted[0] += 1.5e-6
    shifted[3] -= 1.5e-6
    assert not verdict(shifted)
    nan_vector = base.copy()
    nan_vector[0] = np.nan
    assert not verdict(nan_vector) and not verdict(base[:5])
    # Box bounds at 1e-6 and an aggregate hard row, here a group cap, at 1e-4.
    govt = base[0]
    for slack, accepted in ((5e-7, True), (2e-6, False)):
        boxed = replace(outcome.constraints, max_weights=pd.Series(
            [govt - slack] + [1.0] * 5, index=TICKERS))
        assert validate_solution(base, 'optimal', boxed, n=len(TICKERS)).accepted is accepted
    private = base[pair].sum()
    for slack, accepted in ((5e-5, True), (2e-4, False)):
        grouped = replace(outcome.constraints, group_lower_upper_constraints=(
            op.GroupLowerUpperConstraints(
                group_loadings=pd.DataFrame({'Private': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]},
                                            index=TICKERS),
                group_min_allocation=None,
                group_max_allocation=pd.Series({'Private': private - slack}))))
        checked = validate_solution(base, 'optimal', grouped, n=len(TICKERS))
        assert checked.accepted is accepted
        assert accepted or checked.reason.startswith('hard constraint group_weight:Private')
    # Rejections without a solution log at WARNING, the others at ERROR.
    with captured('optimalportfolios.optimization.solver_diagnostics', 'solver_diag') as seen:
        verdict(None)
        verdict(base, 'infeasible')
        verdict(nan_vector)
        verdict(1.5e6 * base)
        verdict(base, 'optimal_inaccurate')
        verdict(base, 'optimal_inaccurate', accept_inaccurate=False)
        verdict(base)
    assert [(level, item.outcome) for level, item in seen] == [
        (logging.WARNING, 'rejected'), (logging.WARNING, 'rejected'),
        (logging.ERROR, 'rejected'), (logging.ERROR, 'rejected'),
        (logging.WARNING, 'accepted_inaccurate'), (logging.WARNING, 'rejected'),
        (logging.DEBUG, 'accepted')]

    caps = pd.Series(0.10, index=distinct.index)
    impossible = op.Constraints(is_long_only=True, max_weights=caps)
    prior = pd.Series(0.20, index=distinct.index)
    fallback, rejected = op.wrapper_quadratic_optimisation(
        distinct, impossible, weights_0=prior, optimiser_config=config, context='caps sum to 0.5')
    print(rejected.accepted, rejected.status, rejected.reason, rejected.fallback_source)
    print(rejected.compliant, rejected.residuals_frame().query('not passed')['violation'].tolist())
    sources = []
    for prior_weights, benchmark in ((prior, prior), (None, prior), (None, None)):
        _, attempt = op.wrapper_quadratic_optimisation(
            distinct, replace(impossible, benchmark_weights=benchmark), weights_0=prior_weights,
            optimiser_config=config)
        sources.append(attempt.fallback_source)
    print(sources)

    # Five caps of 0.10 sum to 0.5; CLARABEL reports infeasible with no solution, the prior
    # comes back unprojected, and each of its five 0.20 weights breaks its cap by 0.10.
    assert 5 * 0.10 < 1.0
    assert (rejected.accepted, rejected.status, rejected.reason, rejected.fallback_source) == (
        False, 'infeasible', 'w.value is None', 'weights_0')
    pd.testing.assert_series_equal(fallback, prior, check_names=False)
    violations = rejected.residuals_frame().query('not passed')['violation'].tolist()
    assert not rejected.compliant and violations == [0.1] * 5
    assert sources == ['weights_0', 'benchmark_weights', 'zeros']
    # Proposition 3: the elastic program of diagnose_infeasibility needs the caps to give 0.5 in
    # total, the shortfall 1 - 5 * 0.10, spread here as 0.10 on each cap.
    from optimalportfolios.optimization.solver_diagnostics import diagnose_infeasibility
    slack = diagnose_infeasibility(rejected.constraints)
    assert sorted(slack) == sorted(f'box_max:{name}' for name in distinct.index)
    np.testing.assert_allclose(sum(slack.values()), 1.0 - 5 * 0.10, rtol=0.0, atol=1e-8)
    # The same program solved independently by SciPy's linprog: variables (w, s), minimise sum s.
    from scipy.optimize import linprog
    eye = np.eye(5)
    elastic = linprog(np.r_[np.zeros(5), np.ones(5)], A_ub=np.hstack([eye, -eye]),
                      b_ub=np.full(5, 0.10), A_eq=np.r_[np.ones(5), np.zeros(5)][None, :],
                      b_eq=[1.0], bounds=[(0.0, None)] * 10)
    assert elastic.status == 0 and abs(elastic.fun - 0.5) < 1e-9
    # A non-finite prior is skipped in favour of the benchmark.
    from optimalportfolios.optimization.solver_diagnostics import _compute_fallback
    unusable = replace(rejected.constraints, weights_0=prior.where(prior.index != 'Govt'),
                       benchmark_weights=prior)
    assert _compute_fallback(unusable, list(distinct.index), 5)[1] == 'benchmark_weights'
    # validate_inputs and diagnose_infeasibility reach only the alpha-over-TE wrapper: its input
    # contract runs before each solve, and its diagnosis after a rejection, with the status.
    tactical = replace(impossible, benchmark_weights=prior, tracking_err_vol_constraint=0.05)
    contracts, diagnoses = alpha_over_tre_hooks(config, distinct, tactical)
    assert len(contracts) == 1 and diagnoses == ['infeasible']
    quiet = replace(config, validate_inputs=False, diagnose_infeasibility=False)
    assert alpha_over_tre_hooks(quiet, distinct, tactical) == ([], [])
    for name in ('general.quadratic', 'general.max_sharpe', 'general.max_diversification',
                 'general.minimum_tracking_error', 'general.carra_mixture',
                 'saa.min_variance_target_return', 'saa.max_return_target_vol',
                 'taa.maximise_alpha_with_target_yield', 'risk_allocation.risk_budgeting'):
        module = importlib.import_module(f'optimalportfolios.optimization.{name}')
        source = inspect.getsource(module)
        assert 'diagnose_infeasibility' not in source and 'validate_inputs' not in source
        assert 'max_constraint_relaxation' not in source
    # The input contract only logs: the structural finding is recorded and the solve still runs,
    # falling back to the benchmark that the wrapper injects.
    with captured('optimalportfolios.optimization.solver_diagnostics', 'input_contract') as seen:
        _, tactical_outcome = op.wrapper_maximise_alpha_over_tre(
            distinct, pd.Series(0.01, index=distinct.index), prior, tactical,
            optimiser_config=config)
    assert len(seen) == 1 and seen[0][1].structural[0].startswith('box caps sum to 0.5000')
    assert not tactical_outcome.accepted
    assert tactical_outcome.fallback_source == 'benchmark_weights'
    # The diagnosis sends an infeasible status to the elastic program and any other rejected
    # status to the conditioning report.
    diagnostics = importlib.import_module('optimalportfolios.optimization.solver_diagnostics')
    routed = []
    with replaced(diagnostics,
                  diagnose_infeasibility=lambda *args, **kwargs: routed.append('elastic'),
                  check_covar_conditioning=lambda *args, **kwargs: routed.append('conditioning')):
        diagnostics.diagnose_solver_failure('infeasible_inaccurate', rejected.constraints)
        diagnostics.diagnose_solver_failure('solver_error', rejected.constraints,
                                            covar=distinct.to_numpy())
    assert routed == ['elastic', 'conditioning']

    from optimalportfolios.optimization.solver_diagnostics import (
        DroppedGroupSummary, RelaxationSummary)
    book = ['Private equity', 'Equity', 'Bonds', 'Gold']
    book_covar = pd.DataFrame(np.diag([0.20, 0.16, 0.06, np.nan]) ** 2, index=book, columns=book)
    groups = op.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'Illiquid': [1.0, 0.0, 0.0, 0.0],
                                     'Commodities': [0.0, 0.0, 0.0, 1.0]}, index=book),
        group_min_allocation=None,
        group_max_allocation=pd.Series({'Illiquid': 0.20, 'Commodities': 0.10}))
    mandate = op.Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=book),
                             max_weights=pd.Series(1.0, index=book),
                             group_lower_upper_constraints=groups,
                             tracking_err_vol_constraint=0.05)
    records = logging.getLogger('optimalportfolios.optimization.constraints')
    relaxations, dropped = RelaxationSummary(), DroppedGroupSummary()
    records.setLevel(logging.DEBUG)
    records.addHandler(relaxations)
    records.addHandler(dropped)
    held, held_outcome = op.wrapper_maximise_alpha_over_tre(
        book_covar, pd.Series([0.0, 0.3, 0.1, 0.2], index=book),
        pd.Series([0.15, 0.45, 0.40, 0.0], index=book), mandate,
        weights_0=pd.Series([0.25, 0.40, 0.35, 0.0], index=book),
        rebalancing_indicators=pd.Series([0, 1, 1, 1], index=book),
        optimiser_config=op.OptimiserConfig(max_constraint_relaxation=0.02), context='frozen PE')
    records.removeHandler(relaxations)
    records.removeHandler(dropped)
    records.setLevel(logging.NOTSET)
    print(relaxations.records[0].items, relaxations.records[0].breached_tol)
    print(dropped.records[0].groups, held_outcome.accepted, held.round(4).tolist())

    # The frozen 0.25 overhangs the 0.20 cap by 0.05, so the cap rises to 0.25 + 1e-8 for this
    # solve; 0.05 exceeds 0.02, so the record is flagged. Gold's missing variance drops it and
    # leaves the Commodities group empty, which is recorded, not warned.
    record = relaxations.records[0]
    assert record.items == (('Illiquid', 'group_max', 0.20, 0.25000001),)
    assert abs(record.max_relaxation - 0.05) < 1e-7 and record.breached_tol
    assert not record.breached_budget and record.context == 'frozen PE'
    assert dropped.records[0].groups == ('Commodities',)
    assert not dropped.records[0].no_groups_remain
    assert held_outcome.accepted and held_outcome.compliant
    assert held.round(4).tolist() == [0.25, 0.7038, 0.0462, 0.0]
    aligned = held_outcome.constraints.group_lower_upper_constraints
    assert aligned.group_max_allocation.to_dict() == {'Illiquid': 0.25000001}
    frame = held_outcome.residuals_frame().set_index('constraint_type')
    assert frame.loc['tracking_error', 'actual'] <= 0.05 + 1e-4
    assert frame.loc['group_weight', 'upper'] == 0.25000001
    assert frame.loc['tracking_error', 'tolerance'] == 1e-4
    assert frame.loc['group_weight', 'tolerance'] == 1e-4
    # The severity ladder: ERROR above max_constraint_relaxation, INFO for a material waiver
    # without a limit, and a sub-material reconciliation at DEBUG with no record at all.
    states = []
    for current, tolerance in ((0.25, 0.02), (0.25, None), (0.20005, None)):
        with captured('optimalportfolios.optimization.constraints', 'relaxation') as seen:
            mandate.update_with_valid_tickers(
                valid_tickers=book, weights_0=pd.Series([current, 0.4, 0.6 - current, 0.0],
                                                        index=book),
                rebalancing_indicators=pd.Series([0, 1, 1, 1], index=book),
                max_relaxation_tol=tolerance)
        states.append([level for level, _ in seen])
    assert states == [[logging.ERROR], [logging.INFO], []]

    dates = pd.date_range('2024-03-31', periods=4, freq='QE')
    months = pd.date_range('2023-12-31', '2024-12-31', freq='ME')
    growth = [0.002, 0.004, 0.008, 0.006, 0.003]
    prices = pd.DataFrame(100 * np.exp(np.outer(np.arange(len(months)), growth)),
                          index=months, columns=distinct.index)
    yields = pd.DataFrame([[0.03, 0.045, 0.06, 0.07, 0.02]] * 4, index=dates,
                          columns=distinct.index)
    signals = pd.DataFrame([[0.010, 0.020, 0.030, 0.015, 0.025]] * 4, index=dates,
                           columns=distinct.index)
    rolling = op.rolling_maximise_alpha_with_target_return(
        prices, signals, yields, pd.Series([0.035, 0.035, 0.09, 0.035], index=dates),
        op.Constraints(is_long_only=True, max_weights=pd.Series(0.5, index=distinct.index)),
        {date: distinct for date in dates}, optimiser_config=config)
    for row in rolling.attrs['optimization_outcomes']:
        print(row['date'], row['accepted'], row['status'], row['fallback_source'], row['compliant'])

    # One record per rebalance with seven serialisable fields. The 9% target exceeds every
    # yield, so the third solve is rejected and returns the second portfolio drifted to its date,
    # which breaks the 50% cap and is therefore not compliant.
    log = rolling.attrs['optimization_outcomes']
    assert [row['date'] for row in log] == [str(date.date()) for date in dates]
    assert all(set(row) == {'date', 'accepted', 'status', 'solver', 'reason',
                            'fallback_source', 'compliant'} for row in log)
    assert [row['accepted'] for row in log] == [True, True, False, True]
    assert [row['compliant'] for row in log] == [True, True, False, True]
    assert (log[2]['status'], log[2]['fallback_source']) == ('infeasible', 'weights_0')
    assert max(yields.iloc[2]) < 0.09
    vertex = pd.Series([0.0, 0.0, 0.5, 0.0, 0.5], index=distinct.index)
    for position in (0, 1, 3):
        np.testing.assert_allclose(rolling.iloc[position], vertex, rtol=0.0, atol=1e-6)
    expected = drifted(rolling.iloc[1], prices, dates[1], dates[2])
    np.testing.assert_allclose(rolling.iloc[2], expected, rtol=0.0, atol=1e-12)
    assert rolling.iloc[2]['Equity'] > 0.5
    # Other rolling functions return plain weight tables.
    plain = op.rolling_quadratic_optimisation(prices, op.Constraints(is_long_only=True),
                                              {date: distinct for date in dates},
                                              optimiser_config=config)
    assert 'optimization_outcomes' not in plain.attrs

    # Pitfall: the default solve refuses a materially indefinite covariance with ValueError
    # before any fallback, while the legacy quad_form path accepts it as if it were convex.
    indefinite = pd.DataFrame(covar.to_numpy() - 1e-6 * np.outer(null, null), index=TICKERS,
                              columns=TICKERS)
    message = assert_raises(ValueError, op.wrapper_quadratic_optimisation, indefinite,
                            op.Constraints(is_long_only=True), optimiser_config=config)
    assert message.startswith('covar is materially indefinite')
    assert assert_raises(ValueError, op.wrapper_minimise_tracking_error, indefinite,
                         pd.Series(1.0 / len(TICKERS), index=TICKERS),
                         op.Constraints(is_long_only=True)).startswith('covar is materially')
    _, unchecked = op.wrapper_quadratic_optimisation(
        indefinite, op.Constraints(is_long_only=True),
        optimiser_config=replace(config, factorize_covar=False))
    assert unchecked.accepted and unchecked.status == 'optimal'
    # Proposition 2 needs zero-sum riskless directions: two assets at correlation one with
    # volatilities 12% and 15% combine into the riskless long-short portfolio (5, -4).
    unequal = covariance([0.12, 0.15], [[1.0, 1.0], [1.0, 1.0]], ['A', 'B']).to_numpy()
    riskless = np.array([5.0, -4.0])
    assert riskless.sum() == 1.0 and abs(riskless @ unequal @ riskless) < 1e-15
    # configure_run_logging(attach_summary=True) returns a RunDiagnostics that owns the five
    # tallies; its fallback gate defaults to 5%.
    root = logging.getLogger()
    levels = {name: logging.getLogger(name).level for name in (
        '', 'optimalportfolios.optimization.solver_diagnostics',
        'optimalportfolios.optimization.constraints')}
    bundle = diagnostics.configure_run_logging(console_level=logging.CRITICAL,
                                               capture_warnings=False, attach_summary=True)
    try:
        assert isinstance(bundle, diagnostics.RunDiagnostics)
        assert isinstance(bundle.rejections, diagnostics.SolverRejectionSummary)
        assert isinstance(bundle.relaxations, RelaxationSummary)
        assert isinstance(bundle.dropped_groups, DroppedGroupSummary)
        assert isinstance(bundle.contract, diagnostics.InputContractSummary)
        assert isinstance(bundle.warnings_summary, diagnostics.WarningSummary)
        assert isinstance(bundle.to_frame(), pd.DataFrame) and isinstance(bundle.summary(), str)
    finally:
        bundle.close()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
    assert root.level == levels['']
    gate = inspect.signature(diagnostics.RunDiagnostics.check_fallback_gate).parameters
    assert gate['max_fraction'].default == 0.05
    # The six configuration fields this page owns, with their defaults.
    default = op.OptimiserConfig()
    assert (default.solver, default.verbose, default.diagnose_infeasibility,
            default.validate_inputs, default.max_constraint_relaxation,
            default.factorize_covar) == ('CLARABEL', False, True, True, None, True)
    print('solver_numerics_and_outcomes: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the spectrum before and after flooring, and conditioning by gap.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spectrum, conditioning = spectrum_and_conditioning()
    cap = conditioning['max_eigenvalue'] / FLOOR
    above = spectrum['raw'] > FLOOR
    table = pd.concat([
        spectrum.reset_index().rename(columns={'rank': 'x'}).assign(panel='spectrum'),
        conditioning[['raw', 'floored']].reset_index().rename(columns={'gap': 'x'}).assign(
            panel='condition_number'),
    ], ignore_index=True)[['panel', 'x', 'raw', 'floored']]

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    ranks = spectrum.index.to_numpy()
    left.scatter(ranks, spectrum['raw'], s=40, color=blue, label='input', zorder=4)
    left.scatter(ranks, spectrum['floored'], s=150, facecolors='none', edgecolors=orange,
                 linewidths=2.0, label='after the floor', zorder=3)
    left.axhline(FLOOR, color=muted, linestyle='--', linewidth=1.2)
    left.text(1.0, FLOOR * 2.5, 'floor 1e-10', ha='left', va='bottom', color=ink, fontsize=10)
    smallest = spectrum['raw'].iloc[-1]
    left.annotate('', xy=(ranks[-1], FLOOR * 0.6), xytext=(ranks[-1], smallest * 1.8),
                  arrowprops={'arrowstyle': '->', 'color': muted, 'linewidth': 1.2})
    left.text(ranks[-1] - 0.15, smallest, f'raised from {smallest:.0e}', ha='right',
              va='center', color=ink, fontsize=10)
    left.set_yscale('log')
    left.set_ylim(1e-16, 1.0)
    left.set_xlim(0.5, len(ranks) + 0.5)
    left.set_xticks(ranks)
    left.set_xlabel('Eigenvalue rank')
    left.set_ylabel('Eigenvalue (annual variance)')
    left.set_title('Eigenvalues before and after the floor', loc='left', color=ink)
    left.legend(frameon=False, loc='lower left', fontsize=10, labelcolor=ink)

    gaps = conditioning.index.to_numpy()
    right.plot(gaps, conditioning['raw'], color=blue, marker='o', markersize=5, linewidth=1.6,
               label='input')
    right.plot(gaps, conditioning['floored'], color=orange, marker='s', markersize=4,
               linewidth=1.6, label='after the floor')
    right.axhline(cap.iloc[-1], color=muted, linestyle='--', linewidth=1.2)
    right.text(gaps[0], cap.iloc[-1] * 2.2, 'largest eigenvalue / floor', ha='left',
               va='bottom', color=ink, fontsize=10)
    right.text(1e-9, cap.iloc[-1] / 6.0, 'the floor binds', ha='left', va='top', color=ink,
               fontsize=10)
    right.set_xscale('log')
    right.set_yscale('log')
    right.invert_xaxis()
    right.set_ylim(10.0, 1e16)
    right.set_xlabel('1 - correlation of the private pair')
    right.set_ylabel('Condition number')
    right.set_title('Condition number as correlation nears 1', loc='left', color=ink)
    right.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=10,
                 labelcolor=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    unfloored = conditioning['raw_min_eigenvalue'] > FLOOR
    checks = {
        'floored_spectrum_at_or_above_floor': bool(
            (spectrum['floored'] >= FLOOR * (1 - 1e-6)).all()),
        'eigenvalues_above_floor_unchanged': bool(np.allclose(
            spectrum.loc[above, 'floored'], spectrum.loc[above, 'raw'], rtol=1e-10, atol=0.0)),
        'one_eigenvalue_raised': bool((~above).sum() == 1),
        'floored_condition_number_capped': bool(
            (conditioning['floored'] <= cap * (1 + 1e-9)).all()),
        'conditioning_unchanged_above_floor': bool(np.allclose(
            conditioning.loc[unfloored, 'floored'], conditioning.loc[unfloored, 'raw'],
            rtol=1e-12, atol=0.0)),
        'raw_condition_number_exceeds_1e12': bool(conditioning['raw'].iloc[-1] > 1e12),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
