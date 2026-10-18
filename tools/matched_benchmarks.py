"""Offline matched-convention B0 benchmarks; no production algorithms or figures.

Reuse docs_analytics guards/provenance and the existing seeded OP simulator.
Competitors are optional maintainer tools. See README_matched_benchmarks.md.
"""
# ruff: noqa: E402
# The import-cost timestamp intentionally precedes scientific-library imports.
from time import perf_counter
_IMPORT_STARTED = perf_counter()

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.docs_analytics.validate import (
    digest, environment_identity, environment_snapshot, file_record, offline, output_boundary,
    source_fingerprint, write_json,
)
from tools.matched_benchmark_references import (
    factor_oracle, factorlasso_covariance, forecast_scores, ledger_reference, verify_optimizer,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261017
SKFOLIO_VERSION = '1.9.3'
SOLVER_OPTIONS = {'tol_gap_abs': 1e-10, 'tol_gap_rel': 1e-10,
                  'tol_feas': 1e-10, 'max_iter': 500}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')


def array_hash(*arrays):
    """Fingerprint shapes, dtype and full numeric bytes, including missing-price locations."""
    digest = hashlib.sha256()
    for array in arrays:
        if isinstance(array, pd.DataFrame):
            digest.update(json.dumps([list(map(str, array.index)),
                                      list(map(str, array.columns))]).encode())
        values = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def peak_rss():
    """Return process-lifetime peak resident bytes, including native solver allocations."""
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            """Native PROCESS_MEMORY_COUNTERS layout on 32/64-bit Windows."""
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in
                ('peak', 'working', 'qpp', 'qp', 'qnp', 'qn', 'page', 'peakpage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not function(get_process(), ctypes.byref(counters), counters.cb):
            raise OSError('Cannot measure process peak working set')
        return int(counters.peak)
    import resource
    multiplier = 1 if sys.platform == 'darwin' else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)


def hardware():
    """Record machine identity and CPU/thread configuration without a benchmark threshold."""
    processor = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            processor = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    memory_bytes = None
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            """Native MEMORYSTATUSEX physical-memory layout."""
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page', 'available_page', 'virtual',
                 'available_virtual', 'extended')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError('Cannot read physical RAM')
        memory_bytes = int(memory.total)
    elif hasattr(os, 'sysconf'):
        memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = []
    return {'native_threadpools': pools, 'total_ram_bytes': memory_bytes,
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': processor,
            'logical_cpus': os.cpu_count(),
            'thread_limits': {name: os.environ.get(name) for name in THREADS}}


@contextmanager
def matched_solver():
    """Supply and capture identical tolerances at the common CVXPY solve boundary.

    OP's public solver has no solver-options argument. This benchmark-only instrumentation
    supplies the same options without editing its implementation, API or objective.
    """
    import cvxpy as cp
    original = cp.Problem.solve
    records = []

    def solve(problem, *args, **kwargs):
        """Reject backend drift and retain native solve status/options."""
        assert kwargs.get('solver') == 'CLARABEL'
        kwargs.update(SOLVER_OPTIONS)
        result = original(problem, *args, **kwargs)
        records.append({'solver': 'CLARABEL', 'status': problem.status,
                        'options': dict(SOLVER_OPTIONS),
                        'solve_seconds': float(problem.solver_stats.solve_time or 0)})
        assert problem.status == 'optimal'
        return result

    with patch.object(cp.Problem, 'solve', solve):
        yield records


def optimizer_inputs(n, objective):
    """Build a well-conditioned SPD interior problem with fixed supplied moments."""
    covariance = np.diag(np.linspace(0.01, 0.03, n)) + 0.002 * np.ones((n, n))
    means = np.zeros(n) if objective == 'minimum_variance' else np.linspace(.001, .003, n) / n
    gamma = 2.0 if objective == 'minimum_variance' else 3.0
    sample = np.random.default_rng(SEED).normal(0, .01, (252, n))
    return covariance, means, gamma, 2 / n, sample


def fit_optimizer(n, method, objective, inputs=None):
    """Solve the same explicitly supplied objective/moments through each public entry point."""
    covariance, means, gamma, cap, sample = inputs or optimizer_inputs(n, objective)
    algorithm_started = perf_counter()
    if method == 'optimalportfolios':
        import optimalportfolios as op
        outcome = op.cvx_quadratic_optimisation(
            portfolio_objective=(op.PortfolioObjective.MIN_VARIANCE
                                 if objective == 'minimum_variance'
                                 else op.PortfolioObjective.QUADRATIC_UTILITY),
            covar=covariance, means=means, carra=gamma,
            constraints=op.Constraints(is_long_only=True, max_weights=pd.Series(cap,
                                                                             index=range(n))),
            solver='CLARABEL', factorize_covar=False, context='B0 matched moments')
        assert outcome.accepted and outcome.compliant and outcome.fallback_source is None
        weights = outcome.weights
    else:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, ObjectiveFunction
        from skfolio.prior import BasePrior, ReturnDistribution

        class SuppliedPrior(BasePrior):
            """Benchmark-only fixed prior; does not estimate or transform supplied moments."""
            def __init__(self, mu, covariance):
                """Keep constructor arguments unchanged for sklearn cloning."""
                self.mu, self.covariance = mu, covariance

            def fit(self, X, y=None, **fit_params):
                """Retain the exact shared matrices and supplied finite scenario rows."""
                self.n_features_in_ = X.shape[1]
                self.return_distribution_ = ReturnDistribution(
                    mu=self.mu.copy(), covariance=self.covariance.copy(),
                    returns=np.asarray(X).copy())
                return self

        model = MeanRisk(
            objective_function=(ObjectiveFunction.MINIMIZE_RISK
                                if objective == 'minimum_variance'
                                else ObjectiveFunction.MAXIMIZE_UTILITY),
            risk_measure=RiskMeasure.VARIANCE, risk_aversion=gamma / 2,
            prior_estimator=SuppliedPrior(means, covariance), min_weights=0, max_weights=cap,
            budget=1, transaction_costs=0, management_fees=0,
            solver='CLARABEL', solver_params=dict(SOLVER_OPTIONS),
            scale_objective=1.0, scale_constraints=1.0, save_problem=True).fit(sample)
        fitted = model.prior_estimator_.return_distribution_
        np.testing.assert_array_equal(fitted.mu, means)
        np.testing.assert_array_equal(fitted.covariance, covariance)
        weights = model.weights_
    fit_seconds = perf_counter() - algorithm_started
    checks = verify_optimizer(weights, covariance, means, gamma, cap)
    return {**checks, 'fit_seconds': fit_seconds, 'weights': weights.tolist(),
            'input_hash': array_hash(covariance, means, sample), 'cap': cap, 'gamma': gamma}


def factor_inputs(n):
    """Reuse the seeded OP simulator and reserve a chronological 252-observation holdout."""
    from examples.covar_estimation.simulate_factor_returns import simulate_factor_model_returns
    before = np.random.get_state()
    try:
        data = simulate_factor_model_returns(n_assets=n, n_factors=4,
                                            n_periods=756, seed=SEED, dt=1 / 260)
    finally:
        np.random.set_state(before)
    x, y = data['factor_returns'], data['asset_returns']
    assert np.isfinite(x).all().all() and np.isfinite(y).all().all()
    assert y.min().min() > -1
    return x.iloc[:504], y.iloc[:504], y.iloc[504:]


def fit_factor(n, method, inputs=None):
    """Fit the restricted OLS time-series problem, retaining native residual denominators."""
    factors, responses, heldout = inputs or factor_inputs(n)
    algorithm_started = perf_counter()
    if method == 'factorlasso':
        from factorlasso import LassoModel
        model = LassoModel(reg_lambda=0.0, span=None, demean=True,
                           warmup_period=None, solver='CLARABEL').fit(x=factors, y=responses)
        beta = model.coef_.to_numpy()
        covariance = factorlasso_covariance(model, factors)
        residual_ddof = 0
    else:
        from sklearn.linear_model import LinearRegression
        from skfolio.moments import EmpiricalCovariance
        from skfolio.prior import EmpiricalPrior, LoadingMatrixRegression, TimeSeriesFactorModel
        model = TimeSeriesFactorModel(
            loading_matrix_estimator=LoadingMatrixRegression(
                linear_regressor=LinearRegression(fit_intercept=True), n_jobs=1),
            factor_prior_estimator=EmpiricalPrior(
                covariance_estimator=EmpiricalCovariance(ddof=0, nearest=False))).fit(
                    responses, factors=factors)
        beta = model.loading_matrix_estimator_.loading_matrix_
        covariance = model.return_distribution_.covariance
        residual_ddof = 1
    fit_seconds = perf_counter() - algorithm_started
    expected_beta, expected_covariance = factor_oracle(factors, responses, residual_ddof)
    np.testing.assert_allclose(beta, expected_beta, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(covariance, expected_covariance, atol=2e-10, rtol=2e-5)
    assert len(heldout) == 252 and len(responses) == 504
    return {'input_hash': array_hash(factors, responses, heldout),
            'fit_seconds': fit_seconds, 'train_observations': 504, 'heldout_observations': 252,
            'factor_covariance_ddof': 0, 'residual_ddof': residual_ddof,
            'beta_max_error': float(np.max(np.abs(beta - expected_beta))),
            'covariance_max_error': float(np.max(np.abs(covariance - expected_covariance))),
            'covariance': covariance.tolist(), **forecast_scores(covariance, heldout)}


def native_portfolio(prices, targets, lag, cost, *, drift=True):
    """Evaluate explicitly segmented skfolio intervals; no execution engine is invented.

    A supplied lag shifts segment boundaries externally. Missing quotes are marked at the
    previous quote for returns, but Portfolio cannot enforce QIS's unpriced-position freeze.
    Its native per-observation transaction costs remain unchanged and are reported as such.
    """
    from skfolio import Portfolio
    returns = prices.ffill().pct_change(fill_method=None)
    positions = prices.index.get_indexer(targets.index) + lag
    values = pd.Series(100.0, index=prices.index)
    wealth, previous = 100.0, np.zeros(prices.shape[1])
    for i, position in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(prices) - 1
        block = returns.iloc[position + 1:end + 1]
        if block.empty:
            continue
        model = Portfolio(X=block, weights=targets.iloc[i].to_numpy(), previous_weights=previous,
                          transaction_costs=cost, weight_drift=drift, compounded=True)
        nav = wealth * np.cumprod(1 + model.returns)
        values.loc[block.index] = nav
        wealth = float(nav[-1])
        previous = model.ending_weights
    return values.to_numpy()


def accounting_cases(include_skfolio=True):
    """Compare six two-asset cases to QIS's independent ledger and native skfolio returns."""
    import qis
    prices = pd.DataFrame({'A': [100., 110., 121., 121., 133.1],
                           'B': [100., 90., 90., 99., 99.]},
                          index=pd.bdate_range('2024-01-01', periods=5))
    targets = pd.DataFrame([[.5, .5], [.25, .75]],
                           index=prices.index[[0, 2]], columns=prices.columns)
    specs = [('zero_cost_rebalance', 0, 0., False, False),
             ('buy_hold_drift', 0, 0., False, True),
             ('entry_trade_cost', 0, .01, False, True),
             ('rebalance_trade_cost', 0, .01, False, False),
             ('implementation_lag_one', 1, 0., False, False),
             ('unquoted_position_freeze', 0, 0., True, False)]
    records = []
    for name, lag, cost, frozen, hold in specs:
        panel, schedule = prices.copy(), targets.iloc[:1].copy() if hold else targets.copy()
        if frozen:
            panel.iloc[2, 1] = np.nan
        portfolio = qis.backtest_model_portfolio(
            prices=panel, weights=schedule, rebalancing_freq=None, initial_nav=100,
            rebalancing_costs=cost, weight_implementation_lag=lag,
            is_rebalanced_at_first_date=True, ticker=name)
        reference = ledger_reference(panel, schedule, lag, cost)
        np.testing.assert_allclose(portfolio.nav, reference['nav'], atol=1e-11, rtol=0)
        np.testing.assert_allclose(portfolio.units, reference['units'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(portfolio.realized_costs, reference['costs'], atol=1e-12)
        if name == 'zero_cost_rebalance':
            np.testing.assert_allclose(reference['nav'], [100, 100, 105.5, 113.4125, 116.05])
        if name == 'entry_trade_cost':
            np.testing.assert_allclose(reference['nav'], [99, 99, 104.5, 109, 115.05])
        record = {'case': name, 'lag': lag, 'cost_rate': cost,
                  'input_hash': array_hash(panel, schedule),
                  'qis_nav': reference['nav'].tolist(),
                  'qis_costs': reference['costs'].tolist(),
                  'ledger_source': reference['source'], 'ledger_passed': True}
        if include_skfolio:
            native = native_portfolio(panel, schedule, lag, cost)
            difference = float(np.max(np.abs(native - reference['nav'])))
            expected_match = cost == 0 and not frozen
            if expected_match:
                np.testing.assert_allclose(native, reference['nav'], atol=1e-11, rtol=0)
            else:
                assert difference > 1e-4
            record.update(skfolio_nav=native.tolist(), max_nav_difference=difference,
                          expected_match=expected_match,
                          semantics=('same no-cost fully-quoted intervals; lag supplied externally'
                                     if expected_match else 'native cost/freeze mismatch'))
        records.append(record)
    if include_skfolio:
        fixed = native_portfolio(prices, targets.iloc[:1], 0, 0., drift=False)
        drifted = records[1]['qis_nav']
        assert np.max(np.abs(fixed - drifted)) > .1
        records[1]['skfolio_fixed_weight_nav'] = fixed.tolist()
    return records


def worker(kind, method, n, repeats, objective):
    """Separate imports, preparation, first fit, warm samples and lifetime peak RSS."""
    import importlib
    for module in ('optimalportfolios', 'factorlasso', 'qis',
                   'skfolio.optimization', 'skfolio.prior', 'sklearn.linear_model',
                   'examples.covar_estimation.simulate_factor_returns'):
        importlib.import_module(module)
    import_seconds = perf_counter() - _IMPORT_STARTED
    started = perf_counter()
    inputs = optimizer_inputs(n, objective) if kind == 'optimizer' else factor_inputs(n)
    preparation = perf_counter() - started
    baseline_rss = peak_rss()
    call = (lambda: fit_optimizer(n, method, objective, inputs)) if kind == 'optimizer' else (
        lambda: fit_factor(n, method, inputs))
    samples = []
    with offline(), matched_solver() as solves:
        result = call()
        cold = result['fit_seconds']
        for _ in range(repeats):
            repeated = call()
            samples.append(repeated['fit_seconds'])
            assert repeated['input_hash'] == result['input_hash']
    assert not solves or all(item['status'] == 'optimal' for item in solves)
    return {'kind': kind, 'method': method, 'assets': n, 'objective': objective,
            'import_seconds': import_seconds, 'preparation_seconds': preparation,
            'first_fit_seconds': cold, 'warm_fit_seconds': samples,
            'warm_median_seconds': float(np.median(samples)),
            'warm_min_seconds': min(samples), 'warm_max_seconds': max(samples),
            'warm_q25_seconds': float(np.quantile(samples, .25)),
            'warm_q75_seconds': float(np.quantile(samples, .75)),
            'peak_rss_bytes': peak_rss(), 'pre_fit_peak_rss_bytes': baseline_rss,
            'peak_rss_scope': ('lifetime: imports, preparation, reference validation '
                               'and native arrays; not isolated fit allocations'),
            'hardware': hardware(), 'solver_records': solves, 'result': result}


def run_bundle(output, repeats=5, sizes=(10, 100)):
    """Write validated source/input/environment/runtime evidence; never overwrite a run."""
    if (repeats < 3 or any(n not in (10, 100) for n in sizes) or not sizes
            or len(set(sizes)) != len(sizes)):
        raise ValueError('Use at least three warm repeats and the reviewed 10/100 asset workloads')
    from importlib.metadata import version
    if version('skfolio') != SKFOLIO_VERSION:
        raise ValueError(f'Optional benchmark requires skfolio=={SKFOLIO_VERSION}')
    output = output_boundary(output, ROOT)
    output.mkdir(parents=True)
    source = benchmark_source()
    dependencies = {'optimalportfolios': 'optimalportfolios', 'qis': 'qis',
                    'factorlasso': 'factorlasso', 'numpy': 'numpy', 'pandas': 'pandas',
                    'scipy': 'scipy', 'cvxpy': 'cvxpy', 'clarabel': 'clarabel',
                    'skfolio': 'skfolio', 'sklearn': 'scikit-learn'}
    with offline():
        environment = environment_snapshot(dependencies)
        accounting = accounting_cases()
    records = []
    for n in sizes:
        jobs = [('optimizer', method, objective) for objective in
                ('minimum_variance', 'quadratic_utility') for method in
                ('optimalportfolios', 'skfolio')]
        jobs += [('factor', method, 'ols_time_series') for method in ('factorlasso', 'skfolio')]
        for kind, method, objective in jobs:
            destination = output / f'{kind}-{objective}-{method}-{n}.json'
            env = os.environ.copy()
            env.update({name: '1' for name in THREADS})
            started = perf_counter()
            command = [sys.executable, '-m', 'tools.matched_benchmarks', '--worker', kind,
                       '--method', method, '--assets', str(n), '--objective', objective,
                       '--repeats', str(repeats), '--worker-output', str(destination)]
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                     text=True, encoding='utf-8', errors='replace', timeout=1800)
            (output / (destination.stem + '.log')).write_text(process.stdout + process.stderr,
                                                             encoding='utf-8')
            if process.returncode:
                raise RuntimeError(f'Benchmark worker failed: {command}\n{process.stderr[-4000:]}')
            record = json.loads(destination.read_text(encoding='utf-8'))
            record['fresh_process_wall_seconds'] = perf_counter() - started
            records.append(record)
    for kind in ('optimizer', 'factor'):
        for n in sizes:
            for objective in {r['objective'] for r in records if r['kind'] == kind}:
                pair = [r for r in records if r['kind'] == kind and r['assets'] == n
                        and r['objective'] == objective]
                assert len(pair) == 2
                assert pair[0]['result']['input_hash'] == pair[1]['result']['input_hash']
                if kind == 'factor':
                    factors, responses, _ = factor_inputs(n)
                    _, cov0 = factor_oracle(factors, responses, 0)
                    _, cov1 = factor_oracle(factors, responses, 1)
                    np.testing.assert_allclose(
                        np.array(pair[1]['result']['covariance'])
                        - np.array(pair[0]['result']['covariance']), cov1 - cov0,
                        rtol=1e-4, atol=2e-10)
    if benchmark_source() != source:
        raise ValueError('Source changed during benchmark generation')
    if environment_identity(environment_snapshot(dependencies)) != environment_identity(
            environment):
        raise ValueError('Imported dependency source changed during benchmark generation')
    for n in sizes:
        directory = output / 'inputs' / str(n)
        directory.mkdir(parents=True)
        covariance, means, _, _, sample = optimizer_inputs(n, 'quadratic_utility')
        frames = {'optimizer_covariance': pd.DataFrame(covariance),
                  'optimizer_utility_means': pd.DataFrame({'mu': means}),
                  'optimizer_scenarios': pd.DataFrame(sample)}
        x, y, heldout = factor_inputs(n)
        frames.update(factor_train=x, response_train=y, response_heldout=heldout)
        for name, frame in frames.items():
            frame.rename_axis('row').to_csv(directory / (name + '.csv'))
    summary = pd.DataFrame([{key: r[key] for key in
                            ('kind', 'method', 'assets', 'objective', 'import_seconds',
                             'preparation_seconds', 'first_fit_seconds', 'warm_median_seconds',
                             'warm_min_seconds', 'warm_max_seconds', 'peak_rss_bytes',
                             'fresh_process_wall_seconds')} for r in records])
    summary.to_csv(output / 'runtime_summary.csv', index=False)
    pd.DataFrame([{**{k: r[k] for k in ('method', 'assets')},
                   **{k: v for k, v in r['result'].items()
                      if k not in ('weights', 'covariance')}} for r in records
                  if r['kind'] == 'factor']).to_csv(output / 'forecast_scores.csv', index=False)
    write_json(output / 'accounting.json', {'cases': accounting})
    write_json(output / 'bundle.json', {
        'schema_version': 1, 'stage': 'B0',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'verified', 'seed': SEED,
        'sizes': list(sizes), 'warm_repeats': repeats, 'source': source,
        'environment': environment, 'hardware': hardware(), 'solver_options': SOLVER_OPTIONS,
        'conventions': {'returns': 'synthetic simple returns, per observation, no annualisation',
                        'optimizer': ('min variance or mu*w-(gamma/2)*variance; '
                                      'sum=1, 0<=w<=2/N'),
                        'estimation': ('504 training rows, 252 untouched future rows; '
                                       '4 factors, OLS'),
                        'normalization': 'factor covariance ddof=0; FL residual ddof=0, skfolio=1',
                        'accounting': ('QIS pre-cost trade sizing and one-off cash costs; '
                                       'skfolio per-observation costs; external lag segmentation'),
                        'timing': 'one fresh process per case; first fit once, fresh objects warm',
                        'memory': 'absolute lifetime peak resident memory, not allocation count'},
        'not_assessed': ['cross-sectional characteristics engine', '1000/5000 asset workloads',
                         'universal timing thresholds', 'financial outperformance'],
        'records': records, 'outputs': {p.relative_to(output).as_posix(): content_record(p)
                                       for p in output.rglob('*')
                                        if p.is_file()},
    })
    verify_bundle(output)
    return output


def content_record(path):
    """Reuse the analytics CSV validator; hash Python/JSON/log bytes without a new format."""
    if path.suffix == '.csv':
        return file_record(path)
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def benchmark_source():
    """Extend existing source identity to the two tool modules and owned reference files."""
    from tools.matched_benchmark_references import source_root
    import factorlasso
    import qis
    record = source_fingerprint(ROOT)
    paths = [ROOT / 'tools/matched_benchmarks.py',
             ROOT / 'tools/matched_benchmark_references.py',
             source_root(factorlasso) / 'examples/docs/quickstart.py',
             source_root(qis)
             / 'src/qis/portfolio/tests/backtester_missing_prices_test.py']
    record['benchmark_and_owned_references'] = {str(p): content_record(p) for p in paths}
    record['source_versions'] = {}
    import optimalportfolios
    for name, package in [('qis', qis), ('factorlasso', factorlasso),
                          ('optimalportfolios', optimalportfolios)]:
        import re
        metadata = (source_root(package) / 'pyproject.toml').read_text(encoding='utf-8')
        record['source_versions'][name] = re.search(
            r'(?m)^version\s*=\s*["\']([^"\']+)', metadata)[1]
    return record


def verify_bundle(bundle):
    """Reject missing or modified results, including every numeric input snapshot."""
    manifest = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    assert manifest['stage'] == 'B0' and manifest['status'] == 'verified'
    for name, expected in manifest['outputs'].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError('Unsafe output path in benchmark manifest')
        if not path.is_file() or content_record(path) != expected:
            raise ValueError(f'Benchmark output changed: {name}')
    return manifest


def main(argv=None):
    """Run the public bundle command or one isolated worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 100])
    parser.add_argument('--worker', choices=['optimizer', 'factor'])
    parser.add_argument('--method')
    parser.add_argument('--assets', type=int)
    parser.add_argument('--objective')
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        verify_bundle(args.verify)
        print('Benchmark output hashes and CSV contracts passed')
    elif args.worker:
        with offline():
            result = worker(args.worker, args.method, args.assets,
                            args.repeats, args.objective)
        write_json(args.worker_output, result)
    elif args.output_root is None:
        parser.error('--output-root is required')
    else:
        print(run_bundle(args.output_root, args.repeats, tuple(args.sizes)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
