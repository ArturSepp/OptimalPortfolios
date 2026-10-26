"""Run Yahoo factors, FM/FL covariance, the OP backtest and QIS reports."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from examples.oss_portfolio_pipeline.attribution import run_attribution, save_attribution
from examples.oss_portfolio_pipeline.backtest import BacktestConfig, run_backtest, save_backtest
from examples.oss_portfolio_pipeline.config import PipelineConfig
from examples.oss_portfolio_pipeline.covariance import estimate_covariances, save_covariances
from examples.oss_portfolio_pipeline.data import default_cache_directory
from examples.oss_portfolio_pipeline.diagnostics import (
    prepare_investment_diagnostics, save_investment_reports,
)
from examples.oss_portfolio_pipeline.factsheets import save_native_factsheets
from examples.oss_portfolio_pipeline.pipeline import run_pipeline, save_inputs
from examples.oss_portfolio_pipeline.reporting import save_reports
from examples.oss_portfolio_pipeline.stress import build_stress_study, save_stress_report


def main() -> None:
    """Build the data bundle, using an exclusive end to omit the current incomplete session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', default=datetime.now(timezone.utc).date().isoformat(),
                        help='Exclusive Yahoo end date (default: today in UTC)')
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--refresh', action='store_true', help='Refresh the matching Yahoo cache')
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--inputs-only', action='store_true', help='Stop before covariance fitting')
    stages.add_argument('--risk-only', action='store_true',
                        help='Stop before portfolio optimisation')
    parser.add_argument('--no-reports', action='store_true',
                        help='Save the backtest tables without rendering QIS reports')
    parser.add_argument('--no-attribution', action='store_true',
                        help='Skip standalone risk/signal layers and model attribution')
    parser.add_argument('--feature-attribution', action='store_true',
                        help='Rerun four combinations of doubled momentum span and 50%% larger TE')
    parser.add_argument('--stress-report', action='store_true',
                        help='Stress the last optimal target book with its dated FM/FL risk model')
    parser.add_argument('--solver', choices=('MOSEK', 'CLARABEL'), default='MOSEK',
                        help='Production uses MOSEK; CLARABEL is an explicit public alternative')
    parser.add_argument('--tracking-error', type=float, default=0.03,
                        help='Annual ex-ante TE limit in fractional units (default: 0.03)')
    parser.add_argument('--transaction-cost-bps', type=float, default=5.,
                        help='Cost per bought/sold notional, for both portfolios (default: 5 bp)')
    parser.add_argument('--implementation-lag', type=int, default=0,
                        help='Price observations from decision to execution (default: same close)')
    args = parser.parse_args()
    if args.stress_report and (args.no_reports or args.inputs_only or args.risk_only):
        parser.error('--stress-report requires the full backtest with reporting enabled')
    if args.feature_attribution and (args.no_attribution or args.inputs_only or args.risk_only):
        parser.error('--feature-attribution requires the full backtest and attribution stages')
    config = PipelineConfig(end=args.end)
    settings = BacktestConfig(solver=args.solver, tracking_error=args.tracking_error,
                              transaction_cost=args.transaction_cost_bps / 10000.,
                              implementation_lag=args.implementation_lag)
    inputs = run_pipeline(config, args.cache_dir, args.refresh)
    output_root = (Path(os.environ['AGENT_LOCAL_ROOT']) / 'outputs' / 'oss_portfolio_pipeline'
                   if os.environ.get('AGENT_LOCAL_ROOT')
                   else default_cache_directory().parent / 'outputs' / 'oss_portfolio_pipeline')
    output = args.output_dir or output_root / args.end
    path = save_inputs(inputs, config, output)
    print(f'Universe: {len(inputs.universe)} ETFs; factors: {inputs.factor_navs.shape[1]}')
    print(f'Factor history: {inputs.factor_navs.index[0].date()} '
          f'to {inputs.factor_navs.index[-1].date()}')
    print(f'Benchmark: {inputs.benchmark_weights.sum():.0%}; '
          f'{len(inputs.benchmark_targets)} monthly targets including inception')
    if not args.inputs_only:
        print(f'Estimating ME covariances with {args.solver}...', flush=True)
        risk = estimate_covariances(inputs, config, solver=args.solver)
        save_covariances(risk, path)
        print(f'Covariance updates: {len(risk.schedule)}; '
              f'{risk.schedule.index[0].date()} to {risk.schedule.index[-1].date()}')
        if not args.risk_only:
            print('Optimising and executing the monthly active portfolio...', flush=True)
            result = run_backtest(inputs, risk, config, settings)
            save_backtest(result, path)
            print(f'Accepted portfolio decisions: {len(result.executed.targets)}; '
                  f'max ex-ante TE: {result.diagnostics.ex_ante_tracking_error.max():.2%}')
            if not args.no_reports:
                print('Rendering QIS performance and portfolio reports...', flush=True)
                report = save_reports(inputs, result, path)
                print(f'QIS report: {report}')
            if args.stress_report:
                print('Stress testing the last optimal target allocation...', flush=True)
                stress = build_stress_study(inputs, risk, result)
                artifacts = save_stress_report(stress, path)
                print(f'QIS stress report: {artifacts.pdf_path}')
            alternatives = None
            if not args.no_attribution:
                print('Backtesting standalone layers and computing QIS attribution...', flush=True)
                study = run_attribution(inputs, risk, config, result,
                                        include_features=args.feature_attribution)
                attribution_path = save_attribution(study, path, render=not args.no_reports)
                print(f'Model attribution: {attribution_path}')
                alternatives = [run.full for run in study.scenarios.values()]
            if not args.no_reports:
                print('Rendering native QIS factsheets and investment diagnostics...', flush=True)
                native = save_native_factsheets(inputs, result, path, alternatives)
                diagnostics = prepare_investment_diagnostics(inputs, risk, result, alternatives)
                investment = save_investment_reports(diagnostics, path)
                for name, report_path in {**native, **investment}.items():
                    print(f'{name}: {report_path}')
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
