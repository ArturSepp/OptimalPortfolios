"""
Solve the 8 paper-mandate optimisations and produce the efficient-frontier
figure (paper Exhibit 20 region).

Reads only ``paper_inputs.xlsx`` via the ``PaperInputs`` container.
No dependency on ``rosaa``.

Method
------
For each of the 8 benchmark mandates:
    max_w  alphaᵀ w
    s.t.   benchmark·(1−band) ≤ w ≤ benchmark·(1+band)
           ‖w − benchmark‖_Σ ≤ tracking_err_vol_constraint
           sum w = 1, w ≥ 0  (forced via Constraints)

Outputs
-------
- ``efficient_frontier`` figure saved via ``qis.save_fig``
- (optional) ``cma_portfolios.pdf`` PortfolioOptimisationResult report

Run:
    python run_optimisation.py --output-path figures/
    python run_optimisation.py --output-path figures/ --pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import qis as qis
from optimalportfolios import (
    Constraints, ConstraintEnforcementType,
    PortfolioOptimisationResult,
    plot_efficient_frontier,
    wrapper_maximise_alpha_over_tre,
)

from paper_inputs import PaperInputs
from universe import load_paper_assets_short


def solve_optimal_portfolio(benchmark_weights: pd.Series,
                            pd_covar: pd.DataFrame,
                            cmas: pd.Series,
                            band: float = 0.5,
                            tracking_err_vol_constraint: float = 0.015,
                            ) -> pd.Series:
    """Solve a single mandate's tracking-error-constrained alpha maximisation.

    Parameters
    ----------
    benchmark_weights : pd.Series
        Mandate benchmark weights, indexed by ticker.
    pd_covar : pd.DataFrame
        17 × 17 asset covariance (annualised).
    cmas : pd.Series
        Total CMAs (alphas) per asset.
    band : float
        Box constraint width (default 0.5 = ±50% of benchmark weight).
    tracking_err_vol_constraint : float
        Annualised tracking-error vol cap (default 1.5%).
    """
    constraints = Constraints(
        min_weights=(1.0 - band) * benchmark_weights,
        max_weights=(1.0 + band) * benchmark_weights,
        benchmark_weights=benchmark_weights,
        weights_0=benchmark_weights.rename('Current'),
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
        group_lower_upper_constraints=None,
        tracking_err_vol_constraint=tracking_err_vol_constraint,
        turnover_constraint=None,
    )
    weights = wrapper_maximise_alpha_over_tre(
        pd_covar=pd_covar,
        alphas=cmas,
        benchmark_weights=benchmark_weights,
        constraints=constraints,
        weights_0=None,
    ).rename('OptimalPortfolio')
    return weights


def run_optimisation(paper_inputs_xlsx: Path,
                     output_path: Path,
                     band: float = 0.5,
                     tracking_err_vol_constraint: float = 0.015,
                     produce_pdf: bool = False,
                     ) -> PortfolioOptimisationResult:
    """Solve all 8 mandates, build a PortfolioOptimisationResult, save the
    efficient-frontier figure and optionally the PDF report.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    inputs = PaperInputs.load(paper_inputs_xlsx)
    assets = load_paper_assets_short()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tickers = list(assets.index)
    pd_covar = inputs.y_covar.loc[tickers, tickers].copy()
    cmas = inputs.cma_metadata.loc[tickers, 'base_total_cma'].rename('alpha')

    benchmark_weights = inputs.benchmark_weights.loc[tickers].copy()
    asset_class = inputs.cma_metadata.loc[tickers, 'asset_class'].rename(
        assets.to_dict(), axis=0)

    # Solve each mandate
    optimal_weights: Dict[str, pd.Series] = {}
    for portfolio in benchmark_weights.columns:
        bw = benchmark_weights[portfolio]
        # Drop zero-weight assets from this mandate's optimisation
        nonzero_tickers = bw.index[bw > 0]
        if len(nonzero_tickers) < len(bw):
            bw_active = bw.loc[nonzero_tickers]
            pd_covar_active = pd_covar.loc[nonzero_tickers, nonzero_tickers]
            cmas_active = cmas.loc[nonzero_tickers]
        else:
            bw_active, pd_covar_active, cmas_active = bw, pd_covar, cmas

        w_active = solve_optimal_portfolio(
            benchmark_weights=bw_active,
            pd_covar=pd_covar_active,
            cmas=cmas_active,
            band=band,
            tracking_err_vol_constraint=tracking_err_vol_constraint,
        )
        # Pad back to full universe with 0 for inactive assets
        w_full = pd.Series(0.0, index=tickers)
        w_full.loc[nonzero_tickers] = w_active
        optimal_weights[portfolio] = w_full

    optimal_weights_df = pd.DataFrame.from_dict(
        optimal_weights, orient='columns').rename(assets.to_dict(), axis=0)
    print("\nOptimal weights:")
    print(optimal_weights_df.round(4))

    # Build PortfolioOptimisationResult
    covar_data_filtered = _build_covar_data(inputs, tickers, assets)
    result = PortfolioOptimisationResult(
        weights=optimal_weights_df,
        benchmark_weights=benchmark_weights.rename(assets.to_dict(), axis=0),
        covar_data=covar_data_filtered,
        group_attributions={'asset_class': asset_class},
        current_weights=None,
        expected_return=cmas.rename(assets.to_dict(), axis=0),
    )

    # Risk + group-attribution summaries (printed for inspection)
    risk_summary = result.compute_risk_summary()
    group_attrib = result.compute_group_attribution()
    print("\nRisk summary:")
    if isinstance(risk_summary, dict):
        for k, v in risk_summary.items():
            print(f"--- {k} ---")
            print(v.round(4) if hasattr(v, 'round') else v)
    else:
        print(risk_summary.round(4))
    print("\nGroup attribution:")
    if isinstance(group_attrib, dict):
        for k, v in group_attrib.items():
            print(f"--- {k} ---")
            print(v.round(4) if hasattr(v, 'round') else v)
    else:
        print(group_attrib.round(4))

    # Efficient frontier figure
    profiles = {
        'w/o Alts':  ['Income w/o Alts',  'Low w/o Alts',
                      'Balanced w/o Alts', 'Growth w/o Alts'],
        'with Alts': ['Income with Alts', 'Low with Alts',
                      'Balanced with Alts', 'Growth with Alts'],
    }
    with sns.axes_style('darkgrid'):
        fig = plot_efficient_frontier(result, profiles=profiles, fontsize=12)
    qis.save_fig(fig=fig, file_name='efficient_frontier',
                 local_path=str(output_path))

    # Optional PDF report
    if produce_pdf:
        try:
            from optimalportfolios.reports.portfolio_result_pybloqs import (
                generate_portfolio_report,
            )
            file_path = qis.get_local_file_path(
                file_name='cma_portfolios',
                file_type=qis.FileTypes.PDF,
                local_path=str(output_path),
                add_current_date=True,
            )
            report = generate_portfolio_report(
                result=result, report_name='cma_portfolios')
            report.save(file_path)
            print(f"PDF report: {file_path}")
        except ImportError as exc:
            print(f"[skip] PDF report not generated: {exc}")

    return result


def _build_covar_data(inputs: PaperInputs,
                      tickers: list[str],
                      assets: pd.Series):
    """Construct the CurrentFactorCovarData object that
    PortfolioOptimisationResult expects, with the asset axis already
    renamed to the paper's display names.
    """
    from factorlasso import CurrentFactorCovarData

    name_map = assets.to_dict()
    return CurrentFactorCovarData(
        x_covar=inputs.x_covar.copy(),
        y_betas=inputs.y_betas.loc[tickers].rename(name_map, axis=0).copy(),
        y_variances=inputs.y_variances.loc[tickers].rename(name_map, axis=0).copy(),
    )


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-inputs-xlsx",
                   default=Path("data") / "paper_inputs.xlsx",
                   type=Path)
    p.add_argument("--output-path", default=Path("figures"), type=Path)
    p.add_argument("--band", type=float, default=0.5,
                   help="Box-constraint width (default 0.5 = ±50%% of benchmark)")
    p.add_argument("--te-vol", type=float, default=0.015,
                   help="Tracking-error vol constraint, annualised (default 1.5%%)")
    p.add_argument("--pdf", action='store_true',
                   help="Also produce cma_portfolios.pdf via pybloqs")
    args = p.parse_args()

    run_optimisation(
        paper_inputs_xlsx=args.paper_inputs_xlsx,
        output_path=args.output_path,
        band=args.band,
        tracking_err_vol_constraint=args.te_vol,
        produce_pdf=args.pdf,
    )


if __name__ == "__main__":
    _cli()
