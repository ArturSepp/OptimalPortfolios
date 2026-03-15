"""
Local tests for LassoModel estimator.

Synthetic data examples demonstrating all three LassoModelType variants,
NaN masking for assets with different history lengths, sign constraints,
and regression diagnostics via LassoEstimationResult.

Convention:
    beta is (N x M) where N = assets, M = factors.
    estimated_betas DataFrame has index=asset_names, columns=factor_names.

Each test generates synthetic factor returns (x) and asset returns
(y = x @ true_betas.T + noise), fits a LassoModel, and prints estimated
betas vs true betas for visual verification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from enum import Enum

from optimalportfolios.lasso.lasso_estimator import (
    LassoModel,
    LassoModelType,
    LassoEstimationResult,
    get_x_y_np,
    solve_lasso_cvx_problem,
    solve_group_lasso_cvx_problem,
)


class LocalTests(Enum):
    LASSO_BASIC = 1
    LASSO_WITH_NANS = 2
    GROUP_LASSO_PREDEFINED = 3
    GROUP_LASSO_CLUSTERS_HCGL = 4
    SIGN_CONSTRAINTS = 5
    GET_X_Y_NP_STANDALONE = 6
    SOLVER_STANDALONE_WITH_NANS = 7


def generate_synthetic_data(
    n_dates: int = 200,
    n_factors: int = 3,
    n_assets: int = 6,
    noise_std: float = 0.02,
    seed: int = 42,
    nan_start_indices: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Generate synthetic factor and asset return data.

    Args:
        n_dates: Number of time periods.
        n_factors: Number of risk factors (M).
        n_assets: Number of assets (N).
        noise_std: Standard deviation of idiosyncratic noise.
        seed: Random seed for reproducibility.
        nan_start_indices: Dict mapping asset index -> first valid row index.
            Rows before this index are set to NaN (simulating shorter history).

    Returns:
        Tuple of:
            - x_prices: Factor prices (cumulative returns), DatetimeIndex.
            - y_prices: Asset prices (cumulative returns), DatetimeIndex.
            - true_betas: Ground truth factor loadings (N x M) = (n_assets x n_factors).
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range('2020-01-01', periods=n_dates, freq='W-WED')
    factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

    # Factor returns: mean ~0.1% weekly, vol ~2%
    x_returns = rng.normal(loc=0.001, scale=0.02, size=(n_dates, n_factors))
    x_df = pd.DataFrame(x_returns, index=dates, columns=factor_names)

    # True betas (N x M) = (6 assets x 3 factors)
    # Each row is one asset's loadings to the 3 factors
    true_betas = np.array([
        [0.8, 0.0, 0.3],   # Asset_1: loads on F1, F3
        [0.5, 0.6, 0.0],   # Asset_2: loads on F1, F2
        [0.0, 0.9, 0.0],   # Asset_3: loads on F2 only
        [1.2, 0.0, 0.4],   # Asset_4: loads on F1, F3
        [0.0, 0.7, 0.5],   # Asset_5: loads on F2, F3
        [0.3, 0.0, 0.8],   # Asset_6: loads on F1, F3
    ])

    # Asset returns: y = x @ beta' + noise
    # x is (T x M), beta' is (M x N), result is (T x N)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_dates, n_assets))
    y_returns = x_returns @ true_betas.T + noise
    y_df = pd.DataFrame(y_returns, index=dates, columns=asset_names)

    # Introduce NaN for assets with shorter history
    if nan_start_indices is not None:
        for asset_idx, start_row in nan_start_indices.items():
            y_df.iloc[:start_row, asset_idx] = np.nan

    # Convert to prices (cumulative returns starting at 100)
    x_prices = (1.0 + x_df).cumprod() * 100.0
    y_prices = (1.0 + y_df).cumprod() * 100.0
    if nan_start_indices is not None:
        for asset_idx, start_row in nan_start_indices.items():
            y_prices.iloc[:start_row, asset_idx] = np.nan

    return x_prices, y_prices, true_betas


def print_beta_comparison(true_betas: np.ndarray,
                          estimated_betas: pd.DataFrame,
                          title: str = '') -> None:
    """
    Print side-by-side comparison of true vs estimated betas.

    Args:
        true_betas: Ground truth (N x M).
        estimated_betas: DataFrame (N x M), index=assets, columns=factors.
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    true_df = pd.DataFrame(true_betas, index=estimated_betas.index, columns=estimated_betas.columns)
    print(f"\nTrue betas (N x M):\n{true_df.round(3)}")
    print(f"\nEstimated betas (N x M):\n{estimated_betas.round(3)}")
    print(f"\nAbsolute error:\n{(estimated_betas.values - true_betas).round(3)}")


def print_diagnostics(result: LassoEstimationResult,
                      asset_names: pd.Index,
                      title: str = 'Diagnostics') -> None:
    """Print in-sample diagnostics from LassoEstimationResult."""
    print(f"\n{title}:")
    print(f"  R²:             {pd.Series(result.r2, index=asset_names).round(3).to_dict()}")
    print(f"  Residual var:   {pd.Series(result.ss_res, index=asset_names).round(6).to_dict()}")
    print(f"  Alpha:          {pd.Series(result.alpha, index=asset_names).round(6).to_dict()}")
    print(f"  Total var:      {pd.Series(result.ss_total, index=asset_names).round(6).to_dict()}")


def run_local_test(local_test: LocalTests):
    """
    Run local tests for development and debugging purposes.

    These are integration tests using synthetic data to verify
    LassoModel estimation across all model types.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if local_test == LocalTests.LASSO_BASIC:
        """
        Basic LASSO: estimate sparse betas from synthetic data, no NaNs.
        Verifies that L1 regularisation recovers the true sparse structure
        and that estimation_result_ diagnostics are populated.
        """
        x_prices, y_prices, true_betas = generate_synthetic_data()

        x_returns = x_prices.pct_change().dropna()
        y_returns = y_prices.pct_change().dropna()
        common_idx = x_returns.index.intersection(y_returns.index)
        x_returns = x_returns.loc[common_idx]
        y_returns = y_returns.loc[common_idx]

        model = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-5,
            span=52,
            demean=True,
            warmup_period=12
        )
        model.fit(x=x_returns, y=y_returns, verbose=False)
        print_beta_comparison(true_betas, model.estimated_betas, title='LASSO_BASIC')

        # Diagnostics from estimation_result_
        assert model.estimation_result_ is not None, "estimation_result_ should be set after fit()"
        print_diagnostics(model.estimation_result_, y_returns.columns)

        # Verify estimation_result_ fields have correct shape (N x M) = (6 x 3)
        r = model.estimation_result_
        n_assets, n_factors = 6, 3
        assert r.estimated_beta.shape == (n_assets, n_factors), f"beta shape: {r.estimated_beta.shape}"
        assert r.alpha.shape == (n_assets,), f"alpha shape: {r.alpha.shape}"
        assert r.r2.shape == (n_assets,), f"r2 shape: {r.r2.shape}"
        assert r.ss_total.shape == (n_assets,), f"ss_total shape: {r.ss_total.shape}"
        assert r.ss_res.shape == (n_assets,), f"ss_res shape: {r.ss_res.shape}"
        print("\nPASS: estimation_result_ shape checks")

        # Verify estimated_betas DataFrame orientation
        assert list(model.estimated_betas.index) == list(y_returns.columns), "index should be assets"
        assert list(model.estimated_betas.columns) == list(x_returns.columns), "columns should be factors"
        print("PASS: estimated_betas DataFrame orientation (index=assets, columns=factors)")

    elif local_test == LocalTests.LASSO_WITH_NANS:
        """
        LASSO with NaN masking: Asset_4 starts at row 50, Asset_5 starts at row 100.
        Verifies that the valid_mask correctly excludes missing observations
        without discarding valid data for other assets, and that diagnostics
        for short-history assets are properly computed.
        """
        nan_starts = {3: 50, 4: 100}  # Asset_4 from row 50, Asset_5 from row 100
        x_prices, y_prices, true_betas = generate_synthetic_data(nan_start_indices=nan_starts)

        x_returns = x_prices.pct_change().dropna()
        y_returns = y_prices.pct_change().reindex(x_returns.index)  # keep NaNs in y

        model = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-5,
            span=52,
            demean=True,
            warmup_period=12
        )
        model.fit(x=x_returns, y=y_returns, verbose=False)
        print_beta_comparison(true_betas, model.estimated_betas, title='LASSO_WITH_NANS')

        # Show valid_mask summary
        valid_counts = model.valid_mask_.sum(axis=0).astype(int)
        print(f"\nValid observation counts per asset: {dict(zip(y_returns.columns, valid_counts))}")

        # Diagnostics from estimation_result_
        r = model.estimation_result_
        print_diagnostics(r, y_returns.columns)

        # Full-history assets should have valid diagnostics
        assert not np.isnan(r.r2[0]), "Asset_1 (full history) should have valid R²"
        # Short-history assets should still have diagnostics (if above warmup)
        assert not np.isnan(r.r2[3]), "Asset_4 (50 obs, above warmup=12) should have valid R²"
        print("\nPASS: NaN-masked diagnostics checks")

    elif local_test == LocalTests.GROUP_LASSO_PREDEFINED:
        """
        Group LASSO with predefined groups: assets are assigned to 2 groups.
        Group penalty encourages entire groups of betas to be zero together.
        """
        x_prices, y_prices, true_betas = generate_synthetic_data()

        x_returns = x_prices.pct_change().dropna()
        y_returns = y_prices.pct_change().dropna()
        common_idx = x_returns.index.intersection(y_returns.index)
        x_returns = x_returns.loc[common_idx]
        y_returns = y_returns.loc[common_idx]

        group_data = pd.Series([1, 1, 1, 2, 2, 2], index=y_returns.columns)

        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            group_data=group_data,
            reg_lambda=1e-5,
            span=52,
            demean=True,
            warmup_period=12
        )
        model.fit(x=x_returns, y=y_returns, verbose=False)
        print_beta_comparison(true_betas, model.estimated_betas, title='GROUP_LASSO_PREDEFINED')

        print_diagnostics(model.estimation_result_, y_returns.columns)
        print(f"\nGroup assignments:\n{group_data}")

        r = model.estimation_result_
        assert r.ss_total.shape == (6,), f"ss_total shape: {r.ss_total.shape}"
        print("PASS: Group LASSO estimation_result_ shape checks")

    elif local_test == LocalTests.GROUP_LASSO_CLUSTERS_HCGL:
        """
        HCGL: Group LASSO with hierarchical clustering.
        Clusters are automatically derived from the asset correlation matrix.
        """
        x_prices, y_prices, true_betas = generate_synthetic_data(
            n_dates=300, n_assets=6, noise_std=0.015
        )

        x_returns = x_prices.pct_change().dropna()
        y_returns = y_prices.pct_change().dropna()
        common_idx = x_returns.index.intersection(y_returns.index)
        x_returns = x_returns.loc[common_idx]
        y_returns = y_returns.loc[common_idx]

        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-5,
            span=52,
            demean=True,
            warmup_period=12
        )
        model.fit(x=x_returns, y=y_returns, verbose=False)
        print_beta_comparison(true_betas, model.estimated_betas, title='GROUP_LASSO_CLUSTERS (HCGL)')

        print_diagnostics(model.estimation_result_, y_returns.columns)
        print(f"\nDiscovered clusters:\n{model.clusters}")
        print(f"Cluster cutoff distance: {model.cutoff:.4f}")

    elif local_test == LocalTests.SIGN_CONSTRAINTS:
        """
        Group LASSO with per-element sign constraints.
        Verifies that sign constraints are satisfied in estimated betas.

        Sign constraint matrix is (N x M) = (assets x factors):
            Factor_1: all assets non-negative (1.0)
            Factor_2: Asset_1, Asset_4, Asset_6 forced zero (0.0); others free (NaN)
            Factor_3: all assets free (NaN)
        """
        x_prices, y_prices, true_betas = generate_synthetic_data()

        x_returns = x_prices.pct_change().dropna()
        y_returns = y_prices.pct_change().dropna()
        common_idx = x_returns.index.intersection(y_returns.index)
        x_returns = x_returns.loc[common_idx]
        y_returns = y_returns.loc[common_idx]

        # Signs matrix (N x M) = (assets x factors)
        signs = pd.DataFrame(
            [
                [1.0,  0.0,    np.nan],   # Asset_1: F1>=0, F2=0, F3 free
                [1.0,  np.nan, np.nan],   # Asset_2: F1>=0, F2 free, F3 free
                [1.0,  np.nan, np.nan],   # Asset_3: F1>=0, F2 free, F3 free
                [1.0,  0.0,    np.nan],   # Asset_4: F1>=0, F2=0, F3 free
                [1.0,  np.nan, np.nan],   # Asset_5: F1>=0, F2 free, F3 free
                [1.0,  0.0,    np.nan],   # Asset_6: F1>=0, F2=0, F3 free
            ],
            index=y_returns.columns,
            columns=x_returns.columns
        )

        group_data = pd.Series([1, 1, 1, 2, 2, 2], index=y_returns.columns)

        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            group_data=group_data,
            reg_lambda=1e-5,
            span=52,
            demean=True,
            warmup_period=12,
            factors_beta_loading_signs=signs
        )
        model.fit(x=x_returns, y=y_returns, verbose=False)
        print_beta_comparison(true_betas, model.estimated_betas, title='SIGN_CONSTRAINTS')

        # est is (N x M): rows=assets, columns=factors
        est = model.estimated_betas.values
        print(f"\nSign constraint matrix (N x M):\n{signs}")
        print(f"\nFactor_1 column (should be >= 0): {est[:, 0].round(4)}")
        print(f"Factor_2 at constrained-zero assets [Asset_1, Asset_4, Asset_6] (should be ~0): "
              f"{est[[0, 3, 5], 1].round(6)}")

        # Factor_1 (column 0): all assets non-negative
        assert np.all(est[:, 0] >= -1e-8), "Factor_1 sign constraint violated"
        # Factor_2 (column 1): Asset_1 (row 0), Asset_4 (row 3), Asset_6 (row 5) forced zero
        assert np.allclose(est[[0, 3, 5], 1], 0.0, atol=1e-6), "Factor_2 zero constraint violated"
        print("PASS: All sign constraints satisfied")

        print_diagnostics(model.estimation_result_, y_returns.columns)

    elif local_test == LocalTests.GET_X_Y_NP_STANDALONE:
        """
        Test get_x_y_np as a standalone function.
        Verifies index alignment assertion, NaN mask creation (including x all-NaN rows),
        and dimensionality after EWMA demeaning.
        """
        dates = pd.bdate_range('2020-01-01', periods=50, freq='W-WED')
        x = pd.DataFrame(
            np.random.randn(50, 2),
            index=dates,
            columns=['F1', 'F2']
        )
        y = pd.DataFrame(
            np.random.randn(50, 3),
            index=dates,
            columns=['A1', 'A2', 'A3']
        )
        y.iloc[:10, 1] = np.nan
        y.iloc[:20, 2] = np.nan

        print(f"\n{'='*60}")
        print(f" GET_X_Y_NP_STANDALONE")
        print(f"{'='*60}")
        print(f"\nInput shapes: x={x.shape}, y={y.shape}")
        print(f"NaN counts per asset: {y.isna().sum().to_dict()}")

        # EWMA demeaning (drops first row)
        x_np, y_np, valid_mask = get_x_y_np(x=x, y=y, span=26, demean=True)
        print(f"\nAfter get_x_y_np (span=26):")
        print(f"  x_np shape: {x_np.shape}  (T-1 due to EWMA demeaning)")
        print(f"  y_np shape: {y_np.shape}")
        print(f"  valid_mask shape: {valid_mask.shape}")
        print(f"  valid_mask sum per asset: {valid_mask.sum(axis=0).astype(int)}")
        print(f"  y_np has no NaNs: {not np.any(np.isnan(y_np))}")
        assert x_np.shape[0] == 49, "Should drop 1 row for EWMA demeaning"

        # Simple demeaning (no row drop)
        x_np2, y_np2, valid_mask2 = get_x_y_np(x=x, y=y, span=None, demean=True)
        print(f"\nAfter get_x_y_np (span=None, simple demeaning):")
        print(f"  x_np shape: {x_np2.shape}  (same T, no row drop)")
        print(f"  valid_mask sum per asset: {valid_mask2.sum(axis=0).astype(int)}")
        assert x_np2.shape[0] == 50, "No row drop for simple demeaning"

        # x all-NaN row masking
        x_with_nan_row = x.copy()
        x_with_nan_row.iloc[0, :] = np.nan
        x_np3, y_np3, valid_mask3 = get_x_y_np(x=x_with_nan_row, y=y, span=None, demean=True)
        print(f"\nAfter get_x_y_np with x all-NaN first row:")
        print(f"  valid_mask[0, :] (should be all 0): {valid_mask3[0, :]}")
        assert np.all(valid_mask3[0, :] == 0.0), "x all-NaN row should invalidate all assets"
        print("PASS: x all-NaN row correctly masked")

        # Index assertion
        print(f"\nTesting index mismatch assertion...")
        try:
            y_bad = y.iloc[:-5]
            get_x_y_np(x=x, y=y_bad)
            print("  ERROR: assertion should have fired!")
        except AssertionError as e:
            print(f"  Caught expected AssertionError: {e}")

    elif local_test == LocalTests.SOLVER_STANDALONE_WITH_NANS:
        """
        Test solver functions standalone with raw numpy arrays containing NaNs.
        When valid_mask is not provided, solvers derive it internally.
        Returns LassoEstimationResult with beta in (N x M) convention.
        """
        rng = np.random.default_rng(42)
        t, m, n = 150, 2, 4  # T observations, M factors, N assets

        x = rng.normal(0, 0.02, size=(t, m))

        # True betas (N x M) = (4 assets x 2 factors)
        true_b = np.array([
            [0.8, 0.0],   # Asset_0: loads on F0
            [0.0, 0.7],   # Asset_1: loads on F1
            [0.5, 0.0],   # Asset_2: loads on F0
            [0.3, 0.6],   # Asset_3: loads on both
        ])

        # y = x @ beta' + noise: (T x M) @ (M x N) = (T x N)
        y = x @ true_b.T + rng.normal(0, 0.01, size=(t, n))

        y[:30, 1] = np.nan
        y[:60, 2] = np.nan

        print(f"\n{'='*60}")
        print(f" SOLVER_STANDALONE_WITH_NANS")
        print(f"{'='*60}")
        print(f"\nTrue betas (N x M):\n{true_b.round(3)}")

        # Test with valid_mask=None (derived internally)
        result1 = solve_lasso_cvx_problem(
            x=x, y=y,
            valid_mask=None,
            reg_lambda=1e-5,
            span=52,
        )
        assert isinstance(result1, LassoEstimationResult), "Should return LassoEstimationResult"
        assert result1.estimated_beta.shape == (n, m), f"beta shape should be ({n},{m}), got {result1.estimated_beta.shape}"
        print(f"\nLASSO (valid_mask=None):")
        print(f"  Estimated betas (N x M):\n{result1.estimated_beta.round(3)}")
        print(f"  R²:    {result1.r2.round(3)}")
        print(f"  Alpha: {result1.alpha.round(6)}")
        print(f"  ss_total: {result1.ss_total.round(6)}")
        print(f"  ss_res:   {result1.ss_res.round(6)}")

        # Test with explicit valid_mask
        nan_mask = np.isnan(y)
        y_filled = np.where(nan_mask, 0.0, y)
        valid_mask = (~nan_mask).astype(float)
        result2 = solve_lasso_cvx_problem(
            x=x, y=y_filled,
            valid_mask=valid_mask,
            reg_lambda=1e-5,
            span=52,
        )
        print(f"\nLASSO (valid_mask explicit):")
        print(f"  Estimated betas (N x M):\n{result2.estimated_beta.round(3)}")
        print(f"  R²: {result2.r2.round(3)}")

        # Verify consistency
        max_diff_beta = np.nanmax(np.abs(result1.estimated_beta - result2.estimated_beta))
        max_diff_r2 = np.nanmax(np.abs(result1.r2 - result2.r2))
        max_diff_alpha = np.nanmax(np.abs(result1.alpha - result2.alpha))
        print(f"\nMax |diff| betas: {max_diff_beta:.2e}")
        print(f"Max |diff| R²:    {max_diff_r2:.2e}")
        print(f"Max |diff| alpha: {max_diff_alpha:.2e}")
        assert max_diff_beta < 1e-6, f"Beta results differ! max_diff={max_diff_beta}"
        assert max_diff_r2 < 1e-6, f"R² results differ! max_diff={max_diff_r2}"
        print("PASS: Both approaches produce identical results.")

        # Diagnostics finite for full-history assets
        assert not np.isnan(result1.ss_total[0]), "Asset 0 should have valid ss_total"
        assert not np.isnan(result1.ss_res[0]), "Asset 0 should have valid ss_res"
        print("PASS: Diagnostics finite for full-history assets")


if __name__ == '__main__':
    for test in LocalTests:
        print(f"\n\n{'#'*70}")
        print(f"# Running: {test.name}")
        print(f"{'#'*70}")
        try:
            run_local_test(local_test=test)
        except Exception as e:
            print(f"\n  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
