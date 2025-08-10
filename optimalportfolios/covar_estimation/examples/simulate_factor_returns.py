import numpy as np
import pandas as pd
from typing import Tuple, Optional


def simulate_factor_model_returns(
        n_factors: int = 4,
        n_assets: int = 8,
        n_periods: int = 252,
        factor_vol: float = 0.15,
        idio_vol_range: Tuple[float, float] = (0.01, 0.15),
        beta_range: Tuple[float, float] = (-1.5, 1.5),
        factor_corr: Optional[np.ndarray] = None,
        dt: float = 1.0 / 260.0,
        seed: Optional[int] = 42
) -> dict:
    """
    Simulate asset returns using a factor model.

    The factor model is: R_i,t = α_i + Σ(β_i,k * F_k,t) + ε_i,t

    Args:
        n_factors: Number of common factors (default: 4)
        n_assets: Number of assets (default: 8)
        n_periods: Number of time periods (default: 252 for daily)
        factor_vol: Volatility of each factor (default: 0.02)
        idio_vol_range: Range for idiosyncratic volatilities (min, max)
        beta_range: Range for factor loadings (min, max)
        factor_corr: Optional factor correlation matrix (n_factors x n_factors)
        dt: Time step for scaling volatilities (default: 1/260 for daily from annual)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - 'asset_returns': DataFrame of simulated asset returns (n_periods x n_assets)
            - 'factor_returns': DataFrame of factor returns (n_periods x n_factors)
            - 'betas': DataFrame of factor loadings (n_factors x n_assets)
            - 'idio_vol': Series of idiosyncratic volatilities (n_assets,)
            - 'factor_covar': Factor covariance matrix (n_factors x n_factors)
            - 'residual_returns': DataFrame of idiosyncratic returns (n_periods x n_assets)
            - 'theoretical_asset_covar': Theoretical asset covariance matrix (n_assets x n_assets)
    """
    if seed is not None:
        np.random.seed(seed)

    # Scale volatilities by time step
    factor_vol = factor_vol * np.sqrt(dt)
    idio_vol_range = [x * np.sqrt(dt) for x in idio_vol_range]

    # Create asset and factor names
    asset_names = [f'Asset_{i + 1}' for i in range(n_assets)]
    factor_names = [f'Factor_{i + 1}' for i in range(n_factors)]

    # Generate factor correlation matrix if not provided
    if factor_corr is None:
        # Create moderate correlation between factors
        factor_corr = np.eye(n_factors)
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                corr_val = np.random.uniform(-0.6, 0.6)
                factor_corr[i, j] = corr_val
                factor_corr[j, i] = corr_val

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(factor_corr)
        eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
        factor_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    # Create factor covariance matrix
    factor_vol_vec = np.full(n_factors, factor_vol)
    factor_covar = np.outer(factor_vol_vec, factor_vol_vec) * factor_corr

    # Generate factor returns using multivariate normal
    factor_returns = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=factor_covar,
        size=n_periods
    )

    # Generate factor loadings (betas)
    betas = np.random.uniform(
        low=beta_range[0],
        high=beta_range[1],
        size=(n_factors, n_assets)
    )

    # Generate idiosyncratic volatilities
    idio_vol = np.random.uniform(
        low=idio_vol_range[0],
        high=idio_vol_range[1],
        size=n_assets
    )

    # Generate idiosyncratic returns
    residual_returns = np.random.normal(
        loc=0,
        scale=idio_vol.reshape(1, -1),
        size=(n_periods, n_assets)
    )

    # Calculate systematic returns: F @ β
    systematic_returns = factor_returns @ betas

    # Calculate total asset returns: systematic + idiosyncratic
    asset_returns = systematic_returns + residual_returns

    # Convert to DataFrames
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='B')

    asset_returns_df = pd.DataFrame(
        asset_returns,
        index=dates,
        columns=asset_names
    )

    factor_returns_df = pd.DataFrame(
        factor_returns,
        index=dates,
        columns=factor_names
    )

    betas_df = pd.DataFrame(
        betas,
        index=factor_names,
        columns=asset_names
    )

    idio_vol_series = pd.Series(
        idio_vol,
        index=asset_names,
        name='Idiosyncratic_Vol'
    )

    residual_returns_df = pd.DataFrame(
        residual_returns,
        index=dates,
        columns=asset_names
    )

    factor_covar_df = pd.DataFrame(
        factor_covar,
        index=factor_names,
        columns=factor_names
    )

    # Calculate theoretical asset covariance matrix
    # Asset covariance = β' @ factor_covar @ β + diag(idio_var^2)
    theoretical_asset_covar = betas.T @ factor_covar @ betas + np.diag(idio_vol ** 2)

    theoretical_asset_covar_df = pd.DataFrame(
        theoretical_asset_covar,
        index=asset_names,
        columns=asset_names
    )

    return {
        'asset_returns': asset_returns_df,
        'factor_returns': factor_returns_df,
        'betas': betas_df,
        'idio_vol': idio_vol_series,
        'factor_covar': factor_covar_df,
        'residual_returns': residual_returns_df,
        'theoretical_asset_covar': theoretical_asset_covar_df
    }


def verify_factor_model(simulation_results: dict) -> dict:
    """
    Verify the factor model by reconstructing asset returns and computing statistics.

    Args:
        simulation_results: Output from simulate_factor_model_returns()

    Returns:
        Dictionary containing:
            - 'asset_stats': DataFrame with verification statistics for each asset
            - 'covariance_comparison': DataFrame comparing theoretical vs empirical covariances
            - 'covariance_errors': DataFrame with absolute and relative errors
    """
    factor_returns = simulation_results['factor_returns']
    betas = simulation_results['betas']
    residual_returns = simulation_results['residual_returns']
    asset_returns = simulation_results['asset_returns']
    theoretical_covar = simulation_results['theoretical_asset_covar']

    # Calculate empirical asset covariance matrix
    empirical_covar = asset_returns.cov()

    # Reconstruct asset returns
    systematic_returns = factor_returns @ betas
    reconstructed_returns = systematic_returns + residual_returns

    # Calculate verification metrics for each asset
    verification_stats = []

    for asset in asset_returns.columns:
        actual = asset_returns[asset]
        reconstructed = reconstructed_returns[asset]
        systematic = systematic_returns[asset]
        idiosyncratic = residual_returns[asset]

        # Calculate R-squared (systematic vs total variance)
        total_var = np.var(actual)
        systematic_var = np.var(systematic)
        idio_var = np.var(idiosyncratic)
        r_squared = systematic_var / total_var

        # Calculate correlation between actual and reconstructed
        correlation = np.corrcoef(actual, reconstructed)[0, 1]

        # Compare theoretical vs empirical variance
        theoretical_var = theoretical_covar.loc[asset, asset]
        empirical_var = empirical_covar.loc[asset, asset]
        var_error = abs(theoretical_var - empirical_var)
        var_rel_error = var_error / theoretical_var

        verification_stats.append({
            'Asset': asset,
            'Theoretical_Variance': theoretical_var,
            'Empirical_Variance': empirical_var,
            'Variance_Error': var_error,
            'Variance_Rel_Error': var_rel_error,
            'Systematic_Variance': systematic_var,
            'Idiosyncratic_Variance': idio_var,
            'R_Squared': r_squared,
            'Reconstruction_Correlation': correlation,
            'Mean_Return': np.mean(actual),
            'Volatility': np.std(actual)
        })

    asset_stats_df = pd.DataFrame(verification_stats).set_index('Asset')

    # Create covariance comparison matrices
    n_assets = len(asset_returns.columns)
    covariance_comparison = []
    covariance_errors = []

    for i, asset_i in enumerate(asset_returns.columns):
        for j, asset_j in enumerate(asset_returns.columns):
            theoretical_cov = theoretical_covar.iloc[i, j]
            empirical_cov = empirical_covar.iloc[i, j]
            abs_error = abs(theoretical_cov - empirical_cov)
            rel_error = abs_error / abs(theoretical_cov) if theoretical_cov != 0 else np.nan

            covariance_comparison.append({
                'Asset_i': asset_i,
                'Asset_j': asset_j,
                'Theoretical_Cov': theoretical_cov,
                'Empirical_Cov': empirical_cov,
                'Is_Diagonal': (i == j)
            })

            covariance_errors.append({
                'Asset_i': asset_i,
                'Asset_j': asset_j,
                'Absolute_Error': abs_error,
                'Relative_Error': rel_error,
                'Is_Diagonal': (i == j)
            })

    covariance_comparison_df = pd.DataFrame(covariance_comparison)
    covariance_errors_df = pd.DataFrame(covariance_errors)

    return {
        'asset_stats': asset_stats_df,
        'covariance_comparison': covariance_comparison_df,
        'covariance_errors': covariance_errors_df,
        'theoretical_covar_matrix': theoretical_covar,
        'empirical_covar_matrix': empirical_covar
    }


# Example usage and demonstration
if __name__ == "__main__":
    # Simulate factor model returns
    print("Simulating factor model with 4 factors and 8 assets...")
    simulation = simulate_factor_model_returns(
        n_factors=4,
        n_assets=8,
        n_periods=252,
        seed=42
    )

    # Display basic information
    print("\n=== Factor Model Simulation Results ===")
    print(f"Asset returns shape: {simulation['asset_returns'].shape}")
    print(f"Factor returns shape: {simulation['factor_returns'].shape}")
    print(f"Beta matrix shape: {simulation['betas'].shape}")

    # Show factor loadings
    print("\n=== Factor Loadings (Betas) ===")
    print(simulation['betas'].round(3))

    # Show idiosyncratic volatilities
    print("\n=== Idiosyncratic Volatilities ===")
    print(simulation['idio_vol'].round(4))

    # Show factor covariance matrix
    print("\n=== Factor Covariance Matrix ===")
    print(simulation['factor_covar'].round(6))

    # Show theoretical asset covariance matrix
    print("\n=== Theoretical Asset Covariance Matrix ===")
    print(simulation['theoretical_asset_covar'].round(6))

    # Verify the model
    print("\n=== Model Verification ===")
    verification = verify_factor_model(simulation)
    print("Asset Statistics:")
    print(verification['asset_stats'].round(4))

    # Show covariance matrix comparison
    print("\n=== Theoretical vs Empirical Covariance Matrices ===")
    print("Theoretical:")
    print(verification['theoretical_covar_matrix'].round(6))
    print("\nEmpirical:")
    print(verification['empirical_covar_matrix'].round(6))

    # Show covariance errors summary
    print("\n=== Covariance Error Summary ===")
    errors_df = verification['covariance_errors']
    print("Diagonal Elements (Variances):")
    diagonal_errors = errors_df[errors_df['Is_Diagonal'] == True]
    print(f"  Mean Absolute Error: {diagonal_errors['Absolute_Error'].mean():.6f}")
    print(f"  Mean Relative Error: {diagonal_errors['Relative_Error'].mean():.4f}")
    print(f"  Max Absolute Error: {diagonal_errors['Absolute_Error'].max():.6f}")

    print("\nOff-Diagonal Elements (Covariances):")
    off_diagonal_errors = errors_df[errors_df['Is_Diagonal'] == False]
    print(f"  Mean Absolute Error: {off_diagonal_errors['Absolute_Error'].mean():.6f}")
    print(f"  Mean Relative Error: {off_diagonal_errors['Relative_Error'].mean():.4f}")
    print(f"  Max Absolute Error: {off_diagonal_errors['Absolute_Error'].max():.6f}")

    # Sample of asset returns
    print("\n=== Sample Asset Returns (First 10 Days) ===")
    print(simulation['asset_returns'].head(10).round(4))

    # Theoretical vs empirical correlations
    print("\n=== Theoretical vs Empirical Correlations ===")


    def cov_to_corr(cov_matrix):
        std_devs = np.sqrt(np.diag(cov_matrix))
        return cov_matrix / np.outer(std_devs, std_devs)


    theoretical_corr = cov_to_corr(verification['theoretical_covar_matrix'].values)
    empirical_corr = cov_to_corr(verification['empirical_covar_matrix'].values)

    print("Theoretical Correlations:")
    print(pd.DataFrame(theoretical_corr,
                       index=simulation['asset_returns'].columns,
                       columns=simulation['asset_returns'].columns).round(3))

    print("\nEmpirical Correlations:")
    print(pd.DataFrame(empirical_corr,
                       index=simulation['asset_returns'].columns,
                       columns=simulation['asset_returns'].columns).round(3))