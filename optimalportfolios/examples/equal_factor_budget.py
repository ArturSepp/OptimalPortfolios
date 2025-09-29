import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class FactorRiskParity:
    """
    Implementation of Factor-Based Risk Parity following Roncalli & Weisang (2016) approach.
    This class implements equal risk contribution based on risk factors rather than assets.
    """

    def __init__(self, returns_data):
        """
        Initialize with returns data

        Parameters:
        -----------
        returns_data : pd.DataFrame
            DataFrame with asset returns, columns are assets, rows are time periods
        """
        self.returns = returns_data
        self.cov_matrix = returns_data.cov().values
        self.n_assets = len(returns_data.columns)
        self.asset_names = list(returns_data.columns)

    def estimate_factor_loadings(self, n_factors=None, method='pca'):
        """
        Estimate factor loadings matrix using PCA or other methods

        Parameters:
        -----------
        n_factors : int, optional
            Number of factors to extract. If None, uses Kaiser criterion (eigenvalues > 1)
        method : str
            Method for factor extraction ('pca', 'statistical_factors')

        Returns:
        --------
        loadings : np.array
            Factor loadings matrix (n_assets x n_factors)
        factor_cov : np.array
            Factor covariance matrix
        """
        if method == 'pca':
            pca = PCA()
            pca.fit(self.returns)

            if n_factors is None:
                # Kaiser criterion: eigenvalues > 1
                eigenvalues = pca.explained_variance_
                n_factors = np.sum(eigenvalues > 1)
                if n_factors == 0:
                    n_factors = min(3, self.n_assets)  # fallback

            # Extract loadings (scaled by sqrt of eigenvalues)
            loadings = pca.components_[:n_factors].T * np.sqrt(pca.explained_variance_[:n_factors])

            # Factor covariance is identity for PCA factors
            factor_cov = np.eye(n_factors)

            # Store factor returns for analysis
            self.factor_returns = pca.transform(self.returns)[:, :n_factors]

        self.loadings = loadings
        self.factor_cov = factor_cov
        self.n_factors = n_factors

        return loadings, factor_cov

    def calculate_portfolio_variance(self, weights):
        """Calculate portfolio variance given weights"""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def calculate_asset_risk_contributions(self, weights):
        """
        Calculate risk contributions of individual assets

        Parameters:
        -----------
        weights : np.array
            Portfolio weights

        Returns:
        --------
        risk_contributions : np.array
            Risk contribution of each asset
        """
        portfolio_variance = self.calculate_portfolio_variance(weights)
        marginal_contrib = np.dot(self.cov_matrix, weights)
        risk_contrib = np.multiply(weights, marginal_contrib) / portfolio_variance
        return risk_contrib

    def calculate_factor_risk_contributions(self, weights):
        """
        Calculate risk contributions of factors

        Parameters:
        -----------
        weights : np.array
            Portfolio weights

        Returns:
        --------
        factor_risk_contrib : np.array
            Risk contribution of each factor
        """
        # Portfolio exposure to factors
        factor_exposures = np.dot(self.loadings.T, weights)

        # Portfolio variance via factor model
        portfolio_var_factors = np.dot(factor_exposures.T,
                                       np.dot(self.factor_cov, factor_exposures))

        # Factor risk contributions
        marginal_factor_contrib = np.dot(self.factor_cov, factor_exposures)
        factor_risk_contrib = np.multiply(factor_exposures, marginal_factor_contrib) / portfolio_var_factors

        return factor_risk_contrib, factor_exposures

    def risk_parity_objective_assets(self, weights):
        """
        Objective function for asset-level risk parity
        Minimizes squared deviations from equal risk contribution
        """
        risk_contrib = self.calculate_asset_risk_contributions(weights)
        target_risk = 1.0 / self.n_assets  # Equal risk contribution target
        return np.sum((risk_contrib - target_risk) ** 2)

    def risk_parity_objective_factors(self, weights):
        """
        Objective function for factor-level risk parity
        Minimizes squared deviations from equal factor risk contribution
        """
        factor_risk_contrib, _ = self.calculate_factor_risk_contributions(weights)
        target_risk = 1.0 / self.n_factors  # Equal factor risk contribution target
        return np.sum((factor_risk_contrib - target_risk) ** 2)

    def optimize_risk_parity(self, method='factors', bounds_constraint=True):
        """
        Optimize portfolio for risk parity

        Parameters:
        -----------
        method : str
            'assets' for asset-level risk parity, 'factors' for factor-level risk parity
        bounds_constraint : bool
            Whether to constrain weights to be non-negative

        Returns:
        --------
        result : dict
            Optimization results including weights and risk contributions
        """
        # Initial equal weights
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: long-only portfolio
        if bounds_constraint:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(self.n_assets)]

        # Choose objective function
        if method == 'factors':
            if not hasattr(self, 'loadings'):
                raise ValueError("Must call estimate_factor_loadings() first for factor-based optimization")
            objective_func = self.risk_parity_objective_factors
        else:
            objective_func = self.risk_parity_objective_assets

        # Optimize
        result = minimize(objective_func, initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-9, 'maxiter': 1000})

        if not result.success:
            print(f"Optimization warning: {result.message}")

        optimal_weights = result.x

        # Calculate final risk contributions
        asset_risk_contrib = self.calculate_asset_risk_contributions(optimal_weights)

        results = {
            'weights': optimal_weights,
            'asset_risk_contributions': asset_risk_contrib,
            'portfolio_volatility': np.sqrt(self.calculate_portfolio_variance(optimal_weights)),
            'optimization_result': result
        }

        if method == 'factors':
            factor_risk_contrib, factor_exposures = self.calculate_factor_risk_contributions(optimal_weights)
            results['factor_risk_contributions'] = factor_risk_contrib
            results['factor_exposures'] = factor_exposures

        return results

    def compare_methods(self):
        """
        Compare asset-level vs factor-level risk parity

        Returns:
        --------
        comparison : dict
            Results from both methods
        """
        # Estimate factors first
        self.estimate_factor_loadings()

        # Optimize both methods
        asset_rp = self.optimize_risk_parity(method='assets')
        factor_rp = self.optimize_risk_parity(method='factors')

        return {
            'asset_risk_parity': asset_rp,
            'factor_risk_parity': factor_rp
        }

    def plot_risk_contributions(self, results, method='assets'):
        """
        Plot risk contributions

        Parameters:
        -----------
        results : dict
            Results from optimize_risk_parity()
        method : str
            'assets' or 'factors'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot portfolio weights
        ax1.bar(self.asset_names, results['weights'])
        ax1.set_title('Portfolio Weights')
        ax1.set_ylabel('Weight')
        ax1.tick_params(axis='x', rotation=45)

        # Plot risk contributions
        if method == 'assets':
            ax2.bar(self.asset_names, results['asset_risk_contributions'])
            ax2.set_title('Asset Risk Contributions')
            ax2.axhline(y=1 / self.n_assets, color='r', linestyle='--',
                        label=f'Equal Risk Target ({1 / self.n_assets:.3f})')
        else:
            factor_names = [f'Factor {i + 1}' for i in range(self.n_factors)]
            ax2.bar(factor_names, results['factor_risk_contributions'])
            ax2.set_title('Factor Risk Contributions')
            ax2.axhline(y=1 / self.n_factors, color='r', linestyle='--',
                        label=f'Equal Risk Target ({1 / self.n_factors:.3f})')

        ax2.set_ylabel('Risk Contribution')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_periods = 252  # Daily returns for 1 year

    # Create correlated returns
    true_loadings = np.random.randn(n_assets, 3)  # 3 factors
    factor_returns = np.random.randn(n_periods, 3)
    idiosyncratic = np.random.randn(n_periods, n_assets) * 0.3

    returns = np.dot(factor_returns, true_loadings.T) + idiosyncratic
    returns_df = pd.DataFrame(returns,
                              columns=[f'Asset_{i + 1}' for i in range(n_assets)])

    # Initialize and run factor risk parity
    frp = FactorRiskParity(returns_df)

    # Compare methods
    comparison = frp.compare_methods()

    print("=== Asset-Level Risk Parity ===")
    print(f"Portfolio Volatility: {comparison['asset_risk_parity']['portfolio_volatility']:.4f}")
    print(f"Asset Risk Contributions: {comparison['asset_risk_parity']['asset_risk_contributions']}")

    print("\n=== Factor-Level Risk Parity ===")
    print(f"Portfolio Volatility: {comparison['factor_risk_parity']['portfolio_volatility']:.4f}")
    print(f"Factor Risk Contributions: {comparison['factor_risk_parity']['factor_risk_contributions']}")
    print(f"Asset Risk Contributions: {comparison['factor_risk_parity']['asset_risk_contributions']}")

    # Plot results
    frp.plot_risk_contributions(comparison['asset_risk_parity'], method='assets')
    frp.plot_risk_contributions(comparison['factor_risk_parity'], method='factors')
