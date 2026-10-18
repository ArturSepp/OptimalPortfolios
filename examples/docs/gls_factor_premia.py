"""Offline synthetic prices -> LASSO loadings/alpha -> provider-implied GLS premia.

Monthly simple excess-return NAVs, uniform weights, CLARABEL, no missing data.
The small example teaches the two stages; it does not substitute for MATF production
price preparation, expert priors, horizons or data. No files or network access.
"""

import numpy as np
import pandas as pd
from scipy.stats import t

from factorlasso import LassoModel, LassoModelType
from tradablefactors.premia import infer_factor_premia


def main():
    """Fit synthetic histories and independently check premia and confidence intervals."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2006-09-30", periods=241, freq="ME")
    factors = ["Equity", "Rates", "Carry FX"]
    assets = [f"Index {i+1}" for i in range(8)]
    true_b = rng.uniform(-.2, 1., (8, 3))
    x = rng.normal(0., [.035, .015, .01], (240, 3))
    y = x @ true_b.T + rng.normal(0., .012, (240, 8)) + .0005
    factor_prices = pd.DataFrame(100*np.vstack([np.ones(3), np.cumprod(1+x, axis=0)]),
                                 index=dates, columns=factors)
    index_prices = pd.DataFrame(100*np.vstack([np.ones(8), np.cumprod(1+y, axis=0)]),
                                index=dates, columns=assets)
    x = factor_prices.pct_change(fill_method=None).iloc[1:]
    y = index_prices.pct_change(fill_method=None).iloc[1:]
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                       span=None, demean=True, solver="CLARABEL").fit(x=x, y=y)
    betas = model.coef_
    historical_residual = y - x @ betas.T
    # Arithmetic annualisation of monthly simple excess returns; no compounding bridge.
    alpha = 12*historical_residual.mean()
    residual_variances = 12*historical_residual.var(ddof=0)
    provider_cma = pd.Series([.08, .065, .06, .07, .05, .09, .055, .075], index=assets)
    views = {}
    for weight in (0., .5, 1.):
        views[weight] = infer_factor_premia(
            betas, residual_variances, provider_cma, cash_rate=.03,
            alpha=alpha, alpha_weight=weight)
    result = views[1.]

    # Independent weighted QR reference, not the production SVD implementation.
    design = betas.to_numpy()/np.sqrt(residual_variances.to_numpy())[:, None]
    target = (provider_cma-.03-alpha)/np.sqrt(residual_variances)
    q, r = np.linalg.qr(design, mode="reduced")
    reference = np.linalg.solve(r, q.T @ target)
    error = target - design @ reference
    inverse_r = np.linalg.solve(r, np.eye(3))
    covariance = float(error @ error)/5 * inverse_r @ inverse_r.T
    np.testing.assert_allclose(result.premia, reference, atol=1e-12)
    np.testing.assert_allclose(result.covariance, covariance, atol=1e-12)
    np.testing.assert_allclose(result.factors.upper,
                               reference+t.ppf(.975, 5)*np.sqrt(np.diag(covariance)))
    np.testing.assert_allclose(result.assets.fitted+result.assets.residual
                               +result.assets.cash+result.assets.credited_alpha, provider_cma)
    np.testing.assert_allclose(views[0.].premia-views[1.].premia,
                               result.mimicking_weights.T @ alpha)
    assert result.summary["rank"] == 3 and result.summary["residual_df"] == 5
    print(pd.DataFrame({f"alpha_{w:g}": value.premia for w, value in views.items()}))
    print(result.factors[["estimate", "lower", "upper", "variance_inflation"]])
    return result


if __name__ == "__main__":
    main()
