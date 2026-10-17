"""Native factor coordinates must reach the fit and covariance together."""
import numpy as np
import pandas as pd
import pytest
from factorlasso import LassoModel, LassoModelType
from qis.perfstats.config import ReturnTypes
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator


@pytest.mark.parametrize('responses', [1, 3])
def test_point_slopes_covariance_and_log_legacy(responses):
    """Known positive and negative point slopes recover without a fake NAV."""
    rng = np.random.default_rng(123)
    dates = pd.date_range('2018-01-31', periods=80, freq='ME')
    levels = pd.DataFrame({'VIX': rng.uniform(12, 45, len(dates))}, index=dates)
    slopes = np.array([-.002, .001, -.004])[:responses]
    y = pd.DataFrame(levels.VIX.diff().to_numpy()[:, None]*slopes,
                     index=dates, columns=[f'A{k}' for k in range(responses)])
    def fit(mapping):
        """Use the owning estimator in a zero-penalty reference configuration."""
        return FactorCovarEstimator(lasso_model=LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=0., demean=False,
            solver='CLARABEL', span=24, warmup_period=12),
            factor_returns_freq='ME', factor_covar_span=24,
            factor_return_types=mapping).fit_current_factor_covars(
                levels, {'ME': y}, assets=list(y), estimation_date=dates[-1])
    point = fit({'VIX': ReturnTypes.DIFFERENCE})
    np.testing.assert_allclose(point.y_betas.VIX, slopes, atol=1e-7)
    assert point.x_covar.iloc[0,0] > 100.
    default, log = fit(None), fit({'VIX': ReturnTypes.LOG})
    pd.testing.assert_frame_equal(default.y_betas, log.y_betas)
    pd.testing.assert_frame_equal(default.x_covar, log.x_covar)
