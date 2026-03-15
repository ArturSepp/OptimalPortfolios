
import pandas as pd
import qis as qis
from typing import Optional

def compute_ra_carry_alphas(prices: pd.DataFrame,
                            carry: pd.DataFrame,
                            returns_freq: str = 'W-WED',
                            vol_span: Optional[int] = 13,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                            ) -> pd.DataFrame:
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    ewm_vol = qis.compute_ewm_vol(data=returns,
                                  span=vol_span,
                                  mean_adj_type=mean_adj_type,
                                  annualize=True)
    ra_carry = carry.reindex(index=ewm_vol.index, method='ffill').divide(ewm_vol)
    # momentum = qis.map_signal_to_weight(signals=momentum, loc=0.0, slope_right=0.5, slope_left=0.5, tail_level=3.0)
    alphas = qis.df_to_cross_sectional_score(df=ra_carry)
    return alphas