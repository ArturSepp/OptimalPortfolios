"""
implement annualisation
"""

import pandas as pd
from typing import Union
from qis import get_annualization_factor


def get_conversion_factor(from_freq: Union[str, pd.Timestamp],
                          to_freq: Union[str, pd.Timestamp]) -> float:
    """
    Get factor to convert between pandas frequencies.

    Args:
        from_freq: Source frequency
        to_freq: Target frequency

    Returns:
        Conversion factor (multiply source data by this factor)

    Examples:
        >>> get_conversion_factor('QE', 'ME')  # Quarterly to Monthly
        0.3333333333333333
        >>> get_conversion_factor('ME', 'QE')  # Monthly to Quarterly
        3.0
        >>> get_conversion_factor('B', 'ME')  # Business Daily to Monthly
        21.666666666666668
    """
    from_periods = get_annualization_factor(from_freq)
    to_periods = get_annualization_factor(to_freq)

    return from_periods / to_periods
