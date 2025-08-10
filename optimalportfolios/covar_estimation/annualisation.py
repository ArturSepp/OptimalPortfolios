"""
implement annualisation
"""

import pandas as pd
from typing import Union

# Standard pandas frequency mappings to annualization factors
FREQUENCY_MAPPINGS = {
    # Daily frequencies
    'D': 365,  # Calendar daily
    'B': 260,  # Business daily

    # Weekly frequencies
    'WE': 52,  # Weekly ending (replaces 'W')
    'W-MON': 52,  # Weekly ending Monday
    'W-TUE': 52,  # Weekly ending Tuesday
    'W-WED': 52,  # Weekly ending Wednesday
    'W-THU': 52,  # Weekly ending Thursday
    'W-FRI': 52,  # Weekly ending Friday
    'W-SAT': 52,  # Weekly ending Saturday
    'W-SUN': 52,  # Weekly ending Sunday

    # Monthly frequencies
    'ME': 12,  # Month end (replaces 'M')
    'MS': 12,  # Month start
    'BME': 12,  # Business month end (replaces 'BM')
    'BMS': 12,  # Business month start

    # Quarterly frequencies
    'QE': 4,  # Quarter end (replaces 'Q')
    'QS': 4,  # Quarter start
    'BQE': 4,  # Business quarter end (replaces 'BQ')
    'BQS': 4,  # Business quarter start
    'QE-DEC': 4,  # Quarter ending December (replaces 'Q-DEC')
    'QE-JAN': 4,  # Quarter ending January
    'QE-FEB': 4,  # Quarter ending February
    'QE-MAR': 4,  # Quarter ending March

    # Annual frequencies
    'YE': 1,  # Year end (replaces 'A')
    'YS': 1,  # Year start (replaces 'AS')
    'BYE': 1,  # Business year end (replaces 'BA')
    'BYS': 1,  # Business year start (replaces 'BAS')
    'YE-DEC': 1,  # Year ending December (replaces 'A-DEC')
    'YE-JAN': 1,  # Year ending January
    'YE-FEB': 1,  # Year ending February
    'YE-MAR': 1,  # Year ending March
    'YE-APR': 1,  # Year ending April
    'YE-MAY': 1,  # Year ending May
    'YE-JUN': 1,  # Year ending June
    'YE-JUL': 1,  # Year ending July
    'YE-AUG': 1,  # Year ending August
    'YE-SEP': 1,  # Year ending September
    'YE-OCT': 1,  # Year ending October
    'YE-NOV': 1,  # Year ending November

    # Legacy frequency support (deprecated but still supported)
    'W': 52,  # Legacy weekly (use 'WE')
    'M': 12,  # Legacy monthly (use 'ME')
    'Q': 4,  # Legacy quarterly (use 'QE')
    'A': 1,  # Legacy annual (use 'YE')
    'BM': 12,  # Legacy business month (use 'BME')
    'BQ': 4,  # Legacy business quarter (use 'BQE')
    'BA': 1,  # Legacy business annual (use 'BYE')
}


def get_annualization_factor(freq: Union[str, pd.Timestamp]) -> float:
    """
    Get factor to annualize from given pandas frequency.

    Args:
        freq: Pandas frequency string (e.g., 'ME', 'QE', 'B') or frequency object

    Returns:
        Number of periods per year for the given frequency

    Examples:
        >>> get_annualization_factor('ME')
        12
        >>> get_annualization_factor('QE')
        4
        >>> get_annualization_factor('B')
        260
    """
    if isinstance(freq, pd.Timestamp):
        freq = freq.freq

    freq_str = str(freq).upper()

    if freq_str in FREQUENCY_MAPPINGS:
        return FREQUENCY_MAPPINGS[freq_str]
    else:
        raise ValueError(f"Unsupported frequency: {freq_str}. "
                         f"Supported frequencies: {list(FREQUENCY_MAPPINGS.keys())}")


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
