# Factor Model Asset Variance Estimation - Mathematical Framework

## Overview

This document outlines the mathematical framework for estimating asset variances using factor models and handling annualization when factor returns and beta estimation occur at different frequencies.

## Core Factor Model for Asset Variance

### Asset Variance Decomposition

**Individual Asset Variance:**
```
asset_variance_i = beta_i.T @ factor_covar @ beta_i + idio_var_i
```

**All Asset Variances (Diagonal of Covariance Matrix):**
```
asset_variances = diag(betas.T @ factor_covar @ betas) + idio_var
```

Where:
- `beta_i` = factor loadings for asset i (K × 1 vector)
- `betas` = factor loadings matrix for all assets (K × N)
- `factor_covar` = factor covariance matrix (K × K)
- `idio_var` = idiosyncratic variances vector (N × 1)
- `asset_variances` = estimated asset variances vector (N × 1)

### Full Asset Covariance Matrix
```
asset_covariance_matrix = betas.T @ factor_covar @ betas + diag(idio_var)
```

## Annualization Framework

### Standard Same-Frequency Annualization

When factors and betas are estimated at the same frequency `f`:

**Annualized Asset Variances:**
```
annual_asset_variances = diag(betas.T @ (factor_covar × ann_f) @ betas) + (idio_var × ann_f)
                       = ann_f × [diag(betas.T @ factor_covar @ betas) + idio_var]
                       = ann_f × base_asset_variances
```

### Mixed-Frequency Annualization

When factor covariance and idiosyncratic variance are estimated at different frequencies:

**Setup:**
- Factor returns estimated at frequency `f1` → annualization factor `ann_f1`
- Beta estimation (residuals) at frequency `f2` → annualization factor `ann_f2`

**Annualized Asset Variances:**
```
annual_asset_variances = diag(betas.T @ (factor_covar × ann_f1) @ betas) + (idio_var × ann_f2)
```

**Component Breakdown:**
- **Systematic variance component:** `diag(betas.T @ (factor_covar × ann_f1) @ betas)`
- **Idiosyncratic variance component:** `idio_var × ann_f2`

## Practical Example: Illiquid Factors

### Scenario
- **Factor covariance:** Estimated from monthly factor returns
- **Beta estimation:** Quarterly regression residuals for idiosyncratic variance
- **Target:** Monthly asset variance estimates

### Mathematical Solution

**Monthly Asset Variances:**
```
monthly_asset_variances = diag(betas.T @ monthly_factor_covar @ betas) + (quarterly_idio_var × 1/3)
```

**Breakdown:**
- `monthly_factor_covar` used directly (already at target frequency)
- `quarterly_idio_var × 1/3` converts quarterly idiosyncratic variance to monthly
- `1/3` factor accounts for quarterly (4 periods/year) to monthly (12 periods/year) conversion

### Alternative Annualization Approach

**Step 1: Annualize to Common Base (Annual)**
```
annual_factor_covar = monthly_factor_covar × 12
annual_idio_var = quarterly_idio_var × 4
annual_asset_variances = diag(betas.T @ annual_factor_covar @ betas) + annual_idio_var
```

**Step 2: Scale to Target Frequency (Monthly)**
```
monthly_asset_variances = annual_asset_variances / 12
```

## Implementation Framework

### Frequency Conversion Factors
```python
import pandas as pd
from typing import Union

# Standard pandas frequency mappings to annualization factors (pandas 2.0+ style)
FREQUENCY_MAPPINGS = {
    # Daily frequencies
    'D': 365,          # Calendar daily
    'B': 260,          # Business daily
    
    # Weekly frequencies  
    'WE': 52,          # Weekly ending (replaces 'W')
    'WE-MON': 52,      # Weekly ending Monday
    'WE-TUE': 52,      # Weekly ending Tuesday
    'WE-WED': 52,      # Weekly ending Wednesday
    'WE-THU': 52,      # Weekly ending Thursday
    'WE-FRI': 52,      # Weekly ending Friday
    'WE-SAT': 52,      # Weekly ending Saturday
    'WE-SUN': 52,      # Weekly ending Sunday
    
    # Monthly frequencies
    'ME': 12,          # Month end (replaces 'M')
    'MS': 12,          # Month start
    'BME': 12,         # Business month end (replaces 'BM')
    'BMS': 12,         # Business month start
    
    # Quarterly frequencies
    'QE': 4,           # Quarter end (replaces 'Q')
    'QS': 4,           # Quarter start
    'BQE': 4,          # Business quarter end (replaces 'BQ')
    'BQS': 4,          # Business quarter start
    'QE-DEC': 4,       # Quarter ending December (replaces 'Q-DEC')
    'QE-JAN': 4,       # Quarter ending January
    'QE-FEB': 4,       # Quarter ending February
    'QE-MAR': 4,       # Quarter ending March
    
    # Annual frequencies
    'YE': 1,           # Year end (replaces 'A')
    'YS': 1,           # Year start (replaces 'AS')
    'BYE': 1,          # Business year end (replaces 'BA')
    'BYS': 1,          # Business year start (replaces 'BAS')
    'YE-DEC': 1,       # Year ending December (replaces 'A-DEC')
    'YE-JAN': 1,       # Year ending January
    'YE-FEB': 1,       # Year ending February
    'YE-MAR': 1,       # Year ending March
    'YE-APR': 1,       # Year ending April
    'YE-MAY': 1,       # Year ending May
    'YE-JUN': 1,       # Year ending June
    'YE-JUL': 1,       # Year ending July
    'YE-AUG': 1,       # Year ending August
    'YE-SEP': 1,       # Year ending September
    'YE-OCT': 1,       # Year ending October
    'YE-NOV': 1,       # Year ending November
    
    # Legacy frequency support (deprecated but still supported)
    'W': 52,           # Legacy weekly (use 'WE')
    'M': 12,           # Legacy monthly (use 'ME')
    'Q': 4,            # Legacy quarterly (use 'QE')
    'A': 1,            # Legacy annual (use 'YE')
    'BM': 12,          # Legacy business month (use 'BME')
    'BQ': 4,           # Legacy business quarter (use 'BQE')
    'BA': 1,           # Legacy business annual (use 'BYE')
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
```

### Asset Variance Estimation Function
```python
def estimate_asset_variances(
    betas: np.ndarray,
    factor_covar: np.ndarray,
    idio_var: np.ndarray,
    factor_freq: Union[str, pd.Timestamp],
    idio_freq: Union[str, pd.Timestamp],
    target_freq: Union[str, pd.Timestamp] = 'YE'
) -> np.ndarray:
    """
    Estimate asset variances using factor model with mixed frequencies.
    
    Args:
        betas: Factor loadings matrix (K x N)
        factor_covar: Factor covariance matrix (K x K) 
        idio_var: Idiosyncratic variances (N,)
        factor_freq: Pandas frequency of factor covariance estimation (e.g., 'ME', 'QE', 'B')
        idio_freq: Pandas frequency of idiosyncratic variance estimation (e.g., 'ME', 'QE', 'B')
        target_freq: Target frequency for asset variances (default: 'YE' for annual)
        
    Returns:
        Asset variances at target frequency (N,)
        
    Examples:
        >>> # Monthly factors, quarterly idiosyncratic, annual target
        >>> variances = estimate_asset_variances(betas, factor_cov, idio_var, 'ME', 'QE', 'YE')
        >>> # Business daily factors, monthly idiosyncratic, monthly target  
        >>> variances = estimate_asset_variances(betas, factor_cov, idio_var, 'B', 'ME', 'ME')
    """
    # Scale factor covariance to target frequency
    factor_scale = get_conversion_factor(factor_freq, target_freq)
    scaled_factor_covar = factor_covar * factor_scale
    
    # Scale idiosyncratic variance to target frequency
    idio_scale = get_conversion_factor(idio_freq, target_freq)
    scaled_idio_var = idio_var * idio_scale
    
    # Calculate systematic variance component
    systematic_var = np.diag(betas.T @ scaled_factor_covar @ betas)
    
    # Total asset variances
    asset_variances = systematic_var + scaled_idio_var
    
    return asset_variances
```

## Key Relationships

### Variance Attribution
```
Total Asset Variance = Systematic Variance + Idiosyncratic Variance
                    = β.T @ F @ β + σ²_idio
```

### Risk Decomposition
```
Systematic Risk % = (β.T @ F @ β) / Total Variance
Idiosyncratic Risk % = σ²_idio / Total Variance
```

### Cross-Asset Covariance
```
Cov(Asset_i, Asset_j) = β_i.T @ F @ β_j  (if i ≠ j)
                      = β_i.T @ F @ β_i + σ²_idio,i  (if i = j)
```

## Validation Checks

### Model Consistency
```python
# 1. Variance decomposition should sum correctly
assert np.allclose(systematic_var + idio_var, total_asset_var)

# 2. Systematic variance should be non-negative
assert np.all(systematic_var >= 0)

# 3. Scaling relationships should hold
annual_var = monthly_var * 12  # For monthly to annual conversion
```

### Statistical Properties
```python
# 4. Correlation structure preservation
monthly_corr = cov_to_corr(monthly_asset_covar)
annual_corr = cov_to_corr(annual_asset_covar) 
assert np.allclose(monthly_corr, annual_corr, atol=1e-10)
```

## Summary

This framework enables robust estimation of asset variances using factor models when dealing with mixed-frequency data sources. The key insight is that systematic and idiosyncratic variance components can be scaled independently based on their respective estimation frequencies, allowing for flexible model construction in real-world scenarios where data availability varies across factors.