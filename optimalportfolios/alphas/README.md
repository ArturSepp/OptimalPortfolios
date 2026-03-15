# Alphas Module — `optimalportfolios.alphas`

Alpha signal building blocks for the `optimalportfolios` package.

This module provides individual alpha signal computation functions and
the `AlphasData` container. Each signal function is a standalone building
block that computes a cross-sectional score from asset prices — ready for
use in any portfolio optimisation workflow.

**This module does not contain aggregation logic.** The routing of assets
to signals, combination rules, and product-specific configurations belong
in the consuming application (e.g., a private `rosaa` package). This
module provides the bricks; the application builds the house.

## Architecture

```
optimalportfolios/alphas/
├── signals/
│   ├── momentum.py           # compute_momentum_alpha()
│   ├── low_beta.py           # compute_low_beta_alpha()
│   └── managers_alpha.py     # compute_managers_alpha()
├── alpha_data.py             # AlphasData container
└── tests/
    └── signals_test.py       # Per-signal tests
```

## Naming Conventions

The module distinguishes three stages in the alpha pipeline:

| Stage | What it is | Example |
|-------|-----------|---------|
| **Raw signal** | Observable quantity with units | Cumulative return, EWMA beta, regression residual |
| **Score** | Cross-sectional rank/z-score, dimensionless | Momentum z-score, negated beta rank |
| **Alpha** | Portfolio-ready signal after combination and CDF mapping | Combined score mapped to [-1, 1] |

Pipeline: **raw signal → score → alpha**.

Functions are named `compute_*_alpha()` because their primary output is
a score ready for aggregation into the final alpha vector. The raw signal
is returned as the second element for diagnostics.

## Signal Functions

Each signal function has the same interface:

```python
def compute_*_alpha(
    prices: pd.DataFrame,
    ...,                                    # signal-specific inputs
    returns_freq: Union[str, pd.Series],    # single or mixed frequency
    group_data: Optional[pd.Series],        # for within-group scoring
    **signal_params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (score, raw_signal)"""
```

### Momentum

Cross-sectional momentum score from EWMA-filtered risk-adjusted excess returns.

```python
from optimalportfolios.alphas import compute_momentum_alpha

score, raw = compute_momentum_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    long_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `long_span` | 12 | EWMA span for the momentum signal |
| `short_span` | None | Short-term reversal subtraction |
| `vol_span` | 13 | EWMA span for vol normalisation |

### Low Beta

Cross-sectional low-beta score ("betting against beta").

```python
from optimalportfolios.alphas import compute_low_beta_alpha

score, raw_beta = compute_low_beta_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    beta_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_span` | 12 | EWMA span for beta estimation |

### Managers Alpha

Cross-sectional score from factor model regression residuals.

```python
from optimalportfolios.alphas import compute_managers_alpha

score, raw_alpha = compute_managers_alpha(
    prices=asset_prices,
    risk_factor_prices=factor_prices,
    estimated_betas=rolling_data.get_y_betas(),
    returns_freq='ME',
    alpha_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_span` | 12 | EWMA span for smoothing excess returns |

### Mixed-Frequency Support

All functions accept `returns_freq` as string or per-asset `pd.Series`:

```python
# mixed: equities monthly, alternatives quarterly
returns_freq = pd.Series({'SPY': 'ME', 'EZU': 'ME', 'HF_Macro': 'QE', 'PE': 'QE'})
score, raw = compute_momentum_alpha(prices, returns_freq=returns_freq, ...)
```

### Within-Group Scoring

When `group_data` is provided, cross-sectional scoring is computed within
each group independently:

```python
group_data = pd.Series({'SPY': 'Equity', 'TLT': 'Bonds', 'GLD': 'Real Assets'})
score, raw = compute_momentum_alpha(prices, group_data=group_data, ...)
```

## AlphasData

Container for alpha computation results:

```python
from optimalportfolios.alphas import AlphasData

data = AlphasData(
    alpha_scores=combined_scores,   # (T × N) — input to optimiser
    momentum_score=mom_score,       # component diagnostics
    beta_score=beta_score,
    managers_scores=mgr_score,
    momentum=raw_momentum,          # raw signals
    beta=raw_beta,
    managers_alphas=raw_alpha,
)

# snapshot at a single date
snapshot = data.get_alphas_snapshot(date=pd.Timestamp('2024-12-31'))
```

## Test Coverage

| Test | What it verifies |
|------|-----------------|
| `MOMENTUM_SINGLE_FREQ` | Basic momentum at single frequency |
| `MOMENTUM_MIXED_FREQ` | Mixed-frequency dispatch |
| `MOMENTUM_GROUPED` | Within-group vs global scoring |
| `LOW_BETA_SINGLE_FREQ` | Basic low-beta |
| `LOW_BETA_MIXED_FREQ` | Mixed-frequency low-beta |
| `MANAGERS_ALPHA` | Regression alpha with FactorCovarEstimator |

## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation
Using Multi-Asset Tradable Factors",
*Working Paper*.