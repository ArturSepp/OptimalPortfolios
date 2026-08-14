"""
seeded NaN masks over the committed fixture.

`examples/data/multiasset.py` ships a clean panel: 292 monthly observations, 19 instruments, zero
NaN. `AGENTS.md` states the backtest layer is NaN-aware by design, so a fixture with no NaN cannot
exercise the property the repository says matters most.

These helpers put the defects back. They mask the committed fixture rather than generating a new
panel, deliberately: `qis/tests/synthetic_data.py` already exists, a second synthetic-panel
generator in this stack is the duplication the shared code core exists to prevent, and
`qis.datasets.generate_synthetic_universe` is not available at this package's declared
`qis>=5.0.5` floor. Masking keeps the numbers real and lets each test choose the defect it wants
instead of inheriting a generator's global design.

Every function is deterministic given its seed, takes numpy and pandas only, and never imports the
library under test.
"""
# packages
from typing import List

import numpy as np
import pandas as pd


def mask_late_listings(prices: pd.DataFrame, n_instruments: int = 3, n_periods: int = 60,
                       seed: int = 1) -> pd.DataFrame:
    """
    blank the first ``n_periods`` rows of ``n_instruments`` columns: a ragged start.

    The commonest shape in a real panel — an instrument that did not exist yet. Anything reading
    the first row to size the universe sees a different universe from anything reading the last.

    Args:
        prices: the clean panel
        n_instruments: how many columns start late
        n_periods: how many leading observations to blank
        seed: fixes which columns are chosen

    Returns:
        a copy with the leading block masked
    """
    out = prices.copy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(prices.columns.to_numpy(), size=n_instruments, replace=False)
    out.loc[out.index[:n_periods], chosen] = np.nan
    return out


def mask_delistings(prices: pd.DataFrame, n_instruments: int = 2, n_periods: int = 40,
                    seed: int = 2) -> pd.DataFrame:
    """
    blank the last ``n_periods`` rows of ``n_instruments`` columns: a delisting.

    The mirror of a late listing, and the one that breaks a backtest differently: the instrument
    is present when weights are first set and gone when they are next applied.

    Args:
        prices: the clean panel
        n_instruments: how many columns end early
        n_periods: how many trailing observations to blank
        seed: fixes which columns are chosen

    Returns:
        a copy with the trailing block masked
    """
    out = prices.copy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(prices.columns.to_numpy(), size=n_instruments, replace=False)
    out.loc[out.index[-n_periods:], chosen] = np.nan
    return out


def mask_scattered_gaps(prices: pd.DataFrame, fraction: float = 0.02,
                        seed: int = 3) -> pd.DataFrame:
    """
    blank a fraction of interior cells at random: missing observations mid-sample.

    Leading and trailing gaps are structural and usually handled. Interior gaps are the ones that
    reach an estimator, because a column with a hole in the middle still looks alive.

    Args:
        prices: the clean panel
        fraction: share of interior cells to blank, in [0, 1)
        seed: fixes which cells are chosen

    Returns:
        a copy with scattered interior cells masked
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction!r}")
    out = prices.copy()
    rng = np.random.default_rng(seed)
    interior = out.iloc[1:-1]
    mask = rng.random(interior.shape) < fraction
    out.iloc[1:-1] = interior.where(~mask)
    return out


def instruments_alive_at(prices: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    """
    the columns carrying a price at ``date``.

    The universe a point-in-time optimisation is entitled to see, and the reference a test
    compares an inclusion indicator against.

    Args:
        prices: the panel, masked or not
        date: the observation date

    Returns:
        column labels with a non-NaN price at that date, in panel order
    """
    row = prices.loc[date]
    return [str(c) for c in prices.columns[row.notna().to_numpy()]]
