"""
The shared paper universe: one source of truth for both paper packages.

Defines the 18-asset universe of the MATF-CMA (JPM) and Achievable Sharpe
(FAJ) papers: tickers, sleeve names, asset classes, the paper admission
policy, the legacy nine-factor and custom eleven-factor panels, and the common
bootstrap window. The FAJ paper pins the nine-factor panel; the JPM paper pins
the custom eleven-factor panel. Both papers import these definitions through
cma_data; neither paper defines its own universe. Estimated quantities (betas,
alphas, vols, premia) do NOT live
here — they live in versioned snapshots (see loaders.py).

Units and conventions: tickers are Bloomberg tickers with the ' Index'
suffix. The bootstrap window is 300 months, matching Appendix B of the JPM
paper. Admission weights are the PAPER policy (PE recut to w = 0.5).

Does not belong here: benchmark weights (benchmarks.py) and data loading
(loaders.py).
"""
# packages
import pandas as pd
from typing import Dict

# The 18-asset paper universe: ticker -> paper sleeve name.
PAPER_UNIVERSE: Dict[str, str] = {
    # Fixed income (5)
    'LGTRTRUH Index': 'Global Government',
    'LGCPTRUH Index': 'Global IG Bonds',
    'H23059US Index': 'Global HY Bonds',
    'H04386US Index': 'EM HC Bonds',
    'LF94TRUH Index': 'Global Inflation-Linked',
    # Equities (7)
    'NDDUUS Index': 'US',
    'MSDEXKSN Index': 'Europe ex-UK',
    'NDDLUK Index': 'UK',
    'NDDLSZ Index': 'Switzerland',
    'NDDLJN Index': 'Japan',
    'M1APJ Index': 'Asia ex-Japan',
    'M1EFZ Index': 'EM ex-Asia',
    # Alternatives (6)
    'MP503001 Index': 'Private Equity',
    'MP503008 Index': 'Private Credit',
    'MP503009 Index': 'Real Estate',
    'EHFI804 Index': 'Insurance-Linked',
    'HFRIFWI Index': 'Hedge Funds',
    'BCOMGCTR Index': 'Gold',
}

ASSET_CLASSES: Dict[str, str] = {
    **{t: 'Bonds' for t in list(PAPER_UNIVERSE)[:5]},
    **{t: 'Equities' for t in list(PAPER_UNIVERSE)[5:12]},
    **{t: 'Alternatives' for t in list(PAPER_UNIVERSE)[12:]},
}

# Paper admission policy (JPM R2 Decision Two): PE recut from 1.0 to 0.5.
ADMISSION_POLICY: Dict[str, float] = {
    'MP503001 Index': 0.50,   # Private Equity, recut
    'MP503008 Index': 0.50,   # Private Credit
    'MP503009 Index': 0.00,   # Real Estate, not admitted
    'EHFI804 Index': 1.00,    # Insurance-Linked
    'HFRIFWI Index': 1.00,    # Hedge Funds
    'BCOMGCTR Index': 0.25,   # Gold
}

# The nine-factor MATF panel, canonical order (matches production sheets).
FACTORS = ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation', 'Commodities',
           'Private Equity', 'Rates Vol', 'Fx']

# JPM MATF_CUSTOM adoption decision D1, 2026-08-12; FAJ continues to use FACTORS.
FACTORS_CUSTOM = ['Equity', 'Rates', 'Credit', 'Credit EM', 'Carry G10', 'Carry EM',
                  'Inflation', 'Commodities', 'Private Equity', 'Rates Vol', 'Fx']

# Common bootstrap window for both papers: 300 months.
BOOTSTRAP_START = '2001-07-31'
BOOTSTRAP_END = '2026-06-30'
BOOTSTRAP_MONTHS = 300


def get_universe() -> pd.Series:
    """the 18-asset universe as a Series: index = tickers, values = sleeve names."""
    return pd.Series(PAPER_UNIVERSE, name='sleeve')


def get_admission_policy() -> pd.Series:
    """paper admission weights w_i over the full universe (zero outside alternatives)."""
    w = pd.Series(0.0, index=list(PAPER_UNIVERSE), name='w_paper')
    for ticker, weight in ADMISSION_POLICY.items():
        w[ticker] = weight
    return w
