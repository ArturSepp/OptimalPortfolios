"""
Universe definition for the MATF-CMA paper exhibits.

The 17-asset paper universe used by both portfolio optimisation
(`run_optimisation.py`) and the bootstrap exhibits (`run_bootstrap.py`):
5 fixed income + 7 equities + 5 alternatives. The bond sleeve is broadly
aligned with the Bloomberg Multiverse Index composition (Securitized and
Government-Related folded equally into Global Government and Global IG),
augmented with EM hard-currency and Global Inflation-Linked sleeves.
The equity sleeve covers the US, EMU (MSCI Europe ex UK ex Switzerland),
UK, Switzerland, Japan, Asia ex-Japan, and EM ex-Asia, providing complete
geographic coverage of developed and emerging equity markets. The
alternatives sleeve uses 'LGT_REAL_ASSETS' as the broad commodities +
REITs aggregator. This keeps the universe well-identified at
T_eff ≈ 25 monthly observations and avoids double-counting commodity
exposure across separate sub-indices.
"""
from __future__ import annotations

import pandas as pd


def load_paper_assets_short(add_crypto: bool = False) -> pd.Series:
    """Return the 17-asset paper universe as a Series mapping ticker → name.

    Parameters
    ----------
    add_crypto : bool, default False
        If True, append Bitcoin (XBTUSD Curncy) as an 18th asset for crypto-
        sensitivity analyses. Not used in the headline paper exhibits.

    Returns
    -------
    pd.Series
        Index = Bloomberg tickers ('LGTRTRUH Index', 'LGCPTRUH Index', ...),
        values = display names ('Global Government', 'Global IG Bonds', ...).
        Order matches the canonical row order used in all paper exhibits:
        5 fixed income + 7 equities + 5 alternatives.
    """
    instrument_ticker_map = {
        # Fixed income (5)
        'LGTRTRUH Index':        'Global Government',
        'LGCPTRUH Index':        'Global IG Bonds',
        'H23059US Index':        'Global HY Bonds',
        'EMUSTRUU Index':        'EM HC Bonds',
        'LF94TRUH Index':        'Global Inflation-Linked',
        # Equities (7)
        'NDDUUS Index':          'MSCI US',
        'MSDEE15N Index':        'MSCI Europe',
        'NDDLUK Index':          'MSCI UK',
        'SLIC Index':            'Swiss SLI',
        'NDDLJN Index':          'MSCI Japan',
        'M1APJ Index':           'MSCI Asia Ex-Japan',
        'M1EFZ Index':           'MSCI EM ex-Asia',
        # Alternatives (5)
        'MP503001 Index':        'Private Equity',
        'MP503008 Index':        'Private Credit',
        'LGT_ILS Index':         'Insurance-Linked',
        'HFRXGL Index':          'Hedge Funds',
        'LGT_REAL_ASSETS Index': 'Real Assets',
    }
    if add_crypto:
        instrument_ticker_map['XBTUSD Curncy'] = 'Bitcoin'
    return pd.Series(instrument_ticker_map)
