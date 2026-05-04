"""
Universe definition for the MATF-CMA paper exhibits.

The 17-asset paper universe used by both portfolio optimisation
(`run_optimisation.py`) and the bootstrap exhibits (`run_bootstrap.py`).
Granular fixed-income sub-asset classes are aggregated into 'Other Fixed
Income' (LGT_OTHERFI), and broad commodities + REITs are aggregated into
'Real Assets' (LGT_REAL_ASSETS). This keeps the universe well-identified
at T_eff ≈ 26 monthly observations and avoids double-counting commodity
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
        7 fixed income + 5 equities + 5 alternatives.
    """
    instrument_ticker_map = {
        # Fixed income (7)
        'LGTRTRUH Index':        'Global Government',
        'LGCPTRUH Index':        'Global IG Bonds',
        'H23059US Index':        'Global HY Bonds',
        'EMUSTRUU Index':        'EM HC Bonds',
        'LGT_OTHERFI Index':     'Other Fixed Income',
        'LF94TRUH Index':        'Global Inflation-Linked',
        'H24641US Index':        'Global Convertibles',
        # Equities (5)
        'NDDUUS Index':          'MSCI US',
        'MSDEE15N Index':        'MSCI Europe',
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
