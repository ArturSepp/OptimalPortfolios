"""
The Consensus provider vector: Horizon Actuarial 2025 survey mapped to the paper universe.

Builds the Consensus column of the providers table from the survey averages in
reference/horizon_actuarial_2025_average_assumptions.csv. Returns are the
survey's printed 10-year ARITHMETIC nominal total returns for a USD investor
(the providers-table basis), so no geometric-to-arithmetic lift is applied.
Every sleeve carries a source flag: PUBLISHED where the survey line matches the
sleeve in kind, CONVERTED where a proxy mapping applies judgment, and
HELD_AT_MATF where the survey has no line (the sleeve is held at the MATF value
and carries no cross-provider tilt, per the paper's provider-table convention).
Main entry points: build_consensus_provider() and get_horizon_survey().

Does not belong here: the anonymised provider vectors A-D (licensed inputs,
built by the untracked _local_build_providers_csv.py) and any MATF quantity
(snapshots via loaders.py).
"""
# packages
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional

from .universe import PAPER_UNIVERSE

REFERENCE_PATH = Path(__file__).parent / 'reference'
SURVEY_FILE = 'horizon_actuarial_2025_average_assumptions.csv'
DISTRIBUTIONS_FILE = 'horizon_actuarial_2025_distributions_10y.csv'
CONSENSUS_LABEL = 'Consensus'
SURVEY_VINTAGE = '2025-01'   # assumption sets effective around January 2025 (survey published 2025-08)


class ProviderSource(str, Enum):
    PUBLISHED = 'published'          # provider line matches the sleeve in kind
    CONVERTED = 'converted'          # proxied from a published line, mapping judgment applied
    HELD_AT_MATF = 'held_at_matf'    # no provider line; held at MATF, no cross-provider tilt


# sleeve ticker -> (survey category or None, source flag, mapping note)
HORIZON_MAP: Dict[str, Tuple[Optional[str], ProviderSource, str]] = {
    'LGTRTRUH Index': ('Non-US Debt - Developed', ProviderSource.CONVERTED,
                       'survey line is ex-US developed sovereigns; index is global incl. US, USD-hedged'),
    'LGCPTRUH Index': ('US Corporate Bonds - Core', ProviderSource.CONVERTED,
                       'US core aggregate proxies global IG hedged USD'),
    'H23059US Index': ('US Corporate Bonds - High Yield', ProviderSource.CONVERTED,
                       'US high yield proxies global HY hedged USD'),
    'H04386US Index': ('Non-US Debt - Emerging', ProviderSource.PUBLISHED, ''),
    'LF94TRUH Index': ('TIPS (Inflation-Protected)', ProviderSource.CONVERTED,
                       'US TIPS proxy global inflation-linked hedged USD'),
    'NDDUUS Index': ('US Equity - Large Cap', ProviderSource.PUBLISHED, ''),
    'MSDEXKSN Index': ('Non-US Equity - Developed', ProviderSource.CONVERTED,
                       'one survey line spans the four developed ex-US sleeves'),
    'NDDLUK Index': ('Non-US Equity - Developed', ProviderSource.CONVERTED,
                     'one survey line spans the four developed ex-US sleeves'),
    'NDDLSZ Index': ('Non-US Equity - Developed', ProviderSource.CONVERTED,
                     'one survey line spans the four developed ex-US sleeves'),
    'NDDLJN Index': ('Non-US Equity - Developed', ProviderSource.CONVERTED,
                     'one survey line spans the four developed ex-US sleeves'),
    'M1APJ Index': ('Non-US Equity - Emerging', ProviderSource.CONVERTED,
                    'index mixes EM Asia with developed Pacific; EM line is the closer proxy'),
    'M1EFZ Index': ('Non-US Equity - Emerging', ProviderSource.CONVERTED,
                    'one survey line spans both EM sleeves'),
    'MP503001 Index': ('Private Equity', ProviderSource.PUBLISHED, ''),
    'MP503008 Index': ('Private Debt', ProviderSource.PUBLISHED, ''),
    'MP503009 Index': ('Real Estate', ProviderSource.PUBLISHED, ''),
    'EHFI804 Index': (None, ProviderSource.HELD_AT_MATF,
                      'survey has no insurance-linked line'),
    'HFRIFWI Index': ('Hedge Funds', ProviderSource.PUBLISHED, ''),
    'BCOMGCTR Index': ('Commodities', ProviderSource.CONVERTED,
                       'broad commodity basket proxies gold; indicative only (paper footnote convention)'),
}


def get_horizon_survey() -> pd.DataFrame:
    """the transcribed survey averages: index = survey category, decimal returns and vols."""
    df = pd.read_csv(REFERENCE_PATH / SURVEY_FILE, index_col='category')
    if len(df.index) != 18:  # 17 asset classes + inflation
        raise ValueError(f"survey table row count changed, got {len(df.index)}")
    return df


def get_horizon_distributions() -> pd.DataFrame:
    """the survey's 10-year geometric return distributions: min/p25/p50/p75/max per category."""
    return pd.read_csv(REFERENCE_PATH / DISTRIBUTIONS_FILE, index_col='category')


def build_consensus_provider() -> pd.DataFrame:
    """the Consensus rows of the providers table over the 18-asset universe.

    Columns: provider, sleeve, total_cma_arith (decimal, NaN where held at
    MATF), source, survey_category, vintage, note. Index = tickers.
    """
    survey = get_horizon_survey()
    rows = {}
    for ticker, sleeve in PAPER_UNIVERSE.items():
        category, source, note = HORIZON_MAP[ticker]
        if category is not None and category not in survey.index:
            raise ValueError(f"unknown survey category, got {category!r} for {ticker!r}")
        rows[ticker] = {'provider': CONSENSUS_LABEL,
                        'sleeve': sleeve,
                        'total_cma_arith': survey.loc[category, 'arith_10y'] if category else float('nan'),
                        'source': source.value,
                        'survey_category': category or '',
                        'vintage': SURVEY_VINTAGE,
                        'note': note}
    return pd.DataFrame.from_dict(rows, orient='index')
