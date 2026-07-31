"""
Mandate benchmark weights for the shared paper universe.

Implements the two-level benchmark construction of the JPM paper (Appendix
F): top-level asset-class allocations per mandate, mapped to instrument
weights through within-class decomposition weights, with alternatives funded
proportionally from bonds and equities. Weight of instrument i is
w_i = W_class(mandate) * d_i with sum(d_i) = 1 within each class.

Provenance of the decomposition weights: fixed income and equities from the
'universe weight' sheet of the FAJ universe_snapshot workbook (Bloomberg
Multiverse and MSCI ACWI decompositions), with the Europe and Switzerland
lines carried to the 18-universe tickers. Alternatives split the former
Real Assets weight (0.20) into Real Estate (0.10) and Gold (0.10),
matching the JPM paper's Balanced benchmark.

CONFIRMED DEFECT in the R2 exhibit build (checked 2026-07-30, register D8):
Panel A of the R2 provider table prints Asia ex-Japan 0.9 / EM ex-Asia 4.5,
a pairwise exchange of this construction's 4.52 / 0.88. Every other Panel A
row matches this file to rounding, and the printed optima are internally
consistent with the transposed pair (caps and floors of 0.9 and 4.5), so
the swap sits in the benchmark INPUT of the exhibit build, not in a label.
This file carries the correct, ACWI-consistent values; fix the exhibit
build's benchmark and regenerate the mandate exhibits.

Does not belong here: universe identity (universe.py) and data (snapshots).
"""
# packages
import pandas as pd
from typing import Dict

from .universe import PAPER_UNIVERSE, ASSET_CLASSES

MANDATES = ['Income w/o Alts', 'Low w/o Alts', 'Balanced w/o Alts', 'Growth w/o Alts',
            'Income with Alts', 'Low with Alts', 'Balanced with Alts', 'Growth with Alts']

# Top-level class allocations per mandate (Bonds, Equities, Alternatives).
CLASS_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    'Income w/o Alts': {'Bonds': 1.00, 'Equities': 0.00, 'Alternatives': 0.00},
    'Low w/o Alts': {'Bonds': 0.70, 'Equities': 0.30, 'Alternatives': 0.00},
    'Balanced w/o Alts': {'Bonds': 0.40, 'Equities': 0.60, 'Alternatives': 0.00},
    'Growth w/o Alts': {'Bonds': 0.00, 'Equities': 1.00, 'Alternatives': 0.00},
    'Income with Alts': {'Bonds': 0.90, 'Equities': 0.00, 'Alternatives': 0.10},
    'Low with Alts': {'Bonds': 0.56, 'Equities': 0.24, 'Alternatives': 0.20},
    'Balanced with Alts': {'Bonds': 0.28, 'Equities': 0.42, 'Alternatives': 0.30},
    'Growth with Alts': {'Bonds': 0.00, 'Equities': 0.60, 'Alternatives': 0.40},
}

# Within-class decomposition weights d_i, sum to 1 per class.
DECOMPOSITION_WEIGHTS: Dict[str, float] = {
    # Fixed income (Bloomberg Multiverse decomposition)
    'LGTRTRUH Index': 0.5430,
    'LGCPTRUH Index': 0.3210,
    'H23059US Index': 0.0400,
    'H04386US Index': 0.0400,
    'LF94TRUH Index': 0.0560,
    # Equities (MSCI ACWI regional decomposition)
    'NDDUUS Index': 0.6827,
    'MSDEXKSN Index': 0.0886,
    'NDDLUK Index': 0.0315,
    'NDDLSZ Index': 0.0200,
    'NDDLJN Index': 0.0486,
    'M1APJ Index': 0.1076,
    'M1EFZ Index': 0.0210,
    # Alternatives (indicative allocation; Real Assets 0.20 split RE/Gold)
    'MP503001 Index': 0.50,
    'MP503008 Index': 0.10,
    'MP503009 Index': 0.10,
    'EHFI804 Index': 0.10,
    'HFRIFWI Index': 0.10,
    'BCOMGCTR Index': 0.10,
}


def get_benchmark_weights(mandate: str = 'Balanced with Alts') -> pd.Series:
    """instrument-level benchmark weights for one mandate over the 18-asset universe."""
    if mandate not in CLASS_ALLOCATIONS:
        raise ValueError(f"unknown mandate, got {mandate!r}; choose from {MANDATES}")
    alloc = CLASS_ALLOCATIONS[mandate]
    weights = {t: alloc[ASSET_CLASSES[t]] * DECOMPOSITION_WEIGHTS[t] for t in PAPER_UNIVERSE}
    return pd.Series(weights, name=mandate)


def get_all_benchmarks() -> pd.DataFrame:
    """all eight mandate benchmarks: index = tickers, one column per mandate."""
    return pd.concat([get_benchmark_weights(mandate=m) for m in MANDATES], axis=1)
