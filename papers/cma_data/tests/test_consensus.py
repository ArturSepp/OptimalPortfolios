"""
Tests for the Consensus provider vector (Horizon 2025 survey mapping).

Checks the transcribed survey table against spot values printed in the source
PDF (Exhibit 17 and Exhibit 20), the mapping's coverage of the 18-asset
universe, and the providers-table schema of build_consensus_provider().
Does not belong here: tests of the MATF snapshot (test_cma_data.py).
"""
# packages
import numpy as np

from cma_data.consensus import (build_consensus_provider, get_horizon_survey,
                                get_horizon_distributions, ProviderSource)
from cma_data.universe import PAPER_UNIVERSE


def test_survey_spot_values():
    survey = get_horizon_survey()
    # Exhibit 17 printed values (10-year, all 41 advisors)
    assert np.isclose(survey.loc['US Equity - Large Cap', 'arith_10y'], 0.0767)
    assert np.isclose(survey.loc['US Equity - Large Cap', 'geom_10y'], 0.0639)
    assert np.isclose(survey.loc['Hedge Funds', 'geom_10y'], 0.0592)
    assert np.isclose(survey.loc['Private Equity', 'arith_10y'], 0.1151)
    assert np.isclose(survey.loc['Commodities', 'stdev'], 0.1783)
    assert np.isclose(survey.loc['Inflation', 'geom_10y'], 0.0238)


def test_survey_arithmetic_exceeds_geometric():
    survey = get_horizon_survey()
    assert (survey['arith_10y'] >= survey['geom_10y'] - 1e-12).all()
    assert (survey['arith_20y'] >= survey['geom_20y'] - 1e-12).all()


def test_distribution_tails_match_paper_fragments():
    # the R2 draft quotes these two tails (Exhibit 20 maxima, geometric 10Y)
    dist = get_horizon_distributions()
    assert np.isclose(dist.loc['Hedge Funds', 'max'], 0.085)
    assert np.isclose(dist.loc['Commodities', 'max'], 0.075)
    ordered = dist[['min', 'p25', 'p50', 'p75', 'max']].values
    assert (np.diff(ordered, axis=1) >= -1e-12).all()


def test_consensus_covers_universe():
    consensus = build_consensus_provider()
    assert list(consensus.index) == list(PAPER_UNIVERSE)
    held = consensus['source'] == ProviderSource.HELD_AT_MATF.value
    assert list(consensus.index[held]) == ['EHFI804 Index']   # ILS only
    assert consensus.loc[~held, 'total_cma_arith'].notna().all()
    assert consensus.loc[held, 'total_cma_arith'].isna().all()


def test_consensus_spot_values():
    consensus = build_consensus_provider()
    assert np.isclose(consensus.loc['NDDUUS Index', 'total_cma_arith'], 0.0767)
    assert np.isclose(consensus.loc['BCOMGCTR Index', 'total_cma_arith'], 0.0620)
    assert np.isclose(consensus.loc['MP503008 Index', 'total_cma_arith'], 0.0858)
    # the four developed ex-US equity sleeves share one survey line
    dev = ['MSDEXKSN Index', 'NDDLUK Index', 'NDDLSZ Index', 'NDDLJN Index']
    assert consensus.loc[dev, 'total_cma_arith'].nunique() == 1
