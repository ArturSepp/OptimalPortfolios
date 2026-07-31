"""
Tests for the shared paper-data layer: universe counts, benchmark identities,
manifest integrity, and loader parity. Run from the cma_data folder:
    python -m pytest tests/ -q
No production data or network required; snapshot tests skip when no
snapshot is present.
"""
# packages
import importlib.util
import sys
import pytest
from pathlib import Path

CMA_DATA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cma_data', CMA_DATA / '__init__.py',
                                              submodule_search_locations=[str(CMA_DATA)])
cma_data = importlib.util.module_from_spec(spec)
sys.modules['cma_data'] = cma_data
spec.loader.exec_module(cma_data)


def test_universe_counts():
    universe = cma_data.get_universe()
    assert len(universe) == 18
    classes = [cma_data.ASSET_CLASSES[t] for t in universe.index]
    assert classes.count('Bonds') == 5
    assert classes.count('Equities') == 7
    assert classes.count('Alternatives') == 6


def test_admission_policy_matches_paper():
    w = cma_data.get_admission_policy()
    assert w['MP503001 Index'] == 0.50      # PE recut
    assert w['EHFI804 Index'] == 1.00
    assert w['BCOMGCTR Index'] == 0.25
    assert float(w[[t for t, c in cma_data.ASSET_CLASSES.items()
                    if c != 'Alternatives']].abs().sum()) == 0.0


def test_benchmarks_sum_to_one():
    benchmarks = cma_data.get_all_benchmarks()
    for mandate in cma_data.MANDATES:
        assert abs(benchmarks[mandate].sum() - 1.0) < 1e-9


def test_balanced_with_alts_matches_paper_panel():
    bench = cma_data.get_benchmark_weights(mandate='Balanced with Alts')
    assert abs(bench['NDDUUS Index'] - 0.2867) < 5e-4      # US 28.7 in Panel A
    assert abs(bench['MP503001 Index'] - 0.15) < 1e-9      # PE 15.0
    assert abs(bench[[t for t, c in cma_data.ASSET_CLASSES.items()
                      if c == 'Bonds']].sum() - 0.28) < 1e-9


def test_unknown_mandate_raises():
    with pytest.raises(ValueError):
        cma_data.get_benchmark_weights(mandate='Aggressive')


SNAPSHOT = CMA_DATA / 'snapshots' / '2026q2'


@pytest.mark.skipif(not SNAPSHOT.exists(), reason='no snapshot present')
def test_snapshot_manifest_verifies():
    """every present file matches its hash; absent files are reported, not fatal.

    The three return panels are not redistributed publicly, so a public checkout
    reports them as absent. Config files are always present or verify_manifest
    raises.
    """
    manifest, absent = cma_data.verify_manifest(snapshot_path=SNAPSHOT)
    assert manifest['tag'] == '2026q2'
    assert len(manifest['file_sha256']) >= 6
    assert set(absent).issubset(set(cma_data.loaders.PANEL_FILES))
    for name in cma_data.loaders.CONFIG_FILES:
        assert name not in absent


@pytest.mark.skipif(not SNAPSHOT.exists(), reason='no snapshot present')
def test_snapshot_loads_and_aligns():
    inputs = cma_data.load_snapshot(tag='2026q2')
    assert list(inputs.assets.index) == list(cma_data.get_universe().index)
    assert inputs.betas.shape == (18, 9)
    assert inputs.factor_covar.shape == (9, 9)
    # the return panels are not redistributed publicly; assert only when present
    if inputs.has_panel('asset_excess_logreturns'):
        assert len(inputs.asset_excess_logreturns) >= 290    # ~300 month rows on the window
    # tampering detection: corrupting one byte must fail verification
    target = SNAPSHOT / 'betas.csv'
    original = target.read_bytes()
    try:
        target.write_bytes(original + b' ')
        with pytest.raises(ValueError):
            cma_data.verify_manifest(snapshot_path=SNAPSHOT)
    finally:
        target.write_bytes(original)
