"""
the cross-signal comparison tables in ``alphas.signal_diagnostics``.

``compare_signal_diagnostics`` and ``compare_signal_ic_ir`` take a dict of per-signal
``qis.SignalDiagnosticsResult`` objects and stack them into one table for a research
write-up. Neither had a collected test.

The interesting behaviour is entirely in the *shape* of the result, not the statistics —
those come from ``qis``. Given several signals and several horizons, the pair either returns
a ``(signal, horizon)`` MultiIndex or, when a single horizon is requested, a flat index named
``signal``. A signal that lacks the requested horizon is skipped with a warning rather than
raising, so one short-history signal cannot take down a comparison over a dozen others — and
that skip is the branch most likely to swallow something silently.

The fixtures construct ``SignalDiagnosticsResult`` directly, which is a plain dataclass. The
numbers are stated, so the assertions are about routing and labelling.
"""
# packages
import logging
from typing import Dict
import numpy as np
import pandas as pd
import qis
# optimalportfolios
from optimalportfolios.alphas.signal_diagnostics import (
    compare_signal_diagnostics, compare_signal_ic_ir)

HORIZONS = ['1M', '3M', '12M']


def make_result(horizons=None, seed: int = 0) -> qis.SignalDiagnosticsResult:
    """A diagnostics result whose pooled table carries one row per horizon."""
    horizons = horizons or HORIZONS
    rng = np.random.default_rng(seed)
    pooled = pd.DataFrame({'beta': rng.normal(0.02, 0.01, len(horizons)),
                           't_stat': rng.normal(2.0, 0.5, len(horizons)),
                           'r2': rng.uniform(0.01, 0.05, len(horizons))},
                          index=pd.Index(horizons, name='horizon'))
    return qis.SignalDiagnosticsResult(
        pooled_universe=pooled, per_group={}, pairs=pd.DataFrame(),
        horizon_labels=list(horizons), group_order=[])


def make_results(**overrides) -> Dict[str, qis.SignalDiagnosticsResult]:
    """Two signals over the same horizons unless overridden."""
    results = {'momentum': make_result(seed=1), 'low_beta': make_result(seed=2)}
    results.update(overrides)
    return results


# --------------------------------------------------------------------------- #
# compare_signal_diagnostics
# --------------------------------------------------------------------------- #
def test_comparison_stacks_every_signal_and_horizon() -> None:
    """with no horizon requested the table keeps the (signal, horizon) MultiIndex"""
    table = compare_signal_diagnostics(make_results())
    assert isinstance(table.index, pd.MultiIndex)
    assert table.index.names == ['signal', 'horizon']
    assert len(table) == 2 * len(HORIZONS)
    assert set(table.index.get_level_values('signal')) == {'momentum', 'low_beta'}
    assert 'beta' in table.columns


def test_comparison_restricted_to_one_horizon_is_indexed_by_signal() -> None:
    """asking for a single horizon flattens the index, which is the write-up shape"""
    table = compare_signal_diagnostics(make_results(), horizon='3M')
    assert not isinstance(table.index, pd.MultiIndex)
    assert table.index.name == 'signal'
    assert list(table.index) == ['momentum', 'low_beta']


def test_comparison_skips_a_signal_missing_the_requested_horizon(caplog) -> None:
    """a short-history signal is dropped with a warning, not allowed to raise"""
    results = make_results(short=make_result(horizons=['1M'], seed=3))
    with caplog.at_level(logging.WARNING):
        table = compare_signal_diagnostics(results, horizon='12M')
    assert 'short' in caplog.text and '12M' in caplog.text
    assert 'short' not in table.index
    assert set(table.index) == {'momentum', 'low_beta'}


def test_comparison_of_no_signals_is_an_empty_frame() -> None:
    """an empty input is an empty table, not a KeyError"""
    assert compare_signal_diagnostics({}).empty


def test_comparison_where_every_signal_is_skipped_is_empty(caplog) -> None:
    """if nothing survives the horizon filter the result is empty rather than malformed"""
    with caplog.at_level(logging.WARNING):
        table = compare_signal_diagnostics(make_results(), horizon='not_a_horizon')
    assert table.empty


# --------------------------------------------------------------------------- #
# compare_signal_ic_ir
# --------------------------------------------------------------------------- #
def test_ic_ir_comparison_stacks_every_signal_and_horizon() -> None:
    """the IC-IR table has the same two shapes as the pooled-regression one"""
    table = compare_signal_ic_ir(make_results())
    assert isinstance(table.index, pd.MultiIndex)
    assert table.index.names == ['signal', 'horizon']
    assert set(table.index.get_level_values('signal')) == {'momentum', 'low_beta'}


def test_ic_ir_comparison_restricted_to_one_horizon_is_indexed_by_signal() -> None:
    """same flattening rule as the pooled comparison"""
    table = compare_signal_ic_ir(make_results(), horizon='3M')
    assert table.index.name == 'signal'
    assert list(table.index) == ['momentum', 'low_beta']


def test_ic_ir_comparison_skips_a_signal_missing_the_horizon(caplog) -> None:
    """the skip-with-warning behaviour is shared by both comparison functions"""
    results = make_results(short=make_result(horizons=['1M'], seed=4))
    with caplog.at_level(logging.WARNING):
        table = compare_signal_ic_ir(results, horizon='12M')
    assert 'short' not in table.index


def test_ic_ir_comparison_of_no_signals_is_an_empty_frame() -> None:
    """an empty input is an empty table here too"""
    assert compare_signal_ic_ir({}).empty
