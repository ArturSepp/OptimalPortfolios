"""
align_rolling_clusters: stable ids under relabelling, movement under movement.

The property under test is the reason the function exists. ``fcluster`` numbers
a partition by dendrogram traversal, so two consecutive estimation dates can
describe the same grouping with different integers; a chart coloured by the raw
label would then show migrations that never happened. These checks pin the two
halves of the contract — a pure relabelling must produce zero reassignments, and
a real move must produce a non-zero count — and the second is what stops the
function from passing by always returning a constant.

Seeded and self-contained: partitions are written out by hand, no estimator runs.

Run:
    pytest optimalportfolios/alphas/signals/tests/align_rolling_clusters_test.py
"""
# packages
import pandas as pd
import pytest
from typing import Dict
# qis / project
from optimalportfolios.alphas.signals.utils import align_rolling_clusters


TICKERS = ['A', 'B', 'C', 'D', 'E', 'F']
DATES = pd.date_range('2025-03-31', periods=3, freq='QE')


def make_rolling(labels_by_date: Dict[pd.Timestamp, Dict[str, str]]
                 ) -> Dict[pd.Timestamp, pd.Series]:
    """the shape ``extract_rolling_clusters`` returns."""
    return {date: pd.Series(labels) for date, labels in labels_by_date.items()}


def test_pure_relabelling_produces_stable_ids_and_no_reassignments() -> None:
    """same grouping, different integers: the aligned ids must not move."""
    grouping = {'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 3, 'F': 3}
    permutations = [{1: 1, 2: 2, 3: 3}, {1: 3, 2: 1, 3: 2}, {1: 2, 2: 3, 3: 1}]
    rolling = make_rolling({
        date: {ticker: f"QE:{permutation[cluster]}"
               for ticker, cluster in grouping.items()}
        for date, permutation in zip(DATES, permutations)})

    aligned, n_reassigned = align_rolling_clusters(rolling)

    first = aligned[DATES[0]]
    for date in DATES:
        assert aligned[date].equals(first), f"labels moved at {date}"
    assert int(n_reassigned.sum()) == 0
    # and the grouping itself survived the relabelling
    assert (first['A'] == first['B']) and (first['A'] != first['C'])


def test_a_real_move_registers() -> None:
    """one member changes cluster: exactly one reassignment, on that date.

    The prove-fail half. A function that always returned constant ids would pass
    the check above and fail this one.
    """
    before = {'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 3, 'F': 3}
    after = dict(before, C=3)                     # C leaves cluster 2 for cluster 3
    rolling = make_rolling({
        DATES[0]: {t: f"QE:{c}" for t, c in before.items()},
        DATES[1]: {t: f"QE:{c}" for t, c in after.items()}})

    aligned, n_reassigned = align_rolling_clusters(rolling)

    assert int(n_reassigned.loc[DATES[0]]) == 0
    assert int(n_reassigned.loc[DATES[1]]) == 1
    assert aligned[DATES[1]]['C'] == aligned[DATES[1]]['E']
    assert aligned[DATES[1]]['C'] != aligned[DATES[0]]['C']
    # the instruments that did not move keep their ids
    for ticker in ('A', 'B', 'D', 'E', 'F'):
        assert aligned[DATES[1]][ticker] == aligned[DATES[0]][ticker], ticker


def test_frequencies_are_aligned_independently() -> None:
    """a monthly bucket and a quarterly one never trade ids.

    The estimator partitions each frequency separately, so an 'ME' cluster and a
    'QE' cluster are not candidates for the same identity. Prefixes are
    re-emitted unchanged.
    """
    rolling = make_rolling({
        DATES[0]: {'A': 'ME:1', 'B': 'ME:1', 'C': 'QE:1', 'D': 'QE:1'},
        DATES[1]: {'A': 'ME:9', 'B': 'ME:9', 'C': 'QE:7', 'D': 'QE:7'}})

    aligned, n_reassigned = align_rolling_clusters(rolling)

    assert int(n_reassigned.sum()) == 0
    for date in (DATES[0], DATES[1]):
        assert aligned[date]['A'].startswith('ME:')
        assert aligned[date]['C'].startswith('QE:')
    assert aligned[DATES[1]]['A'] == aligned[DATES[0]]['A']
    assert aligned[DATES[1]]['C'] == aligned[DATES[0]]['C']


def test_a_new_cluster_takes_a_fresh_id() -> None:
    """a split does not silently reuse the id of the group it left."""
    rolling = make_rolling({
        DATES[0]: {'A': 'QE:1', 'B': 'QE:1', 'C': 'QE:1'},
        DATES[1]: {'A': 'QE:1', 'B': 'QE:1', 'C': 'QE:2'}})

    aligned, _ = align_rolling_clusters(rolling)

    assert aligned[DATES[1]]['A'] == aligned[DATES[0]]['A']
    assert aligned[DATES[1]]['C'] != aligned[DATES[0]]['C']
    assert aligned[DATES[1]]['C'] != aligned[DATES[1]]['A']


def test_a_late_arriving_instrument_does_not_disturb_the_others() -> None:
    """an instrument absent at the first date joins without renumbering anyone."""
    rolling = make_rolling({
        DATES[0]: {'A': 'QE:1', 'B': 'QE:1', 'C': 'QE:2'},
        DATES[1]: {'A': 'QE:2', 'B': 'QE:2', 'C': 'QE:1', 'D': 'QE:1'}})

    aligned, n_reassigned = align_rolling_clusters(rolling)

    assert int(n_reassigned.loc[DATES[1]]) == 0
    assert aligned[DATES[1]]['D'] == aligned[DATES[1]]['C']


def test_empty_input_returns_empty() -> None:
    """no dates in, nothing out — not a crash."""
    aligned, n_reassigned = align_rolling_clusters({})
    assert aligned == {}
    assert n_reassigned.empty


def test_a_non_series_assignment_raises() -> None:
    """the input shape is checked, with the offending type in the message."""
    with pytest.raises(ValueError, match='must be a pd.Series'):
        align_rolling_clusters({DATES[0]: {'A': 'QE:1'}})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
