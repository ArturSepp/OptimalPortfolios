"""Focused construction tests for the U1 ME/36 long-short curiosity run."""
import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_me36_long_short import (
    _dollar_neutral_books,
)


def test_overlap_is_removed_before_exact_dollar_neutral_normalisation() -> None:
    """Cancel an ambiguous asset and retain unit long, unit short, and gross two."""
    dates = pd.DatetimeIndex(["2026-01-31"])
    long_book = pd.DataFrame([[0.5, 0.5, 0.0]], index=dates, columns=list("abc"))
    short_book = pd.DataFrame([[0.5, 0.0, 0.5]], index=dates, columns=list("abc"))
    long_side, short_side, signed, diagnostics = _dollar_neutral_books(
        long_book, short_book
    )
    assert long_side.loc[dates[0]].to_dict() == {"a": 0.0, "b": 1.0, "c": 0.0}
    assert short_side.loc[dates[0]].to_dict() == {"a": 0.0, "b": 0.0, "c": 1.0}
    assert signed.loc[dates[0]].to_dict() == {"a": 0.0, "b": 1.0, "c": -1.0}
    assert diagnostics.loc[0, "net_exposure"] == 0.0
    assert diagnostics.loc[0, "gross_exposure"] == 2.0
