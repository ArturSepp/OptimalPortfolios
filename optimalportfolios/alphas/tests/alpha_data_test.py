"""
the alpha container's two accessors.

``AlphasData`` is mostly a bag of optional panels, but ``get_alphas_snapshot`` has one branch
worth pinning: when a component panel does not carry the requested date it falls back to that
panel's *last* row rather than to NaN. That is deliberate -- clusters and betas are estimated
on a slower cadence than the scores -- but it is also the one place where a snapshot can
silently mix dates, so the fallback is asserted explicitly rather than left implied. A missing
date in ``alpha_scores`` itself is a different matter and raises.

``to_dict`` exists so the container can go straight to ``qis.save_df_to_excel``, which chokes
on a None value; the test states that contract in those terms.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios.alphas.alpha_data import AlphasData

TICKERS = ['a', 'b', 'c']
DATES = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])


def panel(offset: float = 0.0, dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    """A small panel whose every value is distinguishable by row and column."""
    values = np.arange(len(dates) * len(TICKERS), dtype=float).reshape(len(dates), len(TICKERS))
    return pd.DataFrame(values + offset, index=dates, columns=TICKERS)


def test_the_snapshot_starts_from_the_alpha_scores_column() -> None:
    """With no components populated the snapshot is the score column alone."""
    data = AlphasData(alpha_scores=panel())
    snapshot = data.get_alphas_snapshot(date=DATES[1])
    assert list(snapshot.columns) == ['Alpha Scores']
    assert list(snapshot.index) == TICKERS
    pd.testing.assert_series_equal(snapshot['Alpha Scores'], panel().loc[DATES[1], :],
                                   check_names=False)


def test_populated_components_are_appended_in_the_documented_order() -> None:
    """Scores come first, then the per-signal scores, then the raw signals."""
    data = AlphasData(alpha_scores=panel(), momentum=panel(100.0), momentum_score=panel(200.0),
                      beta=panel(300.0))
    snapshot = data.get_alphas_snapshot(date=DATES[0])
    assert list(snapshot.columns) == ['Alpha Scores', 'Momentum Score', 'Momentum', 'Beta']


def test_a_component_without_the_requested_date_falls_back_to_its_last_row() -> None:
    """A slower-cadence panel contributes its most recent estimate, not NaN.

    This is the deliberate mixed-date behaviour: clusters and betas are estimated less often
    than the scores are, so the snapshot pairs today's score with the latest available beta.
    """
    slow = panel(offset=100.0, dates=pd.DatetimeIndex(['2023-11-30', '2023-12-31']))
    data = AlphasData(alpha_scores=panel(), beta=slow)
    snapshot = data.get_alphas_snapshot(date=DATES[2])
    pd.testing.assert_series_equal(snapshot['Beta'], slow.iloc[-1, :], check_names=False)


def test_a_component_carrying_the_date_uses_that_row_not_the_last_one() -> None:
    """The fallback must not fire when the date is present."""
    data = AlphasData(alpha_scores=panel(), beta=panel(100.0))
    snapshot = data.get_alphas_snapshot(date=DATES[0])
    pd.testing.assert_series_equal(snapshot['Beta'], panel(100.0).loc[DATES[0], :],
                                   check_names=False)
    assert not snapshot['Beta'].equals(panel(100.0).iloc[-1, :])


def test_a_date_missing_from_the_alpha_scores_raises() -> None:
    """The primary panel has no fallback: an unknown snapshot date is a caller error."""
    data = AlphasData(alpha_scores=panel())
    with pytest.raises(KeyError, match='is not in alpha_scores index'):
        data.get_alphas_snapshot(date=pd.Timestamp('2020-06-30'))


def test_to_dict_drops_the_unpopulated_fields() -> None:
    """qis.save_df_to_excel cannot take a None sheet, so None fields never reach it."""
    data = AlphasData(alpha_scores=panel(), momentum=panel(100.0), clusters=panel(1.0))
    as_dict = data.to_dict()
    assert set(as_dict) == {'alpha_scores', 'momentum', 'clusters'}
    assert all(isinstance(value, pd.DataFrame) for value in as_dict.values())


def test_to_dict_keys_are_the_field_names() -> None:
    """The keys are the dataclass field names, which is what the Excel sheets are named for."""
    data = AlphasData(alpha_scores=panel())
    assert list(data.to_dict()) == ['alpha_scores']
