"""Offline contracts for the Bloomberg-only crypto-paper update."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from papers.crypto_allocation_risk_2023.replication.bloomberg_snapshot import (
    ANALYSIS_COLUMNS,
    create_bloomberg_snapshot,
    load_bloomberg_prices,
    load_bloomberg_risk_free,
    verify_bloomberg_snapshot,
)
from papers.crypto_allocation_risk_2023.replication.published_update import (
    _common_reporting_start,
    _input_weights,
    _periods,
    _run_pair,
    summarize_crypto_weights,
)
from optimalportfolios.reports.marginal_backtest import OptimisationType


class FakeBloomberg:
    """Deterministic Bloomberg adapter recording every request contract."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs) -> pd.DataFrame:
        """Return source-shaped synthetic histories for one ticker group."""
        self.calls.append(kwargs)
        names = list(kwargs["tickers"].values())
        start = pd.Timestamp(kwargs["start_date"])
        end = pd.Timestamp(kwargs["end_date"])
        if names == ["RiskFree"]:
            index = pd.date_range(start, end, freq="B")
            return pd.DataFrame({"RiskFree": np.linspace(1.0, 4.0, len(index))}, index=index)
        if "SPY" in names:
            index = pd.date_range(start, end, freq="B")
            return pd.DataFrame(
                {
                    name: 100.0 * np.exp((0.0001 + 0.00001 * offset) * np.arange(len(index)))
                    for offset, name in enumerate(names)
                },
                index=index,
            )

        index = pd.date_range(start, end, freq="D")
        frame = pd.DataFrame(index=index, columns=names, dtype=float)
        for offset, name in enumerate(names):
            if name == "Macro":
                dates = pd.date_range(start, end, freq="ME")
            elif name in ("HFs", "SG CTA"):
                dates = pd.date_range(start, end, freq="B")
            else:
                dates = index
            values = 100.0 * np.exp((0.0002 + 0.00001 * offset) * np.arange(len(dates)))
            frame.loc[dates, name] = values
        frame.loc[frame.index < pd.Timestamp("2020-06-01"), "ETH"] = np.nan
        return frame


@pytest.fixture()
def fake_bloomberg() -> FakeBloomberg:
    """Return a fresh deterministic Bloomberg adapter."""
    return FakeBloomberg()


def test_snapshot_requests_and_transforms_are_bloomberg_only(
    tmp_path: Path, fake_bloomberg: FakeBloomberg
) -> None:
    """All source groups use explicit fields, dates, adjustment flags, and native frequency."""
    paths = create_bloomberg_snapshot(
        start_date="2020-01-01",
        as_of="2021-02-05",
        data_path=tmp_path,
        fetcher=fake_bloomberg,
    )

    assert len(fake_bloomberg.calls) == 3
    equity_call, index_call, rate_call = fake_bloomberg.calls
    assert equity_call["field"] == index_call["field"] == rate_call["field"] == "PX_LAST"
    assert equity_call["freq"] is index_call["freq"] is rate_call["freq"] is None
    assert equity_call["CshAdjNormal"] is equity_call["CshAdjAbnormal"] is True
    assert equity_call["CapChg"] is True
    for call in (index_call, rate_call):
        assert call["CshAdjNormal"] is call["CshAdjAbnormal"] is False
        assert call["CapChg"] is False
    assert pd.Timestamp(equity_call["end_date"]) == pd.Timestamp("2021-02-05")

    manifest = verify_bloomberg_snapshot(tag=paths.root.name, data_path=tmp_path)
    assert manifest["provider"] == "Bloomberg Desktop API"
    assert manifest["inclusive_as_of"] == "2021-02-05"

    legacy = load_bloomberg_prices(
        tag=paths.root.name, use_legacy_eth_proxy=True, data_path=tmp_path
    )
    observed = load_bloomberg_prices(
        tag=paths.root.name, use_legacy_eth_proxy=False, data_path=tmp_path
    )
    assert tuple(legacy.columns) == ANALYSIS_COLUMNS
    assert legacy.index.is_monotonic_increasing and not legacy.index.has_duplicates
    assert legacy.index.max() == pd.Timestamp("2021-02-05")
    assert legacy["ETH"].first_valid_index() == legacy["BTC"].first_valid_index()
    assert observed["ETH"].first_valid_index() == pd.Timestamp("2020-06-01")
    assert observed.loc[:"2020-05-31", "ETH"].isna().all()
    assert legacy.dropna().gt(0.0).all().all()

    # The Sunday month-end macro observation must become available on the following business day,
    # rather than being silently discarded by a bare asfreq('B').
    assert legacy.loc["2021-02-01", "Macro"] > legacy.loc["2021-01-29", "Macro"]

    risk_free = load_bloomberg_risk_free(tag=paths.root.name, data_path=tmp_path)
    assert risk_free.between(0.01, 0.04).all()


def test_manifest_detects_tampering(tmp_path: Path, fake_bloomberg: FakeBloomberg) -> None:
    """A modified licensed input fails before it can reach the analysis."""
    paths = create_bloomberg_snapshot(
        start_date="2020-01-01",
        as_of="2021-02-05",
        data_path=tmp_path,
        fetcher=fake_bloomberg,
    )
    paths.risk_free.write_text(
        paths.risk_free.read_text(encoding="utf-8") + "2021-02-08,0.99\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bloomberg_snapshot(tag=paths.root.name, data_path=tmp_path)


def test_missing_bloomberg_series_fails_closed(tmp_path: Path) -> None:
    """A requested but all-NaN Bloomberg security cannot create a snapshot."""
    fake = FakeBloomberg()

    def missing_eth(**kwargs) -> pd.DataFrame:
        frame = fake(**kwargs)
        if "ETH" in frame:
            frame["ETH"] = np.nan
        return frame

    with pytest.raises(ValueError, match="ETH"):
        create_bloomberg_snapshot(
            start_date="2020-01-01",
            as_of="2021-02-05",
            data_path=tmp_path,
            fetcher=missing_eth,
        )


def test_weight_summary_checks_published_statistics() -> None:
    """Weight summaries retain min, median, mean, max, last, and full-investment checks."""
    index = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"])
    weights = pd.DataFrame(
        {"Other": [0.99, 0.97, 0.95], "BTC": [0.01, 0.03, 0.05]}, index=index
    )
    summary = summarize_crypto_weights(weights=weights, crypto_asset="BTC")
    assert summary["min_crypto_weight"] == pytest.approx(0.01)
    assert summary["median_crypto_weight"] == pytest.approx(0.03)
    assert summary["mean_crypto_weight"] == pytest.approx(0.03)
    assert summary["max_crypto_weight"] == pytest.approx(0.05)
    assert summary["last_crypto_weight"] == pytest.approx(0.05)
    assert summary["max_weight_sum_error"] == pytest.approx(0.0)


def test_observed_history_waits_for_a_common_60_month_warmup() -> None:
    """All methods start together at the first quarter after 60 monthly returns."""
    index = pd.date_range("2018-02-28", periods=70, freq="ME")
    prices = pd.DataFrame(
        {
            "HFs": np.exp(0.003 * np.arange(len(index))),
            "ETH": np.exp(0.008 * np.arange(len(index))),
        },
        index=index,
    )

    assert _common_reporting_start(prices) == pd.Timestamp("2023-03-31")


def test_post_published_sample_starts_at_paper_cutoff() -> None:
    """The incremental study starts from the paper's 30 June 2023 NAV boundary."""
    periods = _periods(pd.Timestamp("2026-09-04"))

    index = pd.to_datetime(["2023-06-29", "2023-06-30", "2023-07-03", "2026-09-04"])
    located = periods["post_published_sample"].locate(pd.Series(index=index, dtype=float))

    assert located.index[0] == pd.Timestamp("2023-06-30")
    assert located.index[-1] == pd.Timestamp("2026-09-04")


def test_max_div_with_and_without_crypto_are_fully_invested() -> None:
    """MaxDiv must estimate matching covariance universes for both portfolio variants."""
    index = pd.date_range("2008-01-01", "2018-12-31", freq="B")
    t = np.arange(len(index), dtype=float)
    returns = pd.DataFrame(
        {
            "HFs": 0.00010 + 0.0010 * np.sin(t / 17.0),
            "PE": 0.00020 + 0.0013 * np.cos(t / 23.0),
            "Gold": 0.00008 + 0.0011 * np.sin(t / 31.0 + 0.4),
            "BTC": 0.00050 + 0.0030 * np.cos(t / 11.0 + 0.7),
        },
        index=index,
    )
    prices = 100.0 * np.exp(returns.cumsum())
    portfolio_without, portfolio_with = _run_pair(
        prices=prices,
        crypto_asset="BTC",
        method=OptimisationType.MAX_DIV,
        universe="alternatives",
        as_of=pd.Timestamp("2018-12-31"),
    )
    for portfolio in (portfolio_without, portfolio_with):
        weights = _input_weights(portfolio)
        assert (weights.sum(axis=1) - 1.0).abs().max() <= 1e-6


def test_balanced_max_sharpe_preserves_fixed_balanced_sleeve() -> None:
    """Both MaxSharpe variants must keep the first sleeve fixed at 75%."""
    index = pd.date_range("2008-01-01", "2018-12-31", freq="B")
    t = np.arange(len(index), dtype=float)
    returns = pd.DataFrame(
        {
            "60/40": 0.00015 + 0.0008 * np.sin(t / 19.0),
            "HFs": 0.00010 + 0.0010 * np.sin(t / 17.0),
            "PE": 0.00020 + 0.0013 * np.cos(t / 23.0),
            "Gold": 0.00008 + 0.0011 * np.sin(t / 31.0 + 0.4),
            "BTC": 0.00050 + 0.0030 * np.cos(t / 11.0 + 0.7),
        },
        index=index,
    )
    prices = 100.0 * np.exp(returns.cumsum())
    portfolio_without, portfolio_with = _run_pair(
        prices=prices,
        crypto_asset="BTC",
        method=OptimisationType.MAX_SHARPE,
        universe="balanced_risk_budget",
        as_of=pd.Timestamp("2018-12-31"),
    )
    for portfolio in (portfolio_without, portfolio_with):
        weights = _input_weights(portfolio)
        assert (weights["60/40"] - 0.75).abs().max() <= 1e-8
        assert (weights.sum(axis=1) - 1.0).abs().max() <= 1e-6
