"""Differential tests for the F7 readable-source reconstruction."""
from __future__ import annotations

from dataclasses import asdict
from enum import Enum

import numpy as np
import pandas as pd
import pytest

from papers.cluster_lineage_2026.replication import configs
from papers.cluster_lineage_2026.replication import run_backtests as reconstructed
from papers.cluster_lineage_2026.replication.recovery_loader import load_executed


def _normalize(value):
    """Convert dataclass registries from independent modules to comparable primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_normalize(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_normalize(item) for item in value)
    return value


@pytest.fixture(scope="module")
def executed_configs():
    """Return the surviving executed configuration registry."""
    return load_executed("configs")


@pytest.fixture(scope="module")
def executed_backtests():
    """Return executed E5 logic with the owner-frozen wrapper behavior applied."""
    executed = load_executed("run_backtests")
    bytecode_eligibility = executed._investable_eligibility

    def owner_frozen_eligibility(data, dates):
        """Apply the post-bytecode owner screen that the former shim overrode."""
        eligibility = bytecode_eligibility(data, dates)
        if data.name == reconstructed.UniverseName.FUTURES:
            excluded = eligibility.columns.intersection(
                reconstructed.FUTURES_INVESTABILITY_EXCLUSIONS
            )
            if len(excluded):
                eligibility = eligibility.copy()
                eligibility.loc[:, excluded] = False
        return eligibility

    executed._investable_eligibility = owner_frozen_eligibility
    executed.generate_alpha_profile_report = lambda *args, **kwargs: None
    return executed


def _sample_dates(universe: reconstructed.UniverseName, count: int = 6) -> pd.DatetimeIndex:
    """Take deterministic spread dates from one frozen baseline cache."""
    dates = reconstructed.load_cached(universe, reconstructed.SmootherName.BASELINE).dates
    positions = np.linspace(0, len(dates) - 1, count, dtype=int)
    return dates[positions]


def test_configs_match_executed_registry(executed_configs) -> None:
    """Prove every executed enum, dataclass registry, and calibration value was recovered."""
    for enum_name in ("UniverseName", "SmootherName"):
        observed = {item.name: item.value for item in getattr(configs, enum_name)}
        expected = {item.name: item.value for item in getattr(executed_configs, enum_name)}
        assert observed == expected

    observed_universes = {
        key.value: _normalize(asdict(value)) for key, value in configs.UNIVERSE_SPECS.items()
    }
    expected_universes = {
        key.value: _normalize(asdict(value))
        for key, value in executed_configs.UNIVERSE_SPECS.items()
    }
    assert observed_universes == expected_universes

    observed_smoothers = {
        key.value: _normalize(asdict(value)) for key, value in configs.SMOOTHER_SPECS.items()
    }
    expected_smoothers = {
        key.value: _normalize(asdict(value))
        for key, value in executed_configs.SMOOTHER_SPECS.items()
    }
    assert observed_smoothers == expected_smoothers
    assert _normalize(configs.M1_STAR_DELTAS) == _normalize(executed_configs.M1_STAR_DELTAS)
    assert configs.PRODUCTION_MOMENTUM_CLUSTER == executed_configs.PRODUCTION_MOMENTUM_CLUSTER


def test_futures_exclusion_is_integrated_once(executed_backtests) -> None:
    """Prove the canonical eleven-contract mask matches the former owner override."""
    assert reconstructed.FUTURES_INVESTABILITY_EXCLUSIONS == frozenset(
        {
            "BMR1 Curncy",
            "CUA1 Comdty",
            "IJ1 Comdty",
            "KC1 Comdty",
            "KM1 Index",
            "MES1 Index",
            "QC1 Index",
            "RS1 Comdty",
            "ST1 Index",
            "UXY1 Comdty",
            "WN1 Comdty",
        }
    )
    assert reconstructed.FUTURES_INVESTABILITY_EXCLUSION_ALIASES == {
        "MMR1 Curncy": "BMR1 Curncy"
    }
    universe = reconstructed.UniverseName.FUTURES
    dates = _sample_dates(universe, count=3)
    data = reconstructed.load_universe(universe)
    observed = reconstructed._investable_eligibility(data, dates)
    expected = executed_backtests._investable_eligibility(data, dates)
    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    present = observed.columns.intersection(reconstructed.FUTURES_INVESTABILITY_EXCLUSIONS)
    assert len(present) > 0
    assert not observed.loc[:, present].to_numpy().any()


@pytest.mark.parametrize("universe", list(reconstructed.UniverseName))
def test_frozen_inputs_and_weights_match_executed(universe, executed_backtests) -> None:
    """Compare frozen data transformations and every configured target weight exactly."""
    dates = _sample_dates(universe)
    data = reconstructed.load_universe(universe)
    observed_eligibility = reconstructed._investable_eligibility(data, dates)
    expected_eligibility = executed_backtests._investable_eligibility(data, dates)
    pd.testing.assert_frame_equal(observed_eligibility, expected_eligibility, check_exact=True)

    pd.testing.assert_frame_equal(
        reconstructed._prices(data),
        executed_backtests._prices(data),
        check_exact=True,
    )
    for vol_adjusted in (False, True):
        observed_scores = reconstructed._raw_momentum_scores(
            data, dates, vol_adjusted=vol_adjusted
        )
        expected_scores = executed_backtests._raw_momentum_scores(
            data, dates, vol_adjusted=vol_adjusted
        )
        pd.testing.assert_frame_equal(observed_scores, expected_scores, check_exact=True)

    scores = (
        reconstructed._raw_momentum_scores(data, dates, vol_adjusted=False)
        .reindex(columns=observed_eligibility.columns)
        .where(observed_eligibility)
    )
    observed, observed_counterfactuals = reconstructed._build_leg_weights(
        data,
        dates,
        scores,
        observed_eligibility,
        0.2,
        reconstructed.IN_BAND[universe],
    )
    expected, expected_counterfactuals = executed_backtests._build_leg_weights(
        data,
        dates,
        scores,
        expected_eligibility,
        0.2,
        reconstructed.IN_BAND[universe],
    )
    assert observed.keys() == expected.keys()
    assert observed_counterfactuals.keys() == expected_counterfactuals.keys()
    for leg in observed:
        pd.testing.assert_frame_equal(observed[leg], expected[leg], check_exact=True)
    for leg in observed_counterfactuals:
        pd.testing.assert_frame_equal(
            observed_counterfactuals[leg],
            expected_counterfactuals[leg],
            check_exact=True,
        )


@pytest.mark.filterwarnings("ignore:.*weight dates trade past the end.*:UserWarning")
@pytest.mark.filterwarnings("ignore:weighted instruments .*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:DataFrame is highly fragmented.*:pandas.errors.PerformanceWarning"
)
@pytest.mark.parametrize("universe", list(reconstructed.UniverseName))
def test_all_window_artifacts_match_executed(
    universe,
    executed_backtests,
    monkeypatch,
) -> None:
    """Differentially reproduce all eleven E5 artifact frames on frozen input slices."""
    monkeypatch.setattr(
        reconstructed,
        "generate_alpha_profile_report",
        lambda *args, **kwargs: None,
    )
    dates = _sample_dates(universe)
    observed = reconstructed._run_window(universe, "f7_differential", dates)
    expected = executed_backtests._run_window(universe, "f7_differential", dates)
    assert observed.keys() == expected.keys()
    for name in observed:
        pd.testing.assert_frame_equal(observed[name], expected[name], check_exact=True)
        observed_bytes = observed[name].to_csv(
            index=False, float_format="%.15g", lineterminator="\n"
        ).encode()
        expected_bytes = expected[name].to_csv(
            index=False, float_format="%.15g", lineterminator="\n"
        ).encode()
        assert observed_bytes == expected_bytes, name
