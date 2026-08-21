"""Independently validate E5b artifacts and the binding construction conventions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from papers.cluster_lineage_2026.replication.run_backtests import load_universe
from papers.cluster_lineage_2026.replication.run_e5b import _root as e5b_output_root


EXPECTED_GROUP_COUNT_ROWS = {"msci_us": 1323, "futures": 2655, "mac": 2272}


def _root() -> Path:
    """Return the E5b output root."""
    return e5b_output_root()


def _check_acceptance() -> None:
    """Assert every measured construction and yardstick acceptance line passes."""
    acceptance = pd.read_csv(_root() / "acceptance.csv")
    assert len(acceptance) == 9
    assert acceptance["status"].eq("PASS").all()
    assert acceptance.loc[
        acceptance["acceptance_line"].eq("weights_sum_to_one"), "measured"
    ].max() <= 1e-12
    assert acceptance.loc[
        acceptance["acceptance_line"].eq("equal_group_budget"), "measured"
    ].max() <= 1e-15
    assert acceptance.loc[
        acceptance["acceptance_line"].eq("no_ew_performance_comparison"), "measured"
    ].eq(0.0).all()


def _check_determinism() -> None:
    """Assert every replayed CSV is byte-identical."""
    replay = pd.read_csv(_root() / "determinism.csv")
    assert len(replay) == 46
    assert replay["byte_identical"].all()
    assert replay["first_sha256"].eq(replay["second_sha256"]).all()


def _check_universe_outputs() -> None:
    """Assert diagnostic coverage, construction labels, and U1 windows."""
    for universe, expected_rows in EXPECTED_GROUP_COUNT_ROWS.items():
        path = _root() / "group_equal" / universe
        counts = pd.read_csv(path / "group_count_per_date.csv")
        summary = pd.read_csv(path / "group_count_summary.csv")
        validation = pd.read_csv(path / "weight_validation.csv")
        payoff = pd.read_csv(path / "payoff_comparison.csv")
        reference = pd.read_csv(path / "ew_reference.csv")
        assert len(counts) == expected_rows
        assert counts["construction"].eq("group_equal").all()
        assert counts["available_group_count"].ge(1).all()
        assert summary["available_group_count_std"].notna().all()
        assert validation["weight_status"].eq("PASS").all()
        assert validation["group_budget_status"].eq("PASS").all()
        assert not payoff.astype(str).apply(
            lambda column: column.str.contains("EW_all.*minus|minus.*EW_all", regex=True)
        ).any().any()
        assert reference["leg"].eq("EW_all").any()
        if universe == "msci_us":
            assert set(counts["analysis_window"]) == {
                "headline_20090831_20260630",
                "full_panel",
            }


def _check_e6_addendum() -> None:
    """Assert exact requested bootstrap row coverage and frozen parameters."""
    root = _root() / "e6_addendum"
    group_equal = pd.read_csv(root / "payoff_bootstrap_group_equal.csv")
    combined = pd.read_csv(root / "payoff_bootstrap_all_constructions.csv")
    assert len(group_equal) == 36
    assert len(combined) == 72
    assert group_equal["construction"].eq("group_equal").all()
    assert set(combined["construction"]) == {"asset_equal", "group_equal"}
    assert combined["block_length"].eq(6).all()
    assert combined["bootstrap_draws"].eq(2000).all()
    assert combined["seed"].eq(20260813).all()
    assert not combined["contrast"].str.contains("EW", case=False).any()


def _check_mac_qe_hold() -> None:
    """Assert U3 QE selections do not change between quarterly QE dates."""
    data = load_universe("mac")
    qe_assets = set(data.asset_returns["QE"].columns)
    qe_dates = set(data.asset_returns["QE"].index)
    weights = pd.read_csv(
        _root() / "group_equal" / "mac" / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    qe_columns = [column for column in weights.columns if column in qe_assets]
    assert len(qe_columns) == 17
    prior: dict[str, tuple[str, ...]] = {}
    violations = []
    for date, row in weights.iterrows():
        leg = str(row["leg"])
        selected = tuple(column for column in qe_columns if row[column] > 0.0)
        if leg in prior and date not in qe_dates and selected != prior[leg]:
            violations.append((date, leg))
        prior[leg] = selected
    assert not violations


def _check_report_quotes_all_bootstrap_rows() -> None:
    """Assert every combined E6 addendum row is quoted verbatim in the report."""
    report_path = (
        Path(__file__).resolve().parents[1]
        / "agents"
        / "2026-08-14_sol_E5b_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    rows = pd.read_csv(
        _root() / "e6_addendum" / "payoff_bootstrap_all_constructions.csv"
    )
    universe_names = {"msci_us": "U1", "futures": "U2", "mac": "U3"}
    window_names = {
        "headline_20090831_20260630": "headline",
        "full_panel": "full panel",
    }
    metric_names = {
        "net_return_annualized_delta": "net return",
        "net_sharpe_delta": "Sharpe",
        "one_way_turnover_annualized_delta": "turnover",
    }
    for row in rows.itertuples(index=False):
        contrast = row.contrast.removeprefix("cluster_")
        contrast = contrast.replace("M1_delta_0.02", "M1_0.02")
        contrast = contrast.replace("M1_delta_0.05", "M1_0.05")
        contrast = contrast.replace("_minus_cluster_baseline", " - baseline")
        contrast = contrast.replace("_minus_global", " - global")
        contrast = contrast.replace("_minus_taxonomy", " - taxonomy")
        expected = (
            f"| {universe_names[row.universe]} | {window_names[row.analysis_window]} | "
            f"{row.construction} | {contrast} | {metric_names[row.metric]} | "
            f"{row.estimate:.6f} | [{getattr(row, '_6'):.6f}, "
            f"{getattr(row, '_7'):.6f}] | "
            f"{'yes' if row.ci_excludes_zero else 'no'} |"
        )
        assert expected in report, expected


def main() -> None:
    """Run the independent E5b validation suite."""
    _check_acceptance()
    _check_determinism()
    _check_universe_outputs()
    _check_e6_addendum()
    _check_mac_qe_hold()
    _check_report_quotes_all_bootstrap_rows()
    print("E5b independent validation: PASS")
    print("acceptance lines: 9/9 PASS")
    print("determinism replay: 46/46 CSV artifacts byte-identical")
    print("E6 payoff bootstrap: 36 group_equal rows; 72 combined rows")
    print("U3 QE hold: 17 assets; 0 non-QE-date selection changes")
    print("E5b report: all 72 bootstrap rows quoted")


if __name__ == "__main__":
    main()
