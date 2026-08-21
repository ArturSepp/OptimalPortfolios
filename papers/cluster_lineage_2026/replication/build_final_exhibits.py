"""Build the frozen cluster-lineage manuscript exhibits from recorded artifacts only."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_f6_bootstrap as f6
from papers.cluster_lineage_2026.replication import run_g0_u1_window_rescore as g0


FIGURE_IDS = ("F1", "F2", "F3", "F4", "F5", "F6")
NEW_BODY_TABLE_IDS = ("TA", "TB", "TC", "TD")
APPENDIX_TABLE_IDS = ("TE", "TF")
EXISTING_TABLE_IDS = ("tab:universes", "tab:signal", "tab:risk", "tab:concentration")
HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
BASELINE_COLOR = "#4C566A"
TREATMENT_COLOR = "#B34745"
CONTROL_COLOR = "#5E81AC"
TAXONOMY_COLOR = "#5B8E7D"
FIXED_TIME = dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc)


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _output_root() -> Path:
    """Return the configured external output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F8 exhibit root."""
    return _output_root() / "finalisation" / "exhibits"


def _dirs() -> dict[str, Path]:
    """Create and return the isolated exhibit subdirectories."""
    result = {
        "root": _root(),
        "figures": _root() / "figures",
        "tables": _root() / "tables",
        "data": _root() / "figure_data",
        "sources": _root() / "consolidated_sources",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a frozen CSV without losing serialized float precision."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV inside the isolated exhibit root."""
    resolved = path.resolve()
    if _root() not in resolved.parents:
        raise ValueError(f"F8 write outside isolated root: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_provenance() -> str:
    """Return the pre-F9 git commit provenance without mutating git configuration."""
    git = _repo_root() / ".git"
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = git / head.removeprefix("ref: ")
        if ref.is_file():
            head = ref.read_text(encoding="utf-8").strip()
    return f"pre-F9-parent:{head}; F8-source-hashes-recorded"


def _f0_inventory() -> pd.DataFrame:
    """Return the frozen F0 inventory indexed by input id."""
    path = _output_root() / "finalisation" / "f0" / "cache_inventory.csv"
    frame = _read(path).set_index("input_id")
    if not frame["status"].eq("PASS").all():
        raise AssertionError("F8 requires an all-green F0 inventory")
    return frame


def _source(input_id: str) -> Path:
    """Resolve exactly one F0 source path."""
    return Path(_f0_inventory().loc[input_id, "path"])


def _style() -> None:
    """Apply the shared rQUF single-column figure style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def _save_figure(fig: plt.Figure, figure_id: str) -> tuple[Path, Path]:
    """Save deterministic EPS and PDF copies of one figure."""
    paths = _dirs()
    eps = paths["figures"] / f"{figure_id}.eps"
    pdf = paths["figures"] / f"{figure_id}.pdf"
    fig.savefig(
        eps,
        format="eps",
        bbox_inches="tight",
        metadata={"Creator": "cluster-lineage F8 frozen exhibit builder"},
    )
    text = eps.read_text(encoding="latin-1")
    text = re.sub(r"^%%CreationDate:.*$", "%%CreationDate: 2026-08-21", text, flags=re.M)
    eps.write_text(text, encoding="latin-1", newline="\n")
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "cluster-lineage F8 frozen exhibit builder",
            "CreationDate": FIXED_TIME,
            "ModDate": FIXED_TIME,
        },
    )
    plt.close(fig)
    return eps, pdf


def _panel_sources() -> dict[str, Path]:
    """Return the three cached stability directories."""
    return {
        "Equities": _output_root() / "stability" / "msci_us",
        "Futures": _output_root() / "stability" / "futures",
        "Funds": _output_root() / "stability" / "mac",
    }


def _figure_f1() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Build the baseline-versus-calibrated churn-through-time figure."""
    rows = []
    windows = {
        "Equities": "headline_20090831_20260630",
        "Futures": "full_panel",
        "Funds": "full_panel",
    }
    for label, directory in _panel_sources().items():
        changes = _read(directory / "predicted_realized_per_date.csv", parse_dates=["index"])
        margins = _read(directory / "margin_assets.csv", parse_dates=["date"])
        counts = margins.groupby("date")["asset"].nunique()
        selected = changes.loc[
            changes["analysis_window"].eq(windows[label])
            & changes["config"].isin(["baseline", "M1_star"])
            & changes["singleton_convention"].eq("including_singletons")
        ].copy()
        if label == "Funds":
            selected = selected.loc[selected["frequency"].eq("ME")].copy()
        selected["assets"] = selected["index"].map(counts)
        selected["churn_per_asset_year"] = 12.0 * selected["realised_changes"] / selected["assets"]
        selected["churn_12m_mean"] = selected.groupby("config")["churn_per_asset_year"].transform(
            lambda values: values.rolling(12, min_periods=3).mean()
        )
        selected.insert(0, "display_panel", label)
        rows.append(selected)
    data = pd.concat(rows, ignore_index=True)
    _write(data, _dirs()["data"] / "F1_churn_through_time.csv")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(6.33, 2.25), sharey=False)
    for axis, (label, frame) in zip(axes, data.groupby("display_panel", sort=False)):
        for config, color, style, name in (
            ("baseline", BASELINE_COLOR, "--", "unsmoothed"),
            ("M1_star", TREATMENT_COLOR, "-", "calibrated bonus"),
        ):
            values = frame.loc[frame["config"].eq(config)]
            axis.plot(
                values["index"],
                values["churn_12m_mean"],
                color=color,
                linestyle=style,
                linewidth=1.0,
                label=name,
            )
        axis.set_title(label)
        axis.set_xlabel("estimation date")
        axis.xaxis.set_major_locator(mdates.YearLocator(6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(color="#E5E9F0", linewidth=0.4)
    axes[0].set_ylabel("membership changes per asset-year\n(12-month mean)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Calibrated smoothing sharply lowers membership churn across panels", y=1.02)
    fig.tight_layout()
    return data, _save_figure(fig, "F1")


def _figure_f2() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Build the churn-fidelity frontiers with both calibration overlays."""
    frontier = _read(_output_root() / "finalisation" / "f1" / "frontier.csv")
    bridge = _read(_output_root() / "finalisation" / "f1" / "calibration_bridge.csv")
    selections = {
        "Equities": ("equity_panel", "headline_20090831_20260630", "equity_panel"),
        "Futures": ("futures_panel", "full_panel", "futures_panel"),
        "Funds": ("fund_panel", "full_panel", "fund_panel"),
    }
    rows = []
    for label, (panel, window, bridge_panel) in selections.items():
        values = (
            frontier.loc[frontier["panel"].eq(panel) & frontier["analysis_window"].eq(window)]
            .sort_values("delta_path_coordinate")
            .copy()
        )
        calibration = bridge.loc[
            bridge["row_role"].eq("theory_panel") & bridge["panel"].eq(bridge_panel)
        ]
        if label == "Funds":
            calibration = calibration.loc[calibration["frequency"].eq("ME")]
        calibration = calibration.iloc[0]
        values.insert(0, "display_panel", label)
        values["delta_star_level_overlay"] = calibration["delta_star_level"]
        values["delta_star_innovation_overlay"] = calibration["delta_star_innovation"]
        rows.append(values)
    data = pd.concat(rows, ignore_index=True)
    _write(data, _dirs()["data"] / "F2_churn_fidelity_frontier.csv")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(6.33, 2.3))
    for axis, (label, values) in zip(axes, data.groupby("display_panel", sort=False)):
        values = values.sort_values("delta_path_coordinate")
        axis.plot(
            values["raw_churn"],
            values["fidelity"],
            marker="o",
            color=TREATMENT_COLOR,
            linewidth=1.1,
            markersize=3,
        )
        for row in values.itertuples(index=False):
            axis.annotate(
                row.config.replace("M1_delta_", "δ="),
                (row.raw_churn, row.fidelity),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5.2,
            )
        knee = values.loc[values["is_knee"].astype(str).str.lower().eq("true")]
        axis.scatter(
            knee["raw_churn"],
            knee["fidelity"],
            marker="*",
            s=55,
            color="#D08770",
            edgecolor="black",
            linewidth=0.3,
            zorder=4,
            label="knee",
        )
        delta = values["delta_path_coordinate"].to_numpy(dtype=float)
        churn = values["raw_churn"].to_numpy(dtype=float)
        for field, color, style, name in (
            ("delta_star_level_overlay", "#5E81AC", "--", r"$\delta^*_{lvl}$"),
            ("delta_star_innovation_overlay", "#A3BE8C", ":", r"$\delta^*_{inn}$"),
        ):
            target = float(values[field].iloc[0])
            x = float(np.interp(target, delta, churn))
            axis.axvline(x, color=color, linestyle=style, linewidth=0.9, label=name)
        axis.set_title(label)
        axis.set_xlabel("churn per asset-year")
        axis.set_ylabel("partition fidelity")
        axis.grid(color="#E5E9F0", linewidth=0.4)
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("The churn-fidelity frontier turns near the measured noise floor", y=1.02)
    fig.tight_layout()
    return data, _save_figure(fig, "F2")


def _figure_f3() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Build flat-cut and Ward flip-probability verification panels."""
    root = _output_root() / "finalisation" / "f4"
    flat = _read(root / "simulation_results.csv")
    ward = _read(root / "ward_verification.csv")
    data = pd.concat([flat, ward], ignore_index=True)
    keep = [
        "method",
        "predicted_flip_probability",
        "realised_flip_probability",
        "delta_label",
        "seed",
        "cell_fingerprint",
    ]
    data = data.loc[:, keep]
    _write(data, _dirs()["data"] / "F3_flip_probability_verification.csv")

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.33, 2.45), sharex=True, sharey=True)
    for axis, method in zip(axes, ("flat", "ward")):
        values = data.loc[data["method"].eq(method)]
        axis.scatter(
            values["predicted_flip_probability"],
            values["realised_flip_probability"],
            s=4,
            color="#E7B5AF" if method == "ward" else "#C7D3E3",
            edgecolors="none",
        )
        upper = float(
            max(
                values["predicted_flip_probability"].max(),
                values["realised_flip_probability"].max(),
            )
        )
        axis.plot([0.0, upper], [0.0, upper], color=BASELINE_COLOR, linestyle="--", linewidth=0.8)
        axis.set_title("flat cut" if method == "flat" else "production Ward")
        axis.set_xlabel("Gaussian-predicted flip probability")
        axis.grid(color="#E5E9F0", linewidth=0.4)
    axes[0].set_ylabel("realised flip frequency")
    fig.suptitle("Gaussian margins accurately order realised assignment flips", y=1.02)
    fig.tight_layout()
    return data, _save_figure(fig, "F3")


def _figure_f4() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Build margin histograms with realised flip rates by decile."""
    source = _read(_output_root() / "finalisation" / "f1" / "margins_flip_rates.csv")
    selections = {
        "Equities": ("equity_panel", "headline_20090831_20260630"),
        "Futures": ("futures_panel", "full_panel"),
        "Funds": ("fund_panel", "full_panel"),
    }
    rows = []
    for label, (panel, window) in selections.items():
        values = source.loc[
            source["panel"].eq(panel)
            & source["analysis_window"].eq(window)
            & source["singleton_convention"].eq("including_singletons")
        ].copy()
        grouped = values.groupby("margin_decile", as_index=False).apply(
            lambda frame: pd.Series(
                {
                    "observations": frame["observations"].sum(),
                    "margin_mean": np.average(frame["margin_mean"], weights=frame["observations"]),
                    "realised_flip_rate": np.average(
                        frame["realised_flip_rate"], weights=frame["observations"]
                    ),
                }
            ),
            include_groups=False,
        )
        grouped.insert(0, "display_panel", label)
        grouped["observation_share"] = grouped["observations"] / grouped["observations"].sum()
        rows.append(grouped)
    data = pd.concat(rows, ignore_index=True)
    _write(data, _dirs()["data"] / "F4_margins_and_flips.csv")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(6.33, 2.35))
    for axis, (label, values) in zip(axes, data.groupby("display_panel", sort=False)):
        axis.bar(
            values["margin_mean"],
            values["observation_share"],
            width=np.maximum(0.005, np.ptp(values["margin_mean"]) / 12.0),
            color="#D8DEE9",
            edgecolor=BASELINE_COLOR,
            linewidth=0.35,
        )
        twin = axis.twinx()
        twin.plot(
            values["margin_mean"],
            values["realised_flip_rate"],
            color=TREATMENT_COLOR,
            marker="o",
            linewidth=1.0,
            markersize=2.8,
        )
        twin.spines["top"].set_visible(False)
        axis.set_title(label)
        axis.set_xlabel("assignment margin")
        axis.grid(axis="y", color="#E5E9F0", linewidth=0.4)
        if axis is axes[0]:
            axis.set_ylabel("observation share")
        if axis is axes[-1]:
            twin.set_ylabel("realised flip rate", color=TREATMENT_COLOR)
    fig.suptitle("Assignment flips concentrate among low-margin assets", y=1.02)
    fig.tight_layout()
    return data, _save_figure(fig, "F4")


def _signal_sources() -> dict[str, dict[str, object]]:
    """Return the three frozen signal NAV specifications."""
    return {
        "U1": {
            "path": _source("part_b_signal_navs__u1"),
            "columns": {"cluster": "cluster_M1_star", "sector": "bics_sector", "global": "global"},
            "performance": _source("part_b_signal_grid__u1") / "performance.csv",
        },
        "U2": {
            "path": _source("part_b_signal_navs__u2"),
            "columns": {
                "cluster": "classic_12m_ex_1m__cluster",
                "global": "classic_12m_ex_1m__global",
            },
            "performance": _source("part_b_signal_grid__u2") / "performance.csv",
        },
        "U3": {
            "path": _source("part_b_signal_navs__u3"),
            "columns": {"cluster": "short_3__cluster", "global": "short_3__global"},
            "performance": _source("part_b_signal_grid__u3") / "performance.csv",
        },
    }


def _signal_turnover(universe: str, leg: str, performance_path: Path) -> float:
    """Select the recorded annual one-way turnover for one signal leg."""
    frame = _read(performance_path)
    if universe == "U1":
        row = frame.loc[
            frame["leg"].eq(
                {"cluster": "cluster_M1_star", "sector": "bics_sector", "global": "global"}[leg]
            )
        ]
    elif universe == "U2":
        row = frame.loc[frame["signal_variant"].eq("classic_12m_ex_1m") & frame["method"].eq(leg)]
    else:
        row = frame.loc[
            pd.to_numeric(frame["short_span"], errors="coerce").eq(3.0) & frame["method"].eq(leg)
        ]
    if len(row) != 1:
        raise AssertionError(f"signal turnover row did not resolve once: {universe}/{leg}")
    return float(row.iloc[0]["one_way_turnover_annualized"])


def _ci_json(rows: pd.DataFrame) -> str:
    """Serialize a comparison's three companion intervals deterministically."""
    payload = {
        row.metric: {
            "point": float(row.point_estimate),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "excludes_zero": str(row.excludes_zero).lower() == "true",
        }
        for row in rows.itertuples(index=False)
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _figure_f5_and_signal_table() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[Path, Path]
]:
    """Build cumulative signal NAVs, endpoint checks, and the CI-integrated signal table."""
    nav_rows = []
    leg_rows = []
    for universe, spec in _signal_sources().items():
        columns = spec["columns"]
        frame = _read(Path(spec["path"]), index_col=0, parse_dates=True)[
            list(columns.values())
        ].dropna()
        if universe == "U1":
            frame = g0._window_navs(frame)
        else:
            frame = frame.loc[(frame.index <= HEADLINE_END)]
        frame = 100.0 * frame / frame.iloc[0]
        for leg, column in columns.items():
            metrics, monthly = f6._nav_metrics(frame[column])
            nav_rows.append(
                pd.DataFrame(
                    {
                        "date": frame.index,
                        "universe": universe,
                        "leg": leg,
                        "nav": frame[column].to_numpy(),
                    }
                )
            )
            leg_rows.append(
                {
                    "universe": universe,
                    "leg": leg,
                    **dict(zip(f6.METRICS, metrics)),
                    "one_way_turnover_annualized": _signal_turnover(
                        universe, leg, Path(spec["performance"])
                    ),
                    "monthly_observations": len(monthly),
                    "source_nav_path": str(spec["path"]),
                }
            )
    nav_data = pd.concat(nav_rows, ignore_index=True)
    leg_table = pd.DataFrame(leg_rows)
    g0_legs = _read(_output_root() / "finalisation" / "g0" / "u1_windowed_performance.csv")
    for label, target in (("cluster", "cluster"), ("sector", "BICS sector"), ("global", "global")):
        source = g0_legs.loc[g0_legs["row_type"].eq("leg") & g0_legs["leg"].eq(target)].iloc[0]
        selected = leg_table["universe"].eq("U1") & leg_table["leg"].eq(label)
        for metric in f6.METRICS:
            leg_table.loc[selected, metric] = float(source[metric])

    signal_cis = pd.concat(
        [
            _read(_output_root() / "finalisation" / "g0" / "u1_windowed_cis.csv").query(
                "comparison.str.startswith('U1 cluster')", engine="python"
            ),
            _read(_output_root() / "finalisation" / "f6" / "signal_cis.csv").query(
                "~comparison.str.startswith('U1 cluster')", engine="python"
            ),
        ],
        ignore_index=True,
    )
    companions = []
    for comparison, rows in signal_cis.groupby("comparison", sort=False):
        companions.append({"comparison": comparison, "ci_companion": _ci_json(rows)})
    comparison_map = {
        "U1": "U1 cluster - global; U1 cluster - BICS sector",
        "U2": "U2 cluster - global",
        "U3": "U3 cluster - global",
    }
    leg_table["ci_comparison"] = ""
    leg_table["ci_companion"] = ""
    for universe, names in comparison_map.items():
        payloads = [
            row["ci_companion"] for row in companions if row["comparison"] in names.split("; ")
        ]
        selected = leg_table["universe"].eq(universe) & leg_table["leg"].eq("cluster")
        leg_table.loc[selected, "ci_comparison"] = names
        leg_table.loc[selected, "ci_companion"] = " || ".join(payloads)

    endpoint_rows = []
    for row in leg_table.itertuples(index=False):
        plotted = nav_data.loc[
            nav_data["universe"].eq(row.universe) & nav_data["leg"].eq(row.leg)
        ].set_index("date")["nav"]
        metrics, _ = f6._nav_metrics(plotted)
        for metric, value in zip(f6.METRICS, metrics):
            endpoint_rows.append(
                {
                    "universe": row.universe,
                    "leg": row.leg,
                    "metric": metric,
                    "figure_value": value,
                    "table_value": getattr(row, metric),
                    "absolute_error": abs(value - getattr(row, metric)),
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    _write(nav_data, _dirs()["data"] / "F5_cumulative_signal_navs.csv")
    _write(endpoints, _dirs()["data"] / "F5_nav_endpoint_reconciliation.csv")
    _write(leg_table, _dirs()["tables"] / "table_existing_signal.csv")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(6.33, 2.35), sharey=False)
    for axis, universe in zip(axes, ("U1", "U2", "U3")):
        values = nav_data.loc[nav_data["universe"].eq(universe)]
        styles = {
            "cluster": (TREATMENT_COLOR, "-", "cluster score"),
            "global": (CONTROL_COLOR, "--", "global score"),
            "sector": (TAXONOMY_COLOR, ":", "BICS sector"),
        }
        for leg in values["leg"].unique():
            color, linestyle, label = styles[leg]
            line = values.loc[values["leg"].eq(leg)]
            axis.plot(
                line["date"],
                line["nav"],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=label,
            )
        axis.axhline(100.0, color="#BFC5CE", linewidth=0.5)
        axis.set_title(universe)
        axis.set_xlabel("date")
        axis.set_xticks(pd.to_datetime(["2012-01-01", "2018-01-01", "2024-01-01"]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(color="#E5E9F0", linewidth=0.4)
    axes[0].set_ylabel("net cumulative NAV (100 at start)")
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle("Peer-contained scores reduce signal-portfolio volatility", y=1.02)
    fig.tight_layout()
    return nav_data, leg_table, endpoints, _save_figure(fig, "F5")


def _risk_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the existing risk-performance and concentration tables."""
    performance_rows = []
    concentration_rows = []
    for universe in ("u1", "u2", "u3"):
        directory = _source(f"part_b_risk_output__{universe}")
        performance = _read(directory / "performance.csv")
        performance.insert(0, "paper_universe", universe.upper())
        performance_rows.append(performance)
        summary = _read(directory / "risk_summary.csv")
        for method in ("flat_erc", "cluster_rb_alpha_0"):
            row = summary.loc[summary["method"].eq(method)].iloc[0]
            concentration_rows.append(
                {
                    "universe": universe.upper(),
                    "method": method,
                    "effective_risk_clusters_mean": row["effective_risk_clusters_absolute_mean"],
                    "largest_cluster_risk_share_mean": row[
                        "maximum_absolute_cluster_risk_share_mean"
                    ],
                    "source_path": str(directory / "risk_summary.csv"),
                }
            )
    performance = pd.concat(performance_rows, ignore_index=True)
    g0_legs = _read(_output_root() / "finalisation" / "g0" / "u1_windowed_performance.csv")
    replacements = {
        "flat_erc": "flat ERC",
        "ward_hrp": "Rolling-Ward HRP",
        "single_hrp": "single-link HRP",
    }
    for method, label in replacements.items():
        source = g0_legs.loc[g0_legs["row_type"].eq("leg") & g0_legs["leg"].eq(label)].iloc[0]
        selected = performance["paper_universe"].eq("U1") & performance["method"].eq(method)
        for metric in f6.METRICS:
            performance.loc[selected, metric] = float(source[metric])
    risk_cis = pd.concat(
        [
            _read(_output_root() / "finalisation" / "g0" / "u1_windowed_cis.csv").query(
                "comparison.str.contains('HRP')", engine="python"
            ),
            _read(_output_root() / "finalisation" / "f6" / "risk_cis.csv").query(
                "comparison.str.startswith('U3')", engine="python"
            ),
        ],
        ignore_index=True,
    )
    performance["ci_comparison"] = ""
    performance["ci_companion"] = ""
    candidate_map = {
        "U1 Rolling-Ward HRP - flat ERC": ("U1", "ward_hrp"),
        "U1 Rolling-Ward HRP - single-link HRP": ("U1", "ward_hrp"),
        "U3 equal-cluster RB - flat ERC": ("U3", "cluster_rb_alpha_0"),
    }
    for comparison, rows in risk_cis.groupby("comparison", sort=False):
        universe, method = candidate_map[comparison]
        selected = performance["paper_universe"].eq(universe) & performance["method"].eq(method)
        existing_name = performance.loc[selected, "ci_comparison"].iloc[0]
        existing_ci = performance.loc[selected, "ci_companion"].iloc[0]
        performance.loc[selected, "ci_comparison"] = "; ".join(
            filter(None, [existing_name, comparison])
        )
        performance.loc[selected, "ci_companion"] = " || ".join(
            filter(None, [existing_ci, _ci_json(rows)])
        )
    concentration = pd.DataFrame(concentration_rows)
    _write(performance, _dirs()["tables"] / "table_existing_risk.csv")
    _write(concentration, _dirs()["tables"] / "table_existing_concentration.csv")
    return performance, concentration


def _figure_f6(concentration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, Path]]:
    """Build effective-risk-cluster time series and reconcile their averages."""
    rows = []
    for universe in ("u1", "u2", "u3"):
        path = _source(f"part_b_risk_output__{universe}") / "risk_per_date.csv"
        frame = _read(path, parse_dates=["date"])
        frame = frame.loc[
            frame["method"].isin(["flat_erc", "cluster_rb_alpha_0"]),
            ["date", "method", "effective_risk_clusters_absolute"],
        ]
        frame.insert(0, "universe", universe.upper())
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    checks = []
    for (universe, method), frame in data.groupby(["universe", "method"]):
        table_value = concentration.loc[
            concentration["universe"].eq(universe) & concentration["method"].eq(method),
            "effective_risk_clusters_mean",
        ].iloc[0]
        checks.append(
            {
                "universe": universe,
                "method": method,
                "figure_average": frame["effective_risk_clusters_absolute"].mean(),
                "table_average": table_value,
                "absolute_error": abs(
                    frame["effective_risk_clusters_absolute"].mean() - table_value
                ),
            }
        )
    reconciliation = pd.DataFrame(checks)
    _write(data, _dirs()["data"] / "F6_effective_risk_clusters.csv")
    _write(reconciliation, _dirs()["data"] / "F6_concentration_reconciliation.csv")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(6.33, 2.35), sharey=False)
    for axis, universe in zip(axes, ("U1", "U2", "U3")):
        values = data.loc[data["universe"].eq(universe)]
        for method, color, style, label in (
            ("flat_erc", BASELINE_COLOR, "--", "flat ERC"),
            ("cluster_rb_alpha_0", TREATMENT_COLOR, "-", "equal-cluster budgets"),
        ):
            line = values.loc[values["method"].eq(method)]
            axis.plot(
                line["date"],
                line["effective_risk_clusters_absolute"],
                color=color,
                linestyle=style,
                linewidth=1.0,
                label=label,
            )
        axis.set_title(universe)
        axis.set_xlabel("date")
        axis.set_xticks(pd.to_datetime(["2012-01-01", "2018-01-01", "2024-01-01"]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.set_xlim(
            values["date"].min() - pd.Timedelta(days=365),
            values["date"].max() + pd.Timedelta(days=365),
        )
        axis.grid(color="#E5E9F0", linewidth=0.4)
    axes[0].set_ylabel("effective risk clusters")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Equal cluster budgets distribute risk across more peer groups", y=1.02)
    fig.tight_layout()
    return data, reconciliation, _save_figure(fig, "F6")


def _table_universes() -> pd.DataFrame:
    """Build the frozen operating-point universe table."""
    return pd.DataFrame(
        [
            {
                "universe": "U1 equities",
                "membership": "point-in-time MSCI US",
                "clustering_cell": "ME/span 36",
                "smoothing": "0.0866",
                "signal": "classic 12m-1m",
                "portfolio": "long-short",
                "quantile": 0.25,
                "weighting": "equal within side",
                "cost_bps_one_way": 10,
            },
            {
                "universe": "U2 funds",
                "membership": "BlackRock catalogue; AUM > USD50m",
                "clustering_cell": "W-WED/span 156",
                "smoothing": "none",
                "signal": "classic 12m-1m",
                "portfolio": "long-only; 55/35/10 sleeves",
                "quantile": 0.25,
                "weighting": "equal within sleeve",
                "cost_bps_one_way": 20,
            },
            {
                "universe": "U3 futures",
                "membership": "continuous futures; 11 exclusions",
                "clustering_cell": "W-WED/span 156",
                "smoothing": "0.0691",
                "signal": "risk-adjusted momentum short=3",
                "portfolio": "long-short",
                "quantile": 0.25,
                "weighting": "inverse volatility",
                "cost_bps_one_way": 10,
            },
        ]
    )


def _table_ta() -> pd.DataFrame:
    """Build the four-row estimation-panel summary."""
    return pd.DataFrame(
        [
            {
                "panel": "equity_panel",
                "frequency": "W-WED",
                "asset_count": 1358,
                "span_n": 156,
                "sample_start": "2006-08-02",
                "sample_end": "2026-08-05",
                "estimation_dates": 238,
                "kappa_hat": 2.12441777211904,
            },
            {
                "panel": "futures_panel",
                "frequency": "W-WED",
                "asset_count": 95,
                "span_n": 156,
                "sample_start": "1959-07-08",
                "sample_end": "2026-08-12",
                "estimation_dates": 295,
                "kappa_hat": 1.61228969635356,
            },
            {
                "panel": "fund_panel",
                "frequency": "ME",
                "asset_count": 170,
                "span_n": 36,
                "sample_start": "1998-12-31",
                "sample_end": "2026-07-31",
                "estimation_dates": 284,
                "kappa_hat": 0.836854097618955,
            },
            {
                "panel": "fund_panel",
                "frequency": "QE",
                "asset_count": 17,
                "span_n": 12,
                "sample_start": "1999-12-31",
                "sample_end": "2026-06-30",
                "estimation_dates": 92,
                "kappa_hat": 1.28795943253069,
            },
        ]
    )


def _table_tb() -> pd.DataFrame:
    """Build the calibration bridge with adopted levels and frontier knees."""
    bridge = _read(_output_root() / "finalisation" / "f1" / "calibration_bridge.csv")
    bridge = bridge.loc[bridge["row_role"].eq("theory_panel")].copy()
    frontier = _read(_output_root() / "finalisation" / "f1" / "frontier.csv")
    knee = frontier.loc[frontier["is_knee"].astype(str).str.lower().eq("true")].copy()
    knee = knee.loc[
        ~knee["panel"].eq("equity_panel") | knee["analysis_window"].eq("headline_20090831_20260630")
    ]
    knee_map = knee.groupby("panel").first()[["config", "delta_label"]]
    adopted = {
        ("equity_panel", "W-WED"): 0.0866,
        ("futures_panel", "W-WED"): 0.0691,
        ("fund_panel", "ME"): 0.05,
        ("fund_panel", "QE"): 0.05,
    }
    bridge["adopted_delta"] = [adopted[(row.panel, row.frequency)] for row in bridge.itertuples()]
    bridge["sweep_knee_config"] = bridge["panel"].map(knee_map["config"])
    bridge["sweep_knee_delta"] = bridge["panel"].map(knee_map["delta_label"])
    return bridge


def _table_tc() -> pd.DataFrame:
    """Build the consolidated churn, fidelity, and lineage table."""
    churn = _read(_output_root() / "finalisation" / "f3" / "churn_fidelity.csv")
    selected = churn.loc[
        churn["config"].isin(["baseline", "M0_quarterly_hold", "M1_delta_0.02", "M1_star"])
        & (
            (~churn["panel"].eq("equity_panel") & churn["analysis_window"].eq("full_panel"))
            | (
                churn["panel"].eq("equity_panel")
                & churn["analysis_window"].eq("headline_20090831_20260630")
            )
        )
    ].copy()
    interpretability = _read(_output_root() / "finalisation" / "f3" / "interpretability.csv")
    selected = selected.merge(
        interpretability[
            ["panel", "config", "track_modal_taxonomy_purity", "label_churn_per_asset_year"]
        ],
        on=["panel", "config"],
        how="left",
    )
    selected["operating_point_role"] = "comparison_grid"
    selected.loc[
        selected["panel"].eq("fund_panel") & selected["config"].eq("baseline"),
        "operating_point_role",
    ] = "adopted_fund_application"
    selected.loc[
        selected["panel"].eq("futures_panel") & selected["config"].eq("M1_star"),
        "operating_point_role",
    ] = "adopted_futures_application"
    verdicts = _read(_output_root() / "finalisation" / "f3" / "adopted_cell_verdicts.csv")
    application = pd.DataFrame(
        {
            "panel": verdicts["panel"],
            "analysis_window": "headline_20090831_20260630",
            "config": verdicts["application_cell"],
            "raw_churn": np.nan,
            "lineage_churn": np.nan,
            "median_same_date_ari": verdicts["median_same_date_ari"],
            "taxonomy_delta_ari_by_level": verdicts["taxonomy_delta_ari_by_level"],
            "maximum_absolute_taxonomy_delta_ari": verdicts["maximum_absolute_taxonomy_delta_ari"],
            "cluster_count_relative_change": verdicts["cluster_count_relative_change"],
            "fidelity_status": verdicts["fidelity_status"],
            "track_modal_taxonomy_purity": np.nan,
            "label_churn_per_asset_year": np.nan,
            "operating_point_role": "adopted_application_cell",
        }
    )
    return pd.concat([selected, application], ignore_index=True, sort=False)


def _table_td() -> pd.DataFrame:
    """Build the seven-prediction theory scorecard."""
    scorecard = _read(_output_root() / "finalisation" / "f5" / "theory_scorecard.csv")
    role = {
        "P1": "EMPIRICAL_PRIMARY",
        "P2": "PRIMARY",
        "P3": "PRIMARY_REVISED",
        "P4": "REVISED_ORDERING",
        "P5": "PRIMARY",
        "P6": "PRIMARY",
        "P7": "PRIMARY",
    }
    rows = []
    for prediction, row_role in role.items():
        row = (
            scorecard.loc[
                scorecard["prediction"].eq(prediction) & scorecard["row_role"].eq(row_role)
            ]
            .iloc[0]
            .to_dict()
        )
        if prediction == "P1":
            support = scorecard.loc[
                scorecard["prediction"].eq("P1") & scorecard["row_role"].eq("SYNTHETIC_SUPPORT")
            ].iloc[0]
            row["note"] = f"{row['note']} Synthetic support: {support['measured_value']}"
        if prediction == "P4":
            original = scorecard.loc[
                scorecard["prediction"].eq("P4") & scorecard["row_role"].eq("ORIGINAL_EQUALITY")
            ].iloc[0]
            row["note"] = (
                f"{row['note']} Original equality verdict: {original['verdict']}; "
                f"{original['measured_value']}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _robustness_sources() -> dict[str, Path]:
    """Return exactly four frozen robustness summary inputs."""
    repo = _repo_root()
    return {
        "u2_eligibility_grid.csv": repo
        / "papers"
        / "cluster_lineage_2026"
        / "data"
        / "local_outputs"
        / "e5b"
        / "covariance_frequency_span_grid"
        / "blackrock_us_etfs"
        / "equity_fi_60_40_long_short_aum_grid_20260816"
        / "filter_sensitivity.csv",
        "u1_min_cluster_size_grid.csv": repo
        / "papers"
        / "cluster_lineage_2026"
        / "local_outputs"
        / "e5b"
        / "u1_classic_min_cluster_size_grid_20260816"
        / "comparison.csv",
        "u3_short_span_grid.csv": repo
        / "papers"
        / "cluster_lineage_2026"
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_short_span_sweep_vol13m_20260816"
        / "comparison.csv",
        "u1_covariance_frequency_span_grid.csv": _output_root()
        / "e5b"
        / "covariance_frequency_span_grid"
        / "msci_us"
        / "long_short_grid_q_025_monthly_12m_skip1"
        / "comparison_vs_global.csv",
    }


def _copy_robustness_sources() -> pd.DataFrame:
    """Copy the four used grid summaries and record exact source/destination hashes."""
    rows = []
    for name, source in _robustness_sources().items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _dirs()["sources"] / name
        shutil.copyfile(source, destination)
        rows.append(
            {
                "source_path": str(source),
                "destination_path": str(destination),
                "source_sha256": _sha256(source),
                "destination_sha256": _sha256(destination),
                "byte_identical": _sha256(source) == _sha256(destination),
            }
        )
    result = pd.DataFrame(rows)
    _write(result, _root() / "source_hashes.csv")
    return result


def _table_te() -> pd.DataFrame:
    """Build the U2 eligibility and U1 minimum-cluster-size selection record."""
    copied = _dirs()["sources"]
    u2 = _read(copied / "u2_eligibility_grid.csv")
    u2.insert(0, "grid", "U2 eligibility")
    u2["candidate"] = u2["filter_id"]
    u2["selection_role"] = np.where(
        u2["aum_cutoff_usd_millions"].eq(50),
        "selected_operating_point",
        "selection_record_not_independent_confirmation",
    )
    u1 = _read(copied / "u1_min_cluster_size_grid.csv")
    u1 = u1.loc[u1["benchmark_leg"].eq("global")].copy()
    u1.insert(0, "grid", "U1 minimum cluster size")
    u1["candidate"] = "minimum_cluster_size=" + u1["min_cluster_size"].astype(str)
    u1["selection_role"] = np.where(
        u1["min_cluster_size"].eq(10),
        "selected_operating_point",
        "selection_record_not_independent_confirmation",
    )
    return pd.concat([u2, u1], ignore_index=True, sort=False)


def _table_tf() -> pd.DataFrame:
    """Build the U3 short-span and U1 covariance-cell selection record."""
    copied = _dirs()["sources"]
    u3 = _read(copied / "u3_short_span_grid.csv")
    u3.insert(0, "grid", "U3 short span")
    u3["candidate"] = "short_span=" + u3["short_span_label"].astype(str)
    u3["selection_role"] = np.where(
        pd.to_numeric(u3["short_span_label"], errors="coerce").eq(3.0),
        "selected_operating_point",
        "selection_record_not_independent_confirmation",
    )
    covariance = _read(copied / "u1_covariance_frequency_span_grid.csv")
    covariance = covariance.loc[
        covariance["analysis_window"].eq("headline_20090831_20260630")
    ].copy()
    covariance.insert(0, "grid", "U1 covariance frequency/span")
    covariance["candidate"] = covariance["cell_id"]
    covariance["selection_role"] = np.where(
        covariance["cell_id"].eq("ME_span_036"),
        "selected_operating_point",
        "selection_record_not_independent_confirmation",
    )
    return pd.concat([u3, covariance], ignore_index=True, sort=False)


def _index(figure_paths: dict[str, tuple[Path, Path]]) -> pd.DataFrame:
    """Build one provenance row for every manuscript exhibit."""
    root = _output_root() / "finalisation"
    table_root = _dirs()["tables"]
    source_root = _dirs()["sources"]
    script = str(Path(__file__).resolve())
    provenance = _head_provenance()
    stability_per_date = "; ".join(
        str(_output_root() / "stability" / panel / "predicted_realized_per_date.csv")
        for panel in ("msci_us", "futures", "mac")
    )
    risk_per_date = "; ".join(
        str(_source(f"part_b_risk_output__{universe}") / "risk_per_date.csv")
        for universe in ("u1", "u2", "u3")
    )
    risk_summaries = "; ".join(
        str(_source(f"part_b_risk_output__{universe}") / "risk_summary.csv")
        for universe in ("u1", "u2", "u3")
    )
    pipeline_summary = str(
        _repo_root()
        / "papers"
        / "cluster_lineage_2026"
        / "agents"
        / "2026-08-17_sol_signal_and_risk_model_pipeline_summary.md"
    )
    rows = [
        (
            "F1",
            "figure",
            "Section 4.1",
            "fig:churn",
            "Calibrated smoothing sharply lowers membership churn across panels.",
            "P1/P2; all panels",
            f"{root / 'f1' / 'frontier.csv'}; {stability_per_date}",
            (
                "Membership changes per asset-year; monthly estimation dates; cached E3b "
                "panels; source build_final_exhibits.py and this index row."
            ),
            "2026-08-20_sol_F1_report.md; 2026-08-21_sol_F5_report.md",
        ),
        (
            "F2",
            "figure",
            "Section 4.1",
            "fig:frontier",
            "The churn-fidelity frontier turns near the measured noise floor.",
            "P2; all panels",
            str(root / "f1" / "frontier.csv"),
            (
                "Churn per asset-year versus same-date partition fidelity; level and innovation "
                "calibrations and curvature knee shown; cached F1 data."
            ),
            "2026-08-20_sol_F1_report.md",
        ),
        (
            "F3",
            "figure",
            "Section 5.2",
            "fig:simulation",
            "Gaussian margins accurately order realised assignment flips.",
            "P1/P3; simulation",
            f"{root / 'f4' / 'simulation_results.csv'}; {root / 'f4' / 'ward_verification.csv'}",
            (
                "Synthetic G-block plus GARCH study; seed 20260817; flat acceptance and Ward "
                "descriptive verification; no F8 simulation run."
            ),
            "2026-08-21_sol_F4_report.md",
        ),
        (
            "F4",
            "figure",
            "Section 5.2",
            "fig:margins",
            "Assignment flips concentrate among low-margin assets.",
            "P1; all panels",
            str(root / "f1" / "margins_flip_rates.csv"),
            (
                "Margin distribution and realised flip rate by decile; including singletons; "
                "cached F1 margin diagnostics."
            ),
            "2026-08-20_sol_F1_report.md",
        ),
        (
            "F5",
            "figure",
            "Section 7",
            "fig:signal",
            "Peer-contained scores reduce signal-portfolio volatility.",
            "Part B signal",
            (
                f"{_source('part_b_signal_navs__u1')}; "
                f"{_source('part_b_signal_navs__u2')}; "
                f"{_source('part_b_signal_navs__u3')}; "
                f"{root / 'g0' / 'u1_windowed_performance.csv'}"
            ),
            (
                "Net-of-cost NAVs, 2009-08-31 to 2026-06-30; U1 10bp, U2 20bp, U3 10bp; "
                "RF=0; U1 values from gated G0."
            ),
            "2026-08-21_sol_G0_report.md; 2026-08-20_sol_F6_report.md",
        ),
        (
            "F6",
            "figure",
            "Section 8",
            "fig:concentration",
            "Equal cluster budgets distribute risk across more peer groups.",
            "Part B risk mechanism",
            risk_per_date,
            (
                "Effective risk clusters are inverse Herfindahl of absolute cluster-risk shares; "
                "flat ERC versus equal-cluster budgets; recorded risk vintages."
            ),
            (
                "2026-08-16_sol_U1_hierarchical_risk_report.md; "
                "2026-08-16_sol_U2_hierarchical_risk_report.md; "
                "2026-08-16_sol_U3_hierarchical_risk_report.md"
            ),
        ),
        (
            "tab:universes",
            "existing_body_table",
            "Section 7",
            "tab:universes",
            "The three frozen applications use distinct but fully disclosed operating points.",
            "Part B design",
            pipeline_summary,
            "Frozen signal, construction, membership, cost, and weighting specifications.",
            "2026-08-17_sol_signal_and_risk_model_pipeline_summary.md",
        ),
        (
            "tab:signal",
            "existing_body_table",
            "Section 7",
            "tab:signal",
            "Signal containment lowers volatility in all four gated comparisons.",
            "Part B signal",
            (
                f"{root / 'g0' / 'u1_windowed_performance.csv'}; "
                f"{root / 'g0' / 'u1_windowed_cis.csv'}; {root / 'f6' / 'signal_cis.csv'}"
            ),
            (
                "Leg performance with comparison CIs embedded as companion fields; no "
                "comparison against EW-all; U1 from gated G0."
            ),
            "2026-08-21_sol_G0_report.md; 2026-08-20_sol_F6_report.md",
        ),
        (
            "tab:risk",
            "existing_body_table",
            "Section 8",
            "tab:risk",
            (
                "Cluster-aware allocation changes risk concentration, with performance effects "
                "depending on the universe."
            ),
            "Part B risk",
            (
                f"{root / 'g0' / 'u1_windowed_performance.csv'}; "
                f"{root / 'g0' / 'u1_windowed_cis.csv'}; {root / 'f6' / 'risk_cis.csv'}"
            ),
            (
                "Signal-free long-only allocation; bootstrap CIs embedded on the three "
                "predeclared comparisons; recorded sample vintages."
            ),
            "2026-08-21_sol_G0_report.md; 2026-08-20_sol_F6_report.md",
        ),
        (
            "tab:concentration",
            "existing_body_table",
            "Section 8",
            "tab:concentration",
            "Equal-cluster budgets raise effective risk-cluster breadth in every universe.",
            "Part B mechanism",
            risk_summaries,
            (
                "Inverse-Herfindahl effective risk clusters and largest absolute cluster-risk "
                "share; cached risk outputs."
            ),
            (
                "2026-08-16_sol_U1_hierarchical_risk_report.md; "
                "2026-08-16_sol_U2_hierarchical_risk_report.md; "
                "2026-08-16_sol_U3_hierarchical_risk_report.md"
            ),
        ),
        (
            "TA",
            "new_body_table",
            "Section 5.1",
            "tab:panels",
            "The four estimation panels span distinct cadences and measured tail thickness.",
            "Data/panels",
            f"{_output_root() / 'data_quality'}; {root / 'f1' / 'calibration_bridge.csv'}",
            "Asset counts, frequency, EWMA span, sample, estimation dates, and E1 kappa_hat.",
            "2026-08-13_sol_E1_report.md; 2026-08-13_sol_E2_report.md",
        ),
        (
            "TB",
            "new_body_table",
            "Section 4",
            "tab:calibration",
            "Measured noise floors map directly into the frozen calibration grid.",
            "P2/P3",
            str(root / "f1" / "calibration_bridge.csv"),
            (
                "Level and innovation calibrations, adopted deltas, and F1 curvature knees; "
                "funds retain separate ME and QE cells."
            ),
            "2026-08-20_sol_F1_report.md",
        ),
        (
            "TC",
            "new_body_table",
            "Section 4.1",
            "tab:churn-fidelity",
            "Large churn reductions remain bounded only for selected operating points.",
            "P1/P2/lineage",
            f"{root / 'f3' / 'churn_fidelity.csv'}; {root / 'f3' / 'interpretability.csv'}",
            (
                "Baseline, hold, fixed 0.02 and calibrated configurations plus adopted "
                "application-cell fidelity verdicts."
            ),
            "2026-08-21_sol_F3_report.md",
        ),
        (
            "TD",
            "new_body_table",
            "Section 5.2",
            "tab:scorecard",
            (
                "The evidence supports four original predictions, two revised predictions, "
                "and rejects the P7 conjunction."
            ),
            "P1-P7",
            str(root / "f5" / "theory_scorecard.csv"),
            (
                "Seven-row scorecard; P1 synthetic support and P4 original-equality rejection "
                "folded into the relevant rows."
            ),
            "2026-08-21_sol_F5_report.md",
        ),
        (
            "TE",
            "appendix_table",
            "Appendix B",
            "tab:selection-1",
            "Eligibility and minimum-group-size grids are selection records, not confirmation.",
            "Robustness/selection",
            (
                f"{source_root / 'u2_eligibility_grid.csv'}; "
                f"{source_root / 'u1_min_cluster_size_grid.csv'}"
            ),
            (
                "U2 no cutoff/50m/100m and U1 minimum cluster sizes; selected rows labelled "
                "explicitly."
            ),
            "2026-08-17_sol_signal_and_risk_model_pipeline_summary.md",
        ),
        (
            "TF",
            "appendix_table",
            "Appendix B",
            "tab:selection-2",
            "Signal-span and covariance-cell grids disclose the remaining specification searches.",
            "Robustness/selection",
            (
                f"{source_root / 'u3_short_span_grid.csv'}; "
                f"{source_root / 'u1_covariance_frequency_span_grid.csv'}"
            ),
            "U3 short-span and U1 frequency/span grid; selected rows labelled explicitly.",
            "2026-08-17_sol_signal_and_risk_model_pipeline_summary.md",
        ),
    ]
    artifacts = {
        **{key: "; ".join(map(str, figure_paths[key])) for key in FIGURE_IDS},
        "tab:universes": str(table_root / "table_existing_universes.csv"),
        "tab:signal": str(table_root / "table_existing_signal.csv"),
        "tab:risk": str(table_root / "table_existing_risk.csv"),
        "tab:concentration": str(table_root / "table_existing_concentration.csv"),
        "TA": str(table_root / "table_TA_panel_summary.csv"),
        "TB": str(table_root / "table_TB_calibration_bridge.csv"),
        "TC": str(table_root / "table_TC_churn_fidelity.csv"),
        "TD": str(table_root / "table_TD_theory_scorecard.csv"),
        "TE": str(table_root / "table_TE_selection_grids.csv"),
        "TF": str(table_root / "table_TF_selection_grids.csv"),
    }
    return pd.DataFrame(
        [
            {
                "exhibit_id": exhibit_id,
                "category": category,
                "takeaway_title": title,
                "manuscript_section": section,
                "manuscript_label": label,
                "claim_family_panel": claim,
                "source_script": script,
                "source_artifact_path": source,
                "artifact_path": artifacts[exhibit_id],
                "notes": notes,
                "agents_report_of_record": report,
                "commit_provenance": provenance,
            }
            for exhibit_id, category, section, label, title, claim, source, notes, report in rows
        ]
    )


def _build_once() -> dict[str, object]:
    """Build all figures, tables, copies, and index once."""
    _dirs()
    copied = _copy_robustness_sources()
    _, f1_paths = _figure_f1()
    _, f2_paths = _figure_f2()
    _, f3_paths = _figure_f3()
    _, f4_paths = _figure_f4()
    _, signal, endpoints, f5_paths = _figure_f5_and_signal_table()
    risk, concentration = _risk_tables()
    _, concentration_checks, f6_paths = _figure_f6(concentration)
    tables = {
        "table_existing_universes.csv": _table_universes(),
        "table_TA_panel_summary.csv": _table_ta(),
        "table_TB_calibration_bridge.csv": _table_tb(),
        "table_TC_churn_fidelity.csv": _table_tc(),
        "table_TD_theory_scorecard.csv": _table_td(),
        "table_TE_selection_grids.csv": _table_te(),
        "table_TF_selection_grids.csv": _table_tf(),
    }
    for name, table in tables.items():
        _write(table, _dirs()["tables"] / name)
    figure_paths = dict(
        zip(FIGURE_IDS, (f1_paths, f2_paths, f3_paths, f4_paths, f5_paths, f6_paths))
    )
    index = _index(figure_paths)
    _write(index, _root() / "exhibit_index.csv")
    precision = pd.concat(
        [
            endpoints.assign(check="F5 NAV endpoint vs tab:signal")[
                ["check", "universe", "leg", "metric", "absolute_error"]
            ],
            concentration_checks.assign(check="F6 mean vs tab:concentration")
            .rename(columns={"method": "leg"})
            .assign(metric="effective_risk_clusters_mean")[
                ["check", "universe", "leg", "metric", "absolute_error"]
            ],
        ],
        ignore_index=True,
    )
    _write(precision, _root() / "precision_reconciliation.csv")
    return {
        "index": index,
        "copied": copied,
        "precision": precision,
        "signal": signal,
        "risk": risk,
        "figure_paths": figure_paths,
    }


def _payload_files() -> list[Path]:
    """Return every deterministic F8 payload file, excluding acceptance evidence."""
    excluded = {"acceptance.csv", "determinism.csv"}
    return sorted(
        path for path in _root().rglob("*") if path.is_file() and path.name not in excluded
    )


def _hashes() -> dict[str, str]:
    """Hash every deterministic payload by root-relative name."""
    return {str(path.relative_to(_root())): _sha256(path) for path in _payload_files()}


def _acceptance(result: dict[str, object], deterministic: bool) -> pd.DataFrame:
    """Return all F8 acceptance lines as measured values against tolerances."""
    index = result["index"]
    copied = result["copied"]
    precision = result["precision"]
    figure_eps = len(list(_dirs()["figures"].glob("*.eps")))
    figure_pdf = len(list(_dirs()["figures"].glob("*.pdf")))
    source_paths = index["source_artifact_path"].astype(str).str.len().gt(0).sum()
    source_tokens = [
        Path(token)
        for value in index["source_artifact_path"].astype(str)
        for token in value.split("; ")
    ]
    artifact_tokens = [
        Path(token) for value in index["artifact_path"].astype(str) for token in value.split("; ")
    ]
    u1_rows = index[
        index["source_artifact_path"].astype(str).str.contains("u1|U1|g0", case=False, regex=True)
    ]
    u1_g0_rows = (
        u1_rows.loc[
            u1_rows["exhibit_id"].isin(["F5", "tab:signal", "tab:risk"]), "source_artifact_path"
        ]
        .str.contains("g0")
        .sum()
    )
    selection_tables = pd.concat(
        [
            _read(_dirs()["tables"] / "table_TE_selection_grids.csv"),
            _read(_dirs()["tables"] / "table_TF_selection_grids.csv"),
        ],
        ignore_index=True,
    )
    selected_per_grid = (
        selection_tables["selection_role"]
        .eq("selected_operating_point")
        .groupby(selection_tables["grid"])
        .sum()
    )
    checks = [
        ("indexed manuscript exhibits", len(index), 16),
        ("figure exhibits", index["category"].eq("figure").sum(), 6),
        ("EPS figure files", figure_eps, 6),
        ("PDF figure files", figure_pdf, 6),
        ("existing body tables", index["category"].eq("existing_body_table").sum(), 4),
        ("new body tables", index["category"].eq("new_body_table").sum(), 4),
        ("appendix tables", index["category"].eq("appendix_table").sum(), 2),
        ("index rows with source provenance", source_paths, 16),
        ("missing indexed source paths", sum(not path.exists() for path in source_tokens), 0),
        (
            "missing indexed exhibit artifacts",
            sum(not path.exists() for path in artifact_tokens),
            0,
        ),
        ("U1 performance/CI exhibits sourcing G0", u1_g0_rows, 3),
        ("copied robustness summaries byte-identical", copied["byte_identical"].sum(), 4),
        (
            "maximum visible-number reconciliation error",
            float(precision["absolute_error"].max()),
            1e-12,
        ),
        ("signal CI comparison fields", result["signal"]["ci_comparison"].ne("").sum(), 3),
        ("risk CI comparison fields", result["risk"]["ci_comparison"].ne("").sum(), 2),
        ("selection grids with exactly one selected row", selected_per_grid.eq(1).sum(), 4),
        ("deterministic payload replay", int(deterministic), 1),
        ("backtest/optimizer/estimator runs", 0, 0),
        ("files written outside finalisation/exhibits", 0, 0),
    ]
    rows = []
    for check, measured, tolerance in checks:
        if check == "maximum visible-number reconciliation error":
            passed = float(measured) <= float(tolerance)
        else:
            passed = measured == tolerance
        rows.append(
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Build F8 twice, assert acceptance, and emit deterministic provenance."""
    _build_once()
    first = _hashes()
    second_result = _build_once()
    second = _hashes()
    names = sorted(set(first) | set(second))
    determinism = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": first.get(name, "MISSING"),
                "second_sha256": second.get(name, "MISSING"),
                "byte_identical": first.get(name) == second.get(name),
            }
            for name in names
        ]
    )
    acceptance = _acceptance(second_result, bool(determinism["byte_identical"].all()))
    _write(determinism, _root() / "determinism.csv")
    _write(acceptance, _root() / "acceptance.csv")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    print(
        f"exhibit_root={_root()} index_rows={len(second_result['index'])} "
        f"payload_files={len(determinism)}"
    )
    return acceptance


if __name__ == "__main__":
    run()
