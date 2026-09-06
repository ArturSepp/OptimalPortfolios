"""Generate the circulation-note figures from verified 2026 analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis

PAPER_ROOT = Path(__file__).resolve().parent.parent
UPDATE_PATH = PAPER_ROOT / "update_2026"
FIGURE_PATH = UPDATE_PATH / "figures"
OUTPUT_ROOT = PAPER_ROOT / "outputs" / "bbg_20260904" / "published_2024"
LEGACY_OUTPUT = OUTPUT_ROOT / "legacy_eth_proxy"
OBSERVED_OUTPUT = OUTPUT_ROOT / "observed_eth"
RAW_BLOOMBERG_PRICES = PAPER_ROOT / "data" / "bloomberg" / "bbg_20260904" / "raw_bloomberg_px_last.csv"
PUBLISHED_SAMPLE_END = pd.Timestamp("2023-06-30")

SCENARIO_LABELS = {
    ("alternatives", "BTC"): "Alternatives + BTC",
    ("alternatives", "ETH"): "Alternatives + ETH",
    ("balanced_risk_budget", "BTC"): "Balanced/alts + BTC",
    ("balanced_risk_budget", "ETH"): "Balanced/alts + ETH",
}

PUBLISHED_SCENARIO_MEDIANS = {
    # Exact cross-method medians in the workbook underlying Figure 9 of the 2023 paper.
    ("alternatives", "BTC"): 0.034550010958,
    ("alternatives", "ETH"): 0.022348552457,
    ("balanced_risk_budget", "BTC"): 0.045530533822,
    ("balanced_risk_budget", "ETH"): 0.020013524354,
}

COLORS = ("#1F4E79", "#D97706", "#2E7D32", "#8E3B8F")
SOURCE_NOTE = "Source: Bloomberg; OptimalPortfolios paper replication. Data through 4 Sep 2026."


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save a tightly cropped high-resolution PNG and close the figure."""
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    path = FIGURE_PATH / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _source_note(fig: plt.Figure, text: str = SOURCE_NOTE) -> None:
    """Add a consistent source line below an exhibit."""
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7, color="#555555")


def _scenario_summary(path: Path) -> pd.Series:
    """Load scenario-level median crypto weights in canonical order."""
    series = pd.read_csv(path, index_col=[0, 1]).iloc[:, 0]
    expected = list(SCENARIO_LABELS)
    series = series.reindex(expected)
    if series.isna().any():
        raise ValueError(f"Scenario summary is incomplete: {path}")
    series.index = [SCENARIO_LABELS[key] for key in expected]
    return series.astype(float)


def plot_allocation_update() -> Path:
    """Compare the paper's June 2023 and updated 2026 scenario allocations."""
    updated = _scenario_summary(LEGACY_OUTPUT / "scenario_summary.csv")
    published = pd.Series(
        {
            SCENARIO_LABELS[key]: value
            for key, value in PUBLISHED_SCENARIO_MEDIANS.items()
        }
    ).reindex(updated.index)
    frame = pd.concat(
        [
            published.rename("Paper: through 30 Jun 2023"),
            updated.rename("2026 Bloomberg update"),
        ],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    qis.plot_bars(
        df=frame,
        stacked=False,
        title="Crypto allocations remain positive after the paper's sample cutoff",
        yvar_format="{:.1%}",
        colors=list(COLORS[:2]),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median target allocation")
    ax.tick_params(axis="x", labelrotation=12)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:.1%}" for value in container.datavalues], fontsize=8)
    ax.text(
        0.99,
        0.95,
        f"Overall median: {np.median(published):.2%} to {np.median(updated):.2%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F3F6FA", "edgecolor": "#AAB7C4"},
    )
    _source_note(
        fig,
        "Source: workbook underlying Sepp (2023), Figure 9; Bloomberg update through 4 Sep 2026.",
    )
    return _save(fig, "allocation_update_scenarios.png")


def plot_allocation_paths() -> Path:
    """Plot quarterly cross-method median target allocations by scenario."""
    scenario_series: dict[tuple[str, str], pd.Series] = {}
    for universe, asset in SCENARIO_LABELS:
        path = LEGACY_OUTPUT / f"weights_{universe}_{asset}.csv"
        weights = pd.read_csv(path, index_col=0, parse_dates=True)
        scenario_series[(universe, asset)] = weights.median(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, asset in zip(axes, ("BTC", "ETH"), strict=True):
        frame = pd.concat(
            [
                scenario_series[("alternatives", asset)].rename("Alternatives"),
                scenario_series[("balanced_risk_budget", asset)].rename("75/25 balanced"),
            ],
            axis=1,
        )
        qis.plot_time_series(
            df=frame,
            title=f"{asset}: quarterly median across four allocation methods",
            date_format="%Y",
            var_format="{:.1%}",
            colors=list(COLORS[:2]),
            ax=ax,
        )
        ax.set_ylabel("Target allocation")
        ax.tick_params(axis="x", labelrotation=0)
        ax.axvline(PUBLISHED_SAMPLE_END, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].text(
        PUBLISHED_SAMPLE_END,
        axes[0].get_ylim()[1] * 0.92,
        "paper cutoff",
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.09, hspace=0.39)
    _source_note(
        fig,
        "Source: Bloomberg; OptimalPortfolios paper replication. "
        "Targets through 30 Jun 2026; market data through 4 Sep 2026.",
    )
    return _save(fig, "allocation_paths.png")


def plot_crypto_price_history() -> Path:
    """Plot observed BTC and ETH prices before and after the published sample."""
    prices = pd.read_csv(RAW_BLOOMBERG_PRICES, index_col=0, parse_dates=True)[["BTC", "ETH"]]
    prices = prices.sort_index()
    panels = (
        ("BTC", ("#1F4E79", "#D97706")),
        ("ETH", ("#7A5195", "#2E7D32")),
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, (asset, colors) in zip(axes, panels, strict=True):
        series = prices[asset].dropna()
        published = series.where(series.index <= PUBLISHED_SAMPLE_END)
        updated = series.where(series.index >= PUBLISHED_SAMPLE_END)
        frame = pd.concat(
            [
                published.rename(f"{asset} - published sample (through 30 Jun 2023)"),
                updated.rename(f"{asset} - new Bloomberg history"),
            ],
            axis=1,
        )
        qis.plot_time_series(
            df=frame,
            title=f"{asset} observed Bloomberg price",
            date_format="%Y",
            var_format="${:,.0f}",
            colors=list(colors),
            legend_stats=qis.LegendStats.NONE,
            is_log=True,
            linewidth=1.4,
            ax=ax,
        )
        ax.set_ylabel("USD price (log scale)")
        ax.tick_params(axis="x", labelrotation=0)
        ax.axvline(PUBLISHED_SAMPLE_END, color="#777777", linestyle="--", linewidth=1.0)
        cutoff_price = float(series.loc[PUBLISHED_SAMPLE_END])
        latest_price = float(series.iloc[-1])
        ax.text(
            0.99,
            0.05,
            f"30 Jun 2023: USD {cutoff_price:,.0f}   |   4 Sep 2026: USD {latest_price:,.0f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D6DCE2"},
        )
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Crypto prices before and after the paper's 30 June 2023 sample cutoff",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.09, hspace=0.35)
    _source_note(
        fig,
        "Source: Bloomberg XBTUSD Curncy and XETUSD Curncy, unadjusted PX_LAST. "
        "Data through 4 Sep 2026.",
    )
    return _save(fig, "crypto_price_history.png")


def _post_published_sample_deltas() -> pd.DataFrame:
    """Compute post-paper with-minus-without crypto performance deltas."""
    performance = pd.read_csv(LEGACY_OUTPUT / "performance_summary.csv")
    performance = performance.loc[performance["period"] == "post_published_sample"]
    if performance.empty:
        raise ValueError("Performance output has no post_published_sample window; rerun analysis")
    rows = []
    for keys, group in performance.groupby(["universe", "crypto_asset", "method"]):
        indexed = group.set_index("portfolio")
        rows.append(
            {
                "universe": keys[0],
                "crypto_asset": keys[1],
                "method": keys[2],
                "annual_return_delta": (
                    float(indexed.loc["with_crypto", "P.a. return"])
                    - float(indexed.loc["without_crypto", "P.a. return"])
                ),
                "sharpe_delta": (
                    float(indexed.loc["with_crypto", "Log Ex Sharpe"])
                    - float(indexed.loc["without_crypto", "Log Ex Sharpe"])
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_post_update_performance() -> Path:
    """Summarize performance after the paper's 30 June 2023 sample endpoint."""
    deltas = _post_published_sample_deltas()
    method_order = ["ERC", "MaxDiv", "MaxSharpe", "CARA-3"]
    medians = deltas.groupby("method")[["annual_return_delta", "sharpe_delta"]].median()
    medians = medians.reindex(method_order)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    panels = (
        (
            "annual_return_delta",
            "Median annual-return change",
            "{:.1%}",
            "Annual-return change (percentage points)",
        ),
        (
            "sharpe_delta",
            "Median log excess-Sharpe change",
            "{:.2f}",
            "Log excess Sharpe",
        ),
    )
    for ax, (column, title, value_format, ylabel) in zip(axes, panels, strict=True):
        frame = medians[[column]].rename(columns={column: "With minus without crypto"})
        qis.plot_bars(
            df=frame,
            stacked=False,
            title=title,
            yvar_format=value_format,
            colors=[COLORS[0]],
            ax=ax,
        )
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.margins(y=0.12)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=0)
        for container in ax.containers:
            if column == "annual_return_delta":
                labels = [f"{value:+.1%}" for value in container.datavalues]
            else:
                labels = [f"{value:+.3f}" for value in container.datavalues]
            ax.bar_label(container, labels=labels, fontsize=8, padding=3)
    fig.suptitle(
        "Post-paper evidence is mixed (30 Jun 2023 to 4 Sep 2026)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.075,
        "Across all 16 cases: return improved in "
        f"{int((deltas['annual_return_delta'] > 0.0).sum())}; log excess Sharpe improved in "
        f"{int((deltas['sharpe_delta'] > 0.0).sum())}.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.20, wspace=0.18)
    _source_note(fig)
    return _save(fig, "post_update_performance.png")


def plot_observed_eth_sensitivity() -> Path:
    """Compare proxy and observed-ETH central allocation estimates."""
    legacy = _scenario_summary(LEGACY_OUTPUT / "scenario_summary.csv")
    observed = _scenario_summary(OBSERVED_OUTPUT / "scenario_summary.csv")
    legacy_eth = legacy.loc[
        [
            SCENARIO_LABELS[("alternatives", "ETH")],
            SCENARIO_LABELS[("balanced_risk_budget", "ETH")],
        ]
    ]
    observed_eth = observed.loc[legacy_eth.index]
    frame = pd.DataFrame(
        {
            "Legacy ETH proxy": [*legacy_eth, float(np.median(legacy))],
            "Observed ETH": [*observed_eth, float(np.median(observed))],
        },
        index=[*legacy_eth.index, "Overall BTC/ETH headline"],
    )

    observed_manifest = json.loads(
        (OBSERVED_OUTPUT / "ANALYSIS_MANIFEST.json").read_text(encoding="utf-8")
    )
    observed_start = observed_manifest["reporting_start_by_crypto_asset"]["ETH"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    qis.plot_bars(
        df=frame,
        stacked=False,
        title="Observed-ETH shorter-sample sensitivity lowers the central estimate",
        yvar_format="{:.1%}",
        colors=list(COLORS[:2]),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median target allocation")
    ax.tick_params(axis="x", labelrotation=10)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:.1%}" for value in container.datavalues], fontsize=8)
    ax.text(
        0.50,
        0.84,
        f"Observed-ETH common reporting\nstarts {observed_start}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D6DCE2"},
    )
    _source_note(fig, "Source: Bloomberg XETUSD observed from 8 Feb 2018; 60-month common warm-up.")
    return _save(fig, "observed_eth_sensitivity.png")


def main() -> None:
    """Generate every exhibit required by the 2026 internal update note."""
    outputs = [
        plot_allocation_update(),
        plot_allocation_paths(),
        plot_crypto_price_history(),
        plot_post_update_performance(),
        plot_observed_eth_sensitivity(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
