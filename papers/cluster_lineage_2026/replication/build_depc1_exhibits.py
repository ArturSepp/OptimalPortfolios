"""Build the five required de-PC1 exhibits from frozen D4-D5 artifacts."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import qis

import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.run_depc1_strategy_backtests as d5


RUNNER = "papers/cluster_lineage_2026/replication/build_depc1_exhibits.py"
REQUIRED_EXHIBITS = (
    "pc1_explained_share.png",
    "cluster_count_and_taxonomy_ari.png",
    "final_partition_heatmap.png",
    "cumulative_net_nav.png",
    "depc1_vs_raw_instrument_pnl.png",
)


def _root(universe: str) -> Path:
    """Return and create one universe's exhibit directory."""
    root = d4._universe_root(universe) / "exhibits"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save(fig, universe: str, file_name: str) -> Path:
    """Save one qis figure as a deterministic high-resolution PNG."""
    import matplotlib.pyplot as plt

    path = Path(
        qis.save_fig(
            fig=fig,
            file_name=Path(file_name).stem,
            local_path=str(_root(universe)),
            dpi=180,
            add_current_date=False,
        )
    )
    plt.close(fig)
    return path


def _taxonomy_series(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return paired raw/de-PC1 taxonomy-ARI series with concise labels."""
    output = {}
    for raw_column in comparison.columns:
        prefix = "raw_taxonomy_ari_"
        if not raw_column.startswith(prefix):
            continue
        suffix = raw_column.removeprefix(prefix)
        depc_column = f"depc1_taxonomy_ari_{suffix}"
        if depc_column in comparison:
            output[f"raw {suffix}"] = comparison[raw_column]
            output[f"de-PC1 {suffix}"] = comparison[depc_column]
    return pd.DataFrame(output, index=comparison.index)


def _final_partition_table(universe: str) -> pd.DataFrame:
    """Return final-date raw/de-PC1 memberships and the first taxonomy label."""
    inputs, raw, depc1 = d5._partition_panels(universe)
    date = inputs.dates[-1]
    eligible = inputs.eligibility.loc[date].astype(bool)
    assets = inputs.eligibility.columns[eligible]
    taxonomy_name, taxonomy = next(iter(inputs.taxonomy.items()))
    table = pd.DataFrame(
        {
            "asset": assets,
            "raw_cluster": raw.loc[date, assets].to_numpy(),
            "depc1_cluster": depc1.loc[date, assets].to_numpy(),
            "taxonomy_name": taxonomy_name,
            "taxonomy_label": taxonomy.reindex(assets).to_numpy(),
            "date": date,
        }
    )
    return table.sort_values(
        ["taxonomy_label", "raw_cluster", "depc1_cluster", "asset"],
        na_position="last",
    ).reset_index(drop=True)


def _partition_codes(final: pd.DataFrame) -> pd.DataFrame:
    """Factorize arbitrary cluster IDs independently for heat-map display."""
    raw_codes = pd.factorize(final["raw_cluster"], sort=True)[0] + 1
    depc_codes = pd.factorize(final["depc1_cluster"], sort=True)[0] + 1
    return pd.DataFrame(
        [raw_codes, depc_codes],
        index=["raw", "de-PC1"],
        columns=final["asset"],
    )


def _pnl_comparison(instrument_pnl: pd.DataFrame) -> pd.DataFrame:
    """Return one raw-versus-de-PC1 net contribution row per instrument."""
    pivot = instrument_pnl.pivot(
        index="asset", columns="leg", values="net_pnl_pct_of_start"
    )
    required = {"cluster_raw", "cluster_depc1"}
    if not required.issubset(pivot.columns):
        raise AssertionError(f"instrument P&L misses {sorted(required - set(pivot.columns))}")
    output = pivot[["cluster_raw", "cluster_depc1"]].copy()
    output.columns = ["raw_cluster_pnl_pct", "depc1_cluster_pnl_pct"]
    output["depc1_minus_raw_pnl_pct"] = (
        output["depc1_cluster_pnl_pct"] - output["raw_cluster_pnl_pct"]
    )
    return output.reset_index()


def build_universe(universe: str) -> pd.DataFrame:
    """Build five traced exhibits for one universe."""
    started = time.perf_counter()
    source_root = d4._universe_root(universe)
    output_root = _root(universe)
    diagnostics = pd.read_csv(
        source_root / "pc1_diagnostics.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    partition = pd.read_csv(
        source_root / "partition_comparison.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    navs = pd.read_csv(
        source_root / "navs.csv", parse_dates=["date"], float_precision="round_trip"
    ).set_index("date")
    instrument_pnl = pd.read_csv(
        source_root / "instrument_pnl.csv", float_precision="round_trip"
    )

    figures = {}
    figures["pc1_explained_share.png"] = qis.plot_time_series(
        diagnostics["pc1_variance_share"].rename("PC1 explained share"),
        title=f"{universe}: dominant common-mode share",
        ylabel="share of correlation trace",
        var_format="{:.1%}",
        x_date_freq="2YE",
    )
    topology = partition[["raw_cluster_count", "depc1_cluster_count"]].copy()
    taxonomy = _taxonomy_series(partition)
    combined = topology.join(taxonomy)
    figures["cluster_count_and_taxonomy_ari.png"] = qis.plot_time_series(
        combined,
        title=f"{universe}: raw versus de-PC1 topology and taxonomy ARI",
        ylabel="cluster count / ARI",
        var_format="{:.2f}",
        x_date_freq="2YE",
    )
    final = _final_partition_table(universe)
    d4._write(final, output_root / "final_partition.csv")
    figures["final_partition_heatmap.png"] = qis.plot_heatmap(
        _partition_codes(final),
        title=f"{universe}: final raw and de-PC1 memberships",
        date_format=None,
        annot=False,
        top_x_label=False,
        xticklabels=False,
        yticklabels=True,
        cmap="viridis",
    )
    nav_columns = [
        column
        for column in ("cluster_raw", "cluster_depc1", "global")
        if column in navs
    ]
    nav_figure = qis.plot_time_series(
        navs[nav_columns],
        title=f"{universe}: cumulative net NAV",
        ylabel="NAV",
        var_format="{:.3f}",
        x_date_freq="2YE",
        legend_loc="lower right",
        legend_stats=qis.LegendStats.NONE,
        framealpha=0.9,
        facecolor="white",
    )
    nav_legend = nav_figure.axes[0].get_legend()
    if nav_legend is not None:
        nav_legend.set_zorder(100)
        nav_legend.get_frame().set_alpha(1.0)
    figures["cumulative_net_nav.png"] = nav_figure
    pnl = _pnl_comparison(instrument_pnl)
    d4._write(pnl, output_root / "pnl_contribution_comparison.csv")
    figures["depc1_vs_raw_instrument_pnl.png"] = qis.plot_scatter(
        pnl,
        x="raw_cluster_pnl_pct",
        y="depc1_cluster_pnl_pct",
        title=f"{universe}: de-PC1 versus raw instrument net P&L",
        xlabel="raw cluster P&L (percentage points of start NAV)",
        ylabel="de-PC1 cluster P&L (percentage points of start NAV)",
        add_45line=True,
        align_axis=True,
        full_sample_order=1,
        order=1,
        xvar_format="{:.2f}",
        yvar_format="{:.2f}",
    )

    rows = []
    sources = {
        "pc1_explained_share.png": "pc1_diagnostics.csv",
        "cluster_count_and_taxonomy_ari.png": "partition_comparison.csv",
        "final_partition_heatmap.png": "exhibits/final_partition.csv",
        "cumulative_net_nav.png": "navs.csv",
        "depc1_vs_raw_instrument_pnl.png": "exhibits/pnl_contribution_comparison.csv",
    }
    for file_name, fig in figures.items():
        if fig is None:
            raise AssertionError(f"qis returned no figure for {file_name}")
        path = _save(fig, universe, file_name)
        rows.append(
            {
                "universe": universe,
                "exhibit": file_name,
                "path": str(path),
                "source": str(source_root / sources[file_name]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "status": "PASS" if path.stat().st_size > 0 else "FAIL",
            }
        )
    manifest = pd.DataFrame(rows)
    d4._write(manifest, output_root / "exhibit_manifest.csv")
    d4._write(
        pd.DataFrame(
            [{"universe": universe, "runtime_seconds": time.perf_counter() - started}]
        ),
        output_root / "runtime.csv",
    )
    return manifest


def _append_acceptance(universe: str, manifest: pd.DataFrame) -> None:
    """Append the D6 exhibit gate without duplicating earlier stage rows."""
    path = d4._universe_root(universe) / "acceptance.csv"
    acceptance = pd.read_csv(path, float_precision="round_trip")
    if "stage" in acceptance:
        acceptance = acceptance.loc[~acceptance["stage"].eq("D6")]
    row = pd.DataFrame(
        [
            {
                "stage": "D6",
                "check": "required traced exhibits emitted",
                "measured": int(manifest["status"].eq("PASS").sum()),
                "tolerance": len(REQUIRED_EXHIBITS),
                "status": "PASS"
                if manifest["status"].eq("PASS").all()
                and set(manifest["exhibit"]) == set(REQUIRED_EXHIBITS)
                else "FAIL",
            }
        ]
    )
    acceptance = pd.concat([acceptance, row], ignore_index=True)
    d4._write(acceptance, path)


def build_all() -> pd.DataFrame:
    """Build and index every required exhibit in roadmap order."""
    manifests = []
    for universe in d4.UNIVERSES:
        manifest = build_universe(universe)
        _append_acceptance(universe, manifest)
        manifests.append(manifest)
    combined = pd.concat(manifests, ignore_index=True)
    d4._write(combined, d4._output_root() / "exhibit_manifest.csv")
    return combined


def _artifact_hashes() -> dict[str, str]:
    """Hash deterministic exhibit data and PNGs, excluding runtime/replay."""
    paths = []
    for universe in d4.UNIVERSES:
        paths.extend(_root(universe).glob("*.csv"))
        paths.extend(_root(universe).glob("*.png"))
    paths.append(d4._output_root() / "exhibit_manifest.csv")
    return {
        str(path.relative_to(d4._output_root())): d4._sha256(path)
        for path in sorted(paths)
        if path.name not in {"runtime.csv", "exhibit_determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Build twice and require byte-identical exhibit data and images."""
    build_all()
    first = _artifact_hashes()
    build_all()
    second = _artifact_hashes()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    d4._write(replay, d4._output_root() / "exhibit_determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Build and deterministically replay the complete de-PC1 exhibit set."""
    replay = verify_determinism()
    print(f"de-PC1 exhibits: {len(replay)}/{len(replay)} byte-identical", flush=True)


if __name__ == "__main__":
    main()
