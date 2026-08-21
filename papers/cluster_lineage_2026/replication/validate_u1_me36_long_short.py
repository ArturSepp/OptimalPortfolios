"""Independently validate the persisted U1 ME/36 long-short curiosity run."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_me36_long_short import _root


def _read(name: str) -> pd.DataFrame:
    """Read one persisted long-short artifact."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def validate() -> None:
    """Assert exact exposure, expected rows, and deterministic outputs."""
    performance = _read("performance")
    comparison = _read("comparison")
    acceptance = _read("acceptance")
    exposure = _read("exposure_diagnostics")
    replay = _read("determinism")
    assert len(performance) == 4
    assert len(comparison) == 2
    assert len(acceptance) == 4
    assert acceptance["status"].eq("PASS").all()
    assert exposure["net_exposure"].abs().max() <= 1e-12
    assert (exposure["gross_exposure"] - 2.0).abs().max() <= 1e-12
    assert replay["byte_identical"].astype(bool).all()
    headline = performance.loc[
        performance["analysis_window"].eq("headline_20090831_20260630")
    ].set_index("leg")
    cluster = headline.loc["cluster_ME_span_36"]
    global_row = headline.loc["global"]
    print("U1 ME/36 q=0.25 long-short independent validation: PASS")
    print(
        f"cluster: return={cluster['net_return_annualized']:.6f}, "
        f"Sharpe={cluster['sharpe_rf0']:.6f}, "
        f"turnover={cluster['one_way_turnover_annualized']:.6f}"
    )
    print(
        f"global: return={global_row['net_return_annualized']:.6f}, "
        f"Sharpe={global_row['sharpe_rf0']:.6f}, "
        f"turnover={global_row['one_way_turnover_annualized']:.6f}"
    )
    print(
        f"cluster-minus-global: return="
        f"{cluster['net_return_annualized'] - global_row['net_return_annualized']:.6f}, "
        f"Sharpe={cluster['sharpe_rf0'] - global_row['sharpe_rf0']:.6f}"
    )
    print(f"determinism: {len(replay)}/{len(replay)} artifacts byte-identical")


if __name__ == "__main__":
    validate()
