"""Run the all-fund U2 attribution with classic 12m-ex-1m momentum."""

import pandas as pd

from papers.cluster_lineage_2026.replication import (
    run_u2_all_funds_asset_class_attribution as attribution,
)


def main() -> None:
    """Run, replay, and display the classic-momentum attribution."""
    signal = attribution.CLASSIC_SIGNAL
    replay = attribution.verify_determinism(signal)
    performance = pd.read_csv(
        attribution._root(signal) / "performance.csv", float_precision="round_trip"
    )
    asset_classes = pd.read_csv(
        attribution._root(signal) / "asset_class_delta_vs_global.csv",
        float_precision="round_trip",
    )
    print(performance.to_string(index=False), flush=True)
    print(asset_classes.to_string(index=False), flush=True)
    print(
        f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
