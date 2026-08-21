"""Independent stage-E0 checks using scikit-learn and direct transition counts."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication import metrics
from papers.cluster_lineage_2026.replication.sp500_baseline import (
    estimate_rolling,
    extract_partitions,
    load_inputs,
)


def run_validation() -> None:
    """Cross-check ARI with scikit-learn and lineage churn with direct pair counts."""
    from sklearn.metrics import adjusted_rand_score

    covar_data = estimate_rolling("baseline")
    partitions = extract_partitions(covar_data)
    metadata = load_inputs()["metadata"]
    canonical, _ = metrics.ari_metrics(partitions, metadata)
    independent = {}
    for column in metrics.GICS_COLUMNS:
        values = []
        for date in sorted(partitions):
            frame = pd.concat([partitions[date], metadata[column]], axis=1).dropna()
            values.append(adjusted_rand_score(frame.iloc[:, 0], frame.iloc[:, 1]))
        key = f"ari_{column.removeprefix('gics_')}"
        independent[key] = float(pd.Series(values).median())
    max_ari_difference = max(abs(canonical[key] - independent[key]) for key in canonical)

    lineage, _ = metrics.lineage_metrics(covar_data)
    churn_difference = abs(
        lineage["lineage_churn_pair_count"] - lineage["lineage_churn_panel"]
    )
    assert max_ari_difference <= 1e-12
    assert churn_difference <= 1e-12
    print(f"independent ARI maximum difference: {max_ari_difference:.3e}")
    print(f"independent lineage pair/panel difference: {churn_difference:.3e}")


if __name__ == "__main__":
    run_validation()
