"""Recovered E4 interpretability runner backed by the exact executed Python 3.12 module."""
from papers.cluster_lineage_2026.replication.recovery_loader import load_executed

_executed = load_executed("run_interpretability")
globals().update(
    {name: getattr(_executed, name) for name in dir(_executed) if not name.startswith("__")}
)
