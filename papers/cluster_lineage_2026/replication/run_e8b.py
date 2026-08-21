"""Stage E8b entry point for the exact executed backtest and inference module."""
from papers.cluster_lineage_2026.replication.recovery_loader import load_executed

_executed = load_executed("run_e8b")
run = _executed.run
verify_determinism = _executed.verify_determinism

if __name__ == "__main__":
    output = run()
    print(output["performance"].to_string(index=False))
    print(output["inference"].to_string(index=False))
