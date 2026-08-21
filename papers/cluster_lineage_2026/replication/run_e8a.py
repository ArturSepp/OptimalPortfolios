"""Stage E8a entry point for the exact byte-identical executed module."""
from papers.cluster_lineage_2026.replication.recovery_loader import load_executed

_executed = load_executed("run_e8a")
run = _executed.run
verify_determinism = _executed.verify_determinism

if __name__ == "__main__":
    output = run()
    print(output["separability"].to_string(index=False))
    print(output["granularity_summary"].to_string(index=False))
