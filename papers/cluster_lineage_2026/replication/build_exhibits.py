"""Stage E7 entry point for the exact executed canonical-workbook builder."""
from papers.cluster_lineage_2026.replication.recovery_loader import load_executed

_executed = load_executed("build_exhibits")
build_universe = _executed.build_universe
build_all = _executed.build_all

if __name__ == "__main__":
    print(build_all().to_string(index=False))
