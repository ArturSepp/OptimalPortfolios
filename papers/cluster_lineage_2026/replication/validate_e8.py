"""Stage E8 acceptance entry point for the recovered independent validator."""
import csv
from pathlib import Path

from papers.cluster_lineage_2026.replication.recovery_loader import load_executed

_executed = load_executed("validate_e8")


def validate():
    """Return E8 acceptance lines, including the persisted deterministic replay record."""
    result = _executed.validate()
    record = Path(_executed.get_output_path()) / "e8" / "u3m" / "e8b" / "determinism.csv"
    if record.exists():
        with record.open(encoding="utf-8", newline="") as stream:
            identical = sum(
                row.get("byte_identical", "").lower() == "true"
                for row in csv.DictReader(stream)
            )
        mask = result["acceptance_line"].eq("E8b byte-identical artifacts")
        result.loc[mask, "measured"] = identical
        result.loc[mask, "status"] = "PASS" if identical >= 16 else "FAIL"
    return result

if __name__ == "__main__":
    print(validate().to_string(index=False))
