"""
Loaders for the versioned paper-data snapshots.

A snapshot is an immutable folder snapshots/<tag>/ of csv files plus a
MANIFEST.json carrying provenance (source workbook and hash, production
config rows, package versions) and a sha256 per file. load_snapshot()
verifies every hash before returning, so an edited snapshot fails loudly.

Snapshot contents (all decimal per annum unless stated):
    assets.csv                  per-asset config: sleeve, class, frequency,
                                alpha, resid_vol, total_vol, r2, w_workbook,
                                w_paper, factor_excess_cma,
                                equity_regional_addon, rf_rate
    betas.csv                   N x M factor loadings
    factor_covar.csv            M x M factor covariance
    factor_premia.csv           M x 3 (base, stress, upside)
    asset_excess_logreturns.csv estimation returns panel, bootstrap window
    asset_total_returns.csv     reporting returns panel, bootstrap window
    factor_navs.csv             daily factor NAVs (base 100), window-trimmed

Freeze rule: snapshots are append-only. A new cut is a NEW tag; papers pin
their tag and never read a mutable 'latest'.

REDISTRIBUTION. The four config files (assets, betas, factor_covar,
factor_premia) plus MANIFEST.json are the numbers the papers publish and ship
with the public repository. The three RETURN PANELS carry licensed index and
factor histories and are NOT redistributed (see cma_data/.gitignore), so a
public checkout has the config and not the panels. Everything here therefore
treats a panel as OPTIONAL: it loads when present, is None when absent, and
PaperInputs.require_panel() raises a message naming the file and the scripts
that need it. Manifest verification checks the hash of every file that IS
present and reports the absent ones, so tampering is still caught on
everything shipped.

Does not belong here: universe identity (universe.py), benchmarks
(benchmarks.py), any computation on the inputs (the papers' replication).
"""
# packages
import hashlib
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .local_path import get_snapshots_path

# the four config files always ship; the three return panels do not
CONFIG_FILES: Tuple[str, ...] = ('assets.csv', 'betas.csv', 'factor_covar.csv',
                                 'factor_premia.csv')
PANEL_FILES: Tuple[str, ...] = ('asset_excess_logreturns.csv', 'asset_total_returns.csv',
                                'factor_navs.csv')
# which replication scripts need each panel, for the message a public checkout sees
PANEL_CONSUMERS: Dict[str, str] = {
    'asset_excess_logreturns': 'run_consistency_exhibits.py (J4d), run_bootstrap_q2.py (J5)',
    'asset_total_returns': 'no current consumer; carried for completeness',
    'factor_navs': ('run_factor_history_exhibits.py (J2), run_bootstrap_q2.py (J5), '
                    'run_snapshot_tables.py (J1, tab:factor_returns only)'),
}


@dataclass(frozen=True)
class PaperInputs:
    """one frozen paper cut: config, loadings, covariance, premia, and return panels.

    The three return panels are Optional: they are absent in a public checkout
    (licensed histories, see the module docstring). Reach them through
    require_panel() so an absent panel fails with a message rather than an
    AttributeError on None.
    """
    tag: str
    assets: pd.DataFrame            # per-asset config, index = tickers
    betas: pd.DataFrame             # N x 9
    factor_covar: pd.DataFrame      # M x M
    factor_premia: pd.Series        # base premia (M,)
    factor_premia_scenarios: pd.DataFrame   # M x (base, stress, upside)
    asset_excess_logreturns: Optional[pd.DataFrame]   # None when not redistributed
    asset_total_returns: Optional[pd.DataFrame]       # None when not redistributed
    factor_navs: Optional[pd.DataFrame]               # None when not redistributed
    manifest: dict
    absent_files: Tuple[str, ...] = ()   # snapshot files listed in the manifest but not present

    def __post_init__(self):
        if not self.betas.index.equals(self.assets.index):
            raise ValueError(f"betas index misaligned with assets, got {list(self.betas.index)!r}")
        if not self.betas.columns.equals(self.factor_covar.columns):
            raise ValueError(f"factor columns misaligned, got {list(self.betas.columns)!r}")

    def has_panel(self, name: str) -> bool:
        """True when the named return panel is present in this checkout."""
        if name not in PANEL_CONSUMERS:
            raise ValueError(f"unknown panel, got {name!r}; choose from {list(PANEL_CONSUMERS)}")
        return getattr(self, name) is not None

    def require_panel(self, name: str) -> pd.DataFrame:
        """the named return panel, or a clear error naming the file and its consumers."""
        if self.has_panel(name):
            return getattr(self, name)
        raise ValueError(
            f"snapshot panel {name!r} is not in this checkout. The three return panels carry "
            f"licensed index and factor histories and are not redistributed with the public "
            f"repository (cma_data/.gitignore). Scripts needing this one: "
            f"{PANEL_CONSUMERS[name]}. To run them, place the file at "
            f"snapshots/{self.tag}/{name}.csv from the production extract; the manifest hash "
            f"is verified on load, so a wrong file fails loudly.")


def _sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def verify_manifest(snapshot_path: Path) -> Tuple[dict, Tuple[str, ...]]:
    """verify the sha256 of every manifest file that is present; return (manifest, absent files).

    Absence is not tampering: the three return panels are not redistributed
    publicly, so a public checkout is missing them by design. A file that IS
    present must match its hash. A missing CONFIG file is still fatal, because
    the papers' published numbers depend on it.
    """
    manifest_file = snapshot_path / 'MANIFEST.json'
    if not manifest_file.exists():
        raise ValueError(f"MANIFEST.json missing in snapshot, got {str(snapshot_path)!r}")
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    absent: List[str] = []
    for name, expected in manifest.get('file_sha256', {}).items():
        file_path = snapshot_path / name
        if not file_path.exists():
            if name in CONFIG_FILES:
                raise ValueError(f"snapshot config file missing, got {name!r} in "
                                 f"{str(snapshot_path)!r}; the config files always ship")
            absent.append(name)
            continue
        actual = _sha256(file_path)
        if actual != expected:
            raise ValueError(f"snapshot file hash mismatch for {name!r}: "
                             f"expected {expected}, got {actual}. "
                             f"Snapshots are immutable; re-extract under a new tag.")
    return manifest, tuple(absent)


def load_snapshot(tag: str = '2026q2',
                  snapshots_path: Optional[Path] = None,
                  verify: bool = True,
                  ) -> PaperInputs:
    """load one frozen paper cut by tag, verifying the manifest hashes by default."""
    root = get_snapshots_path() if snapshots_path is None else Path(snapshots_path)
    snapshot_path = root / tag
    if not snapshot_path.exists():
        available = sorted(p.name for p in root.glob('*') if p.is_dir()) if root.exists() else []
        raise ValueError(f"snapshot tag not found, got {tag!r}; available: {available}")
    if verify:
        manifest, absent = verify_manifest(snapshot_path=snapshot_path)
    else:
        manifest = json.loads((snapshot_path / 'MANIFEST.json').read_text(encoding='utf-8'))
        absent = tuple(name for name in manifest.get('file_sha256', {})
                       if not (snapshot_path / name).exists())

    def read(name: str, parse_dates: bool = False) -> pd.DataFrame:
        return pd.read_csv(snapshot_path / name, index_col=0,
                           parse_dates=parse_dates)

    def read_panel(name: str) -> Optional[pd.DataFrame]:
        """a return panel, or None when it is not redistributed in this checkout."""
        return None if name in absent else read(name, parse_dates=True)

    assets = read('assets.csv')
    premia = read('factor_premia.csv')
    return PaperInputs(tag=tag,
                       assets=assets,
                       betas=read('betas.csv'),
                       factor_covar=read('factor_covar.csv'),
                       factor_premia=premia['base'],
                       factor_premia_scenarios=premia,
                       asset_excess_logreturns=read_panel('asset_excess_logreturns.csv'),
                       asset_total_returns=read_panel('asset_total_returns.csv'),
                       factor_navs=read_panel('factor_navs.csv'),
                       manifest=manifest,
                       absent_files=absent)
