"""
cma_data — the shared data layer of the paper packages.

    prod pipeline (private) --extract--> snapshots/<tag>/ (frozen, public)
                                             |
                    matf_cma_jpm_2026 <---- loaders ----> achievable_sharpe_faj_2026

One universe, one benchmark construction, one loader API, versioned
immutable snapshots. Papers pin a snapshot tag; regenerating a cut creates
a new tag and moves no published number. The extractor (_local_*.py,
untracked) is the only file that touches production data.

See README.md for the schema and the freeze rules.
"""
from .universe import (PAPER_UNIVERSE, ASSET_CLASSES, ADMISSION_POLICY, FACTORS,
                       BOOTSTRAP_START, BOOTSTRAP_END, BOOTSTRAP_MONTHS,
                       get_universe, get_admission_policy)
from .benchmarks import (MANDATES, get_benchmark_weights, get_all_benchmarks)
from .loaders import (PaperInputs, load_snapshot, verify_manifest)
from .local_path import (get_snapshots_path, load_settings, CMA_DATA_PATH)
from .consensus import (ProviderSource, CONSENSUS_LABEL, HORIZON_MAP,
                        get_horizon_survey, get_horizon_distributions,
                        build_consensus_provider)
