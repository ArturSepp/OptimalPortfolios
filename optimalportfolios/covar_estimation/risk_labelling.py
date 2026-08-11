"""Deprecated compatibility imports for factorlasso cluster-lineage analysis."""

import warnings

from factorlasso.cluster_lineage import (
    RiskClusterReport as RiskClusterReport,
    TaxonomyConfig as TaxonomyConfig,
    _build_tracks as _build_tracks,
    _classify as _classify,
    _match_panel_mcf as _match_panel_mcf,
    _psd_clip as _psd_clip,
    _snapshot_fingerprints as _snapshot_fingerprints,
    analyze_cluster_lineage,
    run_cluster_lineage_report,
    solve_max_weight_matching as solve_max_weight_matching,
)

warnings.warn(
    "optimalportfolios.covar_estimation.risk_labelling is deprecated; import from "
    "factorlasso.cluster_lineage instead",
    DeprecationWarning,
    stacklevel=2,
)

analyze_risk_clusters = analyze_cluster_lineage
run_risk_label_report = run_cluster_lineage_report
