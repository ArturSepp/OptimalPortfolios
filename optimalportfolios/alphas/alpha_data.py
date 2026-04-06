"""
Data container for alpha computation outputs.

Holds the final combined alpha scores and all intermediate components
(raw signals, per-signal scores) for diagnostics and attribution.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass, fields


@dataclass
class AlphasData:
    """
    Container for alpha computation results.

    The primary output is ``alpha_scores`` — the portfolio-ready alpha vector
    that enters the optimiser as the objective in max α'(w - w_b).

    All other fields store intermediate components for diagnostics,
    attribution, and report.

    Attributes:
        alpha_scores: Final combined alpha scores (T × N). Portfolio-ready
            after CDF mapping. This is the input to the TAA optimiser.
        momentum: Raw momentum values (T × N).
        momentum_score: Cross-sectional momentum score (T × N).
        momentum_cluster: Raw momentum for cluster scoring (T × N).
        momentum_cluster_score: Cluster-scored momentum (T × N).
        beta: Raw EWMA beta to benchmark (T × N).
        beta_score: Cross-sectional low-beta score (T × N).
        beta_cluster: Raw EWMA beta for cluster scoring (T × N).
        beta_cluster_score: Cluster-scored low-beta (T × N).
        managers_alphas: Raw regression alpha (T × N).
        managers_scores: Cross-sectional managers alpha score (T × N).
        residual_momentum: Raw EWMA-smoothed residual returns (T × N).
        residual_momentum_score: Cross-sectional residual momentum score (T × N).
        residual_momentum_cluster: Raw residual returns for cluster scoring (T × N).
        residual_momentum_cluster_score: Cluster-scored residual momentum (T × N).
        clusters: Time-varying cluster assignments (T × N). Index=dates,
            columns=tickers, values=cluster_id (integer).
    """
    alpha_scores: pd.DataFrame
    momentum: Optional[pd.DataFrame] = None
    momentum_score: Optional[pd.DataFrame] = None
    momentum_cluster: Optional[pd.DataFrame] = None
    momentum_cluster_score: Optional[pd.DataFrame] = None
    beta: Optional[pd.DataFrame] = None
    beta_score: Optional[pd.DataFrame] = None
    beta_cluster: Optional[pd.DataFrame] = None
    beta_cluster_score: Optional[pd.DataFrame] = None
    managers_alphas: Optional[pd.DataFrame] = None
    managers_scores: Optional[pd.DataFrame] = None
    residual_momentum: Optional[pd.DataFrame] = None
    residual_momentum_score: Optional[pd.DataFrame] = None
    residual_momentum_cluster: Optional[pd.DataFrame] = None
    residual_momentum_cluster_score: Optional[pd.DataFrame] = None
    clusters: Optional[pd.DataFrame] = None

    def get_alphas_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Summary of all alpha components at a single date.

        Args:
            date: Snapshot date. Must exist in alpha_scores.index.

        Returns:
            DataFrame with one row per asset and columns for each
            available component (scores, raw signals).
        """
        if date not in self.alpha_scores.index:
            raise KeyError(f"{date} is not in alpha_scores index")

        snapshot = self.alpha_scores.loc[date, :].to_frame('Alpha Scores')

        for attr, label in [
            ('momentum_score', 'Momentum Score'),
            ('momentum_cluster_score', 'Cluster Mom Total Score'),
            ('beta_score', 'Beta Score'),
            ('beta_cluster_score', 'Cluster Beta Score'),
            ('managers_scores', 'Managers Score'),
            ('residual_momentum_score', 'Residual Mom Score'),
            ('residual_momentum_cluster_score', 'Cluster Residual Mom Score'),
            ('momentum', 'Momentum'),
            ('momentum_cluster', 'Cluster Momentum Total'),
            ('beta', 'Beta'),
            ('beta_cluster', 'Cluster Beta'),
            ('managers_alphas', 'Managers Alpha'),
            ('residual_momentum', 'Residual Momentum'),
            ('residual_momentum_cluster', 'Cluster Residual Momentum'),
            ('clusters', 'Cluster ID'),
        ]:
            data = getattr(self, attr)
            if data is not None:
                if date in data.index:
                    snapshot = pd.concat([snapshot, data.loc[date, :].to_frame(label)], axis=1)
                else:
                    snapshot = pd.concat([snapshot, data.iloc[-1, :].to_frame(label)], axis=1)

        return snapshot

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Serialise to dictionary of DataFrames, excluding None fields.

        Only includes fields that have been populated (non-None),
        so the output is safe for qis.save_df_to_excel().
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }