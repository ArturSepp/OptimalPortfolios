"""
Portfolio optimisation result plotting functions.

optimalportfolios/optimization/portfolio_result_plots.py

Plotting utilities for PortfolioOptimisationResult.
Requires: matplotlib, seaborn, qis.
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis

from optimalportfolios.optimization.portfolio_result import PortfolioOptimisationResult


def plot_efficient_frontier(
        result: PortfolioOptimisationResult,
        profiles: Dict[str, List[str]],
        order: int = 3,
        markersize: int = 40,
        xvar_format: str = '{:.1%}',
        yvar_format: str = '{:.1%}',
        title: Optional[str] = None,
        drop_duplicated_annotations: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
) -> plt.Figure:
    """
    Plot expected return vs volatility frontier for multiple profiles
    (e.g. 'with Alts' vs 'without Alts'), showing both portfolio and benchmark curves.

    Portfolio curves are solid lines, benchmark curves are dotted.
    Each profile gets its own color. Mandate labels are annotated once per point.

    Args:
        result: PortfolioOptimisationResult instance with N portfolios.
        profiles: Dict mapping profile name to list of portfolio names.
                  Example: {'w/o Alts': ['Income w/o Alts', 'Low w/o Alts', ...],
                            'with Alts': ['Income with Alts', 'Low with Alts', ...]}
        order: Polynomial order for scatter fit curve.
        markersize: Marker size for scatter points.
        xvar_format: Format string for x-axis (volatility).
        yvar_format: Format string for y-axis (expected return).
        title: Plot title. None to suppress.
        ax: Optional matplotlib axes. If None, creates new figure.
        **kwargs: Additional kwargs passed to qis.plot_scatter.

    Returns:
        matplotlib Figure.
    """
    dfs, _ = result.compute_efficient_frontier_data(profiles=profiles)

    # Annotation labels: show mandate name only once (first occurrence)
    if drop_duplicated_annotations:
        annotation_labels = (
            dfs['mandate']
            .where(~dfs['mandate'].duplicated(keep='first'), other='')
            .to_list()
        )
    else:
        annotation_labels = dfs['mandate'].tolist()

    # Colors: one per unique curve (profile × portfolio/benchmark)
    unique_curves = dfs['hue'].unique()
    colors = qis.get_n_sns_colors(n=len(unique_curves))
    mandate_color_map = dict(zip(unique_curves, colors))
    annotation_colors = dfs['hue'].map(mandate_color_map).to_list()

    # Linestyles: solid for portfolio, dotted for benchmark
    hue_linestyles = {
        x: '-' if x.split()[-1] == 'portfolio' else ':'
        for x in unique_curves
    }

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        fig = ax.figure

    with sns.axes_style('darkgrid'):
        qis.plot_scatter(
            df=dfs,
            x='total_vol',
            y='exp_return',
            xlabel='Volatility',
            ylabel='Expected Return',
            hue='hue',
            add_hue_model_label=False,
            add_universe_model_label=False,
            order=order,
            markersize=markersize,
            annotation_labels=annotation_labels,
            annotation_colors=annotation_colors,
            xvar_format=xvar_format,
            yvar_format=yvar_format,
            colors=colors,
            hue_linestyles=hue_linestyles,
            title=title,
            ax=ax,
            **kwargs
        )

    return fig