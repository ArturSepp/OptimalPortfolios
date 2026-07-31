"""
Shared exhibit style for the MATF-CMA (JPM) figure build.

One place for the manuscript's visual grammar so the regenerated exhibits sit
beside the frozen R2 panels without a palette seam (roadmap B5): blue
#1f77b4 primary, orange #ff7f0e secondary, gray #7f7f7f auxiliary, direct
value labels in preference to legends, dpi 220, bbox_inches tight, no
seaborn. Captions and notes live in the manuscript tex, never in the PNG, so
figures carry a takeaway title and axis labels only.

Factor display order is the canonical cma_data.FACTORS list, with 'Fx'
rendered as 'FX' (factor_label). Asset display order is the snapshot order,
which is Bonds then Equities then Alternatives.

Units: helpers take decimals per annum and label percent or bp explicitly;
they never convert silently.

Main entry points: save_figure(), factor_label(), sleeve_labels(),
table_figure(), class_boundaries().

Does not belong here: any computation on the snapshot (the run_* scripts own
their mathematics) and the FAJ package's palette, which is different by
design and must not be imported here.
"""
# packages
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence
# qis / project
import qis as qis

matplotlib.use('Agg')

BLUE = '#1f77b4'        # primary series
LIGHT_BLUE = '#7cb5d8'  # secondary shade of the primary channel (blend segments)
ORANGE = '#ff7f0e'      # secondary series (admitted alpha, claimed quantities)
GRAY = '#7f7f7f'        # auxiliary: whiskers, reference lines, annotations
DARK_RED = '#a63603'    # stress series
GREEN = '#2c7a5a'       # upside series

DPI = 220
HEADER_COLOR = '#40466e'        # qis table header, matches the frozen table exhibits
ROW_COLORS = ('#f1f1f2', 'w')

# figures land under the EXACT filenames the manuscript calls, case included
FIGURES_PATH = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FRAGMENTS_PATH = Path(__file__).resolve().parent / 'figures'


def factor_label(factor: str) -> str:
    """manuscript spelling of a factor name: 'Fx' prints as 'FX'."""
    return 'FX' if factor == 'Fx' else factor


def factor_labels(factors: Sequence[str]) -> List[str]:
    """manuscript spellings for a factor sequence, order preserved."""
    return [factor_label(f) for f in factors]


def sleeve_labels(assets: pd.DataFrame) -> pd.Series:
    """per-ticker sleeve names in snapshot order, for axis ticks."""
    if 'sleeve' not in assets.columns:
        raise ValueError(f"assets frame carries no 'sleeve' column, got {list(assets.columns)!r}")
    return assets['sleeve']


def class_boundaries(assets: pd.DataFrame) -> List[int]:
    """positional indices where the asset class changes, for separator lines."""
    classes = list(assets['asset_class'])
    return [i for i in range(1, len(classes)) if classes[i] != classes[i - 1]]


def style_axis(ax: plt.Axes,
               grid_axis: str = 'x',           # 'x', 'y', or 'both'
               fontsize: int = 9,
               ) -> None:
    """manuscript axis furniture: no top/right spine, light grid behind the data."""
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis=grid_axis, color='0.9', lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fontsize)


def save_figure(fig: plt.Figure,
                file_name: str,                            # exact manuscript filename, case included
                local_path: Optional[Path] = None,         # default: ../paper/figures
                ) -> Path:
    """write one figure at manuscript resolution and return its path."""
    folder = FIGURES_PATH if local_path is None else Path(local_path)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / file_name
    fig.savefig(file_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"figure written: {file_path}")
    return file_path


def write_fragment(lines: List[str],
                   file_name: str,
                   local_path: Optional[Path] = None,      # default: replication/figures
                   ) -> Path:
    """write one drop-in tex fragment and return its path."""
    folder = FRAGMENTS_PATH if local_path is None else Path(local_path)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / file_name
    file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"tex fragment written: {file_path}")
    return file_path


def table_figure(df: pd.DataFrame,
                 title: Optional[str] = None,
                 column_width: float = 1.30,
                 first_column_width: float = 3.10,
                 row_height: float = 0.46,
                 fontsize: int = 9,
                 heatmap_columns: Optional[List[int]] = None,
                 special_columns_colors: Optional[List[tuple]] = None,
                 cmap: str = 'RdYlGn',
                 col_widths: Optional[List[float]] = None,
                 first_row_height: Optional[float] = None,   # extra header rows for two-line labels
                 **kwargs,
                 ) -> plt.Figure:
    """render a frame as a table image in the frozen exhibits' grammar (qis.plot_df_table)."""
    if df.empty:
        raise ValueError(f"table frame is empty, got shape {df.shape!r}")
    n_rows, n_cols = df.shape
    widths = col_widths if col_widths is not None \
        else [first_column_width] + [column_width] * n_cols
    # a two-line header needs vertical room; qis sizes rows uniformly, so buy the room
    # in the figure height rather than through first_row_height (which rescales the axes)
    header_rows = 1.9 if first_row_height is None else 1.9 + first_row_height
    fig, ax = plt.subplots(figsize=(sum(widths), row_height * (n_rows + header_rows)))
    qis.plot_df_table(df=df,
                      col_widths=widths,
                      row_height=row_height,
                      fontsize=fontsize,
                      header_color=HEADER_COLOR,
                      row_colors=ROW_COLORS,
                      heatmap_columns=heatmap_columns,
                      special_columns_colors=special_columns_colors,
                      cmap=cmap,
                      title=title,
                      ax=ax,
                      **kwargs)
    return fig


def percent(value: float, decimals: int = 1) -> str:
    """decimal p.a. as a percent string with an explicit sign for deltas."""
    return f"{1e2 * value:.{decimals}f}"


def basis_points(value: float) -> str:
    """decimal p.a. as a signed basis-point string."""
    return f"{1e4 * value:+,.0f}"


LATEX_ESCAPES: Dict[str, str] = {'%': r'\%', '&': r'\&', '_': r'\_'}


def tex_escape(text: str) -> str:
    """escape the three LaTeX-special characters that occur in sleeve and factor names."""
    for raw, escaped in LATEX_ESCAPES.items():
        text = text.replace(raw, escaped)
    return text
