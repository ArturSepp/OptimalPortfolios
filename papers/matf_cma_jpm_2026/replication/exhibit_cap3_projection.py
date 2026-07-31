"""
Exhibits for the Cap 3 governed-set projection (tab:cap3_grid, fig:cap3_implied_premia).

Produces, from the shared cma_data snapshot 2026q2 via governed_cma_projection
(the retired per-paper data/ workbook is gone — superseded 2026-07-30):

  figures/cap3_implied_premia.PNG   two-panel figure: (A) implied premium
      revision lambda_gls - lambda per factor with the identification scale
      sqrt(diag(beta_F^-1)) as a whisker, (B) the same deviation in
      identification units. The takeaway: deviations of hundreds of bp all
      stay inside one identification unit, so the universe cannot separate
      the admitted alpha from premium revisions in those directions.
  figures/exhibit_cap3_draft.tex    drop-in LaTeX for the figure and for the
      Cap 3 projection-grid table, captions and notes in paper style. Homes
      per owner decision O-J3: the table is main-text tab:cap3_grid, the
      figure is Appendix E fig:cap3_implied_premia.

Units: premia in bp in panel A, dimensionless in panel B, weights and shares
in the table as labelled. Colors follow the manuscript exhibits (blue primary,
orange secondary), one series per panel, direct value labels, no legend.

Does not belong here: the projection mathematics (imported from
governed_cma_projection) and manuscript integration (the tex is a draft
fragment, not yet called by the manuscript).
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
# project: paper reproduction package matf_cma_jpm_2026
import exhibit_style as es
from governed_cma_projection import (SNAPSHOT,
                                     KAPPA_GRID,
                                     load_paper_inputs,
                                     compute_sharpe_accounting,
                                     compute_factor_information_matrix,
                                     compute_gls_decomposition,
                                     project_onto_governed_set)

OUTPUT_DIR = es.FRAGMENTS_PATH       # tex fragments stay in replication/figures
FIGURE_DIR = es.FIGURES_PATH         # PNGs go to ../paper/figures under manuscript names (B4)
BLUE = es.BLUE      # manuscript primary series color
ORANGE = es.ORANGE  # manuscript secondary series color
GRAY = es.GRAY


def build_implied_premia_frame(snapshot: str = SNAPSHOT) -> pd.DataFrame:
    """implied premium revisions, identification scales, and standardized deviations per factor."""
    inputs = load_paper_inputs(snapshot=snapshot)
    assets = inputs.assets
    mu_excess = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    lam_gls, _, _ = compute_gls_decomposition(mu_excess=mu_excess, inputs=inputs)
    beta_f_inv = np.linalg.inv(compute_factor_information_matrix(inputs=inputs))
    ident_scale = pd.Series(np.sqrt(np.diag(beta_f_inv)), index=inputs.betas.columns)
    delta = lam_gls - inputs.factor_premia
    return pd.DataFrame({'delta_bp': 1e4 * delta,
                         'ident_scale_bp': 1e4 * ident_scale,
                         'standardized': delta / ident_scale})


def plot_implied_premia(df: pd.DataFrame,
                        file_path: Path = FIGURE_DIR / 'cap3_implied_premia.PNG',
                        ) -> None:
    """two-panel horizontal bar exhibit: raw bp deviations with identification whiskers, and standardized units."""
    factors = list(df.index)[::-1]   # top-to-bottom in canonical order
    d = df.loc[factors]
    factors = ['FX' if f == 'Fx' else f for f in factors]   # manuscript spelling
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)

    ax = axs[0]
    x_span = 1.15 * float(d['ident_scale_bp'].max())
    ax.set_xlim(-x_span, x_span)
    ax.barh(factors, d['delta_bp'], color=BLUE, height=0.62, zorder=3)
    ax.errorbar(x=np.zeros(len(factors)), y=np.arange(len(factors)),
                xerr=d['ident_scale_bp'], fmt='none',
                ecolor=GRAY, elinewidth=1.2, capsize=3, alpha=0.85, zorder=2)
    for i, v in enumerate(d['delta_bp']):
        pad = 0.02 * x_span * (1 if v >= 0 else -1)
        ax.annotate(f"{v:+,.0f}", xy=(v + pad, i + 0.34),
                    ha='left' if v >= 0 else 'right', va='bottom', fontsize=8, color='0.25')
    ax.set_title('(A) Implied premium revision (bp)', fontsize=9.5, loc='left')
    ax.axvline(0.0, color='0.3', lw=0.8)
    ax.grid(axis='x', color='0.9', lw=0.7, zorder=0)
    ax.tick_params(labelsize=9)

    ax = axs[1]
    ax.barh(factors, d['standardized'], color=ORANGE, height=0.62, zorder=3)
    for i, v in enumerate(d['standardized']):
        pad = 0.035 if v >= 0 else -0.035
        ax.annotate(f"{v:+.2f}", xy=(v + pad, i),
                    ha='left' if v >= 0 else 'right', va='center', fontsize=8, color='0.25')
    for ref in (-1.0, 1.0):
        ax.axvline(ref, color=GRAY, lw=1.0, ls='--')
    ax.set_xlim(-1.35, 1.35)
    ax.set_title('(B) In identification units (dashed lines at ±1)', fontsize=9.5, loc='left')
    ax.axvline(0.0, color='0.3', lw=0.8)
    ax.grid(axis='x', color='0.9', lw=0.7, zorder=0)
    ax.tick_params(labelsize=9)

    for ax in axs:
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"figure written: {file_path}")


def build_cap3_grid(snapshot: str = SNAPSHOT) -> pd.DataFrame:
    """the Cap 3 projection grid: production row plus one row per kappa."""
    inputs = load_paper_inputs(snapshot=snapshot)
    accounting = compute_sharpe_accounting(inputs=inputs)
    assets = inputs.assets
    admitted = assets['w_paper'] * assets['alpha']
    raw = float((admitted / assets['resid_vol']).pow(2).sum())
    rows = [{'policy': 'production', 'budget': np.nan, 'theta': 1.0,
             'skill_share': raw / (accounting['attainable'] + raw),
             'w_pe': 0.50, 'w_pc': 0.50, 'w_ils': 1.00, 'w_hf': 1.00, 'w_gold': 0.25,
             'max_cut_bp': 0.0}]
    for kappa in KAPPA_GRID:
        proj = project_onto_governed_set(inputs=inputs, kappa=kappa)
        w = proj['w_projected']
        rows.append({'policy': f"kappa = {kappa:.2f}",
                     'budget': proj.attrs['budget'],
                     'theta': proj.attrs['theta'],
                     'skill_share': proj.attrs['rho_after'],
                     'w_pe': w['MP503001 Index'], 'w_pc': w['MP503008 Index'],
                     'w_ils': w['EHFI804 Index'], 'w_hf': w['HFRIFWI Index'],
                     'w_gold': w['BCOMGCTR Index'],
                     'max_cut_bp': float(proj['cma_change_bp'].abs().max())})
    return pd.DataFrame(rows).set_index('policy')


def write_exhibit_tex(grid: pd.DataFrame,
                      premia: pd.DataFrame,
                      file_path: Path = OUTPUT_DIR / 'exhibit_cap3_draft.tex',
                      ) -> None:
    """drop-in LaTeX draft for the two Decision Two exhibit candidates."""
    g = grid.copy()
    lines = []
    lines.append("% ===== EXHIBITS: Cap 3 governed-set projection =====")
    lines.append("% Source: replication/exhibit_cap3_projection.py on cma_data snapshot 2026q2")
    lines.append("%   (manifest-verified; the retired data/ workbook is superseded 2026-07-30).")
    lines.append("% Homes per owner decision O-J3: tab:cap3_grid main text, fig:cap3_implied_premia Appendix E.")
    lines.append("")
    lines.append(r"\begin{figure}[H]")
    lines.append(r"	\captionof{figure}{Admitted Alpha Reads as Premium Revisions in the Directions the Universe Cannot Identify}\label{fig:cap3_implied_premia}\vspace*{-0.5\baselineskip}")
    lines.append(r"	\begin{center}")
    lines.append(r"		\includegraphics[width=0.95\linewidth]{figures/cap3_implied_premia.PNG}")
    lines.append(r"	\end{center}")
    lines.append(r"	\vspace*{-1.0\baselineskip}")
    lines.append(r"	{\footnotesize Notes: 2026-Q2 production cut, 18 assets, USD. Panel A: the generalized-least-squares implied factor premia of the published excess CMAs minus the calibrated premia, in basis points, with the identification scale $\sqrt{[(\hat{\boldsymbol\beta}^{\intercal}\boldsymbol D^{-1}\hat{\boldsymbol\beta})^{-1}]_{jj}}$ as the whisker. Panel B: the same deviations divided by the identification scale. Every deviation stays inside one identification unit, so the production admissions are statistically indistinguishable from premium revisions on the poorly spanned factors. Method in \citep{SeppKastenholzFAJ2026}.}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"	\captionof{table}{Cap 3 Prices the Diversification Preference: Admission Recuts per Skill-Share Budget}\label{tab:cap3_grid}\vspace*{-0.5\baselineskip}")
    lines.append(r"	\begin{center}")
    lines.append(r"		\footnotesize")
    lines.append(r"		\begin{tabular}{l rrr rrrrr r}")
    lines.append(r"			\toprule")
    lines.append(r"			\textbf{Policy} & \textbf{Budget} & $\theta$ & \textbf{Skill share} & \multicolumn{5}{c}{\textbf{Admission weights $w_i$}} & \textbf{Largest cut} \\")
    lines.append(r"			& & & & PE & PC & ILS & HF & Gold & (bp) \\")
    lines.append(r"			\midrule")
    for policy, r in g.iterrows():
        budget = "--" if np.isnan(r['budget']) else f"{r['budget']:.2f}"
        cut = "--" if r['max_cut_bp'] == 0.0 else f"$-{r['max_cut_bp']:.0f}$"
        name = policy.replace('kappa', r'$\kappa$')
        lines.append(f"			{name} & {budget} & {r['theta']:.2f} & {r['skill_share']:.0%} & "
                     f"{r['w_pe']:.2f} & {r['w_pc']:.2f} & {r['w_ils']:.2f} & {r['w_hf']:.2f} & "
                     f"{r['w_gold']:.2f} & {cut} \\\\".replace('%', r'\%'))
    lines.append(r"			\bottomrule")
    lines.append(r"		\end{tabular}")
    lines.append(r"	\end{center}")
    lines.append(r"	\vspace*{-0.5\baselineskip}")
    lines.append(r"	{\footnotesize Notes: 2026-Q2 production cut, 18 assets, USD. Cap 3 bounds the admitted idiosyncratic squared Sharpe ratio at $\kappa$ times the attainable systematic content $SR^2_{MATFCMA}$, enforced by scaling every admission weight by $\theta = \min(1, \sqrt{\kappa \cdot SR^2_{MATFCMA} / \boldsymbol\alpha_{adm}^{\intercal}\boldsymbol D^{-1}\boldsymbol\alpha_{adm}})$. The skill share is the fraction of the claimed ex-ante squared Sharpe ratio carried by admitted alpha. The largest cut reports the biggest single-sleeve reduction of the excess CMA under the recut, on Insurance-Linked in every row. At every budget shown, the recut admissions also pass Cap~1, so the portfolio-level cap subsumes the sleeve-level failures of Exhibit~\ref{tab:admission_audit}.}")
    lines.append(r"\end{table}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"tex draft written: {file_path}")


class LocalTests(Enum):
    BUILD_DRAFT_EXHIBITS = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 1000)

    if local_test == LocalTests.BUILD_DRAFT_EXHIBITS:
        premia = build_implied_premia_frame()
        print(premia.round(2).to_string())
        plot_implied_premia(df=premia)
        grid = build_cap3_grid()
        print(grid.round(3).to_string())
        write_exhibit_tex(grid=grid, premia=premia)


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.BUILD_DRAFT_EXHIBITS)
