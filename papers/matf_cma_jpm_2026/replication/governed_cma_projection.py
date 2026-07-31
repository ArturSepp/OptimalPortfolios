"""
Governed-set projection of the MATF-CMA vector (the Cap 3 analysis).

Implements, on the frozen 2026-Q2 paper inputs, the consistency decomposition
and the projection of the CMA vector onto the governed set:

    beta_F      = B' D^-1 B                         factor information matrix
    lambda_gls  = beta_F^-1 B' D^-1 mu_e            implied factor premia (GLS cross-section)
    a           = mu_e - B lambda_gls               orthogonal deviation
    SR2_alpha   = a' D^-1 a                         claimed idiosyncratic squared Sharpe
    ceiling     = lambda' SigmaF^-1 lambda          frictionless factor ceiling
    SR2_MATFCMA = lambda' (SigmaF + beta_F^-1)^-1 lambda    attainable systematic content

    Cap 1: w_i alpha_i / sigma_eps_i <= 0.50 per sleeve
    Cap 2: above-factor share of the excess CMA <= 50% per sleeve
    Cap 3: SR2_alpha(admissions) <= kappa * SR2_MATFCMA, portfolio level

The governed-set projection enforces Cap 3 by uniform admission scaling
theta = sqrt(budget / raw claim) applied to every admission weight, which is
the exact projection of the admitted-alpha vector in the D^-1 metric under
channel-proportional scaling. The skill share rho = SR2_alpha / (SR2_MATFCMA +
SR2_alpha) is reported before and after, so the committee reads Cap 3 as a
bound on the share of ex-ante performance resting on claimed skill.

Units: decimal per annum internally; printed tables in percent or bp as
labelled. Input: the shared papers/cma_data layer, snapshot pinned by
SNAPSHOT below and resolved through local_path.py (settings.yaml optional,
defaults work from a fresh clone). Admission weights default to the paper
PRODUCTION policy (PE recut to w = 0.5), carried in the snapshot's w_paper
column. The pre-recut workbook policy is w_workbook (O-J11 / B6).

Does not belong here: mandate-level optimization (the admission dial re-solve
needs the covariance and the optimizer; it joins the exhibit build), figure
styling, and any dependence on the rosaa package.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional
# shared paper-data layer, imported by file location (no sys.path mutation)
from local_path import load_cma_data

_cma_data = load_cma_data()
PaperInputs = _cma_data.PaperInputs

SNAPSHOT = '2026q2'                              # the pinned frozen cut
OUTPUT_DIR = Path(__file__).parent / 'figures'

# frozen paper reference values (MATF-CMA R2 draft, Section 4 / knowledge file);
# printed next to the recomputed values for cross-checking, never asserted:
# the lambda vector of this input file is the post-July production config.
PAPER_REFERENCE = {'ceiling': 0.57, 'attainable': 0.24, 'raw_claim': 1.40, 'gls_claim': 0.63}

KAPPA_GRID = (1.00, 0.50, 0.25)    # Cap 3 budgets as multiples of SR2_MATFCMA
CAP1_IR_LIMIT = 0.50               # max admitted information ratio per sleeve
CAP2_SHARE_LIMIT = 0.50            # max above-factor share of the excess CMA per sleeve


def load_paper_inputs(snapshot: str = SNAPSHOT) -> PaperInputs:
    """load the pinned frozen paper cut from the shared cma_data snapshots (manifest-verified)."""
    return _cma_data.load_snapshot(tag=snapshot)


def compute_sharpe_accounting(inputs: PaperInputs) -> pd.Series:
    """frictionless ceiling, attainable systematic content, and FPIR of the universe."""
    lam = inputs.factor_premia.values
    sigma_f = inputs.factor_covar.values
    beta_f = compute_factor_information_matrix(inputs=inputs)
    ceiling = float(lam @ np.linalg.solve(sigma_f, lam))
    attainable = float(lam @ np.linalg.solve(sigma_f + np.linalg.inv(beta_f), lam))
    return pd.Series({'ceiling': ceiling,
                      'attainable': attainable,
                      'fpir': attainable / ceiling})


def compute_factor_information_matrix(inputs: PaperInputs) -> np.ndarray:
    """beta_F = B' D^-1 B on the paper universe."""
    b = inputs.betas.values
    d_inv = 1.0 / inputs.assets['resid_vol'].values ** 2
    return (b * d_inv[:, None]).T @ b


def compute_gls_decomposition(mu_excess: pd.Series,
                              inputs: PaperInputs,
                              ) -> Tuple[pd.Series, pd.Series, float]:
    """decompose any excess-CMA vector into (lambda_gls, orthogonal deviation a, SR2_alpha).

    lambda_gls is the D^-1-weighted cross-sectional regression of mu_excess on
    the loadings (Shanken second pass). a is the deviation the factor span
    cannot explain, and a' D^-1 a is its squared Sharpe ratio: the claimed
    idiosyncratic content of the vector, the consistency measure.
    """
    if not mu_excess.index.equals(inputs.betas.index):
        raise ValueError(f"mu_excess index misaligned, got {list(mu_excess.index)!r}")
    b = inputs.betas.values
    d_inv = 1.0 / inputs.assets['resid_vol'].values ** 2
    beta_f = compute_factor_information_matrix(inputs=inputs)
    lam_gls = np.linalg.solve(beta_f, (b * d_inv[:, None]).T @ mu_excess.values)
    a = mu_excess.values - b @ lam_gls
    sr2 = float(a @ (d_inv * a))
    return (pd.Series(lam_gls, index=inputs.betas.columns),
            pd.Series(a, index=mu_excess.index),
            sr2)


def compute_solo_premium_like_shares(inputs: PaperInputs) -> pd.Series:
    """per-asset GLS leverage h_i: the share of a solo admitted alpha that is factor premium in disguise."""
    b = inputs.betas.values
    sig_e = inputs.assets['resid_vol'].values
    beta_f_inv = np.linalg.inv(compute_factor_information_matrix(inputs=inputs))
    h = np.array([b[i] @ beta_f_inv @ b[i] / sig_e[i] ** 2 for i in range(len(sig_e))])
    return pd.Series(h, index=inputs.assets.index, name='premium_like_share')


def build_admission_audit(inputs: PaperInputs,
                          admission_weights: Optional[pd.Series] = None,  # default: w_paper column
                          ) -> pd.DataFrame:
    """per-sleeve caps audit: admitted alpha, IR (Cap 1), above-factor share (Cap 2)."""
    assets = inputs.assets
    w = assets['w_paper'] if admission_weights is None else admission_weights
    admitted = w * assets['alpha']
    excess = assets['factor_excess_cma'] + admitted
    audit = pd.DataFrame({'sleeve': assets['sleeve'],
                          'w': w,
                          'alpha': assets['alpha'],
                          'admitted': admitted,
                          'ir': admitted / assets['resid_vol'],
                          'share': admitted / excess.where(excess.abs() > 1e-12)})
    audit['cap1'] = np.where(audit['ir'] <= CAP1_IR_LIMIT + 1e-12, 'pass', 'FAIL')
    audit['cap2'] = np.where(audit['share'].fillna(0.0) <= CAP2_SHARE_LIMIT + 1e-12, 'pass', 'FAIL')
    return audit.loc[w > 0.0]


def project_onto_governed_set(inputs: PaperInputs,
                              kappa: float,
                              admission_weights: Optional[pd.Series] = None,  # default: w_paper column
                              ) -> pd.DataFrame:
    """enforce Cap 3 by uniform admission scaling; return the per-sleeve recut and CMA changes.

    theta = min(1, sqrt(kappa * SR2_MATFCMA / raw claim)) applied to every
    admission weight. Uniform scaling is the exact D^-1-metric projection under
    the constraint that only the admission weights move and move proportionally.
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa!r}")
    assets = inputs.assets
    w = (assets['w_paper'] if admission_weights is None else admission_weights).astype(float)
    admitted = w * assets['alpha']
    raw = float((admitted / assets['resid_vol']).pow(2).sum())
    attainable = compute_sharpe_accounting(inputs=inputs)['attainable']
    budget = kappa * attainable
    theta = min(1.0, float(np.sqrt(budget / raw))) if raw > 0.0 else 1.0
    out = pd.DataFrame({'sleeve': assets['sleeve'],
                        'w': w,
                        'w_projected': theta * w,
                        'ir': admitted / assets['resid_vol'],
                        'ir_projected': theta * admitted / assets['resid_vol'],
                        'cma_change_bp': 1e4 * (theta - 1.0) * admitted})
    out = out.loc[w > 0.0]
    out.attrs['theta'] = theta
    out.attrs['budget'] = budget
    out.attrs['raw'] = raw
    out.attrs['rho_before'] = raw / (attainable + raw)
    out.attrs['rho_after'] = (theta ** 2 * raw) / (attainable + theta ** 2 * raw)
    return out


def run_governed_projection_report(snapshot: str = SNAPSHOT,
                                   save_outputs: bool = True,
                                   ) -> None:
    """full report: Sharpe accounting, GLS decomposition, caps audit, Cap 3 grid."""
    inputs = load_paper_inputs(snapshot=snapshot)
    assets = inputs.assets
    factors = inputs.betas.columns

    print("=" * 78)
    print("Governed-set projection — MATF-CMA paper universe, frozen 2026-Q2 inputs")
    print("=" * 78)

    accounting = compute_sharpe_accounting(inputs=inputs)
    print("\n--- Sharpe accounting (current production lambda) ---")
    print(f"frictionless ceiling   {accounting['ceiling']:.3f}   (paper frozen cut: {PAPER_REFERENCE['ceiling']:.2f})")
    print(f"attainable SR2_MATFCMA {accounting['attainable']:.3f}   (paper frozen cut: {PAPER_REFERENCE['attainable']:.2f})")
    print(f"FPIR                   {accounting['fpir']:.1%}")

    # admission channel under the production policy (w_paper, PE recut to 0.5)
    admitted = assets['w_paper'] * assets['alpha']
    mu_excess = assets['factor_excess_cma'] + admitted
    lam_gls_adm, a_adm, raw_check = compute_gls_decomposition(
        mu_excess=admitted, inputs=inputs)
    honest = float(a_adm @ (a_adm / assets['resid_vol'] ** 2))
    raw = float((admitted / assets['resid_vol']).pow(2).sum())
    print("\n--- Admission channel (production policy w_paper, PE recut to 0.5) ---")
    print(f"raw claimed SR2_alpha  {raw:.3f}   (paper: {PAPER_REFERENCE['raw_claim']:.2f})")
    print(f"GLS-projected claim    {honest:.3f}   (paper: {PAPER_REFERENCE['gls_claim']:.2f})")

    shares = compute_solo_premium_like_shares(inputs=inputs)
    admitted_sleeves = assets.index[assets['w_paper'] > 0.0]
    print("\nsolo premium-like share of each admitted sleeve:")
    for t in admitted_sleeves:
        print(f"  {assets.loc[t, 'sleeve']:<18s} {shares[t]:.0%}")

    # full-vector consistency measure with identification scales
    lam_gls, a_full, sr2_full = compute_gls_decomposition(mu_excess=mu_excess, inputs=inputs)
    beta_f_inv = np.linalg.inv(compute_factor_information_matrix(inputs=inputs))
    ident_scale = pd.Series(np.sqrt(np.diag(beta_f_inv)), index=factors)
    cmp = pd.DataFrame({'lambda_prod_bp': 1e4 * inputs.factor_premia,
                        'lambda_gls_bp': 1e4 * lam_gls,
                        'delta_bp': 1e4 * (lam_gls - inputs.factor_premia),
                        'ident_scale_bp': 1e4 * ident_scale,
                        'delta_standardized': (lam_gls - inputs.factor_premia) / ident_scale})
    print(f"\n--- Full-vector consistency: SR2_alpha(mu) = {sr2_full:.3f} ---")
    print("implied premia vs production (identification scale = sqrt diag beta_F^-1):")
    print(cmp.round({'lambda_prod_bp': 0, 'lambda_gls_bp': 0, 'delta_bp': 0,
                     'ident_scale_bp': 0, 'delta_standardized': 2}).to_string())

    audit = build_admission_audit(inputs=inputs)
    print("\n--- Caps audit (production policy) ---")
    print(audit.round({'w': 2, 'alpha': 4, 'admitted': 4, 'ir': 2, 'share': 2}).to_string())

    print("\n--- Cap 3: governed-set projection grid ---")
    projections = {}
    for kappa in KAPPA_GRID:
        proj = project_onto_governed_set(inputs=inputs, kappa=kappa)
        projections[kappa] = proj
        print(f"\nkappa = {kappa:.2f}: budget = {proj.attrs['budget']:.3f}, "
              f"theta = {proj.attrs['theta']:.3f}, "
              f"skill share {proj.attrs['rho_before']:.0%} -> {proj.attrs['rho_after']:.0%}")
        print(proj.round({'w': 2, 'w_projected': 2, 'ir': 2, 'ir_projected': 2,
                          'cma_change_bp': 0}).to_string())

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / 'governed_projection_2026q2.xlsx'
        with pd.ExcelWriter(out_file) as writer:
            accounting.to_frame('value').to_excel(writer, sheet_name='sharpe_accounting')
            cmp.to_excel(writer, sheet_name='implied_premia')
            audit.to_excel(writer, sheet_name='caps_audit')
            for kappa, proj in projections.items():
                sheet = f"cap3_kappa_{str(kappa).replace('.', '')}"
                proj.to_excel(writer, sheet_name=sheet)
        print(f"\noutputs saved: {out_file}")


class LocalTests(Enum):
    RUN_GOVERNED_PROJECTION_REPORT = 1
    SHARPE_ACCOUNTING_ONLY = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if local_test == LocalTests.RUN_GOVERNED_PROJECTION_REPORT:
        run_governed_projection_report()

    elif local_test == LocalTests.SHARPE_ACCOUNTING_ONLY:
        inputs = load_paper_inputs()
        print(compute_sharpe_accounting(inputs=inputs))


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.RUN_GOVERNED_PROJECTION_REPORT)
