"""build_caches.py — rebuild the three exhibit caches from a production xlsx.

Reconstruction conventions verified against the v0-vintage caches
(this script, pointed at the v0 workbook, reproduces every key to 1e-6):
  alpha (exhibits)   : GLS-orthogonal admitted alpha, ag = a - beta bF^-1 beta' D^-1 a
  mu                 : beta lam + ag
  w_sys/w_idio/w_star: Eq7 blocks at ag, normalized by g = sum(w_sys+w_idio)
  wls                : Sigma^-1 mu, gross-normalized
  tre                : {te: w_b + (te/sr_tan) Sigma^-1 mu}
  sat/sat_order      : identity along benchmark-weight ordering argsort(-w_b)
  mu_fac             : metadata base_excess_factor_cma (includes regional addons)
  cascade srs        : TE-constrained mandates, te=1.5%, band=50%, on (mu, Sigma)
"""
import sys, numpy as np, pandas as pd, cvxpy as cp, openpyxl
import paper_inputs as pi, identity_analytics as ia

def build(xlsx_path, w_b, out_prefix='data/'):
    inp = pi.load_production_inputs(xlsx_path)
    lam, Sig_F, beta, D, a_raw = inp['lam'], inp['Sig_F'], inp['beta'], inp['D'], inp['alpha']
    tickers, FACT = inp['tickers'], inp['factors']
    labels = list(np.load('data/inputs17_v5.npz', allow_pickle=True)['labels'])
    N, M = beta.shape
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    m = wb['cma_metadata']; hdr = {m.cell(1,c).value: c for c in range(1, m.max_column+1)}
    mrow = {m.cell(r,1).value: r for r in range(2, m.max_row+1)}
    mu_fac = np.array([float(m.cell(mrow[f"{t} Index"], hdr['base_excess_factor_cma']).value or 0) for t in tickers])

    ag = ia.gls_orthogonal_alpha(a_raw, beta, D)
    mu = beta@lam + ag
    Sigma = beta@Sig_F@beta.T + np.diag(D)
    bF = ia.factor_information_matrix(beta, D)
    P_F = ia.factor_mimicking_portfolios(beta, D)
    ceiling = ia.frictionless_ceiling(lam, Sig_F)
    identity = ia.matf_identity(lam, Sig_F, beta, D)
    pf_id, pf_ce = ia.per_factor_contributions(lam, Sig_F, beta, D)
    w_sys, w_idio = ia.eq7_weight_blocks(lam, Sig_F, beta, D, ag)
    w_star = w_sys + w_idio; g = w_star.sum()
    assert np.max(np.abs(w_star - np.linalg.solve(Sigma, mu))) < 1e-9
    raw = np.linalg.solve(Sigma, mu); wls = raw/np.abs(raw).sum()
    sr_tan = float(np.sqrt(mu @ raw))
    tre = {te: w_b + (te/sr_tan)*raw for te in (0.01, 0.02, 0.03)}
    order = np.argsort(-w_b); sat = []
    for n in range(1, N+1):
        idx = order[:n]; b = beta[idx]; Di = 1/D[idx]; bf = b.T@(Di[:,None]*b)
        sat.append(float(lam@(bf@np.linalg.solve(np.eye(M)+Sig_F@bf, lam))))
    _, sr_lo = ia.longonly_max_sharpe(mu, Sigma)

    uw = pd.read_excel('data/universe_snapshot.xlsx', 'universe weight', index_col=0)
    mand_cols = [x for x in uw.columns if 'Alts' in x]
    W = uw.loc[[f"{t} Index" for t in tickers], mand_cols].fillna(0).values
    def te_mandate(w_bm, te=0.015, b=0.5):
        w = cp.Variable(N)
        cons = [cp.quad_form(w - w_bm, cp.psd_wrap(Sigma)) <= te**2, cp.sum(w) == 1,
                w >= (1-b)*w_bm, w <= (1+b)*w_bm]
        cp.Problem(cp.Maximize(mu @ w), cons).solve(solver=cp.CLARABEL)
        wv = w.value; return float(mu@wv/np.sqrt(wv@Sigma@wv))
    srs = np.array([te_mandate(W[:,j]) for j in range(8)])
    mandates = [c.replace('\n',' ') for c in mand_cols]

    np.savez(out_prefix+'inputs17_v5.npz', labels=labels, tickers=tickers, w_b=w_b,
             lam=lam, Sig_F=Sig_F, beta=beta, D=D, mu_fac=mu_fac, mu_alpha=a_raw,
             FACT=FACT, ceiling=ceiling, identity=identity, sr_lo=sr_lo,
             pf_id=pf_id, pf_ce=pf_ce, P_F=P_F)
    np.savez(out_prefix+'allfig17.npz', labels=labels, w_sys=w_sys/g, w_idio=w_idio/g,
             w_star=w_star/g, wls=wls, w_b=w_b, tre=tre, sr_unc=sr_tan, sr_tan=sr_tan,
             pf_id=pf_id, pf_ce=pf_ce, FACT=FACT, ceiling=ceiling, identity=identity,
             sat=np.array(sat), sat_order=order, P_F=P_F, alpha=ag, mu=mu, Sigma=Sigma,
             lam=lam, beta=beta, D=D, Sig_F=Sig_F)
    np.savez(out_prefix+'cascade_data.npz', sr_ceiling=np.sqrt(ceiling),
             sr_identity=np.sqrt(identity), sr_lo=sr_lo, mandates=mandates, srs=srs)
    return dict(ceiling=ceiling, identity=identity, sr_tan=sr_tan, sr_lo=sr_lo,
                srs=srs, ag=ag, mu=mu, pf_id=pf_id, a_raw=a_raw, D=D, beta=beta, lam=lam, Sig_F=Sig_F)

if __name__ == '__main__':
    w_b = np.load('data/inputs17_v5.npz', allow_pickle=True)['w_b']
    build(sys.argv[1] if len(sys.argv) > 1 else pi.XLSX, w_b)
