"""
Risk labelling for factor-model covariance estimates.

optimalportfolios.covar_estimation.risk_labelling

Consumes a fully estimated ``RollingFactorCovarData`` and produces persistent,
economically meaningful *risk labels*: it tracks the raw per-date statistical
clusters through time into stable derived clusters, classifies each by its MATF
factor profile and risk evolution, and assigns factor/vol-based labels. Pure
post-processing of the estimated object — no covariance is fitted here.

Reads only ``x_covar`` (Sigma_F), ``y_betas``, ``y_variances`` (``residual_var``),
and ``clusters`` off each snapshot. Offline / full-panel: look-ahead is acceptable
because the output is a label diagnostic, not a tradeable signal.

Matchers
--------
* ``method='mcf'`` (default): global min-cost-flow max-weight vertex-disjoint path
  cover over consecutive + bridge edges (needs ``networkx``). Recommended for
  reporting/labelling — drift-free and consolidates persistent clusters.
* ``method='hungarian'``: per-transition assignment; fast, zero extra dependency,
  but fragments persistent clusters under realistic churn (greedy drift).

Public API: ``analyze_risk_clusters(covar_data, ...) -> RiskClusterReport`` and the
convenience wrapper ``run_risk_label_report(...) -> (figures, tables)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from factorlasso import RollingFactorCovarData, CurrentFactorCovarData, VarianceColumns


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaxonomyConfig:
    """Classification cutoffs (defaults grounded in the MAC 0.40 snapshot)."""
    equity_factor: str = 'Equity'
    high_beta: float = 0.70
    defensive_beta: float = 0.30
    mixed_dominant_frac: float = 0.50      # min modal-dominant share to avoid 'Mixed'
    stable_spread_vol: float = 0.015       # beta-stability: stable if <=
    drifting_spread_vol: float = 0.030     # drifting if >=
    core_coverage: float = 0.70            # persistence: Core if coverage >=
    transient_coverage: float = 0.30       # Transient if coverage <
    vol_low: float = 0.05                  # risk regime (annualized factor vol)
    vol_high: float = 0.12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _psd_clip(m: np.ndarray) -> np.ndarray:
    """Clip eigenvalues at 0 (guards the small negative eigenvalue seen in Sigma_F)."""
    w, v = np.linalg.eigh((m + m.T) / 2.0)
    return (v * np.clip(w, 0.0, None)) @ v.T


def _cluster_series(cd: CurrentFactorCovarData) -> pd.Series:
    """Asset -> cluster label, preferring ``clusters`` and falling back to y_variances."""
    if cd.clusters is not None:
        s = cd.clusters
    else:
        s = cd.y_variances[VarianceColumns.CLUSTER.value]
    return s.dropna()


# --------------------------------------------------------------------------- #
# fingerprint (the node)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Fingerprint:
    members: Tuple[str, ...]
    beta: np.ndarray          # (M,)
    factor_var: float
    idio_var: float
    total_var: float
    r2: float
    dominant: str


def _snapshot_fingerprints(cd: CurrentFactorCovarData,
                           weighting: str = 'equal') -> Tuple[Dict[Any, _Fingerprint], List[str]]:
    """Per-raw-cluster fingerprints at one date, read from the snapshot only."""
    factors = list(cd.x_covar.index)
    sigma = _psd_clip(cd.x_covar.to_numpy())
    betas = cd.y_betas[factors]
    resid_var = cd.y_variances[VarianceColumns.RESIDUAL_VARS.value]
    clusters = _cluster_series(cd)

    out: Dict[Any, _Fingerprint] = {}
    for label, grp in clusters.groupby(clusters):
        members = [a for a in grp.index if a in betas.index]
        if not members:
            continue
        n = len(members)
        if weighting == 'inv_vol':
            tv = cd.get_model_vols(members)[VarianceColumns.TOTAL_VOL.value].to_numpy()
            w = (1.0 / np.where(tv > 0, tv, np.nan))
            w = np.nan_to_num(w, nan=0.0)
            w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        else:  # 'equal'
            w = np.full(n, 1.0 / n)
        b = betas.loc[members].to_numpy()
        beta_c = w @ b
        factor_var = float(beta_c @ sigma @ beta_c)
        idio_var = float(np.sum((w ** 2) * resid_var.loc[members].to_numpy()))
        total_var = factor_var + idio_var
        contrib = beta_c * (sigma @ beta_c)
        dominant = factors[int(np.argmax(contrib))] if np.any(contrib > 0) else factors[0]
        out[label] = _Fingerprint(
            members=tuple(members), beta=beta_c, factor_var=factor_var,
            idio_var=idio_var, total_var=total_var,
            r2=float(factor_var / total_var) if total_var > 0 else 0.0,
            dominant=dominant)
    return out, factors


# --------------------------------------------------------------------------- #
# affinity
# --------------------------------------------------------------------------- #
def _overlap(a: Tuple[str, ...], b: Tuple[str, ...], common: set, metric: str) -> float:
    aset, bset = set(a) & common, set(b) & common
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    if metric == 'jaccard':
        return inter / len(aset | bset)
    return inter / min(len(aset), len(bset))         # overlap coefficient (default)


def _qualifies(fa: _Fingerprint, fb: _Fingerprint, common: set, sigma_bar: np.ndarray,
               *, overlap_metric: str, combine: str,
               overlap_band: Tuple[float, float], spread_vol_cut: float,
               w_overlap: float) -> Tuple[bool, float]:
    """Gated (default) or blended link test; returns (qualifies, weight)."""
    ov = _overlap(fa.members, fb.members, common, overlap_metric)
    d = fb.beta - fa.beta
    spread_vol = float(np.sqrt(max(d @ sigma_bar @ d, 0.0)))
    s_beta = float(np.exp(-(spread_vol ** 2) / (2.0 * spread_vol_cut ** 2)))
    lo, hi = overlap_band
    if combine == 'blend':
        a = w_overlap * ov + (1.0 - w_overlap) * s_beta
        return (a >= 0.5, a)
    # gated
    if ov >= hi:
        return (True, ov + s_beta)                   # continue / heir
    if spread_vol <= spread_vol_cut:                 # mid or low overlap -> beta arbitrates
        return (True, w_overlap * ov + (1.0 - w_overlap) * s_beta)
    return (False, 0.0)


# --------------------------------------------------------------------------- #
# per-transition Hungarian matcher
# --------------------------------------------------------------------------- #
def _match_panel(snapshots: Dict[pd.Timestamp, Dict[Any, _Fingerprint]],
                 x_covars: Dict[pd.Timestamp, np.ndarray],
                 *, overlap_metric: str, combine: str,
                 overlap_band: Tuple[float, float], spread_vol_cut: float,
                 w_overlap: float, bridge_window: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (relabel records, lineage edges)."""
    dates = sorted(snapshots.keys())
    relabel: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    next_did = [0]

    def new_id() -> str:
        next_did[0] += 1
        return f"d{next_did[0]:03d}"

    prev_ids: Dict[Any, str] = {}
    # dormant pool for bridge: derived_id -> (last_date, fingerprint)
    dormant: Dict[str, Tuple[pd.Timestamp, _Fingerprint]] = {}

    for di, date in enumerate(dates):
        fps = snapshots[date]
        if di == 0:
            for label, fp in fps.items():
                did = new_id()
                prev_ids[label] = did
                relabel.append(dict(date=date, raw_label=label, derived_id=did))
                lineage.append(dict(parent_id=None, child_id=did, date=date, event='birth'))
            continue

        prev_date = dates[di - 1]
        sigma_bar = _psd_clip((x_covars[prev_date] + x_covars[date]) / 2.0)
        prev_fps = snapshots[prev_date]
        common = set(a for fp in prev_fps.values() for a in fp.members) & \
                 set(a for fp in fps.values() for a in fp.members)

        A = list(prev_fps.keys())
        B = list(fps.keys())
        links: Dict[Tuple[Any, Any], float] = {}
        for a in A:
            for b in B:
                ok, wt = _qualifies(prev_fps[a], fps[b], common, sigma_bar,
                                    overlap_metric=overlap_metric, combine=combine,
                                    overlap_band=overlap_band, spread_vol_cut=spread_vol_cut,
                                    w_overlap=w_overlap)
                if ok:
                    links[(a, b)] = wt

        # Hungarian backbone on the qualifying subgraph (maximize total weight)
        match: Dict[Any, Any] = {}
        if links:
            big = max(links.values()) + 1.0
            cost = np.full((len(A), len(B)), big)
            for (a, b), wt in links.items():
                cost[A.index(a), B.index(b)] = big - wt
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if (A[r], B[c]) in links:
                    match[A[r]] = B[c]

        next_ids: Dict[Any, str] = {}
        consumed_A: set = set()
        # 1. backbone continuations
        for a, b in match.items():
            next_ids[b] = prev_ids[a]
            consumed_A.add(a)
            lineage.append(dict(parent_id=prev_ids[a], child_id=prev_ids[a],
                                date=date, event='continue'))
        # 2. remaining B: split child, free-parent continuation, bridge revival, or birth
        for b in B:
            if b in next_ids:
                continue
            cand = sorted([a for a in A if (a, b) in links],
                          key=lambda a: links[(a, b)], reverse=True)
            if cand:
                best = cand[0]
                if best in consumed_A:                       # parent already continued -> split
                    did = new_id()
                    next_ids[b] = did
                    lineage.append(dict(parent_id=prev_ids[best], child_id=did,
                                        date=date, event='split'))
                else:                                        # free parent -> continue
                    next_ids[b] = prev_ids[best]
                    consumed_A.add(best)
                    lineage.append(dict(parent_id=prev_ids[best], child_id=prev_ids[best],
                                        date=date, event='continue'))
                continue
            # bridge: try to revive a dormant track within the window
            revived = None
            for did, (ldate, lfp) in dormant.items():
                if (date - ldate).days <= 0:
                    continue
                if (dates.index(date) - dates.index(ldate)) > bridge_window + 1:
                    continue
                ok, _ = _qualifies(lfp, fps[b], set(lfp.members) | set(fps[b].members),
                                   sigma_bar, overlap_metric=overlap_metric, combine=combine,
                                   overlap_band=overlap_band, spread_vol_cut=spread_vol_cut,
                                   w_overlap=w_overlap)
                if ok:
                    revived = did
                    break
            if revived is not None:
                next_ids[b] = revived
                dormant.pop(revived, None)
                lineage.append(dict(parent_id=revived, child_id=revived,
                                    date=date, event='continue'))
            else:
                did = new_id()
                next_ids[b] = did
                lineage.append(dict(parent_id=None, child_id=did, date=date, event='birth'))

        # 3. merges: unconsumed A that still qualifies into some B -> absorbed
        for a in A:
            if a in consumed_A:
                continue
            cand = sorted([b for b in B if (a, b) in links],
                          key=lambda b: links[(a, b)], reverse=True)
            if cand:
                lineage.append(dict(parent_id=prev_ids[a], child_id=next_ids[cand[0]],
                                    date=date, event='merge'))
                consumed_A.add(a)
        # 4. deaths: unconsumed A with no link -> retire (to dormant pool for bridge)
        for a in A:
            if a not in consumed_A:
                dormant[prev_ids[a]] = (prev_date, prev_fps[a])
                lineage.append(dict(parent_id=prev_ids[a], child_id=None,
                                    date=date, event='death'))
        # expire dormant beyond the window
        dormant = {d: (ld, lf) for d, (ld, lf) in dormant.items()
                   if (dates.index(date) - dates.index(ld)) <= bridge_window + 1}

        for b, did in next_ids.items():
            relabel.append(dict(date=date, raw_label=b, derived_id=did))
        prev_ids = next_ids

    return pd.DataFrame(relabel), pd.DataFrame(lineage)


# --------------------------------------------------------------------------- #
# global min-cost-flow disjoint-path-cover matcher (with bridge edges)
# --------------------------------------------------------------------------- #
def _match_panel_mcf(snapshots: Dict[pd.Timestamp, Dict[Any, _Fingerprint]],
                     x_covars: Dict[pd.Timestamp, np.ndarray],
                     *, overlap_metric: str, combine: str,
                     overlap_band: Tuple[float, float], spread_vol_cut: float,
                     w_overlap: float, bridge_window: int, bridge_decay: float = 0.5,
                     scale: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Global max-weight vertex-disjoint path cover via min-cost flow.

    Unlike the per-transition Hungarian (which, with consecutive-only edges,
    separates per transition and equals the greedy backbone), this solves the
    whole panel jointly so bridge edges can route a persistent cluster's
    identity *around* a transient merge/split rather than handing it off locally.
    """
    import networkx as nx
    dates = sorted(snapshots.keys())
    didx = {d: i for i, d in enumerate(dates)}

    # candidate continuation edges over consecutive + bridge gaps
    edges: List[Tuple[Tuple, Tuple, float]] = []
    members_at = {d: {lab: set(fp.members) for lab, fp in snapshots[d].items()} for d in dates}
    clustered_at = {d: set().union(*members_at[d].values()) if members_at[d] else set() for d in dates}
    for i, di in enumerate(dates):
        for gap in range(1, bridge_window + 2):
            j = i + gap
            if j >= len(dates):
                break
            dj = dates[j]
            sigma_bar = _psd_clip((x_covars[di] + x_covars[dj]) / 2.0)
            common = clustered_at[di] & clustered_at[dj]
            decay = bridge_decay ** (gap - 1)
            for a, fa in snapshots[di].items():
                for b, fb in snapshots[dj].items():
                    ok, wt = _qualifies(fa, fb, common, sigma_bar,
                                        overlap_metric=overlap_metric, combine=combine,
                                        overlap_band=overlap_band, spread_vol_cut=spread_vol_cut,
                                        w_overlap=w_overlap)
                    if ok:
                        edges.append(((di, a), (dj, b), wt * decay))

    nodes = [(d, lab) for d in dates for lab in snapshots[d]]
    N = len(nodes)
    G = nx.DiGraph()
    G.add_node('s', demand=-N); G.add_node('t', demand=N)
    G.add_edge('s', 't', capacity=N, weight=0)               # slack: unmatched units
    for v in nodes:
        G.add_edge('s', ('L', v), capacity=1, weight=0)
        G.add_edge(('R', v), 't', capacity=1, weight=0)
    # keep only the best edge per (tail) and per (head) is not needed; flow handles it
    for u, w, wt in edges:
        G.add_edge(('L', u), ('R', w), capacity=1, weight=-int(round(wt * scale)))
    flow = nx.min_cost_flow(G)

    succ: Dict[Tuple, Tuple] = {}
    pred: Dict[Tuple, Tuple] = {}
    for u, w, _ in edges:
        f = flow.get(('L', u), {}).get(('R', w), 0)
        if f == 1:
            succ[u] = w; pred[w] = u

    # qualifying-neighbour lookup for event tagging
    qual_succ: Dict[Tuple, List[Tuple]] = {}
    qual_pred: Dict[Tuple, List[Tuple]] = {}
    for u, w, wt in edges:
        qual_succ.setdefault(u, []).append((w, wt))
        qual_pred.setdefault(w, []).append((u, wt))

    # assign derived ids by walking chains from chain-starts
    relabel: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    did_of: Dict[Tuple, str] = {}
    counter = [0]
    def new_id() -> str:
        counter[0] += 1; return f"d{counter[0]:03d}"
    for v in sorted(nodes, key=lambda x: (didx[x[0]], str(x[1]))):
        if v in pred:
            continue                                         # not a chain start
        did = new_id()
        cur = v
        while cur is not None:
            did_of[cur] = did
            cur = succ.get(cur)
    for v, did in did_of.items():
        relabel.append(dict(date=v[0], raw_label=v[1], derived_id=did))
    # lineage events
    for v in nodes:
        d, lab = v
        if v not in pred:                                    # chain start
            cands = sorted(qual_pred.get(v, []), key=lambda x: x[1], reverse=True)
            if cands:
                lineage.append(dict(parent_id=did_of[cands[0][0]], child_id=did_of[v],
                                    date=d, event='split'))
            else:
                lineage.append(dict(parent_id=None, child_id=did_of[v], date=d, event='birth'))
        else:
            u = pred[v]
            ev = 'continue' if (didx[d] - didx[u[0]]) == 1 else 'bridge'
            lineage.append(dict(parent_id=did_of[u], child_id=did_of[v], date=d, event=ev))
        if v not in succ:                                    # chain end
            cands = sorted(qual_succ.get(v, []), key=lambda x: x[1], reverse=True)
            if cands:
                lineage.append(dict(parent_id=did_of[v], child_id=did_of[cands[0][0]],
                                    date=cands[0][0][0], event='merge'))
            else:
                lineage.append(dict(parent_id=did_of[v], child_id=None, date=d, event='death'))
    return pd.DataFrame(relabel), pd.DataFrame(lineage)


# --------------------------------------------------------------------------- #
# track panels + classification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackPanel:
    betas: pd.DataFrame
    factor_vol: pd.Series
    idio_vol: pd.Series
    total_vol: pd.Series
    r2: pd.Series
    size: pd.Series
    dominant_factor: pd.Series
    members: Dict[pd.Timestamp, List[str]]


def _build_tracks(relabel: pd.DataFrame,
                  snapshots: Dict[pd.Timestamp, Dict[Any, _Fingerprint]],
                  factors: List[str]) -> Dict[str, TrackPanel]:
    tracks: Dict[str, TrackPanel] = {}
    for did, grp in relabel.groupby('derived_id'):
        rows_beta, idx = [], []
        fv, iv, tv, r2, sz, dom, mem = [], [], [], [], [], [], {}
        for _, row in grp.sort_values('date').iterrows():
            fp = snapshots[row['date']][row['raw_label']]
            idx.append(row['date'])
            rows_beta.append(fp.beta)
            fv.append(np.sqrt(fp.factor_var)); iv.append(np.sqrt(fp.idio_var))
            tv.append(np.sqrt(fp.total_var)); r2.append(fp.r2)
            sz.append(len(fp.members)); dom.append(fp.dominant)
            mem[row['date']] = list(fp.members)
        idx = pd.DatetimeIndex(idx)
        tracks[did] = TrackPanel(
            betas=pd.DataFrame(rows_beta, index=idx, columns=factors),
            factor_vol=pd.Series(fv, index=idx), idio_vol=pd.Series(iv, index=idx),
            total_vol=pd.Series(tv, index=idx), r2=pd.Series(r2, index=idx),
            size=pd.Series(sz, index=idx), dominant_factor=pd.Series(dom, index=idx),
            members=mem)
    return tracks


def _classify(tracks: Dict[str, TrackPanel], x_covars: Dict[pd.Timestamp, np.ndarray],
              n_dates: int, cfg: TaxonomyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, trans = [], []
    for did, tp in tracks.items():
        live = list(tp.betas.index)
        coverage = len(live) / n_dates
        lifetime = len(live)
        modal = tp.dominant_factor.value_counts()
        modal_dom = modal.index[0]
        modal_frac = modal.iloc[0] / modal.sum()
        mean_beta = tp.betas.mean(axis=0)
        eq = float(mean_beta.get(cfg.equity_factor, 0.0))
        if modal_frac < cfg.mixed_dominant_frac:
            ttype = 'Mixed'
        elif modal_dom == cfg.equity_factor:
            ttype = ('Equity-HighBeta' if eq >= cfg.high_beta else
                     'Equity-Defensive' if eq <= cfg.defensive_beta else 'Equity-Core')
        else:
            ttype = modal_dom
        # beta-stability: median spread vol from track-mean beta under track-avg Sigma_F
        sig = _psd_clip(np.mean([x_covars[d] for d in live], axis=0))
        mb = mean_beta.to_numpy()
        dists = [float(np.sqrt(max((tp.betas.loc[d].to_numpy() - mb) @ sig @
                                   (tp.betas.loc[d].to_numpy() - mb), 0.0))) for d in live]
        s = float(np.median(dists)) if dists else 0.0
        stab = ('Stable' if s <= cfg.stable_spread_vol else
                'Drifting' if s >= cfg.drifting_spread_vol else 'Transitioning')
        regime = ('Core' if coverage >= cfg.core_coverage else
                  'Transient' if coverage < cfg.transient_coverage else 'Episodic')
        mfv = float(tp.factor_vol.mean())
        vol_regime = ('Low' if mfv < cfg.vol_low else 'High' if mfv > cfg.vol_high else 'Mid')
        if len(live) >= 3:
            xs = np.arange(len(live))
            slope = float(np.polyfit(xs, tp.factor_vol.to_numpy(), 1)[0])
        else:
            slope = 0.0
        rows.append(dict(derived_id=did, track_type=ttype, beta_stability=s,
                         stability_label=stab, coverage=coverage, lifetime=lifetime,
                         persistence=regime, mean_factor_vol=mfv, vol_regime=vol_regime,
                         vol_trend=slope, modal_dom=modal_dom))
        # within-track beta-regime breaks (distance spikes beyond drifting threshold)
        for d, dist in zip(live, dists):
            if dist >= cfg.drifting_spread_vol:
                trans.append(dict(derived_id=did, date=d, kind='beta_break',
                                  detail=f'spread_vol={dist:.3f}'))
    return (pd.DataFrame(rows).set_index('derived_id').sort_values('coverage', ascending=False),
            pd.DataFrame(trans))


# --------------------------------------------------------------------------- #
# report object + public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskClusterReport:
    relabel: pd.DataFrame
    tracks: Dict[str, TrackPanel]
    classification: pd.DataFrame
    lineage: pd.DataFrame
    transitions: pd.DataFrame
    params: Dict[str, Any]
    factor_covar: pd.DataFrame = field(default_factory=pd.DataFrame)

    def label_tracks(self, asset_meta: pd.DataFrame, *,
                     sub_class_col: str = 'Sub Asset Class',
                     class_col: str = 'Asset Class',
                     equity_factor: str = 'Equity',
                     purity_cut: float = 0.50) -> pd.Series:
        """Economic label per derived track: modal sub-asset-class over the
        track's life, qualified by factor regime (beta bucket for equity,
        dominant factor otherwise). ``asset_meta`` is indexed by asset and must
        carry ``sub_class_col`` and ``class_col``. Returns derived_id -> label.
        """
        sac = asset_meta[sub_class_col]; acl = asset_meta[class_col]
        labels: Dict[str, str] = {}
        for did, tp in self.tracks.items():
            insts = [a for mem in tp.members.values() for a in mem]
            s = pd.Series([sac.get(a) for a in insts]).dropna()
            s = s[s.astype(str) != '---']
            a = pd.Series([acl.get(a) for a in insts]).dropna()
            if len(s) == 0:
                labels[did] = self.classification.loc[did, 'track_type']
                continue
            top_sac = s.value_counts(normalize=True)
            theme = (top_sac.index[0] if top_sac.iloc[0] >= purity_cut
                     else (f"Diversified {a.mode().iloc[0]}" if len(a) else 'Mixed'))
            modal_ac = a.mode().iloc[0] if len(a) else None
            if modal_ac == equity_factor:
                eq = float(tp.betas.get(equity_factor, pd.Series(dtype=float)).mean() or 0.0)
                char = ('high-\u03b2' if eq >= 0.70 else 'defensive' if eq <= 0.30 else 'core')
            else:
                char = str(self.classification.loc[did, 'modal_dom'])
            labels[did] = f"{theme} \u00b7 {char}"
        return pd.Series(labels, name='label')

    def factor_labels(self, *, hi_beta: float = 0.70, lo_beta: float = 0.30,
                      secondary_share: float = 0.35,
                      vol_low: float = 0.05, vol_high: float = 0.12) -> pd.Series:
        """Label per track from its MATF betas and volatility only (no external
        metadata): primary (and material secondary) factor by variance
        contribution beta_C ∘ Σ_F beta_C, an exposure descriptor, and a vol
        regime. e.g. 'Equity high-β · high-vol', 'Rates long-duration · low-vol',
        'Equity defensive + Credit · low-vol'.
        """
        factors = list(self.factor_covar.index)
        sigma = self.factor_covar.to_numpy()
        out: Dict[str, str] = {}
        for did, tp in self.tracks.items():
            b = tp.betas.mean(axis=0).reindex(factors).fillna(0.0).to_numpy()
            contrib = np.clip(b * (sigma @ b), 0.0, None)      # marginal factor variances
            if contrib.sum() <= 0:
                out[did] = 'Idiosyncratic'; continue
            order = np.argsort(contrib)[::-1]
            shares = contrib / contrib.sum()
            primary = factors[order[0]]
            secondary = factors[order[1]] if shares[order[1]] >= secondary_share else None
            if primary == 'Equity':
                be = b[factors.index('Equity')]
                desc = 'high-\u03b2' if be >= hi_beta else 'defensive' if be <= lo_beta else 'core'
            elif primary == 'Rates':
                desc = 'long-duration' if b[factors.index('Rates')] > 0 else 'short-duration'
            else:
                desc = 'short' if b[order[0]] < 0 else ''
            fv = float(tp.factor_vol.mean())
            vol = 'low-vol' if fv < vol_low else 'high-vol' if fv > vol_high else 'mid-vol'
            lab = primary + (f" {desc}" if desc else '')
            if secondary:
                lab += f" + {secondary}"
            out[did] = f"{lab} \u00b7 {vol}"
        return pd.Series(out, name='factor_label')

    def to_membership_panel(self) -> pd.DataFrame:
        """Dates (index) x assets (columns) -> assigned derived cluster id.

        NaN where an asset is unclustered at that date. This is the wide
        counterpart of the long-form ``relabel`` map.
        """
        dates = pd.DatetimeIndex(sorted(self.relabel['date'].unique()))
        rows: Dict[pd.Timestamp, Dict[str, str]] = {}
        for did, tp in self.tracks.items():
            for date, members in tp.members.items():
                for a in members:
                    rows.setdefault(date, {})[a] = did
        panel = pd.DataFrame.from_dict(rows, orient='index').reindex(dates)
        return panel.reindex(sorted(panel.columns), axis=1)

    def _labels(self, label_kind: str, asset_meta: Optional[pd.DataFrame]) -> pd.Series:
        if label_kind == 'factor':
            return self.factor_labels()
        if label_kind == 'meta':
            if asset_meta is None:
                raise ValueError("label_kind='meta' requires asset_meta")
            return self.label_tracks(asset_meta)
        if label_kind == 'id':
            return pd.Series({d: d for d in self.tracks}, name='id')
        raise ValueError(f"unknown label_kind {label_kind!r}")

    def to_label_panel(self, label_kind: str = 'factor',
                       asset_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Dates (index) x assets (columns) -> risk label per date (the
        consolidated counterpart of the raw risk_clusters panel)."""
        return self.to_membership_panel().replace(self._labels(label_kind, asset_meta).to_dict())

    def labels_at(self, date, label_kind: str = 'factor',
                  asset_meta: Optional[pd.DataFrame] = None) -> pd.Series:
        """Asset -> risk label at a given rebalancing date (nearest prior date
        if exact date absent). Drives the per-asset ``Cluster Label`` column."""
        panel = self.to_membership_panel()
        d = pd.Timestamp(date)
        if d not in panel.index:
            prior = panel.index[panel.index <= d]
            if len(prior) == 0:
                return pd.Series(dtype=object)
            d = prior[-1]
        row = panel.loc[d].dropna()
        return row.map(self._labels(label_kind, asset_meta).to_dict())

    def to_tables(self) -> Dict[str, pd.DataFrame]:
        return dict(classification=self.classification,
                    membership_panel=self.to_membership_panel(),
                    lineage=self.lineage,
                    transitions=self.transitions,
                    relabel=self.relabel)

    def to_figures(self):
        import matplotlib.pyplot as plt
        figs = []
        # 1. lineage Gantt: derived ids as bands over time
        dates = pd.DatetimeIndex(sorted(self.relabel['date'].unique()))
        ids = list(self.classification.index)
        fig, ax = plt.subplots(figsize=(11, max(3, 0.28 * len(ids))))
        for y, did in enumerate(ids):
            d = self.relabel.loc[self.relabel.derived_id == did, 'date']
            ax.scatter(d, [y] * len(d), s=10)
        ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=6)
        ax.set_title('derived cluster lineage timeline'); figs.append(fig)
        # 2. factor-vol heatmap
        piv = pd.DataFrame({did: tp.factor_vol for did, tp in self.tracks.items()}).reindex(dates)
        fig2, ax2 = plt.subplots(figsize=(11, max(3, 0.28 * len(ids))))
        im = ax2.imshow(piv[ids].T.to_numpy(), aspect='auto', cmap='viridis')
        ax2.set_yticks(range(len(ids))); ax2.set_yticklabels(ids, fontsize=6)
        ax2.set_title('factor vol by derived cluster'); fig2.colorbar(im, ax=ax2); figs.append(fig2)
        return figs


def analyze_risk_clusters(covar_data: RollingFactorCovarData, *,
                             overlap_metric: str = 'overlap',
                             beta_metric: str = 'mahalanobis_sf',
                             combine: str = 'gated',
                             overlap_band: Tuple[float, float] = (0.20, 0.60),
                             spread_vol_cut: float = 0.025,
                             bridge_window: int = 1,
                             bridge_decay: float = 0.5,
                             w_overlap: float = 0.6,
                             weighting: str = 'equal',
                             taxonomy: Optional[TaxonomyConfig] = None,
                             method: str = 'mcf') -> RiskClusterReport:
    cfg = taxonomy or TaxonomyConfig()
    dates = list(covar_data.dates)
    snapshots: Dict[pd.Timestamp, Dict[Any, _Fingerprint]] = {}
    x_covars: Dict[pd.Timestamp, np.ndarray] = {}
    factors: List[str] = []
    for d in dates:
        cd = covar_data[d]
        fps, factors = _snapshot_fingerprints(cd, weighting=weighting)
        snapshots[d] = fps
        x_covars[d] = _psd_clip(cd.x_covar.to_numpy())
    if method in ('mcf', 'graph'):
        relabel, lineage = _match_panel_mcf(snapshots, x_covars, overlap_metric=overlap_metric,
                                            combine=combine, overlap_band=overlap_band,
                                            spread_vol_cut=spread_vol_cut, w_overlap=w_overlap,
                                            bridge_window=bridge_window, bridge_decay=bridge_decay)
    else:
        relabel, lineage = _match_panel(snapshots, x_covars, overlap_metric=overlap_metric,
                                        combine=combine, overlap_band=overlap_band,
                                        spread_vol_cut=spread_vol_cut, w_overlap=w_overlap,
                                        bridge_window=bridge_window)
    tracks = _build_tracks(relabel, snapshots, factors)
    classification, transitions = _classify(tracks, x_covars, len(dates), cfg)
    params = dict(overlap_metric=overlap_metric, beta_metric=beta_metric, combine=combine,
                  overlap_band=overlap_band, spread_vol_cut=spread_vol_cut,
                  bridge_window=bridge_window, weighting=weighting, method=method)
    avg_sigma = pd.DataFrame(np.mean([x_covars[d] for d in dates], axis=0),
                             index=factors, columns=factors)
    return RiskClusterReport(relabel=relabel, tracks=tracks, classification=classification,
                                lineage=lineage, transitions=transitions, params=params,
                                factor_covar=avg_sigma)


def run_risk_label_report(covar_data: RollingFactorCovarData, **kwargs):
    report = analyze_risk_clusters(covar_data, **kwargs)
    return report.to_figures(), report.to_tables()
