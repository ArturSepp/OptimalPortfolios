"""The QIS five-stock, ten-short-option book under OP's four-factor FCGL model.

Run python -m examples.reports.stress_testing_with_options_local --output-dir <fresh directory>.
The first run downloads Yahoo data; subsequent runs replay a hashed local cache.
--refresh explicitly replaces the cache. No Bloomberg access or client data is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qis
import optimalportfolios as opt
import factorlasso as fl
from factorlasso.cluster_lineage import analyze_cluster_lineage

# The portfolio builder and VOP adapter mirror the public QIS teaching example:
# https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/stress_testing_with_options.py
# QIS exports HoldingPayoff, not a VOP-specific mark. Keep this example adapter
# self-contained so an installed QIS distribution does not need its examples checkout.
# User-facing example dependency only; VOP is never imported by the qis package.
import vanilla_option_pricers as vop  # noqa: TID251


STOCKS = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA")
FACTORS = ("SPY", "TLT", "GLD", "USO")
AS_OF = "2025-12-31"
START = "2015-01-01"
SPAN = 52
ANNUALISATION = 52.0
REG_LAMBDA = 1e-5
N_CLUSTERS = 2
FREQUENCY = "W-WED"


def load_prices(cache_dir, as_of=AS_OF, refresh=False):
    """Download explicit raw/adjusted closes once, then replay their checked CSV bytes."""
    import yfinance as yf

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(as_of).normalize()
    path = cache / f"yahoo_{START}_{cut.date()}.csv"
    record = path.with_suffix(".json")
    expected = {"tickers": list(STOCKS + FACTORS), "start": START,
                "as_of": str(cut.date()), "auto_adjust": False, "repair": False}
    if refresh or not path.exists():
        if path.exists() != record.exists() and not refresh:
            raise ValueError("Incomplete cache; use --refresh to rebuild it")
        yf.set_tz_cache_location(str(cache / "yfinance"))
        frame = yf.download(
            list(STOCKS + FACTORS), start=START, end=str((cut + pd.Timedelta(days=1)).date()),
            auto_adjust=False, repair=False, actions=False, threads=False, progress=False,
            group_by="column", timeout=30,
        )
        if frame is None or frame.empty:
            raise ValueError("Yahoo returned no prices; retry later or use an existing cache")
        frame = frame.loc[:, pd.MultiIndex.from_product(
            [["Close", "Adj Close"], STOCKS + FACTORS])].loc[:cut]
        if frame.notna().sum().min() < 260:
            raise ValueError("Every stock and factor needs at least 260 observed daily prices")
        frame.to_csv(path)
        record.write_text(json.dumps({**expected, "yfinance": version("yfinance"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    metadata = json.loads(record.read_text(encoding="utf-8"))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Cache request metadata differs; use the matching cache or --refresh")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Cached prices changed; use --refresh for an intentional new download")
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True,
                        float_precision="round_trip")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame = frame.loc[:cut].sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate price dates")
    frame = frame.dropna()
    if frame.empty or (cut - frame.index[-1]).days > 4 or (frame <= 0).any().any():
        raise ValueError("Missing, stale or nonpositive common price panel")
    return frame, metadata


def fit_risk_model(prices):
    """Fit FCGL through OP and retain the exact partition, linkage and cut distance."""
    adjusted = prices["Adj Close"]
    weekly = qis.to_returns(adjusted, freq=FREQUENCY, is_log_returns=True,
                            is_first_zero=False, drop_first=False).loc[:adjusted.index[-1]]
    date = weekly.index[-1]
    x = weekly[list(FACTORS)].dropna()
    y = weekly[list(STOCKS)]
    if len(x) < 3 * SPAN:
        raise ValueError("Need at least three 52-week spans of complete returns")
    factor_covar = pd.DataFrame(
        ANNUALISATION * qis.compute_ewm_covar(x.to_numpy(), span=SPAN),
        index=FACTORS, columns=FACTORS,
    )
    lasso = fl.LassoModel(
        model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        span=SPAN, cluster_correlation_span=SPAN, warmup_period=SPAN,
        reg_lambda=REG_LAMBDA, n_clusters=N_CLUSTERS, linkage_method="ward",
        distance_transform=fl.DistanceTransform.ONE_MINUS_RHO,
        group_penalty="normalized", l1_weight=0.0, demean=False,
        nonneg=False, auto_sign_constraints=False, solver="CLARABEL",
    )
    estimator = opt.FactorCovarEstimator(
        lasso_model=lasso, factor_returns_freq=FREQUENCY, factor_covar_span=SPAN,
        rebalancing_freq=FREQUENCY, is_apply_vol_normalised_returns=False,
    )
    # Explicit annual covariance preserves the QIS example's zero-mean EWMA convention.
    # OP's internally estimated factor covariance otherwise applies demeaning.
    snapshot = estimator.fit_current_factor_covars(
        risk_factor_prices=adjusted[list(FACTORS)].loc[:date],
        asset_returns_dict={FREQUENCY: y.loc[:date]}, assets=list(STOCKS),
        estimation_date=date, x_covar=factor_covar,
    )
    model = opt.build_risk_model({date: snapshot})
    diagnostics = snapshot.y_variances[["r2"]].copy()
    diagnostics["weekly_observations"] = lasso.valid_mask_.sum(axis=0).astype(int)
    history = qis.to_returns(adjusted[list(FACTORS)], freq="ME", is_log_returns=True,
                             drop_first=True).loc[:adjusted.index[-1]].dropna()
    return model, date, history, diagnostics, snapshot, lasso


def cluster_report_inputs(snapshot):
    """Adapt stored FactorLasso trees and descriptive labels to QIS without reclustering."""
    memberships = fl.get_clusters_by_freq(snapshot.clusters)
    linkages = fl.get_linkages_by_freq(snapshot.linkages)
    cutoffs = snapshot.cutoffs.to_dict()
    lineage = analyze_cluster_lineage(fl.RollingFactorCovarData(
        data={snapshot.estimation_date: snapshot}), weighting="equal")
    labels = lineage.labels_at(snapshot.estimation_date)
    display_labels = {}
    for cadence, members in memberships.items():
        for group in members.unique():
            member_labels = labels.reindex(members.index[members.eq(group)]).dropna().unique()
            if len(member_labels) != 1:
                raise ValueError("Expected one descriptive label per fitted cluster")
            display_labels[f"{cadence}-{group}"] = str(member_labels[0])
    return dict(cluster_memberships=memberships, cluster_linkages=linkages,
                cluster_cutoffs=cutoffs, cluster_labels=display_labels)


@dataclass(frozen=True)
class BsmOptionPayoff:
    """Example-owned, USD European option mark with signed exchange-contract quantity.

    Attributes:
        underlying: Yahoo stock symbol identifying the actual USD quote.
        option_type: C or P; the direction is carried by contracts, not the type.
        strike: Strike in USD per share.
        expiry: Synthetic contract expiry date, strictly after valuation_date.
        valuation_date: Frozen mark date; no time elapses during instantaneous stresses.
        volatility: Assumed annual lognormal implied volatility, fixed under shocks.
        contracts: Signed number of contracts; negative for the short overlay.
        multiplier: Shares per contract, normally 100.
        rate: Continuously compounded USD rate, held fixed.
        dividend_yield: Continuously compounded dividend yield, held fixed.
    """
    underlying: str
    option_type: str
    strike: float
    expiry: pd.Timestamp
    valuation_date: pd.Timestamp
    volatility: float
    contracts: int
    multiplier: int = 100
    rate: float = 0.04
    dividend_yield: float = 0.0
    implementation_id = "qis.example.vop_european_option.v1"
    coverage = "European BSM mark; fixed IV/rate/TTM; no American exercise or margin model."
    boundary_policy = "Smooth BSM delta with positive TTM and volatility; no expiry crossing."

    def __post_init__(self):
        """Reject invalid pricing terms before any report computation."""
        if self.option_type not in ("C", "P") or self.ttm <= 0:
            raise ValueError("Use C/P and an expiry after the frozen valuation date")
        if self.strike <= 0 or self.volatility <= 0 or self.multiplier <= 0:
            raise ValueError("Strike, volatility and multiplier must be positive")

    @property
    def ttm(self):
        """Return ACT/365 time remaining at the frozen valuation date."""
        return (self.expiry - self.valuation_date).days / 365.0

    def unit_prices(self, spots):
        """Call the compiled VOP forward-grid pricer, returning USD per underlying share."""
        forwards = np.asarray(spots, dtype=float) * np.exp(
            (self.rate - self.dividend_yield) * self.ttm)
        return vop.compute_bsm_forward_grid_prices(
            ttm=self.ttm, forwards=forwards, strike=self.strike, vol=self.volatility,
            optiontype=self.option_type, discfactor=np.exp(-self.rate * self.ttm),
        )

    def spot_greeks(self, spot):
        """Convert VOP discounted forward delta and undiscounted forward gamma to spot Greeks."""
        carry = np.exp((self.rate - self.dividend_yield) * self.ttm)
        forward = float(spot * carry)
        discount = np.exp(-self.rate * self.ttm)
        delta = carry * vop.compute_bsm_vanilla_delta(
            self.ttm, forward, self.strike, self.volatility, self.option_type, discount)
        gamma = discount * carry**2 * vop.compute_bsm_vanilla_gamma(
            self.ttm, forward, self.strike, self.volatility)
        return delta, gamma

    def evaluate(self, context: qis.PayoffContext) -> pd.Series:
        """Return signed scenario option marks; all quotes in this example are USD."""
        if (context.reference_currency != "USD"
                or context.quote_currencies[self.underlying] != "USD"):
            raise ValueError("This example payoff supports USD quotes and reference currency only")
        quotes = context.quotes[self.underlying]
        values = self.contracts * self.multiplier * self.unit_prices(quotes.to_numpy())
        return pd.Series(values, index=quotes.index)

    def _jacobian(self, context, spot):
        """Apply the spot-to-log-quote chain rule and the public shared-response mapping."""
        delta, _ = self.spot_greeks(spot)
        dollars = self.contracts * self.multiplier * delta * spot
        return dollars * context.quote_response_jacobian.loc[self.underlying]

    def response_jacobian(self, context: qis.PayoffContext) -> pd.Series:
        """Return current signed dollar delta by underlying response."""
        return self._jacobian(context, context.baseline_quotes[self.underlying])

    def scenario_response_jacobian(self, context: qis.PayoffContext) -> pd.Series:
        """Recompute delta at the stressed spot for QIS conditional risk bands."""
        return self._jacobian(context, context.quotes.iloc[0][self.underlying])


def build_portfolio(prices, model, risk_date):
    """Buy round lots of five stocks and sell one call/put line per stock."""
    date = prices.index[-1]
    spots = prices["Close"].iloc[-1]
    returns = qis.to_returns(prices["Adj Close"][list(STOCKS)], is_log_returns=True,
                             drop_first=True)
    realised = qis.compute_ewm_vol(returns, span=63, annualize=True,
                                   annualization_factor=252).iloc[-1]
    quotes = {stock: qis.Underlying(stock, float(spots[stock]), "USD", stock,
                                    qis.ResponseBasis.LOCAL) for stock in STOCKS}
    holdings, inventory = [], []
    for i, stock in enumerate(STOCKS):
        spot = float(spots[stock])
        lots = int(2_000_000 / (100 * spot))
        if lots < 1:
            raise ValueError("Stock price exceeds the per-name teaching budget")
        shares = 100 * lots
        holdings.append(qis.PortfolioHolding(
            f"{stock} US Equity", stock, shares * spot,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, stock, shares),),
            metadata={"short_name": stock, "sleeve": "Stocks"},
        ))
        month = date.to_period("M") + (2, 3, 4, 5, 8)[i]
        expiry = pd.date_range(month.start_time, month.end_time, freq="W-FRI")[2]
        for kind, moneyness, quantity, skew in (
            ("C", (1.03, 1.05, 1.07, 1.04, 1.08)[i], -lots, 0.0),
            ("P", (0.97, 0.95, 0.93, 0.96, 0.92)[i], -int(1.5 * lots), 0.04),
        ):
            strike = float(5 * np.round(spot * moneyness / 5))
            option = BsmOptionPayoff(stock, kind, strike, expiry, date,
                                    float(max(0.15, 1.15 * realised[stock]) + skew), quantity)
            ticker = f"{stock} US {expiry:%m/%d/%y} {kind}{strike:g} Equity"
            unit_price = float(option.unit_prices(np.array([spot]))[0])
            delta, gamma = option.spot_greeks(spot)
            mark = quantity * 100 * unit_price
            holdings.append(qis.PortfolioHolding(
                ticker, ticker, mark, payoff=option,
                metadata={"short_name": f"{stock} {expiry:%b} {kind}{strike:g}",
                          "sleeve": "Short calls" if kind == "C" else "Short puts",
                          "expiry": str(expiry.date()), "strike": str(strike),
                          "contracts": str(quantity), "multiplier": "100",
                          "iv": str(option.volatility)},
            ))
            inventory.append({"ticker": ticker, "underlying": stock, "type": kind,
                "expiry": str(expiry.date()), "strike": strike, "contracts": quantity,
                "multiplier": 100, "iv": option.volatility, "unit_price": unit_price,
                "mtm_usd": mark, "spot": spot, "unit_spot_delta": delta,
                "dollar_delta": quantity * 100 * delta * spot,
                "dollar_gamma": quantity * 100 * gamma * spot**2})
    denominator = sum(holding.observed_mtm for holding in holdings)
    portfolio = qis.InstrumentPortfolio(
        tuple(holdings), quotes, model, risk_date, date, "USD", denominator,
        denominator_label="Net marked portfolio value",
    )
    return portfolio, pd.DataFrame(inventory).set_index("ticker")


def scenario_inputs():
    """Use independent requests, joint conditional comparisons and four correlated grids."""
    anchors = {f"{factor} {bump:+.0%}": {factor: bump}
               for factor, bumps in (("SPY", (-.3, -.2, -.1, .1, .2, .3)),
                                     ("TLT", (-.1, .1)), ("GLD", (-.2, .2)),
                                     ("USO", (-.3, .3))) for bump in bumps}
    anchors["No move"] = {factor: 0.0 for factor in FACTORS}
    requests = qis.StressScenarios(pd.DataFrame.from_dict(anchors, orient="index"),
                                   convention=qis.ShockConvention.SIMPLE)
    grids = {}
    for factor in FACTORS:
        limit = 30 if factor == "SPY" else 20
        axis = pd.Index(np.arange(-limit, limit + 1) / 100.0, name=f"{factor} simple return")
        grids[factor] = qis.StressScenarios(
            pd.DataFrame({factor: axis.to_numpy()}, index=axis),
            mode=qis.ScenarioMode.CONDITIONAL, convention=qis.ShockConvention.SIMPLE,
        )
    return requests, grids


def verify_example(portfolio, result, inventory, snapshot, lasso):
    """Check FCGL optimality, covariance units, trees, Greeks and additive attribution."""
    from scipy.cluster.hierarchy import fcluster

    beta = result.factor_loadings.to_numpy()
    x = lasso.x_.fillna(0.0).to_numpy()
    y = lasso.y_.fillna(0.0).to_numpy()
    decay = 1.0 - 2.0 / (SPAN + 1)
    w = decay**np.arange(len(x) - 1, -1, -1)
    weights = w[:, None] * lasso.valid_mask_
    residuals = x @ beta.T - y
    norm_weights = weights / weights.sum(axis=0)
    np.testing.assert_allclose(result.residual_variances,
                               ANNUALISATION * (norm_weights * residuals**2).sum(axis=0),
                               rtol=1e-10, atol=1e-12)
    # Independent convex first-order conditions; no second FCGL fit or optimiser.
    gradient = (2.0 / len(x)) * (x.T @ (weights * residuals)).T
    groups = fl.get_clusters_by_freq(snapshot.clusters)[FREQUENCY].reindex(STOCKS)
    for group in groups.unique():
        member = groups.eq(group).to_numpy()
        threshold = REG_LAMBDA * np.sqrt(member.sum() / groups.nunique())
        for column in range(len(FACTORS)):
            block = beta[member, column]
            norm = np.linalg.norm(block)
            grad = gradient[member, column]
            if norm > 1e-3:
                assert np.linalg.norm(grad + threshold * block / norm) < 2e-7
            else:
                assert np.linalg.norm(grad) <= threshold + 2e-7
    assert lasso.model_type == fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO
    np.testing.assert_allclose(result.factor_loadings, snapshot.y_betas)
    np.testing.assert_allclose(result.factor_covariance, snapshot.x_covar)
    pd.testing.assert_frame_equal(
        portfolio.risk_model.covar[portfolio.risk_date], snapshot.get_y_covar())
    np.testing.assert_allclose(snapshot.linkages.to_numpy(), lasso.linkage)
    recovered = fcluster(lasso.linkage, lasso.cutoff, criterion="distance")
    members = lasso.clusters.reindex(lasso.y_.columns).to_numpy()
    np.testing.assert_array_equal(recovered[:, None] == recovered[None, :],
                                   members[:, None] == members[None, :])
    assert snapshot.clusters.index.equals(pd.Index(STOCKS))
    assert 1 <= snapshot.clusters.nunique() <= N_CLUSTERS
    assert len(portfolio.holdings) == 15
    assert inventory.type.value_counts().to_dict() == {"C": 5, "P": 5}
    assert inventory.contracts.lt(0).all() and inventory.dollar_gamma.lt(0).all()
    zero = pd.Series(0.0, index=FACTORS)
    np.testing.assert_allclose(portfolio.get_pnl(zero), 0.0, atol=1e-8)
    for holding in portfolio.holdings:
        if holding.payoff is None:
            continue
        option = holding.payoff
        spot = portfolio.underlyings[option.underlying].spot0
        h = spot * 1e-3
        values = option.unit_prices(np.array([spot - h, spot, spot + h]))
        delta, gamma = option.spot_greeks(spot)
        np.testing.assert_allclose(delta, (values[2] - values[0]) / (2*h), atol=2e-5)
        np.testing.assert_allclose(gamma, (values[2] - 2*values[1] + values[0]) / h**2,
                                   rtol=5e-4, atol=1e-6)
        call = replace(option, option_type="C").unit_prices(np.array([spot]))[0]
        put = replace(option, option_type="P").unit_prices(np.array([spot]))[0]
        parity = spot * np.exp(-option.dividend_yield * option.ttm) - option.strike * np.exp(
            -option.rate * option.ttm)
        np.testing.assert_allclose(call - put, parity, atol=1e-8)
    for factor in FACTORS:
        z = zero.copy()
        z[factor] = 1e-5
        derivative = (portfolio.get_pnl(z) - portfolio.get_pnl(-z)) / 2e-5
        np.testing.assert_allclose(derivative, result.holding_factor_exposures[factor],
                                   rtol=2e-5, atol=0.1)
    for name, value in result.valuations.items():
        np.testing.assert_allclose(result.attribution[name].sum(axis=1),
                                   value.portfolio_pnl, atol=1e-7)
    return {"status": "passed", "holdings": 15, "calls": 5, "puts": 5,
            "checks": ["FCGL block optimality conditions", "weighted residual moments",
                       "OP RiskModel adapter", "fitted cluster/tree partition",
                       "zero-shock marks", "spot delta/gamma finite differences",
                       "put-call parity", "factor delta finite differences", "P&L attribution"]}


def convexity_exhibit(portfolio, result, inventory):
    """Plot full repricing, frozen-delta comparison and option-sleeve contributions through QIS."""
    grid = result.grids["SPY"]
    nav = portfolio.reporting_denominator
    linear = grid.factor_log_shocks @ result.factor_exposures / nav
    curves = pd.DataFrame({"Full option repricing": grid.portfolio_pnl / nav,
                           "Current log-delta approximation": linear})
    sleeves = pd.DataFrame({name: grid.pnl[[holding.holding_id for holding in portfolio.holdings
                            if holding.metadata["sleeve"] == name]].sum(axis=1) / nav
                            for name in ("Stocks", "Short calls", "Short puts")})
    stock_options = pd.DataFrame({stock: grid.pnl[inventory.index[
        inventory.underlying.eq(stock)]].sum(axis=1) / nav for stock in STOCKS})
    table = inventory.groupby("underlying").agg(
        short_contracts=("contracts", "sum"), delta_usd=("dollar_delta", "sum"),
        gamma_usd=("dollar_gamma", "sum")).reindex(STOCKS)
    table["delta_usd"] = table.delta_usd.map(lambda x: f"{x/1e6:.2f}")
    table["gamma_usd"] = table.gamma_usd.map(lambda x: f"{x/1e6:.2f}")
    table.columns = ["Short contracts", "Option delta\n(USDm)", "Gamma x spot²\n(USDm)"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, data, title in ((axes[0, 0], curves, "Correlated SPY: value versus current delta"),
                            (axes[0, 1], sleeves, "Stock and short-option contributions"),
                            (axes[1, 0], stock_options, "Short-option contribution by underlying")):
        qis.plot_line(data, ax=ax, title=title, xlabel="SPY simple return",
                      ylabel="P&L / current net portfolio value", xvar_format="{:.0%}",
                      yvar_format="{:.0%}", linewidth=2, fontsize=10)
    qis.plot_df_table(table, ax=axes[1, 1], fontsize=10,
                      title="Short-option overlay: signed current Greeks")
    fig.suptitle("FCGL risk model: five stocks + ten short options", fontsize=16)
    fig.text(.06, .025, "Other ETFs follow conditional co-moves. Fixed IV, rates and TTM. "
             "Gamma is signed option gamma multiplied by spot squared; premiums are model prices.",
             fontsize=10)
    fig.tight_layout(rect=(.02, .06, .98, .94))
    return fig, curves.join(sleeves), stock_options


def run_example(cache_dir, output_dir=None, as_of=AS_OF, refresh=False):
    """Compute the reproducible book; optionally save a standard report and convexity exhibit."""
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if output is not None and output.exists():
        raise FileExistsError("Use a fresh example output directory")
    if not hasattr(qis, "InstrumentPortfolio"):
        raise RuntimeError("This manual example needs qis>=5.30.2 with instrument stress reports")
    prices, source = load_prices(cache_dir, as_of, refresh)
    model, date, history, diagnostics, snapshot, lasso = fit_risk_model(prices)
    cluster_inputs = cluster_report_inputs(snapshot)
    portfolio, inventory = build_portfolio(prices, model, date)
    requests, grids = scenario_inputs()
    result = qis.run_portfolio_stress_test(
        portfolio, requests, history, grids,
        qis.StressTestConfig(horizon_years=1/12, confidence=.95),
    )
    verification = verify_example(portfolio, result, inventory, snapshot, lasso)
    fig, curves, contributions = convexity_exhibit(portfolio, result, inventory)
    if output is not None:
        output.mkdir(parents=True)
        appendix = inventory[["expiry", "strike", "contracts", "iv", "unit_price"]].copy()
        for column in ("strike", "unit_price"):
            appendix[column] = appendix[column].map(lambda value: f"{value:.2f}")
        appendix.iv = appendix.iv.map(lambda value: f"{value:.1%}")
        config = qis.StressReportConfig(
            title="Five-stock short-option portfolio", model_name="FCGL",
            model_label="SPY / TLT / GLD / USO; FCGL; weekly span 52",
            selected_grids=FACTORS, **cluster_inputs,
            response_diagnostics=diagnostics, write_previews=True,
            appendix_table=appendix, appendix_title="Synthetic option terms and VOP prices",
            appendix_notes=(
                "Bloomberg-style IDs are synthetic teaching contracts, not fetched quotes.",
                "100 shares/contract. Covered calls; puts are 1.5x stock lots, rounded down.",
                "Fixed-IV European proxy; American exercise, margin and volatility shocks omitted.",
            ),
            notes=("Observed Yahoo stock/ETF history; synthetic option positions and model prices.",
                   "Risk uses adjusted closes; option marks use unadjusted closing spot quotes.",
                   "Local Gaussian bands exclude gamma, vega, parameter and regime uncertainty."),
        )
        artifacts = qis.generate_portfolio_stress_report(result, output / "report", config)
        fig.savefig(output / "short_convexity.pdf")
        fig.savefig(output / "short_convexity.png", dpi=150)
        inventory.to_csv(output / "option_inventory.csv")
        diagnostics.to_csv(output / "fit_diagnostics.csv")
        curves.to_csv(output / "spy_convexity.csv")
        contributions.to_csv(output / "spy_option_contributions.csv")
        prices.to_csv(output / "source_prices.csv")
        snapshot.y_betas.to_csv(output / "factor_loadings.csv")
        snapshot.x_covar.to_csv(output / "factor_covariance.csv")
        snapshot.clusters.to_csv(output / "fitted_clusters.csv")
        snapshot.linkages.to_csv(output / "fitted_linkage.csv")
        snapshot.cutoffs.to_csv(output / "fitted_cutoffs.csv")
        (output / "cluster_labels.json").write_text(
            json.dumps(cluster_inputs["cluster_labels"], indent=2), encoding="utf-8")
        provenance = {"source": source, "valuation_date": str(portfolio.valuation_date.date()),
            "risk_date": str(date.date()), "span_weeks": SPAN, "annualisation": ANNUALISATION,
            "reference_currency": "USD", "net_marked_value": portfolio.reporting_denominator,
            "optimalportfolios_distribution_version": version("optimalportfolios"),
            "factorlasso_version": version("factorlasso"),
            "fcgl": {"reg_lambda": REG_LAMBDA, "n_clusters": N_CLUSTERS,
                     "demean": False, "group_penalty": "normalized", "l1_weight": 0.0},
            "qis_distribution_version": version("qis"),
            "qis_file": str(Path(qis.__file__).resolve()),
            "source_sha256": {package.__name__: {
                str(path.relative_to(Path(package.__file__).parent)):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(package.__file__).parent.rglob("*.py"))}
                for package in (opt, fl, qis)},
            "vop_version": version("vanilla-option-pricers"),
            "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "verification": verification}
        (output / "example_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        print("Report:", artifacts.pdf_path)
        print("Convexity exhibit:", output / "short_convexity.pdf")
    plt.close(fig)
    print(f"Valuation {portfolio.valuation_date.date()}; risk {date.date()}; "
          f"net marked value USD {portfolio.reporting_denominator:,.0f}; "
          "5 stocks + 5 short calls + 5 short puts; verification passed.")
    print("Fitted clusters:", snapshot.clusters.to_dict())
    print(curves.loc[[-.3, -.2, -.1, 0., .1, .2, .3]].to_string(float_format="{:.2%}".format))
    return portfolio, result, inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    default_cache = cache_root / "optimalportfolios/options_stress"
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--as-of", default=AS_OF)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run_example(args.cache_dir, args.output_dir, args.as_of, args.refresh)
