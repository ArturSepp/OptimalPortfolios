"""Canonical script of docs/stress_testing_with_options.md.

The page's four Python blocks are excerpts of ``main`` and run here in the same order. They build
a synthetic three-stock book with short calls and puts on a fixed factor model and stress it
through ``optimalportfolios.build_risk_model`` and the QIS instrument stress interface. Every
equation of the page that the book exercises, and every number the page states about it, is
asserted after the blocks against a reference computed a different way: option values by
numerical integration of the lognormal law, deltas by finite differences, put-call parity,
conditional completion by an explicit Schur-complement solve, and the conditional bands and
Euler contributions as explicit sums. The script runs offline after ``pip install
optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.stress_testing_with_options

The page's market-data workflow, with FCGL estimation and Yahoo prices, is the manual runner
``examples/reports/stress_testing_with_options_local.py``; this script does not reproduce it.

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import numpy as np
import pandas as pd

FACTORS = ['Equity', 'Rates', 'Commodities']
# Annual volatilities and correlations of the factor log returns.
FACTOR_VOLS = [0.16, 0.07, 0.25]
FACTOR_CORR = [[1.0, -0.3, 0.3],
               [-0.3, 1.0, -0.1],
               [0.3, -0.1, 1.0]]
STOCKS = ['Stock A', 'Stock B', 'Stock C']
# Log-return loadings of each stock on the factors, and annual residual volatilities.
LOADINGS = [[1.2, -0.3, 0.0],
            [0.9, 0.2, 0.1],
            [1.4, -0.5, 0.2]]
RESIDUAL_VOLS = [0.20, 0.15, 0.25]
SPOTS = [152.0, 83.0, 236.0]
BUDGET_PER_STOCK = 1_000_000  # USD; the book holds the largest 100-share lot count within it
CALL_MONEYNESS = [1.05, 1.04, 1.06]
PUT_MONEYNESS = [0.95, 0.96, 0.94]
MATURITIES = [0.25, 0.40, 0.60]  # years
CALL_VOLS = [0.28, 0.24, 0.36]
PUT_VOL_SPREAD = 0.04  # put volatility = call volatility + 4 volatility points
PUT_RATIO = 1.5  # put contracts = floor(1.5 x stock lots); call contracts = stock lots
RATE = 0.04  # continuously compounded; no dividends
MULTIPLIER = 100
EQUITY_MOVES = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]  # the scenarios of the left panel
GRID_STEP = 0.01  # the right panel's Equity grid runs from -30% to +30% in these steps


def factor_covariance() -> np.ndarray:
    """Return the annual factor covariance from FACTOR_VOLS and FACTOR_CORR."""
    vols = np.array(FACTOR_VOLS)
    return np.outer(vols, vols) * np.array(FACTOR_CORR)


def completed_shock(anchors: dict) -> np.ndarray:
    """Complete simple-return anchors into a full log-shock vector with an explicit solve.

    Each anchor becomes z_A = log(1 + a); the free factors take Sigma_FA Sigma_AA^-1 z_A.

    Args:
        anchors: Factor name to simple-return anchor, including explicit zeros.

    Returns:
        Log shocks in FACTORS order.
    """
    covar = factor_covariance()
    fixed = [FACTORS.index(name) for name in anchors]
    z_fixed = np.log1p(np.array(list(anchors.values()), dtype=float))
    shock = covar[:, fixed] @ np.linalg.solve(covar[np.ix_(fixed, fixed)], z_fixed)
    shock[fixed] = z_fixed
    return shock


def conditional_covariance(anchored: list) -> np.ndarray:
    """Return the Schur complement Sigma_FF - Sigma_FA Sigma_AA^-1 Sigma_AF in full order."""
    covar = factor_covariance()
    fixed = [FACTORS.index(name) for name in anchored]
    result = covar - covar[:, fixed] @ np.linalg.solve(covar[np.ix_(fixed, fixed)],
                                                       covar[fixed, :])
    result[fixed, :] = 0.0
    result[:, fixed] = 0.0
    return result


def lognormal_value(spot: float, strike: float, ttm: float, vol: float, kind: str) -> float:
    """Price a European option per share by integrating its payoff over the lognormal law.

    The terminal spot is spot * exp((RATE - vol^2 / 2) ttm + vol sqrt(ttm) x) with x standard
    normal; the value is the discounted expected payoff. No closed form is used.
    """
    from scipy.integrate import quad

    drift = (RATE - 0.5 * vol**2) * ttm
    stdev = vol * np.sqrt(ttm)
    sign = 1.0 if kind == 'C' else -1.0

    def integrand(x: float) -> float:
        """Payoff at the standard normal draw x times its density."""
        terminal = spot * np.exp(drift + stdev * x)
        return max(sign * (terminal - strike), 0.0) * np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    kink = (np.log(strike / spot) - drift) / stdev
    limits = (kink, kink + 12.0) if kind == 'C' else (kink - 12.0, kink)
    value, _ = quad(integrand, *limits, epsabs=1e-13, epsrel=1e-13, limit=200)
    return float(np.exp(-RATE * ttm) * value)


def reference_book() -> pd.DataFrame:
    """Return the book's terms, one row per holding, rebuilt from the module constants.

    Columns: underlying, kind (S, C or P), strike, ttm, vol and units, the signed number of
    shares for a stock and of shares under the contracts for an option.
    """
    rows = {}
    for stock, spot, call_m, put_m, ttm, vol in zip(STOCKS, SPOTS, CALL_MONEYNESS,
                                                    PUT_MONEYNESS, MATURITIES, CALL_VOLS):
        lots = int(BUDGET_PER_STOCK / (MULTIPLIER * spot))
        rows[stock] = dict(underlying=stock, kind='S', strike=np.nan, ttm=np.nan, vol=np.nan,
                           units=MULTIPLIER * lots)
        for kind, moneyness, contracts, option_vol in (
                ('C', call_m, -lots, vol), ('P', put_m, -int(PUT_RATIO * lots),
                                            vol + PUT_VOL_SPREAD)):
            strike = float(5 * np.round(spot * moneyness / 5))
            rows[f'{stock} {kind}{strike:g}'] = dict(
                underlying=stock, kind=kind, strike=strike, ttm=ttm, vol=option_vol,
                units=MULTIPLIER * contracts)
    return pd.DataFrame.from_dict(rows, orient='index')


def unit_value(row: pd.Series, spot: float) -> float:
    """Value of one unit of a holding at a spot: the spot, or the integrated option value."""
    if row['kind'] == 'S':
        return spot
    return lognormal_value(spot, row['strike'], row['ttm'], row['vol'], row['kind'])


def unit_delta(row: pd.Series, spot: float) -> float:
    """Spot delta of one unit by a central finite difference of ``unit_value``."""
    step = 1e-5 * spot
    return (unit_value(row, spot + step) - unit_value(row, spot - step)) / (2 * step)


def reference_valuation(shocks: dict) -> dict:
    """Revalue the book under complete log-shock vectors without qis or a closed form.

    S_i(z) = S_i(0) exp(B_i z); each holding's P&L is units x (value(S(z)) - value(S(0))),
    and its delta part is units x delta(S(0)) x (S(z) - S(0)).

    Args:
        shocks: Scenario label to a log-shock vector in FACTORS order.

    Returns:
        ``pnl`` and ``delta_part``, scenarios by holdings in USD, and the ``book``.
    """
    book = reference_book()
    spots = pd.Series(SPOTS, index=STOCKS)
    base = {name: unit_value(row, spots[row['underlying']]) for name, row in book.iterrows()}
    deltas = {name: unit_delta(row, spots[row['underlying']]) for name, row in book.iterrows()}
    pnl, delta_part = {}, {}
    for label, shock in shocks.items():
        stressed = spots * np.exp(np.array(LOADINGS) @ np.asarray(shock))
        pnl[label] = {name: row['units'] * (unit_value(row, stressed[row['underlying']])
                                            - base[name]) for name, row in book.iterrows()}
        delta_part[label] = {name: row['units'] * deltas[name]
                             * (stressed[row['underlying']] - spots[row['underlying']])
                             for name, row in book.iterrows()}
    return {'pnl': pd.DataFrame.from_dict(pnl, orient='index'),
            'delta_part': pd.DataFrame.from_dict(delta_part, orient='index'), 'book': book}


def options_delta_line(equity_moves) -> np.ndarray:
    """The options' delta part, as a fraction of N, for conditionally completed Equity moves."""
    book = reference_book()
    options = book[book['kind'] != 'S']
    spots = pd.Series(SPOTS, index=STOCKS)
    sensitivity = pd.Series(0.0, index=STOCKS)
    for _, row in options.iterrows():
        sensitivity[row['underlying']] += row['units'] * unit_delta(row, spots[row['underlying']])
    moves = np.array([spots * np.exp(np.array(LOADINGS) @ completed_shock({'Equity': move}))
                      - spots for move in equity_moves])
    return moves @ sensitivity.to_numpy() / reference_denominator()


def reference_denominator() -> float:
    """Sum of the book's signed marks at the current spots, from the integrated values."""
    book = reference_book()
    spots = pd.Series(SPOTS, index=STOCKS)
    return float(sum(row['units'] * unit_value(row, spots[row['underlying']])
                     for _, row in book.iterrows()))


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd
    import qis
    from factorlasso import CurrentFactorCovarData
    import optimalportfolios as opt

    factors = ["Equity", "Rates", "Commodities"]
    vols = np.array([0.16, 0.07, 0.25])
    correlation = np.array([[1.0, -0.3, 0.3],
                            [-0.3, 1.0, -0.1],
                            [0.3, -0.1, 1.0]])
    stocks = ["Stock A", "Stock B", "Stock C"]
    loadings = pd.DataFrame([[1.2, -0.3, 0.0],
                             [0.9, 0.2, 0.1],
                             [1.4, -0.5, 0.2]], index=stocks, columns=factors)
    snapshot = CurrentFactorCovarData(
        x_covar=pd.DataFrame(np.outer(vols, vols) * correlation,
                             index=factors, columns=factors),
        y_betas=loadings,
        y_variances=pd.DataFrame({"residual_var": [0.20**2, 0.15**2, 0.25**2]},
                                 index=stocks),
    )
    date = pd.Timestamp("2025-12-31")  # Synthetic snapshot label.
    risk_model = opt.build_risk_model({date: snapshot})

    # The block's inputs are the module constants: factor volatilities of 16%, 7% and 25%, and
    # residual volatilities of 20%, 15% and 25%.
    assert factors == FACTORS and stocks == STOCKS
    assert vols.tolist() == FACTOR_VOLS == [0.16, 0.07, 0.25]
    assert correlation.tolist() == FACTOR_CORR and RESIDUAL_VOLS == [0.20, 0.15, 0.25]
    np.testing.assert_array_equal(loadings, LOADINGS)
    np.testing.assert_array_equal(snapshot.x_covar, factor_covariance())
    residual_vars = np.array(RESIDUAL_VOLS) ** 2
    # The adapter keeps B, Sigma_F and d, and assembles Sigma_Y = B Sigma_F B' + diag(d).
    betas = np.array(LOADINGS)
    np.testing.assert_array_equal(risk_model.factor_loadings[date], betas)
    np.testing.assert_array_equal(risk_model.factor_covar[date], factor_covariance())
    np.testing.assert_array_equal(risk_model.residual_vars[date], residual_vars)
    np.testing.assert_allclose(risk_model.covar[date],
                               betas @ factor_covariance() @ betas.T + np.diag(residual_vars),
                               rtol=0.0, atol=1e-15)

    from dataclasses import dataclass

    from scipy.stats import norm

    def bsm(spot, strike, ttm, vol, rate, kind):
        """Black-Scholes-Merton value and spot delta per share, no dividends."""
        spot = np.asarray(spot, dtype=float)
        stdev = vol * np.sqrt(ttm)
        d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / stdev
        d2 = d1 - stdev
        pv_strike = strike * np.exp(-rate * ttm)
        if kind == "C":
            return spot * norm.cdf(d1) - pv_strike * norm.cdf(d2), norm.cdf(d1)
        return pv_strike * norm.cdf(-d2) - spot * norm.cdf(-d1), norm.cdf(d1) - 1.0

    @dataclass(frozen=True)
    class EuropeanOption:
        """Signed European option contracts on one stock, at fixed volatility."""

        underlying: str
        kind: str
        strike: float
        ttm: float
        vol: float
        contracts: int
        rate: float = 0.04
        multiplier: int = 100
        implementation_id = "docs.bsm_european_option.v1"
        coverage = "European BSM value; volatility, rate and maturity held fixed."
        boundary_policy = "Smooth BSM delta; no expiry crossing."

        def evaluate(self, context):
            """Signed position value in each scenario, in USD."""
            spots = context.quotes[self.underlying]
            value, _ = bsm(spots.to_numpy(), self.strike, self.ttm, self.vol,
                           self.rate, self.kind)
            return pd.Series(self.contracts * self.multiplier * value, index=spots.index)

        def dollar_delta(self, context, spot):
            """Dollar sensitivity to each shared log response at one spot."""
            _, delta = bsm(spot, self.strike, self.ttm, self.vol, self.rate, self.kind)
            dollars = self.contracts * self.multiplier * float(delta) * spot
            return dollars * context.quote_response_jacobian.loc[self.underlying]

        def response_jacobian(self, context):
            """Current dollar delta, for QIS exposures and risk."""
            return self.dollar_delta(context, context.baseline_quotes[self.underlying])

        def scenario_response_jacobian(self, context):
            """Dollar delta at the stressed spot, for the conditional bands."""
            return self.dollar_delta(context, context.quotes.iloc[0][self.underlying])

    # The closed form against the integrated risk-neutral expectation, put-call parity at
    # every strike, and each delta against a central difference; values are convex in spot.
    book = reference_book()
    spots_0 = pd.Series(SPOTS, index=STOCKS)
    options_book = book[book["kind"] != "S"]
    for name, row in options_book.iterrows():
        spot = spots_0[row["underlying"]]
        value, delta = bsm(spot, row["strike"], row["ttm"], row["vol"], RATE, row["kind"])
        integrated = lognormal_value(spot, row["strike"], row["ttm"], row["vol"], row["kind"])
        assert abs(value - integrated) < 1e-9 * spot, name
        call, _ = bsm(spot, row["strike"], row["ttm"], row["vol"], RATE, "C")
        put, _ = bsm(spot, row["strike"], row["ttm"], row["vol"], RATE, "P")
        assert abs(call - put - (spot - row["strike"] * np.exp(-RATE * row["ttm"]))) < 1e-11
        step = 1e-5 * spot
        up, _ = bsm(spot + step, row["strike"], row["ttm"], row["vol"], RATE, row["kind"])
        down, _ = bsm(spot - step, row["strike"], row["ttm"], row["vol"], RATE, row["kind"])
        assert abs(delta - (up - down) / (2 * step)) < 1e-9, name
        assert up + down - 2 * value > 0.0  # positive gamma for the long option

    spots = pd.Series([152.0, 83.0, 236.0], index=stocks)
    call_moneyness = pd.Series([1.05, 1.04, 1.06], index=stocks)
    put_moneyness = pd.Series([0.95, 0.96, 0.94], index=stocks)
    maturities = pd.Series([0.25, 0.40, 0.60], index=stocks)  # years
    call_vols = pd.Series([0.28, 0.24, 0.36], index=stocks)

    holdings, share_delta = [], pd.Series(0.0, index=stocks)
    for stock in stocks:
        spot = spots[stock]
        lots = int(1_000_000 / (100 * spot))
        holdings.append(qis.PortfolioHolding(
            stock, stock, 100 * lots * spot,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, stock, 100 * lots),),
        ))
        share_delta[stock] += 100 * lots
        for kind, moneyness, contracts, vol in (
            ("C", call_moneyness[stock], -lots, call_vols[stock]),
            ("P", put_moneyness[stock], -int(1.5 * lots), call_vols[stock] + 0.04),
        ):
            strike = float(5 * np.round(spot * moneyness / 5))
            option = EuropeanOption(stock, kind, strike, maturities[stock], vol, contracts)
            value, delta = bsm(spot, strike, option.ttm, vol, option.rate, kind)
            holding_id = f"{stock} {kind}{strike:g}"
            holdings.append(qis.PortfolioHolding(
                holding_id, holding_id, contracts * 100 * float(value), payoff=option,
            ))
            share_delta[stock] += contracts * 100 * float(delta)

    portfolio = qis.InstrumentPortfolio(
        holdings=tuple(holdings),
        underlyings={stock: qis.Underlying(stock, spots[stock], "USD", stock,
                                           qis.ResponseBasis.LOCAL) for stock in stocks},
        risk_model=risk_model,
        risk_date=date,
        valuation_date=date,
        reference_currency="USD",
        reporting_denominator=sum(holding.observed_mtm for holding in holdings),
        denominator_label="Net marked portfolio value",
    )
    print(pd.Series({h.holding_id: h.observed_mtm for h in holdings}).round(0))
    print(f"N = USD {portfolio.reporting_denominator:,.0f}")

    # The book of the constants: three stocks, a short call and a short put on each, strikes
    # rounded to USD 5, and the page's lots, strikes and contracts.
    assert spots.tolist() == SPOTS and call_moneyness.tolist() == CALL_MONEYNESS
    assert put_moneyness.tolist() == PUT_MONEYNESS and maturities.tolist() == MATURITIES
    assert call_vols.tolist() == CALL_VOLS
    ids = [holding.holding_id for holding in holdings]
    assert ids == book.index.tolist() and len(ids) == 9
    assert ids == ["Stock A", "Stock A C160", "Stock A P145", "Stock B", "Stock B C85",
                   "Stock B P80", "Stock C", "Stock C C250", "Stock C P220"]
    assert book.loc[book["kind"] == "S", "units"].tolist() == [6500, 12000, 4200]
    assert (book.loc[book["kind"] == "C", "units"] / 100).tolist() == [-65, -120, -42]
    assert (book.loc[book["kind"] == "P", "units"] / 100).tolist() == [-97, -180, -63]
    payoffs = {h.holding_id: h.payoff for h in holdings if h.payoff is not None}
    assert all(option.contracts < 0 for option in payoffs.values())
    assert [option.kind for option in payoffs.values()].count("P") == 3
    for name, option in payoffs.items():
        assert option.strike == book.at[name, "strike"] and option.vol == book.at[name, "vol"]
        assert option.contracts * option.multiplier == book.at[name, "units"]
    # Marks equal the integrated option values; N is their signed sum.
    marks = pd.Series({h.holding_id: h.observed_mtm for h in holdings})
    for name, row in book.iterrows():
        reference_mark = row["units"] * unit_value(row, SPOTS[STOCKS.index(row["underlying"])])
        assert abs(marks[name] - reference_mark) < 1e-6, name
    denominator = portfolio.reporting_denominator
    assert abs(denominator - reference_denominator()) < 1e-5
    assert round(denominator) == 2_544_055
    # The net share-equivalent delta of each stock, against finite-difference deltas.
    reference_delta = pd.Series(0.0, index=STOCKS)
    for name, row in book.iterrows():
        reference_delta[row["underlying"]] += row["units"] * unit_delta(
            row, SPOTS[STOCKS.index(row["underlying"])])
    np.testing.assert_allclose(share_delta, reference_delta, rtol=0.0, atol=1e-5)
    # Zero shock preserves every mark: the model baseline equals the mark, with no offset.
    zero = portfolio.evaluate(pd.DataFrame(0.0, index=["No move"], columns=factors))
    assert (zero.pnl.to_numpy() == 0.0).all()
    np.testing.assert_allclose(zero.mtm.iloc[0], marks, rtol=0.0, atol=0.0)

    equity_moves = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    scenarios = qis.StressScenarios(
        pd.DataFrame({"Equity": equity_moves},
                     index=[f"Equity {move:+.0%}" for move in equity_moves]),
        mode=qis.ScenarioMode.CONDITIONAL,
        convention=qis.ShockConvention.SIMPLE,
    )
    result = qis.run_portfolio_stress_test(
        portfolio, scenarios, factor_grids={"Equity": scenarios},
        config=qis.StressTestConfig(horizon_years=1 / 12),
    )
    valuation = result.valuations["requested"]
    spot_moves = np.exp(valuation.factor_log_shocks @ loadings.T) * spots - spots
    delta_pnl = (spot_moves * share_delta).sum(axis=1)
    total_pnl = valuation.portfolio_pnl
    split = pd.DataFrame({
        "Delta": delta_pnl,
        "Gamma": total_pnl - delta_pnl,
        "Total": total_pnl,
    }) / portfolio.reporting_denominator
    print(split.map("{:.2%}".format))

    # Simple anchors become log shocks z_E = log(1 + a); the free factors follow
    # Sigma_FA Sigma_AA^-1 z_A, from an explicit solve.
    shocks = valuation.factor_log_shocks
    labels = shocks.index.tolist()
    np.testing.assert_array_equal(shocks["Equity"], np.log1p(equity_moves))
    completed = np.array([completed_shock({"Equity": move}) for move in equity_moves])
    np.testing.assert_allclose(shocks, completed, rtol=0.0, atol=1e-15)
    # Marks equal the model prices, so no basis offset is added and no premium cash either.
    assert (result.positions["basis_offset"] == 0.0).all()
    assert result.positions.index.tolist() == ids
    # S_i(z) = S_i(0) exp(B_i z) and V(z) - V(0) per holding, against integrated values; the
    # stocks' P&L is shares times the spot change.
    reference = reference_valuation(dict(zip(labels, completed)))
    np.testing.assert_allclose(valuation.pnl, reference["pnl"], rtol=0.0, atol=1e-6)
    stock_pnl = valuation.pnl[STOCKS]
    np.testing.assert_allclose(stock_pnl, spot_moves * book.loc[STOCKS, "units"], rtol=1e-13,
                               atol=1e-8)
    # R_p = sum of P&L / N, as QIS reports it.
    np.testing.assert_allclose(result.summaries["requested"]["portfolio_return"],
                               reference["pnl"].sum(axis=1) / denominator, rtol=0.0, atol=1e-12)
    # The split: the delta part is linear in the spot moves, the gamma part is the options'
    # repricing error of that line; they add to the total, and the gamma part is never
    # positive because every contract is short and each option value is convex in spot.
    reference_split = pd.DataFrame({
        "Delta": reference["delta_part"].sum(axis=1),
        "Gamma": reference["pnl"].sum(axis=1) - reference["delta_part"].sum(axis=1),
        "Total": reference["pnl"].sum(axis=1)}) / denominator
    np.testing.assert_allclose(split, reference_split, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(split["Delta"] + split["Gamma"], split["Total"], rtol=0.0,
                               atol=1e-15)
    options = list(payoffs)
    option_gamma = (reference["pnl"][options] - reference["delta_part"][options]).sum(axis=1)
    np.testing.assert_allclose(split["Gamma"], option_gamma / denominator, rtol=0.0, atol=1e-10)
    moved = split.drop(index="Equity +0%")
    assert (moved["Gamma"] < 0.0).all() and (split.loc["Equity +0%"] == 0.0).all()
    # The page's table, in percent of N.
    table = [[-42.38, -33.41, -75.79],
             [-28.73, -16.56, -45.28],
             [-14.59, -4.37, -18.96],
             [0.00, 0.00, 0.00],
             [15.02, -3.96, 11.06],
             [30.44, -14.02, 16.42],
             [46.26, -27.54, 18.72]]
    np.testing.assert_allclose(100 * split, table, rtol=0.0, atol=0.005)
    # Insight: at +20% the gamma part gives back almost half of the delta gain.
    give_back = -split.at["Equity +20%", "Gamma"] / split.at["Equity +20%", "Delta"]
    assert 0.45 < give_back < 0.5
    # The figure: the options alone reprice to about -34% of N at -30% and -27% at +30%,
    # while their delta line stays within 1% of zero.
    option_total = reference["pnl"][options].sum(axis=1) / denominator
    option_delta = reference["delta_part"][options].sum(axis=1) / denominator
    assert round(100 * option_total["Equity -30%"]) == -34
    assert round(100 * option_total["Equity +30%"]) == -27
    assert (option_delta.abs() < 0.01).all()
    assert abs(options_delta_line(np.arange(-30, 31) * GRID_STEP)).max() < 0.01

    # QIS's own attribution reconciles: its factor columns hold the stocks' exact P&L and the
    # options' log-quote deltas times B z, and its nonlinear column holds the rest.
    attribution = result.attribution["requested"]
    np.testing.assert_allclose(attribution.sum(axis=1), total_pnl, rtol=0.0, atol=1e-7)
    log_moves = shocks @ loadings.T
    log_delta = {name: option.contracts * 100 * unit_delta(book.loc[name], SPOTS[
        STOCKS.index(option.underlying)]) * SPOTS[STOCKS.index(option.underlying)]
        for name, option in payoffs.items()}
    nonlinear = sum(reference["pnl"][name] - log_delta[name] * log_moves[option.underlying]
                    for name, option in payoffs.items())
    np.testing.assert_allclose(attribution["Nonlinear payoff adjustment"], nonlinear, rtol=0.0,
                               atol=1e-3)
    # The current log-delta line, shocks times current factor exposures, is B' J z summed over
    # holdings; at Equity -30% it gives -52.81% against -75.79% from full repricing.
    current_dollars = {name: row["units"] * unit_delta(row, spots_0[row["underlying"]])
                       * spots_0[row["underlying"]] for name, row in book.iterrows()}
    explicit_log_delta = sum(current_dollars[name] * log_moves[row["underlying"]]
                             for name, row in book.iterrows()) / denominator
    log_delta_line = shocks @ result.factor_exposures / denominator
    np.testing.assert_allclose(log_delta_line, explicit_log_delta, rtol=0.0, atol=1e-10)
    assert round(100 * log_delta_line["Equity -30%"], 2) == -52.81

    # Conditional bands: v(z) = h [e' Sigma_F|A e + sum_i q_i^2 d_i] with q_i(z) = J_i(z) / N the
    # stock and both options' dollar deltas at the stressed spots, e = B' q and h = 1/12.
    bands = result.grid_summaries["Equity"]
    free_covar = conditional_covariance(["Equity"])
    for label, shock in zip(labels, completed):
        stressed = spots_0 * np.exp(betas @ shock)
        exposure = pd.Series(0.0, index=STOCKS)
        for name, row in book.iterrows():
            spot = stressed[row["underlying"]]
            exposure[row["underlying"]] += row["units"] * unit_delta(row, spot) * spot
        q = exposure.to_numpy() / denominator
        e = betas.T @ q
        variance = (e @ free_covar @ e + (q**2 * residual_vars).sum()) / 12.0
        assert abs(bands.at[label, "conditional_vol_horizon"] - np.sqrt(variance)) < 1e-9
        # Summing one residual per holding instead of per stock would change the risk.
        per_holding = sum((row["units"] * unit_delta(row, stressed[row["underlying"]])
                           * stressed[row["underlying"]] / denominator) ** 2
                          * residual_vars[STOCKS.index(row["underlying"])]
                          for _, row in book.iterrows())
        assert abs(per_holding - (q**2 * residual_vars).sum()) > 1e-6
    sigma = bands["conditional_vol_horizon"]
    for k in (1, 2):
        np.testing.assert_allclose(bands[f"lower_{k}sigma"], bands["portfolio_return"] - k * sigma,
                                   rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(bands[f"upper_{k}sigma"], bands["portfolio_return"] + k * sigma,
                                   rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(bands["band_half_width"] / sigma, norm.ppf(0.975), rtol=1e-14)
    assert round(norm.ppf(0.975), 2) == 1.96
    # The page's band widths. At -20% the in-the-money short puts at least double every
    # stock's delta; at +30% the in-the-money short calls offset 87% or more of it.
    assert np.round(100 * sigma[["Equity +0%", "Equity -20%", "Equity +30%"]], 2).tolist() == [
        4.15, 6.47, 0.61]
    assert sigma.idxmax() == "Equity -20%"
    shares = book.loc[STOCKS, "units"].to_numpy()
    for label, low, high, kind in (("Equity -20%", 2.0, np.inf, "P"),
                                   ("Equity +30%", 0.0, 0.13, "C")):
        stressed = spots_0 * np.exp(betas @ completed[labels.index(label)])
        net = np.array([sum(row["units"] * unit_delta(row, stressed[stock])
                            for _, row in book[book["underlying"] == stock].iterrows())
                        for stock in STOCKS])
        assert ((net / shares > low) & (net / shares < high)).all(), label
        strikes = book[book["kind"] == kind].set_index("underlying")["strike"]
        moneyness = stressed / strikes.reindex(STOCKS)
        assert ((moneyness < 1.0) if kind == "P" else (moneyness > 1.0)).all(), label

    # An explicit zero stays an anchor; omitted factors are free. INDEPENDENT fills the free
    # factors with zeros, which is what filling the scenario table with zeros does.
    joint = qis.StressScenarios(pd.DataFrame({"Equity": [-0.20], "Rates": [0.0]}, index=["j"]),
                                mode=qis.ScenarioMode.CONDITIONAL,
                                convention=qis.ShockConvention.SIMPLE)
    np.testing.assert_allclose(joint.resolve(risk_model, date).loc["j"],
                               completed_shock({"Equity": -0.20, "Rates": 0.0}), atol=1e-15)
    independent = result.valuations["independent"].factor_log_shocks
    assert (independent[["Rates", "Commodities"]] == 0.0).all().all()
    np.testing.assert_array_equal(independent["Equity"], shocks["Equity"])
    # Pitfall: at Equity -20% the completed moves are Rates +2.97% and Commodities -9.93%;
    # zero-filling them reports -42.41% instead of -45.28%.
    assert np.round(100 * np.expm1(shocks.loc["Equity -20%", ["Rates", "Commodities"]]),
                    2).tolist() == [2.97, -9.93]
    zero_filled = result.summaries["independent"]["portfolio_return"]
    np.testing.assert_allclose(zero_filled, reference_valuation(dict(zip(
        labels, independent.to_numpy())))["pnl"].sum(axis=1) / denominator, atol=1e-12)
    assert round(100 * zero_filled["Equity -20%"], 2) == -42.41

    # Historical factor vectors are complete, so they are replayed without reconditioning.
    history = pd.DataFrame([[-0.05, 0.02, 0.01], [0.03, -0.01, -0.04]],
                           index=pd.to_datetime(["2025-10-31", "2025-11-30"]), columns=factors)
    replay = qis.run_portfolio_stress_test(portfolio, scenarios, history,
                                           config=qis.StressTestConfig(horizon_years=1 / 12))
    pd.testing.assert_frame_equal(replay.historical.factor_log_shocks, history,
                                  check_freq=False)
    np.testing.assert_allclose(replay.historical.pnl, reference_valuation(
        dict(zip(history.index, history.to_numpy())))["pnl"], rtol=0.0, atol=1e-6)

    # Euler contributions over a two-group partition add up to the portfolio volatility.
    current = np.array([sum(row["units"] * unit_delta(row, spots_0[stock]) * spots_0[stock]
                            for _, row in book[book["underlying"] == stock].iterrows())
                        for stock in STOCKS]) / denominator
    exposure_all = betas.T @ current
    sigma_p = np.sqrt(exposure_all @ factor_covariance() @ exposure_all
                      + (current**2 * residual_vars).sum())
    assert abs(result.risk["annual_total_vol"] - sigma_p) < 1e-9
    groups = {"G1": [0, 1], "G2": [2]}
    contributions = sum(
        (betas[members].T @ current[members]) @ (factor_covariance() @ exposure_all) / sigma_p
        + (current[members] ** 2 * residual_vars[members]).sum() / sigma_p
        for members in groups.values())
    assert abs(contributions - sigma_p) < 1e-12
    print("stress_testing_with_options: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the delta and gamma parts by scenario, and the option curve.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    moves = np.round(np.arange(-30, 31) * GRID_STEP * 100) / 100
    reference = reference_valuation({move: completed_shock({'Equity': move}) for move in moves})
    book = reference['book']
    options = book.index[book['kind'] != 'S']
    denominator = reference_denominator()
    pnl, delta_part = reference['pnl'] / denominator, reference['delta_part'] / denominator
    table = pd.DataFrame({
        'portfolio_delta': delta_part.sum(axis=1),
        'portfolio_gamma': pnl.sum(axis=1) - delta_part.sum(axis=1),
        'portfolio_total': pnl.sum(axis=1),
        'options_delta': delta_part[options].sum(axis=1),
        'options_total': pnl[options].sum(axis=1),
    })
    table['options_gamma'] = table['options_total'] - table['options_delta']
    table['left_panel'] = [bool(np.isclose(move, EQUITY_MOVES).any()) for move in table.index]
    table.index.name = 'equity_simple_return'
    bars = table[table['left_panel']]

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    x = np.arange(len(bars))
    width = 0.38
    left.bar(x - width / 2, bars['portfolio_delta'], width, color=blue, label='Delta part')
    left.bar(x + width / 2, bars['portfolio_gamma'], width, color=orange, label='Gamma part')
    left.scatter(x, bars['portfolio_total'], color=ink, s=28, zorder=3,
                 label='Total (full repricing)')
    left.axhline(0.0, color=muted, linewidth=0.8)
    # The unicode minus matches the percent formatter of the right panel's axis.
    left.set_xticks(x, [f'{move:+.0%}'.replace('-', '−') for move in bars.index])
    left.set_xlabel('Equity simple return')
    left.set_title('Book P&L by Equity scenario, % of N', loc='left', color=ink)
    handles, names = left.get_legend_handles_labels()
    order = [names.index(name) for name in ('Delta part', 'Gamma part',
                                            'Total (full repricing)')]
    left.legend([handles[i] for i in order], [names[i] for i in order], frameon=False,
                loc='upper left', fontsize=9, labelcolor=ink)

    right.plot(table.index, table['options_delta'], color=blue, linewidth=1.8, linestyle='--')
    right.plot(table.index, table['options_total'], color=ink, linewidth=2.0)
    right.fill_between(table.index, table['options_total'], table['options_delta'],
                       color=orange, alpha=0.35, linewidth=0)
    right.text(0.29, table['options_delta'].iloc[-1], 'delta line', color=ink, fontsize=9,
               ha='right', va='bottom')
    right.text(-0.29, table['options_total'].iloc[0], 'full repricing', color=ink,
               fontsize=9, ha='left', va='top')
    right.text(0.0, -0.06, 'gamma part', color=ink, fontsize=9, ha='center')
    right.axhline(0.0, color=muted, linewidth=0.8)
    right.set_xlabel('Equity simple return')
    right.set_title('Short options against their delta line, % of N', loc='left', color=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    right.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda value, _: f'{value:+.0%}'.replace('-', '−') if round(value, 6) else '0%'))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    moved = table.drop(index=0.0)
    checks = {
        'parts_sum_to_total': bool(np.allclose(
            table['portfolio_delta'] + table['portfolio_gamma'], table['portfolio_total'],
            rtol=0.0, atol=1e-15)),
        'gamma_part_negative_off_zero': bool((moved['portfolio_gamma'] < 0.0).all()),
        'gamma_part_is_the_options': bool(np.allclose(
            table['portfolio_gamma'], table['options_gamma'], rtol=0.0, atol=1e-12)),
        'options_below_delta_line': bool((moved['options_total'] < moved['options_delta']).all()),
        'options_delta_line_within_one_percent': bool(
            (table['options_delta'].abs() < 0.01).all()
            and np.allclose(table['options_delta'], options_delta_line(table.index), rtol=0.0,
                            atol=1e-12)),
        'zero_shock_zero_pnl': bool(np.allclose(table.loc[0.0, ['portfolio_total',
                                                                  'options_total']], 0.0,
                                                rtol=0.0, atol=1e-15)),
        'left_panel_has_every_scenario': bool(len(bars) == len(EQUITY_MOVES)),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
