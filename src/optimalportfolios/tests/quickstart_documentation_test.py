"""Execute the canonical quickstart and check the guide's numerical and timing claims."""

import contextlib
import importlib
import inspect
import io
import json
import re
import runpy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import qis

from optimalportfolios import EwmaCovarEstimator, FactorCovarEstimator, PortfolioObjective


@pytest.fixture(scope="module")
def article(root):
    """Read the migrated source; the shared root fixture skips installed-wheel runs."""
    return (root / "docs/quickstart.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def canonical(root, article):
    """Resolve the page's include to the existing single source of runnable Python."""
    paths = re.findall(r"^\x60{3}\{literalinclude\} ([^\n]+)$", article, re.MULTILINE)
    assert len(paths) == 1
    script = (root / "docs" / paths[0]).resolve()
    assert script == (root / "examples/getting_started/production_quickstart.py").resolve()
    assert not re.search(r"^\x60{3}python", article, re.MULTILINE)
    return runpy.run_path(str(script))


@pytest.fixture(scope="module")
def pipeline(canonical):
    """Observe real solves and the real QIS backtest without substituting their results."""
    captured = {"outcomes": []}
    original_weights = canonical["compute_rolling_optimal_weights"]
    original_fit = EwmaCovarEstimator.fit_rolling_covars
    original_backtest = qis.backtest_model_portfolio
    quadratic = importlib.import_module("optimalportfolios.optimization.general.quadratic")
    original_single = quadratic.wrapper_quadratic_optimisation

    def observe_fit(self, *args, **kwargs):
        """Retain the actual estimator configuration and covariance output."""
        captured["estimator"] = self
        captured["fit_args"] = kwargs
        result = original_fit(self, *args, **kwargs)
        captured["covars"] = result
        return result

    def observe_weights(*args, **kwargs):
        """Record the caller's target settings and forward the numerical computation."""
        captured["inputs"] = kwargs
        result = original_weights(*args, **kwargs)
        captured["weights"] = result
        return result

    def observe_backtest(*args, **kwargs):
        """Retain the actual holdings, NAV and cost records."""
        captured["backtest_args"] = kwargs
        result = original_backtest(*args, **kwargs)
        captured["portfolio"] = result
        return result

    def observe_single(*args, **kwargs):
        """Keep the outcome for each real single-date optimization."""
        result = original_single(*args, **kwargs)
        captured["outcomes"].append(result[1])
        return result

    stdout = io.StringIO()
    with (
        contextlib.redirect_stdout(stdout),
        patch.dict(canonical["main"].__globals__,
                   {"compute_rolling_optimal_weights": observe_weights}),
        patch.object(EwmaCovarEstimator, "fit_rolling_covars", observe_fit),
        patch.object(qis, "backtest_model_portfolio", observe_backtest),
        patch.object(quadratic, "wrapper_quadratic_optimisation", observe_single),
    ):
        canonical["main"]()
    captured["stdout"] = stdout.getvalue()
    return captured


def setting(article, label):
    """Read a named row's value from the actual guide table."""
    match = re.search(r"^\| " + re.escape(label) + r" \| ([^|\n]+) \|",
                      article, re.MULTILINE)
    assert match, label
    return match[1].strip().replace(chr(96), "")


def numbers(value):
    """Read decimal constants without depending on inline Markdown formatting."""
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]


def test_utility_links_inventory_and_preserved_headings(root, article):
    """Keep portable attribution, source navigation and the legacy article fragments."""
    checker = runpy.run_path(str(root / "tools/check_docs.py"))
    assert not checker["check_document"](article, methodology=False)
    assert not checker["check_local_links"](article, root / "docs/quickstart.md", root)
    assert "# Quickstart\n" in article and "## What to change first\n" in article
    assert not (root / "docs/quickstart.rst").exists()
    inventory = json.loads((root / "tools/docs_inventory.json").read_text(encoding="utf-8"))
    assert inventory["pages"]["docs/quickstart.md"]["form"] == "utility"
    assert "docs/quickstart.rst" not in inventory["pages"]
    installation = (root / "docs/installation.md").read_text(encoding="utf-8")
    assert "(quickstart.rst)" not in installation
    assert installation.count("(quickstart.md)") == 2


def test_canonical_source_and_notebook_remain_one_workflow(root, article, canonical):
    """Exercise the existing notebook contract, including its released-package setup."""
    assert callable(canonical["main"])
    assert "(../examples/getting_started/production_quickstart.py)" in article
    assert "(../examples/getting_started/production_quickstart.ipynb)" in article
    checker = runpy.run_path(str(root / ".github/scripts/check_quickstart_notebook.py"))
    assert checker["main"]() == 0


def test_documented_sample_and_configuration_match_real_inputs(article, pipeline):
    """Tie dates, units, bounds and frequencies to the actual canonical call."""
    inputs = pipeline["inputs"]
    prices = inputs["prices"]
    weights = pipeline["weights"]
    assert setting(article, "Price sample") == (
        f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}"
    )
    assert setting(article, "Decision period") == (
        f"{weights.index[0]:%Y-%m-%d} to {weights.index[-1]:%Y-%m-%d}"
    )
    assert prices.shape == (156, 6) and weights.shape == (31, 6)
    assert setting(article, "Return frequency") == pipeline["estimator"].returns_freq
    assert int(setting(article, "Covariance span")) == pipeline["estimator"].span
    assert setting(article, "Decision frequency") == pipeline["fit_args"]["rebalancing_freq"]
    objective_name = inputs["portfolio_objective"].name
    assert setting(article, "Objective") == f"PortfolioObjective.{objective_name}"
    exposure = float(setting(article, "Target exposure"))
    constraints = inputs["constraints"]
    assert exposure == constraints.min_exposure == constraints.max_exposure
    lower, upper = numbers(setting(article, "Asset bounds"))
    assert constraints.is_long_only and lower == 0.0
    np.testing.assert_allclose(constraints.max_weights, upper)
    assert float(setting(article, "Trading cost rate")) == (
        pipeline["backtest_args"]["rebalancing_costs"]
    )


@pytest.mark.parametrize("position", [0, 15, -1])
def test_annualized_covariance_against_independent_weighted_sum(article, pipeline, position):
    """Verify the EWMA convention at first, intermediate and final decision dates."""
    date = pipeline["weights"].index[position]
    prices = pipeline["inputs"]["prices"].loc[:date]
    span = int(setting(article, "Covariance span"))
    log_returns = np.log(prices).diff().iloc[1:]
    centered = (log_returns - log_returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1.0 - 2.0 / (span + 1.0)
    stated_decay = re.search(r"\\lambda = [^$]+ = ([0-9.]+)\$", article)
    assert stated_decay and float(stated_decay[1]) == pytest.approx(decay)
    stated_factor = re.search(r"monthly covariance by (\d+)", article)
    assert stated_factor
    annualization = int(stated_factor[1])
    assert annualization == 12
    coefficients = (1 - decay) * decay ** np.arange(len(centered) - 1, -1, -1)
    # The QIS tensor is the production path under test. This test-only finite weighted sum,
    # with pandas mean demeaning, provides a separately computed reference instead.
    reference = annualization * np.einsum(
        "t,ti,tj->ij", coefficients, centered.to_numpy(), centered.to_numpy()
    )
    np.testing.assert_allclose(pipeline["covars"][date], reference, rtol=2e-12, atol=2e-15)


@pytest.mark.parametrize("position", [0, 15, -1])
def test_estimates_agree_with_historical_prefix_and_ignore_future(pipeline, position):
    """Check both truncated history and strongly perturbed future prices."""
    prices = pipeline["inputs"]["prices"]
    date = pipeline["weights"].index[position]
    period = qis.TimePeriod(pipeline["weights"].index[0], date)
    estimator = pipeline["estimator"]
    prefix = estimator.fit_rolling_covars(
        prices=prices.loc[:date], time_period=period, rebalancing_freq="QE"
    )
    changed = prices.copy()
    future = changed.index > date
    changed.loc[future] *= np.exp(np.linspace(0.5, 3.0, int(future.sum())))[:, None]
    changed_covars = estimator.fit_rolling_covars(
        prices=changed, time_period=period, rebalancing_freq="QE"
    )
    assert prefix and set(prefix) == set(changed_covars)
    for key in prefix:
        np.testing.assert_allclose(prefix[key], pipeline["covars"][key], rtol=0, atol=1e-14)
        np.testing.assert_allclose(changed_covars[key], pipeline["covars"][key],
                                   rtol=0, atol=1e-14)


def test_real_targets_are_accepted_compliant_and_bounded(article, pipeline):
    """A finite returned weight table must not hide fallback or failed solves."""
    outcomes = pipeline["outcomes"]
    weights = pipeline["weights"]
    assert len(outcomes) == len(weights) == 31
    assert all(item.accepted and item.compliant and item.fallback_source is None
               for item in outcomes)
    lower, upper = numbers(setting(article, "Asset bounds"))
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights >= lower - 1e-7).all().all()
    assert (weights <= upper + 1e-7).all().all()
    np.testing.assert_allclose(weights.sum(axis=1), float(setting(article, "Target exposure")),
                               rtol=0, atol=1e-8)


def test_timeline_uses_next_price_observation(article, pipeline):
    """Compare all real executions and each displayed event with the price-index mapping."""
    portfolio = pipeline["portfolio"]
    weights = pipeline["weights"]
    lag = int(setting(article, "Implementation lag"))
    assert lag == pipeline["backtest_args"]["weight_implementation_lag"]
    index = portfolio.prices.index
    executed = index.take(index.get_indexer(weights.index) + lag)
    actual = portfolio.is_rebalancing[portfolio.is_rebalancing].index
    pd.testing.assert_index_equal(actual, executed, check_names=False)
    first_endpoint = index[index.get_loc(executed[0]) + 1]
    expected = {
        "First decision": weights.index[0], "First execution": executed[0],
        "First invested return endpoint": first_endpoint, "Last decision": weights.index[-1],
        "Last execution": executed[-1], "End of sample": index[-1],
    }
    for event, date in expected.items():
        assert setting(article, event) == date.strftime("%Y-%m-%d")


def test_costs_and_entry_nav_have_independent_notional_reference(article, pipeline):
    """Use observed unit changes to check costs and the first invested holding return."""
    portfolio = pipeline["portfolio"]
    prices = portfolio.prices
    units = portfolio.units
    initial_nav = float(setting(article, "Initial NAV"))
    rate = float(setting(article, "Trading cost rate"))
    changes = units.diff().fillna(units.iloc[0])
    # QIS owns holdings simulation. These test-only identities verify its recorded units/costs
    # and one holding period without implementing another recursive backtester.
    reference_costs = rate * prices * changes.abs()
    np.testing.assert_allclose(portfolio.realized_costs, reference_costs, atol=1e-12, rtol=0)
    np.testing.assert_allclose(changes.loc[~portfolio.is_rebalancing], 0, atol=1e-12)
    assert (units.iloc[0] == 0).all() and portfolio.nav.iloc[0] == initial_nav
    first_trade = portfolio.is_rebalancing[portfolio.is_rebalancing].index[0]
    position = prices.index.get_loc(first_trade)
    np.testing.assert_allclose(
        units.loc[first_trade], initial_nav * pipeline["weights"].iloc[0] / prices.loc[first_trade]
    )
    opening_cost = reference_costs.loc[first_trade].sum()
    assert opening_cost == pytest.approx(initial_nav * rate)
    assert portfolio.nav.loc[first_trade] == pytest.approx(initial_nav - opening_cost)
    stated_cost = re.search(r"opening trade costs ([0-9.]+) NAV", article)
    stated_nav = re.search(r"leaves NAV at ([0-9.]+) on", article)
    assert stated_cost and stated_nav
    assert float(stated_cost[1]) == pytest.approx(opening_cost)
    assert float(stated_nav[1]) == pytest.approx(portfolio.nav.loc[first_trade])
    next_prices = prices.iloc[position + 1]
    pnl = np.dot(units.loc[first_trade], next_prices - prices.loc[first_trade])
    assert portfolio.nav.iloc[position + 1] == pytest.approx(
        portfolio.nav.loc[first_trade] + pnl, abs=1e-10
    )


def test_displayed_stdout_matches_the_canonical_script(article, pipeline):
    """Keep the visible example tied to actual execution while excluding measured runtime."""
    blocks = re.findall(r"^\x60{3}text\n(Price history:.*?)^\x60{3}$",
                        article, re.MULTILINE | re.DOTALL)
    assert len(blocks) == 1
    expected = blocks[0].strip()
    actual = pipeline["stdout"].split("Runtime:", 1)[0].strip()
    assert expected == actual


def test_objective_inventory_and_factor_input_contract(article, pipeline):
    """Keep enum choices complete and expose the factor estimator's different fit interface."""
    documented = set(re.findall(r"\x60([A-Z_]+)\x60", article))
    assert {member.name for member in PortfolioObjective} <= documented | {"MIN_VARIANCE"}
    kwargs = pipeline["fit_args"]
    inspect.signature(EwmaCovarEstimator.fit_rolling_covars).bind(None, **kwargs)
    with pytest.raises(TypeError, match="risk_factor_prices|unexpected keyword"):
        inspect.signature(FactorCovarEstimator.fit_rolling_covars).bind(None, **kwargs)
    parameters = inspect.signature(FactorCovarEstimator.fit_rolling_covars).parameters
    for name in ("risk_factor_prices", "asset_returns_dict", "time_period"):
        assert name in parameters and f"{chr(96)}{name}{chr(96)}" in article
