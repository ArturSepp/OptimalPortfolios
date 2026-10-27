"""Legacy URL coverage and build-time redirect-destination contracts."""
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXTENSION = ROOT / "docs" / "_ext" / "optimalportfolios_indexing.py"
if not EXTENSION.is_file():
    pytest.skip("Documentation sources are not shipped in the wheel.", allow_module_level=True)
SEO = runpy.run_path(str(EXTENSION))


def test_manifest_keeps_all_confirmed_historical_404_urls():
    """Keep the five renamed guides and eighteen removed generated API entry points."""
    manifest = SEO["read_redirect_manifest"]()
    sources = {r["from_url"] for r in manifest["redirects"]}
    guides = {"alpha_profiling", "alphas", "covariance", "examples", "optimisers"}
    names = {
        "CurrentFactorCovarData", "DependenceMeasure", "DistanceTransform", "FactorCovarEstimator",
        "LassoModel", "LassoModelType", "RollingFactorCovarData", "VarianceColumns",
        "build_risk_model", "compute_dependence_matrix", "compute_gerber_matrix",
        "compute_risk_contributions", "estimate_lasso_factor_covar_data",
        "extract_rolling_clusters",
        "filter_covar_and_vectors", "plot_current_covar_data", "plot_hcgl_covar_data",
        "run_rolling_covar_report",
    }
    assert sources == {f"/{name}.html" for name in guides} | {
        f"/generated/optimalportfolios.{name}.html" for name in names
    }
    assert all(r["http_status"] == 301 and r["force"] is False for r in manifest["redirects"])


@pytest.mark.parametrize("change", ["duplicate", "forced", "temporary", "self", "query"])
def test_manifest_rejects_unsafe_or_inconsistent_hosting_rules(tmp_path, change):
    """Reject loops, duplicates, forced redirects and accidental temporary responses."""
    manifest = copy.deepcopy(SEO["read_redirect_manifest"]())
    rule = manifest["redirects"][0]
    if change == "duplicate":
        manifest["redirects"].append(copy.deepcopy(rule))
    elif change == "forced":
        rule["force"] = True
    elif change == "temporary":
        rule["http_status"] = 302
    elif change == "self":
        rule["to_url"] = manifest["site"] + "/en/latest" + rule["from_url"]
    else:
        rule["to_url"] += "?token=not-allowed"
    file = tmp_path / "redirects.json"
    file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        SEO["read_redirect_manifest"](file)


def test_build_guard_rejects_a_removed_local_target_or_anchor(tmp_path, monkeypatch):
    """A future documentation move must not silently break an existing redirect."""
    manifest = {
        "site": "https://optimalportfolios.readthedocs.io",
        "redirects": [{"to_url": "https://optimalportfolios.readthedocs.io/en/latest/new.html#section"}],
    }
    callback = SEO["validate_redirect_targets"]
    monkeypatch.setitem(callback.__globals__, "read_redirect_manifest", lambda: manifest)
    target = tmp_path / "new.html"
    target.write_text(
        '<html><body><h1 id="section">Replacement</h1></body></html>', encoding="utf-8",
    )
    app = SimpleNamespace(builder=SimpleNamespace(name="html"), outdir=str(tmp_path))
    callback(app, None)
    assert list(tmp_path.iterdir()) == [target]
    target.write_text("<html><body>No preserved anchor</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="anchor"):
        callback(app, None)
    target.unlink()
    with pytest.raises(ValueError, match="destination"):
        callback(app, None)


@pytest.mark.parametrize(
    "builder,error", [("latex", None), ("html", RuntimeError("earlier failure"))],
)
def test_failed_or_non_html_build_does_not_validate_missing_outputs(tmp_path, builder, error):
    """Do not obscure an earlier build failure or require HTML from a different builder."""
    app = SimpleNamespace(builder=SimpleNamespace(name=builder), outdir=str(tmp_path))
    SEO["validate_redirect_targets"](app, error)
