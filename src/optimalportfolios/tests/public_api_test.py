"""
enforcement tests for the public surface of optimalportfolios.

The package has no ``__all__``: its public surface is whatever the wildcard imports in
``src/optimalportfolios/__init__.py`` happen to leave in the namespace. That is a contract nobody
wrote down, so these tests write down the parts of it that can be checked cheaply.

Four properties, each of which has a failure mode that is silent today:

    every public name resolves
        A wildcard import that stops re-exporting a symbol removes it from the public surface
        with no error anywhere. A downstream ``AttributeError`` is the first sign.

    the factorlasso re-exports are the same objects as their sources
        ``optimalportfolios.LassoModel is factorlasso.LassoModel`` must hold, or a caller
        constructing one and passing it to the other gets an isinstance failure. CI has checked
        one of the nine inline since 6.1; this checks all nine, in the suite, where it belongs.

    a name shared with qis is the same object, or is on the allowlist below
        Two packages in one stack exporting the same name for different things is how the same
        nominal method comes to give different numbers in different places. New collisions must
        be a deliberate entry here rather than an accident.

    rolling_* and wrapper_* entry points pair up
        Every rolling dispatcher has a single-period wrapper and vice versa. A half-added solver
        is otherwise only discovered by a caller reaching for the missing half.

Deliberately not checked: whether the surface *should* contain what it contains. There is no
recorded export list here, unlike ``qis/api.py``'s ``CORE_API``; adding a name to this package is
not meant to require a second edit. See OP_TEST_SUITE_ROADMAP.md, D1.
"""
# packages
import inspect
import types
from typing import Any, Dict, List, Set

import pytest

# optimalportfolios
import optimalportfolios as op
import qis
import factorlasso


def _public_names() -> List[str]:
    """every name the package exposes, in a stable order."""
    return sorted(name for name in dir(op) if not name.startswith('_'))


PUBLIC_NAMES: List[str] = _public_names()

# The nine symbols optimalportfolios re-exports from factorlasso for backward compatibility.
# Listed here rather than derived, so that dropping one from __init__.py fails this file.
FACTORLASSO_REEXPORTS: List[str] = [
    'CurrentFactorCovarData',
    'DependenceMeasure',
    'DistanceTransform',
    'LassoModel',
    'LassoModelType',
    'RollingFactorCovarData',
    'VarianceColumns',
    'compute_dependence_matrix',
    'compute_gerber_matrix',
]

# Names exported by both optimalportfolios and qis that are NOT the same object. Each entry is a
# deliberate collision with a reason; anything not listed here must be identical in both packages.
#
# `estimate_rolling_ewma_covar` was here until 6.6.0, when the local reimplementation was deleted
# in favour of the qis function. This test is what found it, and the entry's removal is what the
# staleness check below forced once the two names became one object.
QIS_COLLISION_ALLOWLIST: Dict[str, str] = {
    'local_path': 'each package has its own path resolution module; module binding, not a symbol',
    'utils': 'each package has its own utils subpackage; module binding, not a symbol',
    'compute_portfolio_vol': (
        'different functions sharing a name: optimalportfolios takes (covar, weights) and returns '
        'a scalar, qis takes (returns, weights, span) and returns a series'
    ),
}


def test_public_surface_is_not_empty() -> None:
    """a green run here must not be an empty run."""
    assert len(PUBLIC_NAMES) > 100, (
        f"only {len(PUBLIC_NAMES)} public names found; the wildcard imports in __init__.py "
        f"are not populating the namespace")


@pytest.mark.parametrize('name', PUBLIC_NAMES)
def test_public_name_resolves(name: str) -> None:
    """every advertised name is reachable and bound to something."""
    assert hasattr(op, name), (
        f"optimalportfolios.{name} is advertised by dir() but does not resolve")
    assert getattr(op, name) is not None, f"optimalportfolios.{name} is None"


@pytest.mark.parametrize('name', FACTORLASSO_REEXPORTS)
def test_factorlasso_reexport_is_the_same_object(name: str) -> None:
    """
    a re-export is an alias, not a copy.

    If these ever diverge, an isinstance check against one fails for an object built by the other,
    and the failure surfaces far from its cause.
    """
    assert hasattr(op, name), f"optimalportfolios no longer re-exports {name} from factorlasso"
    assert hasattr(factorlasso, name), f"factorlasso no longer exports {name}"
    assert getattr(op, name) is getattr(factorlasso, name), (
        f"optimalportfolios.{name} is not factorlasso.{name}; the re-export has been shadowed")


def test_names_shared_with_qis_are_identical_or_allowlisted() -> None:
    """
    a name in both packages means the same object, unless it is a recorded exception.

    optimalportfolios depends on qis, so a caller with both imported reasonably expects a shared
    name to be one thing. Where it is not, the reason is written down in the allowlist above.
    """
    shared: Set[str] = {n for n in dir(op) if not n.startswith('_')} & \
                       {n for n in dir(qis) if not n.startswith('_')}
    divergent = sorted(n for n in shared if getattr(op, n) is not getattr(qis, n))
    unrecorded = [n for n in divergent if n not in QIS_COLLISION_ALLOWLIST]
    assert not unrecorded, (
        f"names exported by both packages as different objects and not in "
        f"QIS_COLLISION_ALLOWLIST: {unrecorded}. Either call the qis symbol instead of "
        f"reimplementing it, or add an entry saying why the collision is deliberate")


def test_allowlisted_collisions_still_collide() -> None:
    """
    the allowlist does not outlive the collisions it excuses.

    An entry that no longer describes anything real is worse than no entry: it reads as a known
    exception while excusing nothing, and the next reader trusts it.
    """
    stale = []
    for name in QIS_COLLISION_ALLOWLIST:
        if not hasattr(op, name) or not hasattr(qis, name):
            stale.append(f'{name} (no longer in both packages)')
        elif getattr(op, name) is getattr(qis, name):
            stale.append(f'{name} (now the same object)')
    assert not stale, f"QIS_COLLISION_ALLOWLIST entries that no longer apply: {stale}"


def test_rolling_and_wrapper_entry_points_pair_up() -> None:
    """
    every rolling dispatcher has its single-period wrapper, and the reverse.

    The two families are the package's entry points. Adding one half of a pair leaves a solver
    reachable through the rolling backtest but not directly, or the other way round, and nothing
    reports it.
    """
    rolling = {n[len('rolling_'):] for n in dir(op) if n.startswith('rolling_')}
    wrapper = {n[len('wrapper_'):] for n in dir(op) if n.startswith('wrapper_')}
    assert len(rolling) >= 9, f"only {len(rolling)} rolling_* entry points found"
    assert rolling == wrapper, (
        f"rolling_* without a wrapper_*: {sorted(rolling - wrapper)}; "
        f"wrapper_* without a rolling_*: {sorted(wrapper - rolling)}")


def test_portfolio_objective_members_are_stable() -> None:
    """
    the objective enum is part of the public contract.

    Callers persist these by name in configuration; renaming or dropping one breaks a stored
    config with no import error.
    """
    expected = {'MAX_DIVERSIFICATION', 'EQUAL_RISK_CONTRIBUTION', 'MIN_VARIANCE',
                'QUADRATIC_UTILITY', 'MAXIMUM_SHARPE_RATIO', 'MAX_CARA_MIXTURE'}
    actual = {member.name for member in op.PortfolioObjective}
    assert actual == expected, (
        f"PortfolioObjective changed: added {sorted(actual - expected)}, "
        f"removed {sorted(expected - actual)}. Update this test in the same commit, and the "
        f"CHANGELOG, because a stored configuration names these")


def test_exported_modules_are_the_expected_ones() -> None:
    """
    module bindings in the namespace are deliberate.

    A wildcard import pulls in whatever module names the source happened to import, so this list
    drifts silently. Keeping it explicit means an accidental addition is visible in a diff.
    """
    modules = sorted(n for n in PUBLIC_NAMES if isinstance(getattr(op, n), types.ModuleType))
    assert len(modules) <= 12, (
        f"{len(modules)} modules are exported into the top-level namespace: {modules}. "
        f"Wildcard imports are leaking module bindings")


@pytest.mark.parametrize('name', [n for n in PUBLIC_NAMES
                                  if inspect.isclass(getattr(op, n))
                                  or inspect.isfunction(getattr(op, n))])
def test_public_callable_is_introspectable(name: str) -> None:
    """
    every public class or function has a signature a caller can read.

    ``help()`` and every IDE depend on this; a C-level or badly wrapped object silently loses it.
    """
    obj: Any = getattr(op, name)
    try:
        inspect.signature(obj)
    except (ValueError, TypeError) as e:
        pytest.fail(f"optimalportfolios.{name} has no readable signature: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
