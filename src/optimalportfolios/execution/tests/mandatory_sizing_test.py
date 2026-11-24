"""Mandatory intervals must be sized even when no optional ticket improves risk."""
from dataclasses import replace

import numpy as np

from optimalportfolios.execution import wrapper_ranked_execution, audit_execution
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.tests.formulation_test import example, independent_kkt


def test_mandatory_only_sizing_matches_independent_kkt():
    """A synthetic coupled book improves by resizing A without adding optional B/D."""
    p = example()
    current = p.current.copy()
    current[0] -= .02
    current[2] += .02
    target = np.array([.02469727, .03337102, .08390794, .01384931, .04417446])
    covariance = np.array([
        [.01486886, .00670271, .02551962, -.00015602, -.05293816],
        [.00670271, .05477928, .01845108, .02027495, -.00300247],
        [.02551962, .01845108, .07681724, -.01986539, -.07584691],
        [-.00015602, .02027495, -.01986539, .02647381, -.00507137],
        [-.05293816, -.00300247, -.07584691, -.00507137, .22576662],
    ])
    lo, hi = np.minimum(p.base, target), np.maximum(p.base, target)
    lo[[0, 2, 4]], hi[[0, 2, 4]] = [.04, 0., .015], [.1, .06, .04]
    mandatory = np.array([True, False, True, False, False])
    p = replace(p, current=current, target=target, lower=lo, upper=hi,
                full_lower=np.array([.04, 0., 0., 0., .015]),
                full_upper=np.array([.1, .2, .06, .2, .04]), covariance=covariance,
                mandatory=mandatory, categories=('product_min_buy', '', 'product_max_sell', '', ''),
                rules=replace(p.rules, ticket_cap=4, exemption_mask=mandatory),
                groups=np.array([[1., 1., 1., 0., 0.]]),
                group_lower=np.array([.129]), group_upper=np.array([.131]))
    # Active-set enumeration is independent of the CVXPY model and ranked search.
    expected = independent_kkt(p, {0: (.04, .1), 1: (.03, .03),
                                   2: (.005, .06), 3: (.03, .03)})
    result = wrapper_ranked_execution(p)
    assert audit_execution(p, result.weights)['passed']
    assert variance(p, result.weights) <= expected+1e-12
    assert variance(p, result.weights) < variance(p, p.base)-8e-8
    np.testing.assert_allclose(result.weights[[1, 3]], [.03, .03], atol=1e-10, rtol=0)
