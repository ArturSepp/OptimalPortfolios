"""Check opt-in execution and optional solver absence in a fresh process."""

import subprocess
import sys


def test_v2_is_opt_in_and_runs_without_external_solver_or_factor_packages(tmp_path):
    """Keep production exports intact and deliver ranked weights without MOSEK."""
    script = r'''
import importlib.abc
import sys

class BlockExternalImports(importlib.abc.MetaPathFinder):
    """Simulate a base installation even when research extras are installed."""
    def find_spec(self, fullname, path=None, target=None):
        """Reject optional native and external factor/production packages."""
        if fullname.split('.')[0] in {
            'mosek', 'rosaa', 'factormodels', 'factorlasso', 'tradablefactors'
        }:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockExternalImports())
import optimalportfolios as op
import optimalportfolios.execution as production
assert 'optimalportfolios.execution.v2' not in sys.modules
original = {name: getattr(production, name) for name in (
    'solve_ranked_execution', 'solve_guarded_execution', 'ExecutionRankingConfig'
)}
assert 'wrapper_ranked_execution' not in vars(op)
assert 'wrapper_exact_execution' not in vars(production)

from optimalportfolios.execution.v2 import (
    audit_execution, wrapper_ranked_execution, wrapper_exact_execution
)
from optimalportfolios.execution.v2.tests.formulation_test import example
for name, value in original.items():
    assert getattr(production, name) is value
    assert getattr(op, name) is value
assert 'wrapper_ranked_execution' not in vars(op)
assert 'wrapper_exact_execution' not in vars(production)
problem = example()
ranked = wrapper_ranked_execution(problem)
assert audit_execution(problem, ranked.weights)['passed']
exact = wrapper_exact_execution(problem, 'CORRIDOR')
assert exact.diagnostics['native_status'] == 'solver_unavailable'
assert audit_execution(problem, exact.weights)['passed']
assert 'mosek' not in sys.modules
print('opt-in namespace and base-install delivery verified')
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
        text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "base-install delivery verified" in result.stdout
