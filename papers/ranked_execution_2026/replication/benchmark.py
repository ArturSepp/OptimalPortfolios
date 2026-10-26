"""Portable numerical instances and independent checks for Ranked Execution.

The bundle contains supplied decision inputs, not prices or an upstream allocation
model. No private research module or production application is imported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qis import RiskModel
from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules

ARRAYS = ('current', 'base', 'target', 'lower', 'upper', 'full_lower', 'full_upper',
          'groups', 'group_lower', 'group_upper', 'mandatory', 'eligible', 'frozen',
          'covariance')


def sha256(path):
    """Identify exact portable input bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(directory):
    """Reject missing, changed or escaping files before constructing a problem."""
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'ranked-execution-benchmark-1' or not manifest['cases']:
        raise ValueError('Unsupported or empty benchmark')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Input hash mismatch: {name}')
    for case in manifest['cases']:
        if case['array_file'] not in manifest['files']:
            raise ValueError('Unverified case array')
    return manifest


def load_case(directory, record):
    """Read only typed arrays; never unpickle a public data file."""
    with np.load(Path(directory)/record['array_file'], allow_pickle=False) as data:
        values = {name: data[name] for name in ARRAYS}
        rules = ExecutionRules(**record['rules'], holding_floor=data['holding_floor'],
                               exemption_mask=data['exemption_mask'])
    return ExecutionProblem(**values, names=tuple(record['names']), cash=record['cash'],
        categories=tuple(record['categories']), rules=rules,
        provenance={'benchmark': 'ETF302', 'case': record['id']})


def independent_audit(problem, weights):
    """Evaluate original units without the implementation's state or audit routines."""
    if weights is None:
        return {'passed': False, 'violations': ['no_weights']}
    w = np.asarray(weights, dtype=float)
    if w.shape != problem.current.shape or not np.isfinite(w).all():
        return {'passed': False, 'violations': ['invalid_weights']}
    tol, zero = problem.rules.audit_tolerance, problem.rules.zero_tolerance
    residuals = {
        'funding': abs(float(w.sum())-problem.investment),
        'lower': float(np.maximum(problem.lower-w, 0.).max()),
        'upper': float(np.maximum(w-problem.upper, 0.).max()),
        'groups': float(max(np.max(problem.group_lower-problem.groups@w, initial=0.),
                            np.max(problem.groups@w-problem.group_upper, initial=0.))),
    }
    noncash = np.arange(len(w)) != problem.cash
    changes = np.abs(w-problem.current)
    trades = noncash & (changes > zero)
    optional = trades & ~problem.mandatory
    exempt = np.broadcast_to(problem.rules.exemption_mask, w.shape)
    floor = np.broadcast_to(problem.rules.holding_floor, w.shape)
    violations = [name for name, value in residuals.items() if value > tol]
    if np.any(noncash & (w > zero) & (w < floor-tol)):
        violations.append('holding_floor')
    if np.any(trades & ~exempt & (changes < problem.rules.minimum_ticket-tol)):
        violations.append('minimum_ticket')
    if problem.rules.ticket_cap is not None and int(trades.sum()) > problem.rules.ticket_cap:
        violations.append('ticket_cap')
    if (problem.rules.discretionary_allowance is not None
            and int(optional.sum()) > problem.rules.discretionary_allowance):
        violations.append('discretionary_allowance')
    return dict(passed=not violations, violations=violations, residuals=residuals,
                tickets=int(trades.sum()), optional_tickets=int(optional.sum()),
                turnover=float(changes[noncash].sum()))


def tracking_error_bp(problem, weights):
    """Use the owning QIS risk analytics for annual tracking-error reporting."""
    date = pd.Timestamp('2000-01-01')
    covariance = pd.DataFrame(problem.dense_covariance(), index=problem.names,
                              columns=problem.names)
    model = RiskModel(covar={date: covariance})
    return 1e4*float(model.compute_tre_at_date(
        pd.Series(problem.target, index=problem.names),
        pd.Series(weights, index=problem.names), date))
