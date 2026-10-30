"""Public external-risk inputs and a closed-form corridor counterexample.

OR-Library supplies prices only. Holdings, targets and constraints are synthetic;
these cases do not reproduce the original index-tracking objective. Raw source
files remain with their owner and are loaded locally with recorded checksums.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import qis

from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules
from .benchmark import sha256
from .study_design import generated_problem


def read_or_library(path):
    """Read the documented N,T,index,stocks layout without guessing missing prices."""
    values=np.fromstring(Path(path).read_text(encoding='utf-8'),sep=' ')
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Invalid OR-Library numbers')
    n,t=values[:2].astype(int)
    if n<1 or t<2 or not np.array_equal(values[:2],[n,t]) or len(values)!=2+(n+1)*(t+1):
        raise ValueError('Invalid OR-Library dimensions')
    prices=values[2:].reshape(n+1,t+1).T
    if np.any(prices<=0):
        raise ValueError('Nonpositive OR-Library price')
    return prices


def external_problem(case,directory):
    """Use point-in-time QIS EWMA risk and declared synthetic execution inputs."""
    path=Path(directory)/case['source_file']
    if path.name!=case['source_file'] or sha256(path)!=case['source_sha256']:
        raise ValueError('Unverified external source')
    prices=read_or_library(path)
    cut=case['observations']
    if not isinstance(cut,int) or not 53<=cut<len(prices):
        raise ValueError('Invalid covariance observation cut')
    if prices.shape[1]-1!=case['universe']:
        raise ValueError('External universe mismatch')
    # The file has no calendar labels. This is a declared weekly model convention,
    # not a claim that these placeholder dates are the original observation dates.
    frame=pd.DataFrame(prices[:cut+1,1:],
        index=pd.date_range('2000-01-05',periods=cut+1,freq='W-WED'))
    covariances=qis.estimate_rolling_ewma_covar(frame,returns_freq='W-WED',
        rebalancing_freq='W-WED',span=52,demean=True,
        is_apply_vol_normalised_returns=False,apply_an_factor=False)
    last=max(covariances)
    if last!=frame.index[-1]:
        raise ValueError('Risk estimate does not reach the declared observation cut')
    # Annualisation is explicitly 52 model periods; QIS supplies the log-return
    # EWMA estimator. The synthetic cash residual is specified independently.
    risk=52.*covariances[last].to_numpy()
    covariance=np.zeros((case['universe']+1,case['universe']+1))
    covariance[:-1,:-1]=risk; covariance[-1,-1]=1e-6
    p=generated_problem(case)
    return replace(p,covariance=covariance,factor_loadings=None,
        factor_covariance=None,residual_variance=None,
        provenance={'benchmark':'OR-Library-risk-adaptation','source':case['source_file'],
            'source_sha256':case['source_sha256'],'observations':cut,
            'return_convention':'log','model_periods_per_year':52,'span':52})


def corridor_counterexample():
    """Supply a two-risky-asset example whose full-domain solution overshoots target."""
    current=np.array([.2,.2,.6]); target=np.array([.3,.3,.4])
    return ExecutionProblem(names=('AssetA','AssetB','Cash'),cash=2,
        current=current,base=current,target=target,
        lower=np.array([.2,.2,0.]),upper=np.array([.3,.3,1.]),
        full_lower=np.zeros(3),full_upper=np.ones(3),
        groups=np.empty((0,3)),group_lower=np.empty(0),group_upper=np.empty(0),
        mandatory=np.zeros(3,bool),categories=('','',''),eligible=np.ones(3,bool),
        frozen=np.zeros(3,bool),covariance=.01*np.array([[1.,.9,0.],[.9,1.,0.],[0.,0.,0.]]),
        rules=ExecutionRules(minimum_ticket=.005,holding_floor=.0025,
            ticket_cap=1,discretionary_allowance=1,audit_tolerance=1e-7,zero_tolerance=1e-8))


def make_problem(case,directory):
    """Dispatch only explicitly declared input families, with no live fallback."""
    if case['family']=='scaling': return generated_problem(case)
    if case['family']=='external': return external_problem(case,directory)
    if case['family']=='counterexample': return corridor_counterexample()
    raise ValueError('Unknown research input family')
