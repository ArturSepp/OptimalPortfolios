"""Post-study summaries preserve denominators, signs and objective origins."""
from papers.ranked_execution_2026.replication.summarize_study import (
    paired,method_summary,corridor_reference,render_historical,research_pipeline_seconds,
    numerical_bounds,
)


def item(te=None,status='ok',**extra):
    """Return a minimal successful or missing-delivery record."""
    return dict(status=status,audit={'passed':te is not None},te_bp=te,**extra)


def test_missing_delivery_is_not_a_tie_or_zero_risk():
    """Paired comparisons cannot impute a result for a killed worker."""
    rows=[{'id':'a','methods':{'L':item(1.),'R':item(2.)}},
          {'id':'b','methods':{'L':item(None,'process_ceiling'),'R':item(2.)}},
          {'id':'c','methods':{'L':item(3.),'R':item(2.)}}]
    result=paired(rows,'L','R',1e-4)
    assert result['expected']==3 and result['paired']==2 and result['missing']==1
    assert result['left_better']==result['left_worse']==1 and result['ties']==0
    assert result['gap_bp']['minimum']==-1. and result['gap_bp']['maximum']==1.


def test_timed_out_worker_stays_in_process_timing_denominator():
    """Unavailable internal timing and peak memory remain explicitly unobserved."""
    result=method_summary([item(1.,seconds=2.,process_seconds=8.,peak_process_memory_bytes=100.),
                           item(None,'process_ceiling',process_seconds=180.)])
    assert result['expected']==2 and result['audited']==1
    assert result['process_seconds']['n']==2 and result['process_seconds']['median']==94.
    assert result['seconds']['n']==1 and result['peak_process_memory_bytes']['n']==1


def test_native_and_delivered_gaps_use_distinct_incumbents():
    """A better native vector that fails audit does not become the delivered risk."""
    result=method_summary([item(1.,diagnostics={'native_bound':.1,'native_incumbent':.2,
        'variance':.3,'native_gap_relative':.5,'delivered_source':'base'})])
    assert abs(result['native_absolute_gap_variance']['mean']-.1)<1e-14
    assert abs(result['delivery_minus_native_bound']['mean']-.2)<1e-14
    assert result['native_witnesses']==0


def test_best_observed_reference_does_not_mix_full_policy_or_missing_delivery():
    """A lower Full-domain risk cannot silently become the Corridor quality target."""
    row=dict(methods={'Exact60':item(20.),'Exact600':item(10.),
                      'Ranked':item(None),'Full60':item(5.)},
             method_domains={'Exact60':'CORRIDOR','Exact600':'CORRIDOR',
                             'Ranked':'CORRIDOR','Full60':'FULL'})
    assert corridor_reference(row)=={'te_bp':10.,'methods':['Exact600']}


def test_historical_table_retains_small_negative_and_positive_extrema():
    """The presentation rename must not erase a benefit smaller than one basis point."""
    pair=dict(n=3,left_better=1,ties=1,left_worse=1,
              gap_bp={'minimum':-.0009414283,'maximum':2.457878})
    summary={'cohorts':{name:{'paired':{'V2_minus_MIQP':pair}}
                        for name in ('ETF','Legacy','Holdout')}}
    rendered=render_historical(summary)
    assert rendered.count('-0.000941')==3 and rendered.count('2.457878')==3
    assert 'Ranked lower' in rendered and 'Ranked higher' in rendered


def test_complete_research_pipeline_charges_preparation_and_preserves_missing_times():
    """The two-worker pipeline must neither omit preparation nor invent unobserved cost."""
    assert research_pipeline_seconds({'process_seconds':60.},{'process_seconds':8.})==68.
    assert research_pipeline_seconds({'process_seconds':180.},{'process_seconds':8.})==188.
    assert research_pipeline_seconds({}, {'process_seconds':8.}) is None


def test_native_bound_control_uses_only_audited_equivalent_graph_witnesses():
    """A native optimal label cannot hide an accepted better original-unit objective."""
    cones=[dict(optional=24,balanced=False,input_fingerprint='a',variance=.2,
                native={'native_bound':.19},audit={'passed':True}),
           dict(optional=24,balanced=True,input_fingerprint='a',variance=.1,
                native={'native_bound':.09},audit={'passed':True}),
           dict(optional=24,balanced=True,input_fingerprint='a',variance=0.,
                native={},audit={'passed':False})]
    result=numerical_bounds({'cones':cones})
    assert result[0]['best_audited_variance']==.1
    assert abs(result[0]['bound_minus_best']-.09)<1e-14
    assert result[1]['bound_minus_best']<0.


def test_representation_witness_rejects_different_problem_identity():
    """A lower objective on changed inputs cannot invalidate the original native bound."""
    import pytest
    cones=[dict(optional=24,input_fingerprint='a'),dict(optional=24,input_fingerprint='b')]
    with pytest.raises(ValueError,match='identical inputs'):
        numerical_bounds({'cones':cones})
