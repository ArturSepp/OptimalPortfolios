"""Prospective coverage and budgets do not depend on outcome selection."""
import json

from papers.ranked_execution_2026.replication.study_protocol import PROTOCOL,cases


def test_declared_denominators_domains_and_resource_controls():
    """Every planned difficult case and additional arm has an explicit retained slot."""
    spec=json.loads(PROTOCOL.read_text()); rows=cases(spec)
    assert len(rows)==23 and sum(len(r['arms']) for r in rows)==82
    assert len({r['id'] for r in rows})==23
    scaling=[r for r in rows if r['family']=='scaling']
    assert len(scaling)==18
    assert {r['seed'] for r in scaling}=={2026102961,2026102962}
    for n in (300,1000,3000):
        for budget in (10,30,100):
            pair=[r for r in scaling if r['universe']==n and r['budget']==budget]
            assert {r['profile'] for r in pair}=={'wide','tight'}
    assert sum('Exact600' in r['arms'] for r in rows)==3
    assert sum('Dense10' in r['arms'] for r in rows)==3
    assert sum('Full60' in r['arms'] for r in rows)==4
    assert spec['arms']['Ranked']['process_ceiling_seconds']==180
    assert all(spec['arms'][arm]['process_ceiling_seconds']>0 for r in rows for arm in r['arms'])
    assert spec['controls']['audit_tolerance']==1e-7
    assert spec['controls']['native_absolute_gap_variance']==1e-12
