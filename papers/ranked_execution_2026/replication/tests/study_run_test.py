"""Research timeouts, failed preparation and rotated comparisons stay explicit."""
import json
import subprocess

from papers.ranked_execution_2026.replication import study_run as run


def test_process_peak_is_measured():
    """The platform-specific observer returns a finite positive process measurement."""
    assert run.peak_memory_bytes()>0


def test_timeout_kills_only_created_worker_and_retains_missing_delivery(monkeypatch,tmp_path):
    """A stopped solver is not silently replaced by a successful prepared-base delivery."""
    (tmp_path/'logs').mkdir()
    class Process:
        """A single created process that outlives its declared allowance."""
        killed=False
        def wait(self,timeout=None):
            """Timeout once and allow reaping after termination."""
            if timeout is not None: raise subprocess.TimeoutExpired('owned worker',timeout)
            assert self.killed
            return 1
        def kill(self):
            """Mark only this owned process terminated."""
            self.killed=True
    child=Process()
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k: child)
    record=run.isolated_call(tmp_path,'case','Ranked',10)
    assert child.killed and record['status']=='process_ceiling'
    assert not record['audit']['passed'] and record['weights'] is None
    assert json.loads((tmp_path/'case--Ranked.json').read_text())==record


def test_failed_preparation_retains_every_arm(monkeypatch,tmp_path):
    """Preparation failure cannot shrink the study's method denominators."""
    manifest=dict(cases=[{'id':'case'}],specification=dict(preparation_ceiling_seconds=30,
        arms={'Exact':{},'Ranked':{},'Kernel':{}}))
    monkeypatch.setattr(run,'checked_manifest',lambda p:manifest)
    def failed(root,case,arm,ceiling):
        """Prevent any search after unsuccessful preparation."""
        assert arm=='prepare'
        return {'status':'failed','prepared':None}
    monkeypatch.setattr(run,'isolated_call',failed)
    run.run(tmp_path)
    for arm in manifest['specification']['arms']:
        record=json.loads((tmp_path/f'case--{arm}.json').read_text())
        assert record['status']=='preparation_failed' and not record['audit']['passed']
    assert json.loads((tmp_path/'complete.json').read_text())['cases']==1


def test_independent_order_rotates_without_dropping_declared_arms(monkeypatch,tmp_path):
    """Rotation changes order only; each case still gets its own declared attempts."""
    manifest=dict(cases=[{'id':f'case{i}'} for i in range(3)],
        specification=dict(preparation_ceiling_seconds=30,
            arms={a:{'process_ceiling_seconds':60} for a in ('Exact','Ranked','Kernel')}))
    monkeypatch.setattr(run,'checked_manifest',lambda p:manifest)
    calls=[]
    def completed(root,case,arm,ceiling):
        """Observe only scheduling, without running a numerical method."""
        if arm!='prepare': calls.append((case,arm))
        return {'status':'test','prepared':{'exists':True}}
    monkeypatch.setattr(run,'isolated_call',completed)
    run.run(tmp_path)
    assert calls==[('case0','Exact'),('case0','Ranked'),('case0','Kernel'),
                   ('case1','Ranked'),('case1','Kernel'),('case1','Exact'),
                   ('case2','Kernel'),('case2','Exact'),('case2','Ranked')]
