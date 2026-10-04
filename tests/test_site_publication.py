"""Run the actual publication helper in a local remote, including rejected pushes."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'persist_site.py'


def git(cwd,*args):
    return subprocess.run(['git',*args],cwd=cwd,check=True,text=True,capture_output=True).stdout.strip()


def remote_fixture(tmp_path):
    remote=tmp_path/'origin.git';git(tmp_path,'init','--bare','--initial-branch=main',str(remote))
    first=tmp_path/'first';git(tmp_path,'clone',str(remote),str(first))
    for key,value in [('user.name','Test'),('user.email','test@example.test')]:git(first,'config',key,value)
    (first/'README').write_text('seed');git(first,'add','.');git(first,'commit','-m','seed');git(first,'push','origin','main')
    queued=tmp_path/'queued';git(tmp_path,'clone',str(remote),str(queued))
    path=first/'data'/'forecast'/'ledger.jsonl';path.parent.mkdir(parents=True);path.write_text('new committed forecast\n')
    git(first,'add','.');git(first,'commit','-m','predecessor update');git(first,'push','origin','main')
    return remote,first,queued


@pytest.mark.parametrize('mode',['sync','persist'])
def test_queued_checkout_with_no_changes_gets_latest_ledger(tmp_path,mode):
    _,first,queued=remote_fixture(tmp_path)
    result=subprocess.run([sys.executable,str(SCRIPT),mode],cwd=queued,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (queued/'data'/'forecast'/'ledger.jsonl').read_text()=='new committed forecast\n'
    assert git(queued,'rev-parse','HEAD')==git(first,'rev-parse','HEAD')


def test_rejected_push_never_satisfies_publication_barrier(tmp_path):
    remote,_,queued=remote_fixture(tmp_path)
    git(queued,'pull','--rebase')
    journal=queued/'data'/'results'/'results.jsonl';journal.parent.mkdir(parents=True);journal.write_text('new local result\n')
    hook=remote/'hooks'/'pre-receive';hook.write_text('#!/bin/sh\nexit 1\n');hook.chmod(0o755)
    result=subprocess.run([sys.executable,str(SCRIPT),'persist'],cwd=queued,capture_output=True,text=True)
    assert result.returncode!=0
    assert git(queued,'rev-parse','HEAD')!=git(queued,'rev-parse','origin/main')
    assert journal.read_text()=='new local result\n'


def test_workflows_require_persistence_before_site_or_cache_publication():
    yaml=pytest.importorskip('yaml')
    for name,job in [('results.yml','results'),('update.yml','update')]:
        wf=yaml.safe_load((SCRIPT.parents[1]/'.github'/'workflows'/name).read_text())
        steps=wf['jobs'][job]['steps']; barrier=next(s for s in steps if s.get('id')=='persist')
        assert 'persist_site.py persist' in barrier['run']
        for step in steps:
            if step.get('id')=='build' or 'actions/cache/save' in step.get('uses',''):
                assert "steps.persist.outcome == 'success'" in step.get('if','')
