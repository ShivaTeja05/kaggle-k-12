"""Launch a reviewed private free-GPU job and detect Kaggle's exit-zero errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys


def successful_push(returncode, output):
    return returncode==0 and 'successfully pushed' in output.lower() and 'push error' not in output.lower()


def pushed_kernel_ref(output,requested):
    matches=re.findall(r'https://www\.kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)',output)
    return matches[-1] if matches else requested


def json_payload(text):
    decoder=json.JSONDecoder()
    for i,c in enumerate(text):
        if c not in '[{':continue
        try:return decoder.raw_decode(text[i:])[0]
        except ValueError:pass
    raise ValueError('No JSON quota response')


def observed_seconds(path):
    if not path or not Path(path).exists():return 0.
    elapsed=0.
    for line in Path(path).read_text().splitlines():
        try:
            row=json.loads(line)
            value=row.get('seconds',0)
            is_progress=row.get('stage') in {'train','predict','inference'} or ('rows' in row and 'total' in row)
            if is_progress and isinstance(value,(int,float)) and value>=0:elapsed=max(elapsed,float(value))
        except (ValueError,AttributeError):pass
    return elapsed


def projected_hours(used,reservations,timeout,overhead=.1,margin=.25):
    remaining=sum(max(0,r['timeout_seconds']-r.get('observed_seconds',0))/3600+overhead for r in reservations)
    return used+remaining+timeout/3600+overhead+margin


def recent_account_refs(rows,now):
    """A 48-hour horizon exceeds Kaggle's documented maximum session length."""
    refs=[]
    for row in rows:
        stamp=datetime.fromisoformat(row['lastRunTime'].replace('Z','+00:00'))
        if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
        if (now-stamp).total_seconds()<=48*3600:refs.append(row['ref'])
    if len(rows)>=200 and len(refs)==len(rows):
        raise RuntimeError('Account inventory truncated inside the active-session horizon; no safe GPU admission')
    return refs


def account_active_refs():
    listing=subprocess.run(['kaggle','kernels','list','--mine','--sort-by','dateRun',
                            '--page-size','200','--format','json'],capture_output=True,text=True,check=True)
    refs=recent_account_refs(json_payload(listing.stdout),datetime.now(timezone.utc))
    def status(ref):
        r=subprocess.run(['kaggle','kernels','status',ref],capture_output=True,text=True)
        if r.returncode or 'KernelWorkerStatus.' not in r.stdout:
            raise RuntimeError('Cannot establish account session status: '+ref)
        return ref,any(s in r.stdout for s in ['RUNNING','QUEUED'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        return [ref for ref,active in pool.map(status,refs) if active]


def budget_check(folder,timeout,budget_file):
    policy=json.loads(Path(budget_file).read_text())
    if policy.get('gpu_launches_allowed') is False:
        raise SystemExit('GPU launches disabled by the user: '+policy.get('gpu_launch_policy_reason','CPU-only work requested'))
    q=subprocess.run(['kaggle','quota','--format','json'],capture_output=True,text=True,check=True)
    gpu=next(x for x in json_payload(q.stdout) if x['resource']=='GPU')
    if gpu.get('refreshAt')!=policy.get('quota_refresh_at'):
        raise SystemExit('Quota window changed. The 10-hour task cap remains; reconcile prior usage before launching more GPU work.')
    used=float(gpu['used'].removesuffix('h'))
    remaining=float(gpu['remaining'].removesuffix('h'))
    registered={r['kernel'] for r in policy.get('reservations',[])}
    unknown=sorted(set(account_active_refs())-registered)
    if unknown:
        report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'used_hours':used,
                'unregistered_active_sessions':unknown,'admitted':False,
                'reason':'Unknown active sessions can consume account quota. Their remaining maximum is not reserved.'}
        (Path(folder)/'budget_check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));raise SystemExit('Account concurrency is not budgeted. No GPU job launched.')
    active=[]
    for r in policy.get('reservations',[]):
        status=subprocess.run(['kaggle','kernels','status',r['kernel']],capture_output=True,text=True)
        if status.returncode:raise RuntimeError('Cannot establish reservation status: '+r['kernel'])
        if any(s in status.stdout for s in ['RUNNING','QUEUED']):
            active.append({**r,'observed_seconds':observed_seconds(r.get('progress_log'))})
    total=projected_hours(used,active,timeout,policy['per_session_overhead_hours'],policy['safety_margin_hours'])
    minimum_remaining=policy.get('minimum_remaining_hours',0.0)
    projected_remaining=remaining-(total-used)
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'used_hours':used,
            'active_reservations':active,'candidate_timeout_seconds':timeout,
            'projected_worst_case_hours':total,'hard_cap_hours':policy['hard_cap_total_used_hours'],
            'projected_worst_case_remaining_hours':projected_remaining,
            'minimum_remaining_hours':minimum_remaining,
            'admitted':total<=policy['hard_cap_total_used_hours'] and projected_remaining>=minimum_remaining}
    (Path(folder)/'budget_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if projected_remaining<minimum_remaining:
        raise SystemExit(f'GPU job would leave less than {minimum_remaining:g} hours of quota. No job launched.')
    if not report['admitted']:raise SystemExit('GPU cap would be exceeded; wait or reduce the job timeout. No job launched.')
    policy['quota_snapshot_used_hours']=used
    policy['quota_snapshot_remaining_hours']=remaining
    return policy


def launch(folder,timeout,budget_file,check_only=False):
    if not 0<timeout<=21600:raise ValueError('Timeout must be positive and at most six hours')
    folder=Path(folder);meta=json.loads((folder/'kernel-metadata.json').read_text())
    if meta.get('is_private') is not True or meta.get('enable_internet') is not False:
        raise ValueError('Runner requires a private offline notebook')
    if meta.get('machine_shape')!='NvidiaTeslaT4':raise ValueError('Runner is restricted to Kaggle free T4 compute')
    current=subprocess.run(['kaggle','kernels','status',meta['id']],capture_output=True,text=True)
    if any(s in current.stdout for s in ['RUNNING','QUEUED']):
        print(current.stdout);raise SystemExit('Existing version active; no duplicate launched')
    policy=budget_check(folder,timeout,budget_file)
    if check_only:
        print('Budget check only; no GPU job launched')
        return
    result=subprocess.run(['kaggle','kernels','push','-p',str(folder),'-t',str(timeout)],capture_output=True,text=True)
    text=result.stdout+'\n'+result.stderr
    print(text)
    actual=pushed_kernel_ref(text,meta['id'])
    record={'kernel':actual,'requested_kernel':meta['id'],'timeout_seconds':timeout,'launched':successful_push(result.returncode,text),'response':text}
    (folder/'launch_result.json').write_text(json.dumps(record,indent=2))
    if not record['launched']:raise SystemExit(1)
    policy['reservations']=[r for r in policy.get('reservations',[]) if r['kernel'] not in {meta['id'],actual}]
    if actual!=meta['id']:
        meta['id']=actual
        (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    progress=Path('runs/live')/(actual.split('/')[-1]+'.log')
    progress.parent.mkdir(parents=True,exist_ok=True);progress.write_text('')
    policy['reservations'].append({'kernel':actual,'timeout_seconds':timeout,'progress_log':str(progress)})
    Path(budget_file).write_text(json.dumps(policy,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--budget-file',default='reports/gpu_budget.json')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    with Path(a.budget_file).with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        launch(a.folder,a.timeout,a.budget_file,a.check_only)
