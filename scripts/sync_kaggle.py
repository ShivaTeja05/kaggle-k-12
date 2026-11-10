"""One explicit read-only sync of a private Kaggle run; no automatic submissions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def sync(kernel,output):
    status=subprocess.run(['kaggle','kernels','status',kernel],text=True,capture_output=True)
    print(status.stdout);print(status.stderr,file=sys.stderr)
    if status.returncode:raise SystemExit(status.returncode)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'last_status.json').write_text(json.dumps({'kernel':kernel,'status_output':status.stdout},indent=2))
    if 'COMPLETE' in status.stdout or 'ERROR' in status.stdout or 'CANCEL' in status.stdout:
        subprocess.run(['kaggle','kernels','output',kernel,'-p',str(out),'--force',
                        '--file-pattern',r'\.(log|json|jsonl|csv|npz)$'],check=True)
        logs=list(out.glob('*.log'))
        if logs:
            try:
                records=json.loads(logs[0].read_text());text=''.join(x.get('data','') for x in records)
                (out/'readable_log.txt').write_text(text)
                print(text[-6000:])
            except ValueError:pass
    else:print('Job is still running; completed artifacts are not yet available through the CLI.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kernel');p.add_argument('--output',default='runs/latest');a=p.parse_args();sync(a.kernel,a.output)
