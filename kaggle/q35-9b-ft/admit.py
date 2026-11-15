"""Before a GPU launch: find sessions that are active on the Kaggle account but not registered in reports/gpu_budget.json,
read each one's metadata, and register it with zero GPU time only if it is CPU-only (`enable_gpu` false). If any unknown
active session uses a GPU, register nothing for it and exit with code 2: stop and report. Other projects' kernels are only
read (metadata pull), never changed.

    PATH=~/.cache/k12-api-venv/bin:$PATH python kaggle/q35-9b-ft/admit.py <scratch folder>
"""
import json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from launch_kaggle import account_active_refs  # noqa: E402

budget = ROOT / 'reports' / 'gpu_budget.json'
policy = json.loads(budget.read_text())
known = {r['kernel'] for r in policy.get('reservations', [])}
active = account_active_refs()
report, blocked = [], False
for ref in sorted(set(active) - known):
    with tempfile.TemporaryDirectory(dir=sys.argv[1]) as folder:
        pull = subprocess.run(['kaggle', 'kernels', 'pull', ref, '-p', folder, '-m'], capture_output=True, text=True)
        meta_file = Path(folder) / 'kernel-metadata.json'
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else None
    if meta is None or meta.get('enable_gpu') is not False:
        blocked = True
        report.append({'kernel': ref, 'registered': False, 'enable_gpu': None if meta is None else meta.get('enable_gpu'), 'pull': pull.stdout[-200:] + pull.stderr[-200:]})
        continue
    policy.setdefault('reservations', []).append({'kernel': ref, 'timeout_seconds': 0, 'progress_log': None, 'cpu_only': True,
                                                  'note': f"another project's CPU-only session (enable_gpu false in its metadata, machine_shape {meta.get('machine_shape')}); "
                                                          f"zero GPU time; verified {datetime.now(timezone.utc).isoformat(timespec='seconds')}"})
    report.append({'kernel': ref, 'registered': True, 'enable_gpu': False})
if any(r['registered'] for r in report):
    budget.write_text(json.dumps(policy, indent=1))
print(json.dumps({'active': sorted(active), 'already_registered': sorted(set(active) & known), 'checked': report, 'blocked': blocked}, indent=1))
sys.exit(2 if blocked else 0)
