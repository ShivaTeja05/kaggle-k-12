"""Create, inspect and terminate the K-12 RunPod pod under the user's spend cap.

Spend is tracked in reports/runpod_budget.json. `watchdog` terminates the pod
before cumulative spend can pass the cap, even if nobody is watching.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tomllib
import urllib.request

BUDGET = Path('reports/runpod_budget.json')
PUBLIC_KEY = Path.home() / '.runpod/ssh/runpodctl-ssh-key.pub'


def gql(query, variables=None):
    key = tomllib.loads((Path.home() / '.runpod/config.toml').read_text())['apikey']
    request = urllib.request.Request(
        'https://api.runpod.io/graphql', data=json.dumps({'query': query, 'variables': variables or {}}).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}', 'User-Agent': 'k12-cli'})
    reply = json.load(urllib.request.urlopen(request, timeout=60))
    if reply.get('errors'):
        raise RuntimeError(json.dumps(reply['errors']))
    return reply['data']


def load_budget():
    return json.loads(BUDGET.read_text())


def save_budget(budget):
    BUDGET.write_text(json.dumps(budget, indent=2) + '\n')


def now():
    return datetime.now(timezone.utc)


def spent(budget):
    total = 0.
    for pod in budget['pods']:
        end = datetime.fromisoformat(pod['terminated_at']) if pod.get('terminated_at') else now()
        total += (end - datetime.fromisoformat(pod['created_at'])).total_seconds() / 3600 * pod['cost_per_hr']
    return total


def create(args):
    budget = load_budget()
    if sum(not pod.get('terminated_at') for pod in budget['pods']) >= budget['max_active_pods']:
        raise SystemExit(f"{budget['max_active_pods']} pods already active; terminate one first")
    if spent(budget) + budget['minimum_session_usd'] > budget['cap_usd']:
        raise SystemExit(f"Spend {spent(budget):.2f} leaves no room under the {budget['cap_usd']} cap")
    pod = None
    gpu_types = args.gpu_types or budget['gpu_types']
    max_rate = args.max_rate or budget['max_cost_per_hr']
    for attempt in range(args.attempts):
        for gpu in gpu_types:
            try:
                pod = gql('''mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) {
                                id costPerHr machine { gpuDisplayName } } }''',
                          {'input': {'cloudType': 'SECURE', 'gpuTypeId': gpu, 'gpuCount': args.gpu_count,
                                     'name': 'k12-train', 'imageName': args.image, 'containerDiskInGb': args.disk,
                                     'volumeInGb': 0, 'ports': '22/tcp', 'startSsh': True,
                                     'env': [{'key': 'PUBLIC_KEY', 'value': PUBLIC_KEY.read_text().strip()}]}}
                          )['podFindAndDeployOnDemand']
                break
            except RuntimeError as exc:
                if 'SUPPLY_CONSTRAINT' not in str(exc):
                    raise
                print(f'attempt {attempt + 1}: {gpu} unavailable', flush=True)
        if pod:
            break
        time.sleep(30)
    if not pod:
        raise SystemExit('No approved GPU available')
    if pod['costPerHr'] > max_rate:
        gql('mutation($input: PodTerminateInput!) { podTerminate(input: $input) }', {'input': {'podId': pod['id']}})
        raise SystemExit(f"Terminated: {pod['costPerHr']}/hr exceeds the approved rate")
    budget['pods'].append({'id': pod['id'], 'gpu': pod['machine']['gpuDisplayName'], 'cost_per_hr': pod['costPerHr'],
                           'created_at': now().isoformat(), 'terminated_at': None})
    save_budget(budget)
    print(json.dumps(pod))


def status(args):
    budget = load_budget()
    for pod in budget['pods']:
        if not pod.get('terminated_at'):
            info = gql('query($input: PodFilter) { pod(input: $input) { id desiredStatus runtime { uptimeInSeconds '
                       'ports { ip isIpPublic privatePort publicPort type } } } }', {'input': {'podId': pod['id']}})
            print(json.dumps(info['pod'], indent=1))
    print(json.dumps({'spent_usd': round(spent(budget), 2), 'cap_usd': budget['cap_usd']}))


def terminate(args):
    budget = load_budget()
    for pod in budget['pods']:
        if not pod.get('terminated_at') and getattr(args, 'pod', None) in (None, pod['id']):
            gql('mutation($input: PodTerminateInput!) { podTerminate(input: $input) }', {'input': {'podId': pod['id']}})
            pod['terminated_at'] = now().isoformat()
            print('terminated', pod['id'])
    save_budget(budget)
    print(json.dumps({'spent_usd': round(spent(budget), 2), 'cap_usd': budget['cap_usd']}))


def watchdog(args):
    """Terminate once spend reaches the cap minus a margin; poll every minute."""
    while True:
        budget = load_budget()
        if all(pod.get('terminated_at') for pod in budget['pods']):
            print('no active pod'); return
        if spent(budget) >= budget['cap_usd'] - budget['watchdog_margin_usd']:
            print(f'cap reached at {spent(budget):.2f} USD', flush=True)
            terminate(args); return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('create')
    make.add_argument('--image', default='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04')
    make.add_argument('--attempts', type=int, default=10)
    make.add_argument('--gpu-count', type=int, default=1)
    make.add_argument('--gpu-types', nargs='+', help='override the approved list for this pod')
    make.add_argument('--max-rate', type=float, help='per-pod $/hr limit; above the default needs user approval')
    make.add_argument('--disk', type=int, default=120)
    commands.add_parser('status'); commands.add_parser('watchdog')
    commands.add_parser('terminate').add_argument('--pod', help='terminate only this pod (default: all active)')
    args = parser.parse_args()
    {'create': create, 'status': status, 'terminate': terminate, 'watchdog': watchdog}[args.command](args)
