"""Write a Kaggle kernel folder for the Qwen3.5-9B fine-tune rehearsal: run.py = CONFIG + the fine-tune script (embedded,
byte for byte) + runner.py, and kernel-metadata.json. Launch GPU folders only with scripts/launch_kaggle.py.

    python kaggle/q35-9b-ft/build.py inspect|smoke|main|test [key=value ...]
"""
import base64, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OWNER = 'shivateja005'
DATASET = f'{OWNER}/k12-q35-ft-inputs'
MODEL = 'shelterw/qwen3.5/transformers/qwen3.5-9b/1'
COMPETITION = 'misconception-detection-in-k-12-student-math-workings'
TRAIN_KERNEL = f'{OWNER}/k12-q35-9b-ft'

BASE = dict(device='auto', dtype='float16', lean=True, gpu_memory_plans=['9.5GiB,13GiB'], smoke_rows=12, smoke_train_limit=1200, smoke_head_rows=10, smoke_head_limit=600,
            epochs=2, accumulation=8, log_every=5, save_minutes=30, skip_failed_rows=40, load_allowance_seconds=600,
            after_train_reserve_seconds=2100, train_minutes=285, echo_every=4)
MODES = {
    'inspect': dict(folder='q35-9b-inspect', slug='k12-q35-9b-inspect', title='K12 Q35 9B inspect', gpu=False, job_seconds=900,
                    stages=['encode']),
    'smoke': dict(folder='q35-9b-ft', slug='k12-q35-9b-ft', title='K12 Q35 9B ft', gpu=True, job_seconds=2280,
                  stages=['encode', 'smoke_train', 'smoke_head']),
    'main': dict(folder='q35-9b-ft', slug='k12-q35-9b-ft', title='K12 Q35 9B ft', gpu=True, job_seconds=20700,
                 stages=['encode', 'train', 'dev_head']),
    'test': dict(folder='q35-9b-test', slug='k12-q35-9b-test', title='K12 Q35 9B test', gpu=True, job_seconds=11400,
                 stages=['test_head'], adapter_under='k12-q35-9b-ft'),
}


def main():
    mode = sys.argv[1]; spec = dict(MODES[mode]); config = dict(BASE, mode=mode)
    for item in sys.argv[2:]:
        key, value = item.split('=', 1)
        try:
            value = json.loads(value)
        except ValueError:
            pass
        (spec if key in ('folder', 'slug', 'title') else config)[key] = value
    for key in ('job_seconds', 'stages', 'adapter_under'):
        if key in spec and key not in config:
            config[key] = spec[key]
    source = (ROOT / 'src' / 'k12' / 'q38_finetune.py').read_bytes()
    folder = ROOT / 'kaggle' / spec['folder']; folder.mkdir(parents=True, exist_ok=True)
    order_file = folder / 'order.json'  # test mode: score the rows the current system is least sure about first
    order = json.loads(order_file.read_text())['ids'] if mode == 'test' and order_file.exists() else []
    head = (f'CONFIG = {config!r}\nSCRIPT_SHA256 = {hashlib.sha256(source).hexdigest()!r}\n'
            f'SCRIPT_B64 = {base64.b64encode(source).decode()!r}\nORDER = {order!r}\n')
    (folder / 'run.py').write_text(head + (HERE / 'runner.py').read_text())
    meta = {'id': f"{OWNER}/{spec['slug']}", 'title': spec['title'], 'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': True, 'enable_gpu': spec['gpu'], 'enable_internet': False, 'dataset_sources': [DATASET],
            'competition_sources': [COMPETITION], 'kernel_sources': [TRAIN_KERNEL] if mode == 'test' else [], 'model_sources': [MODEL]}
    if spec['gpu']:
        meta['machine_shape'] = 'NvidiaTeslaT4'
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'folder': str(folder), 'kernel': meta['id'], 'config': config, 'script_sha256': hashlib.sha256(source).hexdigest()}, indent=1))


if __name__ == '__main__':
    main()
