"""Mirror the train/test student photos and question diagrams one file at a time.

The competition's bulk archive downloads as an empty zip, so files are fetched
individually with the local Kaggle CLI credentials. Existing files are skipped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi

from k12.core import read_csv

SLUG = 'misconception-detection-in-k-12-student-math-workings'


def wanted(data):
    train, test = read_csv(Path(data) / 'train.csv'), read_csv(Path(data) / 'test.csv')
    files = [f'images/train/{name}' for name in train.image_file]
    files += [f'images/test/{name}' for name in test.image_file]
    diagrams = set(train.question_image_file) | set(test.question_image_file)
    files += [f'images/questions/{name}' for name in sorted(diagrams) if name.strip()]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--output', default='data/images')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    root = Path(args.output)
    missing = [f for f in wanted(args.data) if not (root / 'competition_data' / f).is_file()]
    print(f'{len(missing)} files to fetch', flush=True)

    def fetch(name):
        for attempt in range(5):
            try:
                api.competition_download_file(SLUG, f'competition_data/{name}', quiet=True,
                                              path=str(root / 'competition_data' / Path(name).parent))
                return name
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError(f'Failed to fetch {name}')

    start, failures = time.monotonic(), []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, name) for name in missing]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except RuntimeError as exc:
                failures.append(str(exc))
            if done % 250 == 0:
                print(f'{done}/{len(missing)} in {time.monotonic() - start:.0f}s', flush=True)
    print(f'finished: {len(missing) - len(failures)} fetched, {len(failures)} failed', flush=True)
    for failure in failures:
        print(failure)


if __name__ == '__main__':
    main()
