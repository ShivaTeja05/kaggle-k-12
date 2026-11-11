"""Score any CSV of workings with a saved pod checkpoint (test rows or the unlabelled pool)."""
import argparse
import json
from pathlib import Path

from .core import label_space, read_csv
from .pod_train import predict
from .vision import VisionSystem, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='epoch-N directory written by pod_train')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, help='competition_data directory with taxonomy and images/')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--split', required=True, help='image folder under images/, e.g. test or unlabelled')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    metadata = json.loads((Path(args.checkpoint) / 'metadata.json').read_text())
    classes = label_space(read_csv(Path(args.data) / 'taxonomy_v1.csv'))
    if metadata['classes'] != classes:
        raise ValueError('Checkpoint label order differs from the taxonomy')
    seed_all(metadata['args']['seed'])
    options = argparse.Namespace(**metadata['args'])
    frame = read_csv(args.csv)
    if args.limit:
        frame = frame.head(args.limit)
    system = VisionSystem(args.model, classes, options, checkpoint=args.checkpoint)
    predict(system, frame, Path(args.data), args.split, args.output, args.batch_size)
    print('K12_PREDICT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
