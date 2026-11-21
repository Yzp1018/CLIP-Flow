"""Paper main experiment: train -> held-out calibration -> 19-dataset evaluation.

Default: N+P, low-frequency masking ratio 0.1, 10 epochs.
Select --mode P for proxy-only, ratio 1.0, 30 epochs; --mode N for natural-only.
"""
import argparse
import json
from pathlib import Path

from clipflow import engine
from clipflow.data import disjoint, labelled_images, manifest
from clipflow.settings import BENCHMARKS, Settings


def main(argv=None, encoder=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['N', 'P', 'NP', 'N+P'], default='NP')
    parser.add_argument('--train-manifest', default='data/splits/train.json')
    parser.add_argument('--validation-manifest', default='data/splits/validation.json')
    parser.add_argument('--test-root', default='data/benchmark')
    parser.add_argument('--output', help='Fresh run directory; default runs/main_MODE')
    parser.add_argument('--checkpoint', help='Evaluate a matching main-experiment checkpoint instead of training')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cache-dir')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--epochs', type=int, help='Explicit schedule override; omitted for paper defaults')
    parser.add_argument('--allow-other-count', action='store_true', help='Allow a training set size other than 7200 images')
    args = parser.parse_args(argv)
    mode = 'NP' if args.mode == 'N+P' else args.mode
    cfg_path = Path(__file__).resolve().parent / 'configs' / (mode + '.json')
    cfg = Settings(**json.loads(cfg_path.read_text()))
    cfg.workers, cfg.seed = args.workers, args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.validate()
    output = Path(args.output or ('runs/main_' + mode))

    # Check all splits and all 19 datasets before a potentially long training run.
    validation = manifest(args.validation_manifest)
    training = [] if args.checkpoint else manifest(args.train_manifest)
    if training and not args.allow_other_count and len(training) != 7200:
        raise ValueError('Main experiment requires 7200 natural training images')
    disjoint(training, validation)
    for name in BENCHMARKS:
        folder = Path(args.test_root) / name
        if (folder / 'test').is_dir():
            folder = folder / 'test'
        records = labelled_images(folder)
        disjoint(training, validation, [path for path, _ in records])

    if args.checkpoint:
        import torch
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if state.get('format') != 'clipflow-consolidated-v1':
            raise ValueError('Convert this checkpoint format with python -m clipflow import-legacy first')
        loaded = Settings(**state['settings'])
        primary = (loaded.mode == mode and loaded.dimension == 128 and loaded.normalize
                   and loaded.proxy == 'frequency' and loaded.band == 'low' and loaded.ratio == cfg.ratio)
        if not primary:
            raise ValueError('Checkpoint settings do not match the selected paper main experiment')
        checkpoint = Path(args.checkpoint)
    else:
        print('1/3 Training main experiment: ' + mode, flush=True)
        engine.train(cfg, args.train_manifest, output, args.device, args.cache_dir,
                     encoder=encoder, allow_other_count=args.allow_other_count)
        checkpoint = output / 'last.pt'

    print('2/3 Calibrating one threshold on held-out validation data', flush=True)
    calibration = output / 'calibration.json'
    engine.calibrate(checkpoint, args.validation_manifest, calibration,
                     args.device, args.cache_dir, encoder=encoder)

    print('3/3 Evaluating all 19 benchmark datasets', flush=True)
    result = engine.evaluate(checkpoint, args.test_root, output / 'benchmark',
                             calibration=calibration, device=args.device, cache_dir=args.cache_dir,
                             encoder=encoder)
    print(json.dumps({'mode': mode, 'mAP_percent': 100 * result['mean_ap'],
                      'mean_accuracy_percent': 100 * result['mean_accuracy'],
                      'results': str(output / 'benchmark/metrics.json')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
