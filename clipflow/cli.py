import argparse
import json
import random
from pathlib import Path

from .settings import Settings


def build_parser():
    parser = argparse.ArgumentParser(description='CLIP-Flow training and evaluation')
    sub = parser.add_subparsers(dest='action', required=True)
    split = sub.add_parser('split', help='Prepare disjoint natural-image training/validation manifests')
    split.add_argument('--real-root', required=True)
    split.add_argument('--output', required=True)
    split.add_argument('--train-count', type=int, default=7200)
    split.add_argument('--validation-count', type=int, default=1000)
    split.add_argument('--seed', type=int, default=1000)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--train-manifest', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--allow-other-count', action='store_true')
    cal = sub.add_parser('calibrate')
    cal.add_argument('--checkpoint', required=True)
    cal.add_argument('--validation-manifest', required=True)
    cal.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--test-root', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--calibration', help='Omit for AP-only evaluation')
    evaluate.add_argument('--suite', choices=['benchmark', 'emerging'], default='benchmark')
    evaluate.add_argument('--datasets', nargs='+')
    evaluate.add_argument('--mapping', help='JSON object mapping canonical dataset names to local directories')
    evaluate.add_argument('--allow-other-count', action='store_true')
    legacy = sub.add_parser('import-legacy')
    legacy.add_argument('--source', required=True)
    legacy.add_argument('--config', required=True)
    legacy.add_argument('--train-manifest', required=True)
    legacy.add_argument('--output', required=True)
    for command in (train, cal, evaluate, legacy):
        command.add_argument('--device', default='auto')
        command.add_argument('--cache-dir')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import engine
    if args.action == 'split':
        from .data import images
        paths = images(args.real_root)
        if args.train_count < 1 or args.validation_count < 1 or len(paths) < args.train_count + args.validation_count:
            raise ValueError('Not enough images for requested disjoint splits')
        random.Random(args.seed).shuffle(paths)
        output = Path(args.output)
        engine.write_json(output / 'train.json', list(map(str, paths[:args.train_count])))
        engine.write_json(output / 'validation.json', list(map(str, paths[args.train_count:args.train_count + args.validation_count])))
        engine.write_json(output / 'split.json', {'seed': args.seed, 'train_count': args.train_count,
                          'validation_count': args.validation_count, 'source': str(Path(args.real_root).resolve()),
                          'selection': 'seeded random partition'})
    elif args.action == 'train':
        cfg = Settings(**json.loads(Path(args.config).read_text())).validate()
        engine.train(cfg, args.train_manifest, args.output, args.device, args.cache_dir,
                     allow_other_count=args.allow_other_count)
    elif args.action == 'calibrate':
        engine.calibrate(args.checkpoint, args.validation_manifest, args.output, args.device, args.cache_dir)
    elif args.action == 'evaluate':
        mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else None
        engine.evaluate(args.checkpoint, args.test_root, args.output, args.calibration, args.suite,
                        args.datasets, mapping, args.device, args.cache_dir,
                        allow_other_count=args.allow_other_count)
    elif args.action == 'import-legacy':
        cfg = Settings(**json.loads(Path(args.config).read_text())).validate()
        engine.import_legacy(args.source, cfg, args.train_manifest, args.output, args.device, args.cache_dir)


if __name__ == '__main__':
    main()
