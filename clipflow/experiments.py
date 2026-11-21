"""Plan or run all CLIP-Flow experiment families, with one model per configuration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


GROUPS = {
    'main': ['N', 'P', 'NP'],
    'proxies': ['NP', 'proxy_smooth', 'proxy_sharpen', 'proxy_noise', 'proxy_jitter', 'proxy_phase'],
    'bands': ['NP', 'band_mid', 'band_high'],
    'dimensions': ['dim_768', 'dim_512', 'dim_256', 'NP', 'dim_64', 'dim_32', 'dim_16', 'dim_8', 'dim_4', 'dim_2', 'no_dr'],
    'normalization': ['NP', 'no_normalization'],
    'emerging': ['P'],
}


def plan(groups, train_manifest, validation_manifest, test_root, output, emerging_root=None,
         device='auto', config_root=None):
    config_root = Path(config_root or Path(__file__).resolve().parents[1] / 'configs')
    output = Path(output)
    selected = list(dict.fromkeys(name for group in groups for name in GROUPS[group]))
    commands = []
    for name in selected:
        folder = output / name
        checkpoint, calibration = folder / 'last.pt', folder / 'calibration.json'
        prefix = [sys.executable, '-m', 'clipflow']
        commands.append(prefix + ['train', '--config', str(config_root / (name + '.json')),
                        '--train-manifest', str(train_manifest), '--output', str(folder), '--device', device])
        commands.append(prefix + ['calibrate', '--checkpoint', str(checkpoint),
                        '--validation-manifest', str(validation_manifest), '--output', str(calibration), '--device', device])
        # AP and fixed-threshold accuracy are evaluated together on all 19 datasets.
        commands.append(prefix + ['evaluate', '--checkpoint', str(checkpoint), '--test-root', str(test_root),
                        '--calibration', str(calibration), '--output', str(folder / 'benchmark'), '--device', device])
    if 'emerging' in groups:
        if not emerging_root:
            raise ValueError('--emerging-root is required for the emerging-model experiment')
        commands.append([sys.executable, '-m', 'clipflow', 'evaluate', '--checkpoint', str(output / 'P/last.pt'),
                         '--suite', 'emerging', '--test-root', str(emerging_root),
                         '--output', str(output / 'P/emerging'), '--device', device])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=list(GROUPS) + ['all'], default=['main'])
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--validation-manifest', required=True)
    parser.add_argument('--test-root', required=True)
    parser.add_argument('--emerging-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--execute', action='store_true', help='Without this flag, print the plan only')
    args = parser.parse_args(argv)
    groups = list(GROUPS) if 'all' in args.groups else args.groups
    commands = plan(groups, args.train_manifest, args.validation_manifest, args.test_root,
                    args.output, args.emerging_root, args.device)
    print(json.dumps(commands, indent=2))
    if args.execute:
        from .data import disjoint, manifest
        disjoint(manifest(args.train_manifest), manifest(args.validation_manifest))
        for command in commands:
            if 'train' in command:
                target = Path(command[command.index('--output') + 1])
                if (target / 'last.pt').exists():
                    raise FileExistsError('Use a fresh run directory to preserve existing checkpoints: ' + str(target))
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
