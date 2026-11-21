"""Parse historical text records. These are archival observations, never new-run results."""
import argparse
import json
import re
from pathlib import Path

from .settings import BENCHMARKS

REFERENCE_AP = {
    'table1_P': [99.86,98.40,98.38,89.42,99.84,98.13,92.23,84.18,72.09,98.14,98.79,76.29,98.49,94.33,98.46,95.49,97.07,96.48,97.15],
    'table1_NP': [99.94,99.87,99.24,97.10,99.92,99.50,87.34,80.06,79.48,94.85,96.94,87.78,99.41,96.93,99.44,99.35,99.59,99.28,99.55],
    'table4_mid': [99.50,98.34,97.38,98.83,98.51,96.31,73.16,74.52,57.31,90.51,95.70,75.96,99.39,98.07,99.45,97.98,98.66,98.25,99.79],
    'table4_high': [99.96,99.81,99.12,97.46,99.88,99.30,88.80,80.43,78.31,94.64,97.01,86.36,99.40,97.04,99.41,99.20,99.51,99.12,99.54],
}


def parse_logs(root):
    blocks = []
    for path in sorted(Path(root).rglob('*.txt')):
        if 'result' not in str(path.relative_to(root)):
            continue
        rows, start, context = {}, 0, ''
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = re.match(r'^(\w+)\s*:\s*([\d.]+)', line)
            if not match:
                if line.strip():
                    context = line.strip()
                continue
            key = match[1].lower()
            if key == 'progan':
                rows, start = {}, number
            if key in BENCHMARKS:
                values = re.match(r'^\w+\s*:\s*([\d.]+)\s+([\d.]+)\(([^)]*)\)\s+([\d.]+)\(([^)]*)\)', line)
                rows[key] = {'ap_percent': float(match[2])}
                if values:
                    rows[key].update(best_accuracy_percent=float(values[2]), best_threshold=float(values[3]),
                                     fixed_accuracy_percent=float(values[4]), fixed_threshold=float(values[5]))
            elif key == 'total' and set(rows) == set(BENCHMARKS):
                blocks.append({'file': str(path.relative_to(root)), 'start_line': start,
                               'end_line': number, 'context': context, 'rows': rows, 'logged_total': line.strip()})
                rows = {}
    return blocks


def audit(root):
    blocks = parse_logs(root)
    matches = {}
    for reference, expected in REFERENCE_AP.items():
        matches[reference] = [dict(file=b['file'], start_line=b['start_line'], end_line=b['end_line'],
                                  logged_total=b['logged_total']) for b in blocks
                              if all(abs(b['rows'][name]['ap_percent'] - value) < 0.005 for name,value in zip(BENCHMARKS, expected))]
    return {'kind': 'archived-log-audit', 'paper': 'https://arxiv.org/abs/2508.09477',
            'note': 'Exact matches are at the displayed two-decimal precision; logs do not prove executability or checkpoint identity.',
            'complete_blocks': len(blocks), 'exact_19_dataset_ap_matches': matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='legacy/cflowtest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from .engine import write_json
    write_json(args.output, audit(args.root))


if __name__ == '__main__':
    main()
