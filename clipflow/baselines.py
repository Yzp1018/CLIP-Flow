"""Common evaluation of external-method anomaly-score CSVs; does not implement their training."""
import argparse
import csv
from collections import defaultdict

from .engine import write_json
from .metrics import calibrate_threshold, evaluate_scores


def read_scores(path):
    with open(path, newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records or not {'dataset', 'image_id', 'label', 'score'}.issubset(records[0]):
        raise ValueError('CSV columns: dataset,image_id,label,score; larger scores mean fake')
    keys = [(r['dataset'], r['image_id']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate sample keys in score CSV')
    return records


def compare(test_csv, output, validation_csv=None):
    test = read_scores(test_csv)
    threshold = None
    if validation_csv:
        validation = read_scores(validation_csv)
        # Image IDs must be stable globally across the two exports.
        if {r['image_id'] for r in test} & {r['image_id'] for r in validation}:
            raise ValueError('Validation/test image IDs overlap')
        threshold, _ = calibrate_threshold([int(r['label']) for r in validation], [float(r['score']) for r in validation])
    grouped = defaultdict(list)
    for record in test:
        grouped[record['dataset']].append(record)
    result = {'threshold': threshold, 'datasets': {}}
    for name, records in grouped.items():
        result['datasets'][name] = evaluate_scores([int(r['label']) for r in records], [float(r['score']) for r in records], threshold)
    result['mean_ap'] = sum(r['ap'] for r in result['datasets'].values()) / len(grouped)
    result['mean_accuracy'] = None if threshold is None else sum(r['accuracy'] for r in result['datasets'].values()) / len(grouped)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--validation-csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.test_csv, args.output, args.validation_csv)


if __name__ == '__main__':
    main()
