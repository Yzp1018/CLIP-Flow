"""Render new-run score histograms and dimension ablations; never use paper values as run results."""
import argparse
import json
from pathlib import Path

import numpy as np


def histograms(p_results, np_results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['progan', 'stargan', 'guided', 'ldm_200']
    figure, axes = plt.subplots(2, 4, figsize=(14, 6))
    for row, (mode, root) in enumerate([('P', p_results), ('N+P', np_results)]):
        for col, name in enumerate(names):
            values = np.load(Path(root) / (name + '.npz'), allow_pickle=False)
            labels, scores = values['labels'], values['scores']
            axis = axes[row, col]
            bins = np.histogram_bin_edges(scores, bins=50)
            for label, caption in [(0, 'Natural'), (1, 'AI-generated')]:
                axis.hist(scores[labels == label], bins=bins, density=True, alpha=0.6, label=caption)
            axis.set(title=mode + ': ' + name, xlabel='Anomaly score', ylabel='Density')
    axes[0, 0].legend()
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def dimensions(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .experiments import GROUPS
    labels, aps, counts = [], [], []
    no_dr = None
    for name in GROUPS['dimensions']:
        folder = Path(root) / name
        settings = json.loads((folder / 'training.json').read_text())
        metrics = json.loads((folder / 'benchmark/metrics.json').read_text())
        if name == 'no_dr':
            no_dr = (100 * metrics['mean_ap'], settings['trainable_parameters'])
            continue
        labels.append(str(settings['settings']['dimension']) if name != 'no_dr' else 'No DR')
        aps.append(100 * metrics['mean_ap'])
        counts.append(settings['trainable_parameters'])
    figure, left = plt.subplots(figsize=(8, 4))
    right = left.twinx()
    left.plot(labels, aps, 'o-', color='tab:blue')
    right.plot(labels, counts, 's--', color='tab:orange')
    if no_dr is not None:
        left.axhline(no_dr[0], color='tab:blue', linestyle=':', label='No DR mAP')
        right.axhline(no_dr[1], color='tab:orange', linestyle=':', label='No DR parameters')
        left.legend(loc='lower left')
        right.legend(loc='upper right')
    left.set(xlabel='Feature dimension', ylabel='mAP (%)')
    right.set(ylabel='Trainable parameters')
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    hist = sub.add_parser('histograms')
    hist.add_argument('--p-results', required=True)
    hist.add_argument('--np-results', required=True)
    hist.add_argument('--output', required=True)
    dim = sub.add_parser('dimensions')
    dim.add_argument('--root', required=True)
    dim.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'histograms':
        histograms(args.p_results, args.np_results, args.output)
    else:
        dimensions(args.root, args.output)


if __name__ == '__main__':
    main()
