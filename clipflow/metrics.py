import numpy as np
from sklearn.metrics import average_precision_score


def validate_scores(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError('Expected equal one-dimensional labels/scores with both classes')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite anomaly scores')
    return labels, scores


def calibrate_threshold(labels, scores):
    """Choose maximum balanced validation accuracy in O(n log n), including ties."""
    labels, scores = validate_scores(labels, scores)
    order = np.argsort(scores, kind='stable')
    ordered, sorted_scores = labels[order], scores[order]
    positives, negatives = (ordered == 1).sum(), (ordered == 0).sum()
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(ordered) - 1]
    removed_pos = np.cumsum(ordered == 1)[ends]
    removed_neg = np.cumsum(ordered == 0)[ends]
    qualities = np.r_[0.5, 0.5 * ((positives - removed_pos) / positives + removed_neg / negatives)]
    thresholds = np.r_[sorted_scores[0], np.nextafter(sorted_scores[ends], np.inf)]
    best = int(np.argmax(qualities))
    return float(thresholds[best]), float(qualities[best])


def evaluate_scores(labels, scores, threshold=None):
    labels, scores = validate_scores(labels, scores)
    result = {'ap': float(average_precision_score(labels, scores)),
              'count': int(len(labels)), 'real': int((labels == 0).sum()), 'fake': int((labels == 1).sum())}
    result['accuracy'] = None if threshold is None else float(((scores >= threshold) == labels).mean())
    return result
