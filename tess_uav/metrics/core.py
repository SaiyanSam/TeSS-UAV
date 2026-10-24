"""Probability normalization, semantic entropy, and Jensen-Shannon divergence."""
import numpy as np
def normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    total = float(np.sum(x))

    if total <= eps:
        return None

    return x / total


def entropy_bits(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)

    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(p, num_entropy_classes=None, eps=1e-12):
    """
    Normalized entropy.

    By default, normalization uses the full evidence-vector dimension.

    For experiments where the evidence vector contains an ignored/background
    slot that can never receive evidence, pass num_entropy_classes explicitly.
    Example:
        shared taxonomy indices [0,1,2,3] with 0 ignored
        -> num_entropy_classes = 3
    """
    if num_entropy_classes is None:
        k = len(p)
    else:
        k = int(num_entropy_classes)

    if k <= 1:
        return 0.0

    return float(entropy_bits(p, eps=eps) / np.log2(k))


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence with log base 2.
    Bounded between 0 and 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    return float(0.5 * (kl_pm + kl_qm))

