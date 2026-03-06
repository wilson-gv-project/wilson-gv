"""
Claude Opus 4.6 
"""

import numpy as np


def build_permutation(mapping: dict[int, int], nmodes: int) -> np.ndarray:
    """Build array perm such that perm[h] = a (0-indexed).
    mapping is A(1-indexed) -> H(1-indexed)."""
    perm = np.empty(nmodes, dtype=int)
    for a, h in mapping.items():
        perm[h - 1] = a - 1
    return perm


def reindex_tensor(tensor: np.ndarray, perm: np.ndarray, mode_axes: list[int]) -> np.ndarray:
    """Reindex a tensor from A-numbering to H-numbering along the specified axes."""
    result = tensor
    for ax in mode_axes:
        result = np.take(result, perm, axis=ax)
    return result


def clean_noise(arr, tol=1e-10):
    """
    set to zero vals below tolerace
    """
    return np.where(np.abs(arr) < tol, 0.0, arr)