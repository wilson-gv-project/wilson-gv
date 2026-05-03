"""
Utility functions and classes. Related to different parts of calculations and setup.


"""
import numpy as np

import os
# Get the root directory of the package dynamically
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_package_root() -> str:
    """Returns the absolute path to the package root."""
    return PACKAGE_ROOT


def pairwise_differences(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Chatgpt.

    for vib levels diffs tensors

    # ApBmA[a, b] = ApB[a, b] - A[b] = A[a] + B[b] - A[b]
    
    # from 2d array subtract 1d array => from each row subtract this 1d array
    # then it means: quant2[0,1] is quant2[a,b] - quant1[a] = diff2_1[b,a]

    # state1, state2, state2-state1
    # harmonic state is given by index of NM
    # complex state has a composition, and will have a new label
    
    """
    a = np.asarray(A)
    b = np.asarray(B)

    # Reshape a to (a₁, ..., aₙ, 1, ..., 1) with m trailing 1s
    a_broad = a.reshape(*a.shape, *([1] * b.ndim))

    # Reshape B to (1, ..., 1, b₁, ..., bₘ) with n leading 1s
    b_broad = b.reshape(*([1] * a.ndim), *b.shape)

    return a_broad - b_broad


def coolprint(text: str) -> None:
    """Print yellow text."""
    from rich import print
    print(f"[italic yellow2]{text}[/italic yellow2]")