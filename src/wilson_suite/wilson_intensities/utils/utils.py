"""
Utility functions and classes. Related to different parts of calculations and setup.


"""
import numpy as np

from CQCParse.parsing import CFOURParser, CFOUROutput, GaussianParser, GaussianOutput
import pickle

from CQCParse.parsing import GaussianDataParser, CFOURdataParser, ParsedData
from dataclasses import dataclass, field

import os
# Get the root directory of the package dynamically
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_package_root() -> str:
    """Returns the absolute path to the package root."""
    return PACKAGE_ROOT



@dataclass
class Conditions:
    """Dataclass that collects spectrum simulation configs."""

    Gamma_rc: float
    diag_margin_rc: float
    dynamic_range_n: int|float
    omega1: np.ndarray
    omega2: np.ndarray
    program: str
    data_parser: CFOURdataParser|GaussianDataParser
    molecule: str
    method: str
    basis: str
    new_idx_dict : dict
    el_terms_selected: list
    mech_terms_selected: list
    list2exclude: list = None
    only_modes: list = None
    vpt2settings: dict = field(default_factory=lambda: {'anharmonic_type': 'GVPT2'})
    vib_levels_harmonic: bool = False
    preview: bool = False


def prep_data_load(parsed_data: ParsedData) -> tuple:
    """
    Collect data from parser result, ParsedData. 
    
    Used with TermND.
    """
    # todo? refactor these attribute names?
    ddata = [parsed_data.derivatives.dipgrad,
             parsed_data.derivatives.diphess,
             parsed_data.derivatives.polgrad,
             parsed_data.derivatives.polhess,
             parsed_data.derivatives.cff]
    # naming starts here, internal wilson_intensities naming,
    # later used to set up props data for DataForPrecalc
    deriv_data = dict(zip(['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff'], ddata))

    allstates = parsed_data.vib_states.anharmonic_states
    harmonic_states = parsed_data.vib_states.harmonic_states

    mode_indices = [i for i in np.arange(parsed_data.nmodes) if i not in parsed_data.list2exclude]

    return deriv_data, allstates, harmonic_states, mode_indices


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