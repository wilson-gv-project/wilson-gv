"""
Utility functions and classes. Related to different parts of calculations and setup.

Mainly about calculations
"""
import numpy as np
from collections import Counter
from typing import List
from dataclasses import dataclass, field
from ...wilson_utils.unit_convertor import convNu2Ene
import string
from contextlib import contextmanager
from typing import Dict, Any, Self
from collections.abc import Hashable


def safe_product(parts: list | tuple) -> float:
    """
    Returns 0. when one part of product is 0.
    """
    result = 1.
    if any(parts) == 0.:
        return 0.
    for part in parts:
        result *= part
    return result


def check_energy_unit(value: float) -> str:
    """
    Find a reasonable energy unit for given value
    """
    if value < 1.:
        return 'Hartree'
    else:
        return 'cm-1'


@contextmanager
def debug_mode(level: int):
    """
    Context manager to temporarily set the debug level.
    """
    from ...wilson_utils import printing as debug

    original_level = debug.level
    debug.level = level
    try:
        yield
    finally:
        debug.level = original_level


def make_abc_tuple(in_tuple: tuple, final_len: int) -> tuple:
    """
    Extend ab tuple to abc tuple - num_rescond_abc to num_unique_abc

    num_unique_abc >= num_rescond_abc
    extend tuple with None values
    """
    return tuple([*in_tuple]+[None]*(final_len-len(in_tuple)))


def get_indices(term: dict) -> dict:
    """
     {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
      'vibenediff': ('a+b+c,zero', 'c,a+b'),
      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
      'non_averaged_props': (('F', ('a', 'b', 'c',)),),
      'vibene_denom': ('a','b','c'),
      'termB_pref': 1.,
      'termA_pref': -1/48.}

    indices are in tuples in tuples: resonances, averaged_props, non_averaged_props
    but vibenediff has str in tuples
    """
    res_idx = [[j.split('+') for j in i[0].split(',')] for i in term['resonances']]
    if term['vibenediff'] is not None:
        vd_idx = [[j.split('+') for j in i.split(',')] for i in term['vibenediff']]
    else:
        vd_idx = []
    arvrg_idx = [list(i[1]) for i in term['averaged_props']]
    if term['non_averaged_props'] is not None:
        non_arvrg_idx = [list(i[1]) for i in term['non_averaged_props']]
    else:
        non_arvrg_idx = []
    vibene_idx = list(term['vibene_denom'])

    return {'resonances': res_idx,
            'vibenediff': vd_idx,
            'arvrg_idx': arvrg_idx,
            'non_arvrg_idx': non_arvrg_idx,
            'vibene': vibene_idx}


def flatten_list(nested_list: list) -> list:
    """
    Flatten nested list
    """
    import itertools
    newlist = list(itertools.chain(*nested_list))
    if list in [type(list_in) for list_in in newlist]:
        return flatten_list(newlist)
    else:
        return newlist


def get_allparts_indices(term: dict) -> tuple[int, int]:
    """
    Extract mode indices from term expression, from the whole term (all_idx) or resonance condition part only (res_idx)
    """
    resultdict = get_indices(term)

    s1 = set(flatten_list(resultdict['vibenediff']))
    s2 = set(flatten_list(resultdict['resonances']))
    s3 = set(flatten_list(resultdict['arvrg_idx']))
    s4 = set(flatten_list(resultdict['non_arvrg_idx']))
    s5 = set(resultdict['vibene'])
    sets = {'vibenediff': s1,
            'resonances': s2,
            'arvrg_idx': s3,
            'non_arvrg_idx': s4,
            'vibene_idx': s5}
    for s in sets:
        sets[s].discard("zero")

    all_idx = len(set([j for i in sets for j in sets[i]]))
    res_idx = len(sets['resonances'])

    return all_idx, res_idx

# a list of lowercase letters of alphabet in order
abc_list = list(string.ascii_lowercase)
# dictionary of nulerals to latinized Greek letters 
num_Greek = {0: 'A', 1: 'B', 2: 'G', 3: 'D', 4: 'E', 5: 'Z', 6: 'H', 7: 'T', 8: 'I'}
# list of latinized Greek letters in order
greek_list = list(num_Greek.values())


def make_abc_dict(abc_comb: tuple) -> dict:
    """
    Compliling a dictionary of letter indices to their numerical values.

    In order of restective lists!
    """
    return {letter: number for letter, number in zip(abc_list[: len(abc_comb)], abc_comb)}
