import numpy as np
from collections import Counter
from typing import List
from dataclasses import dataclass, field

from numpy.ma.core import indices

from wilson.spectrum.tools import convNu2Ene
# alldata = [Nnmodes, data, avrg_terms, axes_dict,
#            term_with_data.states_arrays_Eh,
#            term_with_data.harmonic_arrays_Eh]
@dataclass
class DataForPrecalc:
    Nnmodes: int
    props_data: dict
    avrg_terms: np.ndarray
    axes_dict: dict
    states_arrays_Eh: dict
    harmonic_arrays_Eh: dict

@dataclass
class MolProperty:
    """
    MolProperty(name='mu_Q', cart_axes=('B',), nm_indices=('a',)),
    MolProperty(name='mu_QQ', cart_axes=('G',), nm_indices=('a', 'b'))

    MolProperty(name='alpha_Q', cart_axes=('A', 'D'), nm_indices=('b',)),
    MolProperty(name='alpha_QQ', cart_axes=('A', 'D'), nm_indices=('a', 'b')),

    MolProperty(name='CFF', cart_axes=(), nm_indices=('a', 'b', 'c'))

    """
    name: str
    cart_axes: tuple[str]
    nm_indices: tuple[str]
    tensor: np.ndarray = 0.

    @property
    def simple_tuple(self):
        """
        len(self.cart_axes) - number of EL perturbations
        len(self.nm_indices) - number of normal coordinates derivatives
        """
        return tuple([len(self.cart_axes), len(self.nm_indices)])

    # def __repr__(self):
        # return rf'\{self.name.strip("_Q")}_{','.join(list(self.cart_axes))}__dQ{','.join(list(self.nm_indices))}'
        # return f'{self.simple_tuple}'


@dataclass
class VibState:
    quanta_dict: dict
    freq: float

    # def __post_init__(self):
    #     if not isinstance(self.name, str):
    #         raise TypeError("name must be a string")
    #     if not isinstance(self.age, int):
    #         raise TypeError("age must be an int")


@dataclass
class VibStatesDiff:
    diff_type: tuple
    res_cond: bool
    pf_type: tuple = None
    diff_str: str = ''

    def __eq__(self, other):
        if not isinstance(other, VibStatesDiff):
            return False
        return (
            self.diff_type == other.diff_type
        )

    def __hash__(self):
        # Since Counter doesn't have a hash, we convert it to a frozenset of items
        return hash(self.diff_type)

    def __repr__(self):
        s = ''
        if self.res_cond:
            s += f'pf_type: {self.pf_type}) '
        if self.diff_str:
            s += f'diff_str: {self.diff_str}) '
        return f'VibStatesDiff: {self.diff_type}, res_cond? - {self.res_cond}. '+s


@dataclass
class AveragedProps:
    props: List[MolProperty] = field(default_factory=list)

    @property
    def cart_axes(self):
        return tuple([p.cart_axes for p in self.props])

    @property
    def nm_indices(self):
        return tuple([p.nm_indices for p in self.props])

    # def __eq__(self, other):
    #     if not isinstance(other, AveragedProps):
    #         return False
    #     return (
    #         self.cart_axes == other.cart_axes and
    #         Counter(self.nm_indices) == Counter(other.nm_indices)
    #     )

    def __eq__(self, other):
        if not isinstance(other, AveragedProps):
            return False
        return (
            self.cart_axes == other.cart_axes and
            sorted(self.nm_indices) == sorted(other.nm_indices)
        )

    def __hash__(self):
        # Since Counter doesn't have a hash, we convert it to a frozenset of items
        return hash((self.cart_axes, frozenset(Counter(self.nm_indices).items())))

    def __repr__(self):
        return f'\nAveragedProps:\n   {self.cart_axes}\n   {self.nm_indices}\n'

class DoubleDict:
    def __init__(self):
        self.kv = {}
        self.vk = {}

    def add(self, k, v):
        self.kv[k] = v
        self.vk[v] = k

    def get_by_key(self, k):
        return self.kv.get(k)

    def get_by_value(self, v):
        return self.vk.get(v)


def dict2arraydict(states_dict):
    """
    format transformation for vib states freqs data
    """
    states_arrs = {}
    d1 = {k:v for k,v in states_dict.items() if len(k)==1}
    d2 = {k:v for k,v in states_dict.items() if len(k)==2}
    d3 = {k:v for k,v in states_dict.items() if len(k)==3}

    if d1:
        states_arrs[1] = np.array(list(d1.values()))
    if d2:
        states_arrs[2] = np.zeros((len(d1), len(d1)))
        for ab in d2:
            states_arrs[2][(int(ab[0]), int(ab[1]))] = d2[ab]
            states_arrs[2][(int(ab[1]), int(ab[0]))] = d2[ab]
    if d3:
        states_arrs[3] = np.zeros((len(d1), len(d1), len(d1)))
        for abc in d3:
            states_arrs[3][(int(abc[0]), int(abc[1]), int(abc[2]))] = d3[abc]
            states_arrs[3][(int(abc[0]), int(abc[2]), int(abc[1]))] = d3[abc]
            states_arrs[3][(int(abc[1]), int(abc[0]), int(abc[2]))] = d3[abc]
            states_arrs[3][(int(abc[1]), int(abc[2]), int(abc[0]))] = d3[abc]
            states_arrs[3][(int(abc[2]), int(abc[0]), int(abc[1]))] = d3[abc]
            states_arrs[3][(int(abc[2]), int(abc[1]), int(abc[0]))] = d3[abc]

    states_arrs[0] = 0.
    return states_arrs


def mainVibStates2arraydict(listVibStates, Nnmodes):
    """
    vibState {('0',): 1.0}, energy is 3560.764 cm-1
    vibState {('6', '6'): 1.0}, energy is 2591.707 cm-1
    """
    states_arrs = {}
    states_arrs[1] = np.zeros(Nnmodes)
    states_arrs[2] = np.zeros((Nnmodes, Nnmodes))
    states_arrs[3] = np.zeros((Nnmodes, Nnmodes, Nnmodes))

    from itertools import permutations

    for vs in listVibStates:
        if len(vs.s)==1:
            for k_tuple in vs.s:
                perms = set(permutations(tuple([int(i) for i in k_tuple])))
                for p in perms:
                    states_arrs[len(k_tuple)][p] = convNu2Ene(vs.e) if check_energy_unit(vs.e) == 'cm-1' else vs.e
                    # states_arrs[len(k_tuple)][p] = vs.e

    states_arrs[0] = 0.

    return states_arrs

def safe_product(parts):
    result = 1
    for part in parts:
        if part == 0:
            return 0
        result *= part
    return result


def check_energy_unit(value):
    """
    find a reasonable energy unit
    """
    if value < 1.:
        return 'Hartree'
    else:
        return 'cm-1'

from contextlib import contextmanager
@contextmanager
def debug_mode(level):
    """
    Context manager to temporarily set the debug level.
    """
    import wilson.debug as debug

    original_level = debug.level
    debug.level = level
    try:
        yield
    finally:
        debug.level = original_level


def make_abc_tuple(in_tuple, final_len):
    """
    extend ab tuple to abc tuple - num_rescond_abc to num_unique_abc
    num_unique_abc >= num_rescond_abc
    extend tuple with None values
    """
    return tuple([*in_tuple]+[None]*(final_len-len(in_tuple)))


def get_indices(term):
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


def flatten_list(nested_list):
    import itertools
    newlist = list(itertools.chain(*nested_list))
    if list in [type(l) for l in newlist]:
        return flatten_list(newlist)
    else:
        return newlist

def get_allparts_indices(term):

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

    allidx = len(set([j for i in sets for j in sets[i]]))
    res_idx = len(sets['resonances'])
    return allidx, res_idx

import string
abc_list = list(string.ascii_lowercase)
num_Greek = {0: 'A', 1: 'B', 2: 'G', 3: 'D', 4: 'E', 5: 'Z', 6: 'H', 7: 'T', 8: 'I'}
greek_list = list(num_Greek.values())

def make_abc_dict(abc_comb):
    return {l: n for l, n in zip(abc_list[: len(abc_comb)], abc_comb)}
