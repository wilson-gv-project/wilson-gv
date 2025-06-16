import numpy as np
from collections import Counter
from typing import List
from dataclasses import dataclass, field

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


@dataclass
class VibStatesDiff:
    diff_type: tuple

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
        return f'VibStatesDiff: {self.diff_type}'


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

