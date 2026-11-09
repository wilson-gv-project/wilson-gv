"""
plan.py — the compile stage. Runs with zero molecular data.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_derive.abstractions import ResonanceCondition
from wilson_suite.wilson_response_evaluate.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResCondKey,
    ResLocPoint,
    ResonanceMotif,
    generate_LHS_motif,
)
from wilson_suite.wilson_response_evaluate.tests.helpers import (
    AXES,
    MOTIF_A,
    MOTIF_AB,
    MOTIF_B_TWICE,
    MOTIF_MIXED,
    NDIMS,
    coords,
    polprop,
    vibdiff,
)


def vibdiff_keys(coll: FreqTermsCollection) -> list[tuple]:
    """VibDiffTerm compares by identity and the collection deepcopies, so compare by content."""
    return [(tuple(ft.sl.q), tuple(ft.sr.q), ft.is_pert_wf_diff) for ft in coll] # type: ignore


## PropsCollection ----------------------------------------------------------

def test_propscollection_copies_its_inputs():
    original = polprop(ops=(0, 1), inds='a')
    coll = PropsCollection([original])

    original.setInds(['b'])

    assert coll.get_mode_indices() == ('a',)


def test_propscollection_splits_averaged_and_non_averaged():
    avrg = polprop(ops=(0, 1), inds='a')
    non_avrg = polprop(ops=(), inds='abc')
    coll = PropsCollection([non_avrg, avrg])

    assert list(coll.get_averaged_props()) == [avrg]
    assert list(coll.get_non_averaged_props()) == [non_avrg]


def test_propscollection_flattens_axes_and_indices_in_order():
    coll = PropsCollection([polprop(ops=(2, 3), inds='a'), polprop(ops=(0,), inds='ab')])

    assert coll.get_cart_axes() == (2, 3, 0)
    assert coll.get_mode_indices() == ('a', 'a', 'b')


def test_propscollection_sort_puts_non_averaged_last():
    coll = PropsCollection([polprop(inds='abc'), polprop(ops=(3,), inds='a'), polprop(ops=(1,), inds='b')])

    assert [p.ops[0].o if p.ops else None for p in coll.sort()] == [1, 3, None]


def test_propscollection_names_one_per_property_in_order():
    """term 8's averaged part polgrad[b] * dipgrad[a] * dipgrad[c] + cff[a, b, c]: dipgrad is named twice"""
    coll = PropsCollection([polprop(ops=(0, 3), inds='b'), polprop(ops=(1,), inds='a'), polprop(ops=(2,), inds='c'),
                            polprop(inds='abc')])

    assert coll.names == ('polgrad', 'dipgrad', 'dipgrad', 'cff')


def test_propscollection_equal_by_value_and_usable_as_key():
    a = PropsCollection([polprop(ops=(0, 1), inds='a')])
    b = PropsCollection([polprop(ops=(0, 1), inds='a')])

    assert a == b
    assert hash(a) == hash(b)
    assert {a: 'tensor'}[b] == 'tensor'


def test_propscollection_equality_is_symmetric_ignores_order_and_counts_repeats():
    p1, p2 = polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='b')
    one, both = PropsCollection([p1]), PropsCollection([p1, p2])

    assert one != both and both != one
    assert PropsCollection([]) != one and one != PropsCollection([])
    assert PropsCollection([p1, p1]) != one                 # a repeated prop is another product
    assert both == PropsCollection([p2, p1]) and hash(both) == hash(PropsCollection([p2, p1]))


def test_propscollection_equality_counts_the_order_of_slots_in_a_prop():
    """the slots go to the data array's Cartesian axes in order, so (0, 1) and (1, 0) are different props"""
    assert PropsCollection([polprop(ops=(0, 1), inds='a')]) != PropsCollection([polprop(ops=(1, 0), inds='a')])


## FreqTermsCollection ------------------------------------------------------

def test_freqterms_equal_by_value_symmetric_and_usable_as_key():
    a = FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='ab', sr='c', pert=True)])
    b = FreqTermsCollection([vibdiff(sl='ab', sr='c', pert=True), vibdiff(sl='a')])   # other objects, other order

    assert a == b and hash(a) == hash(b)
    assert {a: 'rows'}[b] == 'rows'
    assert FreqTermsCollection([]) != a and a != FreqTermsCollection([])
    assert a != FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='ab', sr='c', pert=False)])
    assert FreqTermsCollection([vibdiff(sl='a')] * 2) != FreqTermsCollection([vibdiff(sl='a')])


def test_compiled_term_compiled_twice_is_equal():
    """CompiledTerm's == compares its collections, so the same term compiled twice is equal"""
    term = _fake_vibpert_term()

    assert CompiledTerm.from_VibPertTerm(term) == CompiledTerm.from_VibPertTerm(term)  # type: ignore


def test_freqterms_split_by_pert_wf_diff():
    harmonic = vibdiff(sl='a')                     # 1/omega_a
    pert = vibdiff(sl='ab', sr='a', pert=True)     # from a perturbed wavefunction
    coll = FreqTermsCollection([harmonic, pert])

    assert vibdiff_keys(coll.get_vibenedenom()) == [(('a',), (), False)]
    assert vibdiff_keys(coll.get_pert_wf_diff()) == [(('a', 'b'), ('a',), True)]


## ResCondKey ---------------------------------------------------------------
# One resonance condition stripped to what identifies it: which quanta sit on each side of the
# energy difference, and which perturbing frequencies enter. plan.py keeps these instead of
# derive's ResonanceCondition so a ResonanceMotif can be hashed, compared and pickled without
# holding derive objects.



def test_rescondkey_pf_order_does_not_matter():
    key = ResCondKey(diff=(('a',), ()), pf=('A', '-B'))
    same = ResCondKey(diff=(('a',), ()), pf=('-B', 'A'))

    assert key == same and hash(key) == hash(same)
    assert ResonanceMotif((key,)) == ResonanceMotif((same,))


## ResonanceMotif -----------------------------------------------------------

def test_resonance_motif_from_conditions_matches_from_tuples():
    """Both constructors reduce each condition to the same ResCondKey: quanta sorted (as
    HarmOscStateSymbolic does), ground state as (), pf list -> tuple."""
    conds = [
        ResonanceCondition(diff=vibdiff(sl='ba'), pf=['1']),              # E_ab - E_0
        ResonanceCondition(diff=vibdiff(sl='a', sr='c'), pf=['-1', '2']),  # E_a - E_c
    ]

    from_conds = ResonanceMotif.from_conditions(conds)
    from_tuples = ResonanceMotif.from_tuples([((('b', 'a'), ()), ('1',)),
                                              ((('a',), ('c',)), ('-1', '2'))])

    assert from_conds == from_tuples
    assert hash(from_conds) == hash(from_tuples)
    assert {from_conds: 'shared'}[from_tuples] == 'shared'     # what makes it a dedup key
    assert from_conds.conditions == (ResCondKey(diff=(('a',), ('c',)), pf=('-1', '2')),   # conditions sorted
                                     ResCondKey(diff=(('a', 'b'), ()), pf=('1',)))


def test_resonance_motif_axes_and_mode_indices():
    # a motif is just a tuple of keys; build one directly to show that
    motif = ResonanceMotif((ResCondKey(diff=(('a',), ()), pf=('-1', '2')),
                            ResCondKey(diff=(('a', 'b'), ('c',)), pf=('2',))))

    assert motif.get_max_different_freq_axes() == {'1', '2'}   # sign on pf is stripped
    assert motif.get_nm_indices() == {'a', 'b', 'c'}          # both sides of every diff
    assert len(motif) == 2
    assert list(motif) == list(motif.conditions)               # iterating yields the keys


def test_resonance_motif_condition_order_does_not_matter():
    """Same conditions in another order is the same motif, so reorderings dedup to one key."""
    forward = ResonanceMotif.from_tuples([((('a',), ()), ('A',)), ((('b',), ('a',)), ('B',))])
    backward = ResonanceMotif.from_tuples([((('b',), ('a',)), ('B',)), ((('a',), ()), ('A',))])

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert forward.conditions == backward.conditions
    assert len({forward, backward}) == 1


def test_resonance_motif_keeps_repeated_conditions():
    once = ResonanceMotif.from_tuples([((('a',), ()), ('A',))])
    twice = ResonanceMotif.from_tuples([((('a',), ()), ('A',)), ((('a',), ()), ('A',))])

    assert len(twice) == 2
    assert once != twice



def test_resonance_motif_from_tuples_sorts_quanta_and_pf():
    """Quanta are a multiset (E_ba is E_ab) and pf a sum of signed axes (A - B is -B + A),
    so both are sorted; signs stay attached to their axis."""
    motif = ResonanceMotif.from_tuples([((('c', 'a'), ('b', 'a')), ('B', '-A'))])

    assert motif.conditions == (ResCondKey(diff=(('a', 'c'), ('a', 'b')), pf=('-A', 'B')),)



def test_resonance_motif_max_state_lvl_counts_quanta_on_either_side():
    motif = ResonanceMotif.from_tuples([((('a',), ()), ('1',)),
                                        ((('b',), ('a', 'b', 'c')), ('2',)),
                                        ((('a', 'b'), ('a',)), ('1', '2'))])

    assert motif.get_max_state_lvl() == 3


def test_resonance_motif_empty_is_the_no_resonance_term():
    """CompiledTerm uses ResonanceMotif(()) for terms without resonance conditions."""
    motif = ResonanceMotif(())

    assert len(motif) == 0
    assert list(motif) == []
    assert motif.get_max_different_freq_axes() == set()
    assert motif.get_nm_indices() == set()
    assert motif.get_max_state_lvl() == 0
    assert motif == ResonanceMotif.from_conditions([])


## generate_LHS_motif ---------------------------------------------------------
# Row i <-> condition i of the motif (conditions are kept sorted), column j <-> j-th of the
# motif's distinct axes, sorted. Entry is -s_j: the equation is written as
# -sum_j s_j w_j = -(E_left - E_right).

@pytest.mark.parametrize('motif, expected, expected_axes', [
    (MOTIF_AB,      [[1., 0.], [0., -1.]],  ('A', 'B')),   # '-A' -> +1
    (MOTIF_A,       [[-1.]],                ('A',)),
    (MOTIF_MIXED,   [[-1., 1.], [0., -1.]], ('A', 'B')),   # '-B' -> +1 ; ('A', '-B') sorts before ('B',)
    (MOTIF_B_TWICE, [[-1.], [-1.]],         ('B',)),       # two conditions, one axis -> 2x1
])
def test_generate_LHS_motif(motif, expected, expected_axes):
    lhs, axes = generate_LHS_motif(ResonanceMotif.from_tuples(motif))

    assert axes == expected_axes
    assert lhs.shape == np.shape(expected)
    np.testing.assert_array_equal(lhs, expected)


def test_generate_LHS_motif_extra_axes_get_zero_columns():
    lhs, axes = generate_LHS_motif(ResonanceMotif.from_tuples(MOTIF_A), axes=('C', 'A', 'B'))

    assert axes == ('A', 'B', 'C')
    np.testing.assert_array_equal(lhs, [[-1., 0., 0.]])


def test_generate_LHS_motif_axes_must_cover_the_motif():
    with pytest.raises(ValueError):
        generate_LHS_motif(ResonanceMotif.from_tuples(MOTIF_AB), axes=('A',))


## ParameterSet -------------------------------------------------------------

def test_parameterset_holds_exactly_the_given_labels():
    """no hidden entry, e.g. none for the ground state"""
    ps = ParameterSet({'a': 0, 'b': 3})

    assert len(ps) == 2
    assert list(ps) == ['a', 'b']
    assert ps.parameter_labels() == ['a', 'b']
    assert ps.indices() == [0, 3]
    assert ps.to_dict() == {'a': 0, 'b': 3}
    assert repr(ps) == "ParameterSet({'a': 0, 'b': 3})"
    for key in ('zero', ''):
        with pytest.raises(KeyError):
            ps[key]


def test_parameterset_rejects_non_mapping():
    with pytest.raises(TypeError):
        ParameterSet([('a', 0)]) # type: ignore


def test_parameterset_equality_is_order_independent():
    assert ParameterSet({'a': 0, 'b': 1}) == ParameterSet({'b': 1, 'a': 0})
    assert hash(ParameterSet({'a': 0, 'b': 1})) == hash(ParameterSet({'b': 1, 'a': 0}))
    assert ParameterSet({'a': 0}) != ParameterSet({'a': 1})



## ResLocPoint --------------------------------------------------------------

@pytest.mark.parametrize('ndim', NDIMS)
def test_res_loc_point_sorts_axes_in_any_dimension(ndim):
    # coordinates given in reverse axis order, e.g. for ndim=3: C=2., B=1., A=0.
    point = ResLocPoint({ax: float(i) for i, ax in reversed(list(enumerate(AXES[:ndim])))})

    assert point.axes == tuple(AXES[:ndim])
    assert point.values == tuple(float(i) for i in range(ndim))
    assert point.dimensionality == ndim
    assert point[AXES[ndim - 1]] == ndim - 1


@pytest.mark.parametrize('ndim', NDIMS)
def test_res_loc_point_equality_and_hash_ignore_axis_order(ndim):
    p1 = ResLocPoint(coords(ndim, 1.))
    p2 = ResLocPoint(dict(reversed(coords(ndim, 1.).items())))

    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert p1 != ResLocPoint(coords(ndim, 1., **{AXES[ndim - 1]: 2.}))


@pytest.mark.parametrize('ndim', NDIMS)
def test_res_loc_point_repr_lists_every_axis(ndim):
    expected = 'Point(' + ', '.join(f'{ax}=1.0' for ax in AXES[:ndim]) + ')'

    assert repr(ResLocPoint(coords(ndim, 1.))) == expected


def test_res_loc_point_rejects_unknown_axis():
    with pytest.raises(KeyError):
        _ = ResLocPoint({'A': 1.})['B']



def test_res_loc_point_ignores_later_changes_to_the_input_dict():
    """The point copies the caller's dict: changing the dict later changes neither the values nor the hash."""
    coord = {'A': 1.}
    point = ResLocPoint(coord)
    before = hash(point)

    coord['A'] = 2.
    coord['B'] = 3.

    assert point.as_dict() == {'A': 1.}
    assert point.dimensionality == 1
    assert hash(point) == before


## CompiledTerm -------------------------------------------------------------

def _fake_vibpert_term(res=(('a', '', ('1',)),)):
    """Only the attributes CompiledTerm.from_VibPertTerm reads.
    res: one (left quanta, right quanta, pf) per resonance condition."""
    return SimpleNamespace(
        coeff=0.5,
        props=[polprop(ops=(0, 1), inds='a'), polprop(inds='ab')],
        freqterms=[vibdiff(sl='a'), vibdiff(sl='ab', sr='a', pert=True)],
        res=[ResonanceCondition(diff=vibdiff(sl=sl, sr=sr), pf=list(pf)) for sl, sr, pf in res],
        tellNonSummSummIndices=lambda: (('a',), ('b',)),
        to_str=lambda *_: 'the fake term',
    )


def test_compiled_term_from_vibpert_term():
    ct = CompiledTerm.from_VibPertTerm(_fake_vibpert_term()) # type: ignore

    assert ct.frac_factor == 0.5
    assert ct.all_props.get_mode_indices() == ('a', 'a', 'b')
    assert vibdiff_keys(ct.freq_vibenedenom) == [(('a',), (), False)]
    assert ct.cmp_resmotf == ResonanceMotif.from_tuples([((('a',), ()), ('1',))])
    assert ct.idx_summ == ('b',)
    assert ct.idx_nonsumm == ('a',)

    indices = set(ct.all_props.get_mode_indices()
                  + tuple(i for vd in ct.cmp_freqdenom for i in (*vd.sl.q, *vd.sr.q))  # type: ignore
                  + tuple(ct.cmp_resmotf.get_nm_indices()))
    assert sorted(set(ct.idx_summ+ct.idx_nonsumm)) == sorted(indices)


def bare_term(avrg=(), non_avrg=()) -> CompiledTerm:
    """only properties: no motif, no denominators, no labels"""
    return CompiledTerm(PropsCollection(list(avrg)), PropsCollection(list(non_avrg)),
                        ResonanceMotif(()), FreqTermsCollection([]), 1., idx_summ=(), idx_nonsumm=())


@pytest.mark.parametrize('avrg, non_avrg, expected', [
    ((),                                      (polprop(inds='abc'), polprop(inds='abc')), ('cff',)),   # repeated
    ((),                                      (polprop(inds='abc'), polprop(inds='ab')),  ('cff', 'hess')),
    ((polprop(ops=(0, 1), inds='a'),),         (polprop(inds='a'),),                        ('polgrad', 'grad')),
    ((polprop(ops=(0,)), polprop(ops=(0, 1))), (),                                          ('dip', 'pol')),  # dord 0
    ((),                                      (),                                          ()),
])
def test_compiled_term_prop_names_one_per_distinct_trivial_name(avrg, non_avrg, expected):
    """what MolSystemData.load requests for the term: each name once, averaged properties first"""
    assert bare_term(avrg, non_avrg).prop_names == expected



# A term's resonance conditions must fix exactly one point, in the axes the term uses.
# number of conditions = number of axes = rank of the condition matrix. No data needed.

@pytest.mark.parametrize('res', [
    pytest.param((('a', '', ('-A',)), ('b', '', ('B',))),        id='-A and B (EVV): point in 2D'),
    pytest.param((('a', '', ('A',)),),                           id='A only: point in 1D'),
    pytest.param((('b', '', ('B',)),),                           id='B only: point in 1D'),
    pytest.param((('', 'a', ('A', '-B')), ('', 'a', ('B',))),    id='A - B and B: point in 2D'),
])
def test_compiled_term_conditions_fixing_one_point_pass(res):
    ct = CompiledTerm.from_VibPertTerm(_fake_vibpert_term(res)) # type: ignore

    assert len(ct.cmp_resmotf) == len(res)


@pytest.mark.parametrize('res', [
    pytest.param((('a', '', ('A', 'B')),),                                  id='one condition on A + B: a line'),
    pytest.param((('a', '', ('B',)), ('b', 'a', ('B',))),                   id='B twice: 2 conditions, 1 axis'),
    pytest.param((('a', '', ('A', 'B')), ('b', '', ('-A', '-B'))),          id='2 conditions, 2 axes, rank 1'),
    pytest.param((('a', '', ('A',)), ('b', '', ('B',)), ('ab', '', ('A', 'B'))),
                                                                            id='3 conditions, 2 axes'),
])
def test_compiled_term_conditions_not_fixing_one_point_raise(res):
    with pytest.raises(ValueError, match='do not fix one point') as err:
        CompiledTerm.from_VibPertTerm(_fake_vibpert_term(res)) # type: ignore
    assert 'the fake term' in str(err.value)        # the message shows the term's to_str


def test_compiled_term_without_conditions_passes():
    """No place on the spectrum (solve_LSE_motif raises there), but the other parts can still be computed."""
    ct = CompiledTerm.from_VibPertTerm(_fake_vibpert_term(res=())) # type: ignore

    assert ct.cmp_resmotf == ResonanceMotif(())
    assert ct.max_state_lvl == 2                    # from the freqterms: sl 'ab'
    assert FreqTermsCollection([]).get_max_state_lvl() == 0
