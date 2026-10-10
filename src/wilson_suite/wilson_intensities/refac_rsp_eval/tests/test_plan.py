"""
plan.py — the compile stage. Runs with zero molecular data.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_derive.abstractions import ResonanceCondition
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResCondKey,
    ResLocPoint,
    ResonanceMotif,
    compile_terms,
    generate_LHS_motif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
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


def test_identify_avrg_motif_erases_inds_on_a_copy_only():
    coll = PropsCollection([polprop(ops=(0, 1), inds='a'), polprop(inds='ab')])

    motif = coll.identify_avrg_motif()

    assert [p.inds for p in motif] == [None] # type: ignore
    assert coll.get_mode_indices() == ('a', 'a', 'b')


def test_propscollection_equal_by_value_and_usable_as_key():
    a = PropsCollection([polprop(ops=(0, 1), inds='a')])
    b = PropsCollection([polprop(ops=(0, 1), inds='a')])

    assert a == b
    assert hash(a) == hash(b)
    assert {a: 'tensor'}[b] == 'tensor'


## FreqTermsCollection ------------------------------------------------------

def test_freqterms_split_by_pert_wf_diff():
    harmonic = vibdiff(sl='a')                     # 1/omega_a
    pert = vibdiff(sl='ab', sr='a', pert=True)     # from a perturbed wavefunction
    coll = FreqTermsCollection([harmonic, pert])

    assert vibdiff_keys(coll.get_vibenedenom()) == [(('a',), (), False)]
    assert vibdiff_keys(coll.get_pert_wf_diff()) == [(('a', 'b'), ('a',), True)]


def test_freqterms_pert_wf_diff_against_ground_is_pert_only():
    coll = FreqTermsCollection([vibdiff(sl='ab', sr='', pert=True)])
    assert vibdiff_keys(coll.get_vibenedenom()) == []
    assert vibdiff_keys(coll.get_pert_wf_diff()) == [(('a', 'b'), (), True)]


def test_freqterms_num_indices_are_sorted_unique_left_quanta():
    coll = FreqTermsCollection([vibdiff(sl='b'), vibdiff(sl='ab'), vibdiff(sl='ab', sr='a', pert=True)])

    assert coll.get_num_indices_vibenedenom() == ('a', 'b')


## ResCondKey ---------------------------------------------------------------
# One resonance condition stripped to what identifies it: which quanta sit on each side of the
# energy difference, and which perturbing frequencies enter. plan.py keeps these instead of
# derive's ResonanceCondition so a ResonanceMotif can be hashed, compared and pickled without
# holding derive objects.

def test_rescondkey_left_right_and_ground_state():
    key = ResCondKey(diff=(('a', 'b'), ()), pf=('1',))     # E_ab - E_0, perturbed by freq '1'

    assert key.left == ('a', 'b')
    assert key.right == ()        # ground state is the empty tuple
    assert key.pf == ('1',)


def test_rescondkey_is_a_frozen_value_usable_as_dict_key():
    key = ResCondKey(diff=(('a',), ('c',)), pf=('-1', '2'))
    same = ResCondKey(diff=(('a',), ('c',)), pf=('-1', '2'))

    assert key == same and hash(key) == hash(same)
    assert {key: 'motif'}[same] == 'motif'
    with pytest.raises(AttributeError):
        key.pf = ()  # type: ignore


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


def test_resonance_motif_is_immutable():
    motif = ResonanceMotif.from_tuples([((('a',), ()), ('1',))])

    with pytest.raises(AttributeError):
        motif.conditions = () # type: ignore


def test_resonance_motif_from_tuples_sorts_quanta_and_pf():
    """Quanta are a multiset (E_ba is E_ab) and pf a sum of signed axes (A - B is -B + A),
    so both are sorted; signs stay attached to their axis."""
    motif = ResonanceMotif.from_tuples([((('c', 'a'), ('b', 'a')), ('B', '-A'))])

    assert motif.conditions == (ResCondKey(diff=(('a', 'c'), ('a', 'b')), pf=('-A', 'B')),)


def test_resonance_motif_accepts_any_sequence_of_keys():
    keys = [ResCondKey(diff=(('a',), ()), pf=('1',))]

    assert ResonanceMotif(keys).conditions == tuple(keys)       # type: ignore  # list -> tuple in __post_init__
    assert ResonanceMotif(keys) == ResonanceMotif(tuple(keys))  # type: ignore


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
    assert motif == ResonanceMotif.from_conditions([])


def test_resonance_motif_repeated_axes_are_counted_once():
    motif = ResonanceMotif.from_tuples([((('a',), ()), ('A', '-B')),
                                        ((('b',), ()), ('B',)),
                                        ((('a',), ('b',)), ('-A',))])

    assert motif.get_max_different_freq_axes() == {'A', 'B'}


def test_resonance_motif_pickle_roundtrip():
    motif = ResonanceMotif.from_tuples([((('a', 'b'), ()), ('A',)), ((('b',), ('a',)), ('-A', 'B'))])

    restored = pickle.loads(pickle.dumps(motif))

    assert restored == motif
    assert hash(restored) == hash(motif)


## generate_LHS_motif ---------------------------------------------------------
# Row i <-> condition i of the motif (conditions are kept sorted), column j <-> j-th of the
# motif's distinct axes, sorted. Entry is -s_j: the equation is written as
# -sum_j s_j w_j = -(E_left - E_right).

@pytest.mark.parametrize('motif, expected, expected_axes', [
    (MOTIF_AB,      [[-1., 0.], [0., -1.]], ('A', 'B')),
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


def test_generate_LHS_motif_ignores_states():
    """Only pf enters the LHS; which states resonate is the RHS's business.
    (States do fix the row order, so both motifs here sort the ('A', '-B') condition first.)"""
    one = ResonanceMotif.from_tuples(((((), ('a',)), ('A', '-B')), ((('b',), ()), ('B',))))
    other = ResonanceMotif.from_tuples(((((), ('b',)), ('A', '-B')), ((('a', 'b'), ('a',)), ('B',))))

    np.testing.assert_array_equal(generate_LHS_motif(one)[0], generate_LHS_motif(other)[0])


## ParameterSet -------------------------------------------------------------

def test_parameterset_injects_zero_and_maps_empty_label():
    ps = ParameterSet({'a': 0, 'b': 3})

    assert ps['zero'] == 'zero'
    assert ps[''] == 'zero'
    assert ps.parameter_labels() == ['a', 'b']
    assert ps.indices() == [0, 3]
    assert len(ps) == 3


def test_parameterset_zero_entry_is_hidden_from_repr_but_kept_in_to_dict():
    """The 'zero' entry is always there: giving it explicitly changes nothing, and pickling keeps it."""
    ps = ParameterSet({'a': 0})

    assert repr(ps) == "ParameterSet({'a': 0})"
    assert ps.to_dict() == {'a': 0, 'zero': 'zero'}
    assert ps == ParameterSet({'a': 0, 'zero': 'zero'})
    assert hash(ps) == hash(ParameterSet({'a': 0, 'zero': 'zero'}))
    assert pickle.loads(pickle.dumps(ps)).to_dict() == {'a': 0, 'zero': 'zero'}


def test_parameterset_rejects_non_mapping():
    with pytest.raises(TypeError):
        ParameterSet([('a', 0)]) # type: ignore


def test_parameterset_equality_is_order_independent():
    assert ParameterSet({'a': 0, 'b': 1}) == ParameterSet({'b': 1, 'a': 0})
    assert hash(ParameterSet({'a': 0, 'b': 1})) == hash(ParameterSet({'b': 1, 'a': 0}))
    assert ParameterSet({'a': 0}) != ParameterSet({'a': 1})


def test_parameterset_orders_alphabetically_by_label():
    assert ParameterSet({'a': 0, 'b': 5}) < ParameterSet({'a': 1, 'b': 0})
    assert not ParameterSet({'a': 1}) < ParameterSet({'a': 1})
    assert min(ParameterSet({'a': 2}), ParameterSet({'a': 0})) == ParameterSet({'a': 0})


def test_parameterset_pickle_roundtrip():
    ps = ParameterSet({'a': 1, 'c': 2})

    assert pickle.loads(pickle.dumps(ps)) == ps


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


def test_res_loc_point_is_not_equal_to_other_types():
    assert ResLocPoint({'A': 1.}) != (('A', 1.),)


def test_res_loc_point_ignores_later_changes_to_the_input_dict():
    """Before, the point kept the caller's dict: the values changed, but the hash did not."""
    coord = {'A': 1.}
    point = ResLocPoint(coord)
    before = hash(point)

    coord['A'] = 2.
    coord['B'] = 3.

    assert point.as_dict() == {'A': 1.}
    assert point.dimensionality == 1
    assert hash(point) == before


def test_res_loc_point_as_dict_returns_a_new_dict():
    point = ResLocPoint({'A': 1., 'B': 2.})

    d = point.as_dict()
    d['A'] = 5.

    assert point.as_dict() == {'A': 1., 'B': 2.}
    assert point['A'] == 1.


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
    assert ct.cmp_freqdenom.get_num_indices_vibenedenom() == ('a',)
    assert ct.cmp_resmotf == ResonanceMotif.from_tuples([((('a',), ()), ('1',))])
    assert ct.idx_summ == ('b',)
    assert ct.idx_nonsumm == ('a',)

    indices = set(ct.all_props.get_mode_indices()
                  + ct.cmp_freqdenom.get_num_indices_vibenedenom()
                  + tuple(ct.cmp_resmotf.get_nm_indices()))
    assert sorted(set(ct.idx_summ+ct.idx_nonsumm)) == sorted(indices)


def test_compile_terms_one_compiled_per_term():
    compiled = compile_terms([_fake_vibpert_term(), _fake_vibpert_term()]) # type: ignore

    assert len(compiled) == 2
    assert all(isinstance(c, CompiledTerm) for c in compiled)


# B7: a term's resonance conditions must fix exactly one point, in the axes the term uses.
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


def test_compile_terms_raises_if_one_term_does_not_fix_one_point():
    bad = _fake_vibpert_term(res=(('a', '', ('A', 'B')),))

    with pytest.raises(ValueError, match='do not fix one point'):
        compile_terms([_fake_vibpert_term(), bad]) # type: ignore
