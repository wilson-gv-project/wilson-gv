"""
plan.py — the compile stage. Runs with zero molecular data (README rule R5).
"""

import pickle
from types import SimpleNamespace

import pytest

from wilson_suite.wilson_derive.abstractions import (
    HarmOscStateSymbolic,
    PolProp,
    QOperator,
    ResonanceCondition,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResCondKey,
    ResonanceMotif,
    compile_terms,
)


def polprop(ops: tuple[int, ...] = (), inds: str = '') -> PolProp:
    """ops -> QOperator labels, inds -> one-letter mode symbols (dord = len(inds))."""
    p = PolProp(ops=[QOperator(o=i) for i in ops], dord=len(inds))
    p.setInds(list(inds))
    return p


def vibdiff(sl: str = '', sr: str = '', pert: bool = False) -> VibDiffTerm:
    return VibDiffTerm(sl=HarmOscStateSymbolic(list(sl)), sr=HarmOscStateSymbolic(list(sr)), is_pert_wf_diff=pert)


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


def test_freqterms_pert_wf_diff_against_ground_counts_as_vibenedenom():
    coll = FreqTermsCollection([vibdiff(sl='ab', sr='', pert=True)])

    assert vibdiff_keys(coll.get_vibenedenom()) == [(('a', 'b'), (), True)]


def test_freqterms_num_indices_are_sorted_unique_left_quanta():
    coll = FreqTermsCollection([vibdiff(sl='b'), vibdiff(sl='ab'), vibdiff(sl='ab', sr='a', pert=True)])

    assert coll.get_num_indices_vibenedenom() == ('a', 'b')


## ResonanceMotif -----------------------------------------------------------

def test_resonance_motif_from_conditions_matches_from_tuples():
    conds = [
        ResonanceCondition(diff=vibdiff(sl='ba'), pf=['1']),
        ResonanceCondition(diff=vibdiff(sl='a', sr='c'), pf=['-1', '2']),
    ]

    from_conds = ResonanceMotif.from_conditions(conds)
    from_tuples = ResonanceMotif.from_tuples([((('b', 'a'), ()), ('1',)),
                                              ((('a',), ('c',)), ('-1', '2'))])

    assert from_conds == from_tuples
    assert hash(from_conds) == hash(from_tuples)
    assert from_conds.conditions[0] == ResCondKey(diff=(('a', 'b'), ()), pf=('1',))


def test_resonance_motif_is_immutable():
    motif = ResonanceMotif.from_tuples([((('a',), ()), ('1',))])

    with pytest.raises(AttributeError):
        motif.conditions = () # type: ignore


def test_resonance_motif_axes_and_mode_indices():
    motif = ResonanceMotif.from_tuples([((('a',), ()), ('-1', '2')),
                                        ((('a', 'b'), ('c',)), ('2',))])

    assert motif.get_max_different_freq_axes() == {'1', '2'}
    assert motif.get_nm_indices() == {'a', 'b', 'c'}
    assert len(motif) == 2


## ParameterSet -------------------------------------------------------------

def test_parameterset_injects_zero_and_maps_empty_label():
    ps = ParameterSet({'a': 0, 'b': 3})

    assert ps['zero'] == 'zero'
    assert ps[''] == 'zero'
    assert ps.parameter_labels() == ['a', 'b']
    assert ps.indices() == [0, 3]
    assert len(ps) == 3


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


## CompiledTerm -------------------------------------------------------------

def _fake_vibpert_term():
    """Only the attributes CompiledTerm.from_VibPertTerm reads."""
    return SimpleNamespace(
        coeff=0.5,
        props=[polprop(ops=(0, 1), inds='a'), polprop(inds='ab')],
        freqterms=[vibdiff(sl='a'), vibdiff(sl='ab', sr='a', pert=True)],
        res=[ResonanceCondition(diff=vibdiff(sl='a'), pf=['1'])],
        tellNonSummSummIndices=lambda: (('b',), ('a',)),
    )


def test_compiled_term_from_vibpert_term():
    ct = CompiledTerm.from_VibPertTerm(_fake_vibpert_term()) # type: ignore

    assert ct.frac_factor == 0.5
    assert ct.cmp_props.get_mode_indices() == ('a', 'a', 'b')
    assert ct.cmp_freqdenom.get_num_indices_vibenedenom() == ('a',)
    assert ct.cmp_resmotf == ResonanceMotif.from_tuples([((('a',), ()), ('1',))])
    assert ct.idx_summ_nonsumm == (('b',), ('a',))


def test_compile_terms_one_compiled_per_term():
    compiled = compile_terms([_fake_vibpert_term(), _fake_vibpert_term()]) # type: ignore

    assert len(compiled) == 2
    assert all(isinstance(c, CompiledTerm) for c in compiled)
