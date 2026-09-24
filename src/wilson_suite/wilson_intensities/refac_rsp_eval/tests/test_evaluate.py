"""
evaluate.py — the numeric stage. Everything here is built by hand from tiny arrays;
no VibPerturbedTerm, no data files.
"""

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.evaluate as evaluate_mod
from wilson_suite.wilson_derive.abstractions import (
    HarmOscStateSymbolic,
    PolProp,
    QOperator,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    ParameterSet,
    PrecalculatedData,
    VibDiff,
    VibState,
    VibStatesData,
    _get_ind_tuple_from_base,
    _make_func_to_compute_avrg,
    _make_hq_states_from_datadict,
    _make_vibdiff_key,
    calculate_avrg_tensor,
    eval_avrg_per_indexdict,
    eval_non_avrg_per_indexdict,
    eval_vibenedenom,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    otf_vibdiffdenom,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    PropsCollection,
    ResonanceMotif,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


def polprop(ops: tuple[int, ...] = (), inds: str = '') -> PolProp:
    p = PolProp(ops=[QOperator(o=i) for i in ops], dord=len(inds))
    p.setInds(list(inds))
    return p


def vibdiff(sl: str = '', sr: str = '', pert: bool = False) -> VibDiffTerm:
    return VibDiffTerm(sl=HarmOscStateSymbolic(list(sl)), sr=HarmOscStateSymbolic(list(sr)), is_pert_wf_diff=pert)


def state(label: str, energy: float) -> VibState:
    return VibState(harm_quanta_coeffs={}, energy=energy, state_label=label)


# Two modes; energies in cm-1 chosen so every difference is distinct.
E0, E1, E01, E00 = 1000., 1500., 2600., 1950.


@pytest.fixture
def states() -> VibStatesData:
    return VibStatesData(allstates=(state('0', E0), state('1', E1), state('0,1', E01), state('0,0', E00)),
                         harmonic_osc_states_labels=(0, 1))


## VibState / VibStatesData -------------------------------------------------

def test_vibstate_serial_roundtrip():
    s = VibState(harm_quanta_coeffs={(0, 1): 0.9, (1,): 0.1}, state_label='0,1')

    assert s.serial_harm_quanta_coeffs == {'0,1': 0.9, '1': 0.1}
    assert s.deserialize_state_dict() == {(0, 1): 0.9, (1,): 0.1}


def test_vibstate_equality_is_label_and_energy():
    assert state('0', 1000.) == state('0', 1000. + 1e-12)
    assert state('0', 1000.) != state('1', 1000.)
    assert state('0', 1000.) != state('0', 1001.)
    assert state('0', 1.) < state('1', 0.)


def test_vibstatesdata_appends_ground_state(states):
    assert 'zero' in states.allstates_map
    assert states.get_state_by_label('zero').energy == 0.
    assert states.get_state_by_label('0,1').energy == E01


def test_vibstatesdata_unknown_label_raises(states):
    with pytest.raises(ValueError):
        states.get_state_by_label('7')


def test_vibstatesdata_harmonic_osc_states_filters_by_label_choice():
    data = VibStatesData(allstates=(state('1', E1), state('0', E0), state('0,1', E01)),
                         harmonic_osc_states_labels=(0,1))

    assert data.get_harmonic_osc_states() == {0: E0, 1: E1}


def test_make_hq_states_from_datadict_joins_quanta_labels():
    harm, anharm = _make_hq_states_from_datadict({'harmonic_states': {('0',): E0, ('0', '1'): E01}})

    assert [s.state_label for s in harm] == ['0', '0,1']
    assert [s.energy for s in harm] == [E0, E01]
    assert anharm == ()


## VibDiff ------------------------------------------------------------------

def test_make_vibdiff_key_maps_symbols_sorts_and_names_ground():
    key = _make_vibdiff_key(vibdiff(sl='ab', sr=''), {'a': 1, 'b': 0})
    assert key == ('0,1', 'zero')

    key = _make_vibdiff_key(vibdiff(sl='', sr='abc'), {'a': 1, 'b': 0, 'c': 2})
    assert key == ('zero', '0,1,2')

def test_vibdiff_normalized_puts_ground_left_then_label_order():
    zero, s0, s1 = state('zero', 0.), state('0', E0), state('1', E1)

    assert VibDiff(s0, zero).normalized() == VibDiff(zero, s0)
    assert VibDiff(s0, zero) != VibDiff(zero, s0)
    assert VibDiff(s0, zero).energy_difference() == s0.energy
    assert VibDiff(zero, s0).energy_difference() == -s0.energy

    assert VibDiff(s1, s0).normalized() == VibDiff(s0, s1)
    assert VibDiff(s1, s0) != VibDiff(s0, s1)
    assert VibDiff(s0, s1).normalized() == VibDiff(s0, s1)


def test_vibdiff_energy_difference_cm1_and_au():
    vd = VibDiff(state('0,1', E01), state('0', E0))

    assert vd.energy_difference() == pytest.approx(E01 - E0)
    assert vd.energy_difference(au=True) == pytest.approx(convNu2Ene(E01 - E0))


def test_vibdiff_from_symbolic_and_from_quanta_agree(states):
    index_dict = {'a': 0, 'b': 1}

    symb = VibDiff.from_symbolic(vibdiff(sl='ab', sr='a'), index_dict, states)
    quanta = VibDiff.from_quanta(('a', 'b'), ('a',), index_dict, states)

    assert symb == quanta
    assert symb.energy_difference() == pytest.approx(E01 - E0)


## MolPropsCollection -------------------------------------------------------

@pytest.fixture
def props() -> MolPropsCollection:
    return MolPropsCollection([
        MolecularProperty(trivial_name='cff', extra_data={'ops': ['g', 'g', 'g']}),
        MolecularProperty(trivial_name='polgrad', extra_data={'ops': ['e', 'e', 'g']}),
    ])


def test_molpropscollection_lookup(props):
    assert props.names() == ['cff', 'polgrad']
    assert 'cff' in props and 'hess' not in props
    assert props['polgrad'] is props.get('polgrad')
    assert len(props) == 2
    assert props.get('hess') is None


def test_molpropscollection_fill_from_and_is_filled(props):
    assert not props.is_filled
    assert props.without_values().names() == ['cff', 'polgrad']

    props.fill_from({'cff': np.zeros(1), 'unrelated': 1})

    assert props['cff'].vals is not None
    assert props.without_values().names() == ['polgrad']
    assert not props.is_filled

## DATA REQUEST

def test_build_data_request_for_term():
    assert False

## Evaluation kernels -------------------------------------------------------

CFF = np.arange(8, dtype=float).reshape(2, 2, 2) + 1.   # cff[a, b, c] = 1 + 4a + 2b + c
POLGRAD_AVRG = np.array([10., 20.])                    # already-averaged <polgrad>[a]


@pytest.fixture
def molsys(states) -> MolSystemData:
    props = MolPropsCollection([MolecularProperty(trivial_name='cff', vals=CFF, extra_data={})])
    return MolSystemData(name='toy', eigenvals=np.array([E0, E1]), eigenvecs=None, mol_props=props, states=states)


def test_eval_non_avrg_reads_tensor_by_symbol_order(molsys):
    expr = PropsCollection([polprop(inds='abc')])

    assert eval_non_avrg_per_indexdict(expr, {'a': 1, 'b': 0, 'c': 1}, molsys) == CFF[1, 0, 1]


def test_eval_non_avrg_short_circuits_on_zero(molsys):
    molsys.mol_props['cff'].vals = np.zeros((2, 2, 2))
    expr = PropsCollection([polprop(inds='abc'), polprop(inds='abc')])

    assert eval_non_avrg_per_indexdict(expr, {'a': 0, 'b': 0, 'c': 0}, molsys) == 0.


def test_eval_non_avrg_missing_property_raises(molsys):
    expr = PropsCollection([polprop(inds='ab')])  # 'hess' is not in molsys

    with pytest.raises(KeyError):
        eval_non_avrg_per_indexdict(expr, {'a': 0, 'b': 0}, molsys)


def test_get_ind_tuple_from_base_distinct_vs_repeated_base_labels():
    """
    `expr` is an averaged property product as it appears in one term, with that term's mode
    labels. `base_expr` is the member of expr's motif whose tensor was actually computed
    (avrg_expr_tensor_mapping[expr] -> base_expr); that tensor has one axis per UNIQUE base
    label, in alphabetical order. get_ind_tuple_from_base returns the position in that tensor
    holding expr's value for the mode assignment index_dict (keyed by expr's labels).

    Two shapes of base:
      - all base labels distinct -> one axis per slot; expr's labels are read slot by slot
      - a base label repeats     -> fewer axes than slots; only the unique labels are read
    A base with more unique labels than expr has slots cannot stand in for expr.
    """
    index_dict = {'a': 1, 'b': 0}

    # all-distinct base: axes (a, b) <-> slots, so expr's labels are read slot by slot
    base = PropsCollection([polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='b')])
    expr = PropsCollection([polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='b')])
    assert _get_ind_tuple_from_base(expr, base, index_dict) == (1, 0)

    # repeated base label: 'a' fills both slots, the tensor has a single axis -> one index
    repeated = PropsCollection([polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='a')])
    assert _get_ind_tuple_from_base(repeated, PropsCollection([polprop(ops=(0, 1), inds='a')]), index_dict) == (1,)

    # more unique base labels than expr has slots: base cannot represent expr
    with pytest.raises(ValueError):
        _get_ind_tuple_from_base(PropsCollection([polprop(ops=(0, 1), inds='a')]), base, index_dict)

def test_eval_avrg_per_indexdict():
    assert False

def test_calculate_avrg_tensor():
    assert False


def test_otf_vibdiffdenom_is_product_of_inverse_au_differences(molsys):
    freqterms = FreqTermsCollection([vibdiff(sl='ab', sr='a', pert=True), vibdiff(sl='b', sr='', pert=True)])

    value = otf_vibdiffdenom(freqterms, {'a': 0, 'b': 1}, molsys)

    assert value == pytest.approx(1. / convNu2Ene(E01 - E0) / convNu2Ene(E1))


def test_eval_vibenedenom_reads_precomputed_tensor():
    tensor = np.array([[1., 2.], [3., 4.]])
    pre = PrecalculatedData(avrg_tensors={}, avrg_expr_tensor_mapping={},
                            vibenedenoms_tensors={('a', 'b'): tensor})
    freqterms = FreqTermsCollection([vibdiff(sl='b'), vibdiff(sl='a')])

    assert eval_vibenedenom(freqterms, {'a': 1, 'b': 0}, pre) == 3.


## One term end to end -------------------------------------------------------

@pytest.fixture
def term_and_precalc():
    """
    0.5 * cff[a,b,c] * <polgrad>[a] * 1/(E_ab - E_a) * 1/omega_a ; sum over b, c ; a fixed.
    """
    avrg = polprop(ops=(0, 1), inds='a')
    term = CompiledTerm(
        avrg_props=PropsCollection([avrg]),
        non_avrg_props=PropsCollection([polprop(inds='abc')]),
        cmp_resmotf=ResonanceMotif(()),
        cmp_freqdenom=FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='ab', sr='a', pert=True)]),
        frac_factor=0.5,
        idx_summ_nonsumm=(('b', 'c'), ('a',)),
    )
    avrg_key = PropsCollection([avrg])
    pre = PrecalculatedData(avrg_tensors={avrg_key: POLGRAD_AVRG},
                            avrg_expr_tensor_mapping={avrg_key: avrg_key},
                            vibenedenoms_tensors=None)  # None -> harmonic denominator on the fly # type: ignore
    return term, pre


def test_evaluate_single_index_dict_multiplies_the_four_factors(term_and_precalc, molsys):
    term, pre = term_and_precalc

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1},
                                               molsys_data=molsys, precalculated_data=pre, zero_tol=1e-18)

    # a=0, b=1, c=1:  0.5 * cff[0,1,1] * <polgrad>[0] * 1/(E_01 - E_0) * 1/omega_0
    assert value == pytest.approx(0.5 * CFF[0, 1, 1] * POLGRAD_AVRG[0] / convNu2Ene(E01 - E0) / convNu2Ene(E0))
    assert set(contribs) == {'NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM'}
    assert contribs['NON_AVRG'] == CFF[0, 1, 1]
    assert contribs['AVRG'] == POLGRAD_AVRG[0]


def test_evaluate_single_index_dict_zero_non_avrg_short_circuits(term_and_precalc, molsys):
    term, pre = term_and_precalc
    molsys.mol_props['cff'].vals = np.zeros((2, 2, 2))

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1},
                                               molsys_data=molsys, precalculated_data=pre, zero_tol=1e-18)

    assert value == 0.
    assert contribs['NON_AVRG'] == 0.


## evaluate_term_coeffs: sum over whichever of a, b, c the caller left unfixed ---------------

@pytest.mark.parametrize('fixed, expected_leaves, expected_total', [
    ({'a': 1, 'b': 0, 'c': 1}, {(1, 0, 1)},                                 101),  # nothing missing
    ({'a': 1, 'b': 0},         {(1, 0, 0), (1, 0, 1)},                       201),  # sum over c
    ({'a': 1},                 {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}, 422),  # sum over b and c
])
def test_evaluate_term_coeffs_enumerates_missing_index_combinations(fixed,
                                                                    expected_leaves, expected_total,
                                                                    molsys, monkeypatch):
    """
    evaluate_term_coeffs takes a term with mode labels (a, b, c) and a list of caller-supplied
    PARTIAL assignments ("relevant indices", e.g. {'a': 1}). For each partial assignment it fills
    every label the caller did not fix with each mode 0..n_modes-1, calls
    evaluate_single_index_dict once per complete assignment (a "leaf"), and returns

        results[ParameterSet(partial)] = (sum of leaf values, {ParameterSet(leaf): contribs})

    Which labels get summed is decided by what is missing from the partial dict, not by the
    term's idx_summ_nonsumm split; the split is read only to learn which labels the term has.

    This test checks the enumeration alone: evaluate_full_index_dict is stubbed to return
    100a + 10b + c, so each leaf value spells out its own assignment and the expected totals
    can be added by hand, e.g. {'a': 1} -> 100 + 101 + 110 + 111 = 422.
    """
    # Stand-in for the per-leaf evaluation: value = 100a + 10b + c, so the sums above are checkable by eye.
    monkeypatch.setattr(evaluate_mod, 'evaluate_full_index_dict',
                        lambda term, idx, *_: (100 * idx['a'] + 10 * idx['b'] + idx['c'], {}))
    # the stub term has no averaged props, so the avrg factory is stubbed out too
    monkeypatch.setattr(evaluate_mod, '_make_func_to_compute_avrg', lambda **_: None)
    # Only the index split is read from the term here; molsys has 2 modes, so each missing index runs over {0, 1}.
    term = CompiledTerm(PropsCollection([]), PropsCollection([]), ResonanceMotif(()), FreqTermsCollection([]), 1.,
                        idx_summ_nonsumm=(('b', 'c'), ('a',)))

    results = evaluate_term_coeff_sumover(term, fixed, precalculated_data=None, molsys_data=molsys, polarization_vec=(1.,1.,1.))

    total, leaves = results[ParameterSet(fixed)]
    assert total == expected_total
    assert {tuple(leaf[k] for k in 'abc') for leaf in leaves} == expected_leaves


def test_evaluate_term_coeffs_total_is_sum_of_single_index_dict_values(term_and_precalc, molsys):
    """
    Same summation, now with the real evaluate_single_index_dict and the fixture's toy term
    (0.5 * cff[a,b,c] * <polgrad>[a] * 1/(E_ab - E_a) * 1/omega_a). Fix a=0 and let b, c run
    over the two modes -> four leaves.

    The value of a single leaf is already covered by test_evaluate_single_index_dict_*; here the
    check is that evaluate_term_coeffs passes leaves through faithfully: the total equals the sum
    of evaluating each leaf assignment independently (nothing dropped, double-counted or
    rescaled), and the contribs stored per leaf are exactly what that call returns.

    Leaves are keyed by ParameterSet, which adds a 'zero' sentinel entry, so
    {k: leaf[k] for k in 'abc'} recovers the plain (a, b, c) dict needed to re-evaluate a leaf.
    """
    term, pre = term_and_precalc

    results = evaluate_term_coeff_sumover(term, {'a': 0}, precalculated_data=pre, molsys_data=molsys)

    total, leaves = results[ParameterSet({'a': 0})]

    per_leaf = {leaf: evaluate_full_index_dict(term, {k: leaf[k] for k in 'abc'}, molsys_data=molsys, precalculated_data=pre, zero_tol=1e-18)
                for leaf in leaves}
    assert len(leaves) == 4             # b, c in {0, 1}
    assert total == pytest.approx(sum(value for value, _ in per_leaf.values()))
    assert leaves == {leaf: contribs for leaf, (_, contribs) in per_leaf.items()}
