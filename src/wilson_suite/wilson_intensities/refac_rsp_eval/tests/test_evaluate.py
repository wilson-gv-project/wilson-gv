"""
evaluate.py — the numeric stage. Everything here is built by hand from tiny arrays;
no VibPerturbedTerm, no data files.
"""

from dataclasses import replace

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.evaluate as evaluate_mod
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    PrecalculatedData,
    _get_ind_tuple_from_base,
    coefficient_compute_loop,
    eval_non_avrg_per_indexdict,
    eval_vibenedenom,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    generate_LHS_motif,
    get_RHS_motif,
    get_motifs_feats,
    make_idx_sets,
    otf_vibdiffdenom,
    resonance_compute_loop,
    solve_LSE_motif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResLocPoint,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    E0,
    E00,
    E01,
    E1,
    E0_eigval,
    E1_eigval,
    polprop,
    toy_states,
    toy_term,
    vibdiff,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibStatesData,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


@pytest.fixture
def states() -> VibStatesData:
    return toy_states()


## Resonance location: shared motifs -------------------------------------------
# A motif is a set of resonance conditions  E_left - E_right = sum_j s_j * w_j , one per
# ResCondKey: `diff` names the two states by quanta labels, `pf` the frequency axes w_j with
# sign s_j ('-B' -> s = -1). Fixing the mode labels (a, b, ...) to modes turns the motif into
# a linear system  LHS @ w = RHS  whose solution is where on the axes the term resonates.
#
# The four motifs below are the ones used for the pre-refactor amplitudes/resonances.py.
# With the `states` fixture and a=0, b=1:  E_a = E0, E_b = E1, E_ab = E01, E_0 = 0.

MOTIF_AB = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))        # one axis per condition
MOTIF_A = (((('a', 'b'), ('a',)), ('A',)),)                                    # single condition
MOTIF_MIXED = ((((), ('a',)), ('B',)), (((), ('a',)), ('A', '-B')))             # A - B in one condition
MOTIF_B_TWICE = ((((), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))           # two conditions, one axis


@pytest.fixture
def params() -> dict:
    return {'a': 0, 'b': 1}

@pytest.fixture
def params_obj() -> ParameterSet:
    return ParameterSet({'a': 0, 'b': 1})

def resonance_residuals(motif: ResonanceMotif,
                        location: ResLocPoint,
                        index_dict: dict, states,
                        unit: str = 'Eh') -> list[float]:
    """E_left - E_right - sum_j s_j w_j for every condition, written out independently of evaluate.py.
    State energies are stored in cm-1; `unit` is the unit of `location` ('Eh' or 'cm-1')."""
    def energy(quanta):
        label = ','.join(str(i) for i in sorted(index_dict[q] for q in quanta))
        e_cm1 = states.get_state_by_label(label).energy if label else 0.
        return convNu2Ene(e_cm1) if unit == 'Eh' else e_cm1

    return [energy(c.left) - energy(c.right)
            - sum((-1. if ax.startswith('-') else 1.) * location[ax.strip('-')] for ax in c.pf)  # type: ignore
            for c in motif]


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


## get_RHS_motif --------------------------------------------------------------
# One entry per condition: -(E_left - E_right), in Eh by default, cm-1 on request.

@pytest.mark.parametrize('motif, expected_cm1', [
    (MOTIF_AB,      [-(E01 - E0), -(E1 - E0)]),
    (MOTIF_A,       [-(E01 - E0)]),
    (MOTIF_MIXED,   [E0, E0]),                    # -(E_0 - E_a)
    (MOTIF_B_TWICE, [E0, -(E1 - E0)]),
])
def test_get_RHS_motif_cm1(motif, expected_cm1, params_obj, states):
    rhs = get_RHS_motif(ResonanceMotif.from_tuples(motif), params_obj, states, unit='cm-1')

    assert rhs == pytest.approx(expected_cm1)


def test_get_RHS_motif_defaults_to_hartree(params_obj, states):
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    assert get_RHS_motif(motif, params_obj, states) == get_RHS_motif(motif, params_obj, states, unit='Eh')
    assert get_RHS_motif(motif, params_obj, states) == pytest.approx([convNu2Ene(-(E01 - E0)), convNu2Ene(-(E1 - E0))])


def test_get_RHS_motif_follows_the_mode_assignment(states):
    """Same motif, a and b swapped -> different states resonate."""
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    swapped = get_RHS_motif(motif, ParameterSet({'a': 1, 'b': 0}), states, unit='cm-1')

    assert swapped == pytest.approx([-(E01 - E1), -(E0 - E1)])


def test_get_RHS_motif_unknown_state_raises(states):
    motif = ResonanceMotif.from_tuples(MOTIF_A)

    with pytest.raises(ValueError):          # a=b=1 needs state '1,1', which `states` lacks
        get_RHS_motif(motif, ParameterSet({'a': 1, 'b': 1}), states)


## solve_LSE_motif ------------------------------------------------------------

@pytest.mark.parametrize('motif, expected', [
    (MOTIF_AB,    {'A': E01 - E0, 'B': E1 - E0}),
    (MOTIF_A,     {'A': E01 - E0}),
    (MOTIF_MIXED, {'A': -2 * E0, 'B': -E0}),     # w_B = -E0 ; w_A - w_B = -E0
])
def test_solve_LSE_motif_cm1(motif, expected, params_obj, states):
    location = solve_LSE_motif(ResonanceMotif.from_tuples(motif), params_obj, states, unit='cm-1')

    assert isinstance(location, ResLocPoint)
    assert location.axes == tuple(sorted(expected))
    assert dict(location.coordinates) == pytest.approx(expected)


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_solve_LSE_motif_location_satisfies_every_condition(motif, params_obj, states):
    """Independent check: plug the solution back into  E_left - E_right = sum_j s_j w_j ."""
    res_motif = ResonanceMotif.from_tuples(motif)

    location = solve_LSE_motif(res_motif, params_obj, states, unit='cm-1')

    assert resonance_residuals(res_motif, location, params_obj.to_dict(), states, unit='cm-1') == pytest.approx([0.] * len(res_motif))


def test_solve_LSE_motif_hartree_is_cm1_converted(params_obj, states):
    motif = ResonanceMotif.from_tuples(MOTIF_MIXED)

    in_cm1 = solve_LSE_motif(motif, params_obj, states, unit='cm-1')
    in_eh = solve_LSE_motif(motif, params_obj, states)

    assert in_eh.axes == in_cm1.axes
    assert in_eh.values == pytest.approx(tuple(convNu2Ene(v) for v in in_cm1.values))  # type: ignore


def test_solve_LSE_motif_inconsistent_system_raises_linalg_error(params_obj, states):
    """MOTIF_B_TWICE asks w_B = -E0 and w_B = E1 - E0 at once: no resonance location exists."""
    with pytest.raises(np.linalg.LinAlgError, match='inconsistent'):
        solve_LSE_motif(ResonanceMotif.from_tuples(MOTIF_B_TWICE), params_obj, states)


def test_solve_LSE_motif_underdetermined_system_raises_linalg_error(params_obj, states):
    """One condition on A + B: the resonance is a line, not a point."""
    with pytest.raises(np.linalg.LinAlgError, match='not a point'):
        solve_LSE_motif(ResonanceMotif.from_tuples(((((), ('a',)), ('A', 'B')),)), params_obj, states)


def test_solve_LSE_motif_single_axis_other_than_A(params_obj, states):
    location = solve_LSE_motif(ResonanceMotif.from_tuples(((((), ('a',)), ('B',)),)), params_obj, states, unit='cm-1')

    assert location == ResLocPoint({'B': -E0})


def test_solve_LSE_motif_empty_motif_raises(params, states):
    with pytest.raises(ValueError, match='no resonance conditions'):
        solve_LSE_motif(ResonanceMotif(()), params, states)


## resonance_compute_loop -----------------------------------------------------
# One solve_LSE_motif call per index set (always in Eh), results keyed by ParameterSet.

SWAPPED = {'a': 1, 'b': 0}
SWAPPED_obj = ParameterSet({'a': 1, 'b': 0})


def test_resonance_compute_loop_one_location_per_index_set(params, params_obj, states):
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    results = resonance_compute_loop([params, SWAPPED], motif, states)

    assert set(results) == {params_obj, SWAPPED_obj}
    assert results[params_obj] == solve_LSE_motif(motif, params_obj, states)
    assert results[SWAPPED_obj] == solve_LSE_motif(motif, SWAPPED_obj, states)


def test_resonance_compute_loop_locations_follow_the_mode_assignment(params, params_obj, states):
    """Same motif, a and b swapped -> different resonance points. Hand values in cm-1, the loop returns Eh."""
    results = resonance_compute_loop([params, SWAPPED], ResonanceMotif.from_tuples(MOTIF_AB), states)

    assert dict(results[params_obj].coordinates) == pytest.approx({'A': convNu2Ene(E01 - E0), 'B': convNu2Ene(E1 - E0)})
    assert dict(results[SWAPPED_obj].coordinates) == pytest.approx({'A': convNu2Ene(E01 - E1), 'B': convNu2Ene(E0 - E1)})


def test_resonance_compute_loop_no_index_sets_gives_empty_dict(states):
    assert resonance_compute_loop([], ResonanceMotif.from_tuples(MOTIF_AB), states) == {}


def test_resonance_compute_loop_accepts_make_idx_sets_output(states):
    """make_idx_sets gives plain dicts; results are still keyed by ParameterSet, as in evaluate_term_coeff_sumover."""
    idx_sets = make_idx_sets(2, ['a'])           # [{'a': 0}, {'a': 1}]

    results = resonance_compute_loop(idx_sets, ResonanceMotif.from_tuples(MOTIF_MIXED), states)

    # w_B = -E_a ; w_A - w_B = -E_a
    assert set(results) == {ParameterSet({'a': 0}), ParameterSet({'a': 1})}
    assert dict(results[ParameterSet({'a': 0})].coordinates) == pytest.approx({'A': convNu2Ene(-2 * E0), 'B': convNu2Ene(-E0)})
    assert dict(results[ParameterSet({'a': 1})].coordinates) == pytest.approx({'A': convNu2Ene(-2 * E1), 'B': convNu2Ene(-E1)})


def test_resonance_compute_loop_missing_state_raises(params, params_obj, states):
    """a=b=1 needs state '1,1', which `states` lacks: the whole loop raises, the set is not skipped."""
    with pytest.raises(ValueError):
        resonance_compute_loop([params, {'a': 1, 'b': 1}], ResonanceMotif.from_tuples(MOTIF_A), states)


## Evaluation kernels -------------------------------------------------------

CFF = np.arange(8, dtype=float).reshape(2, 2, 2) + 1.   # cff[a, b, c] = 1 + 4a + 2b + c
POLGRAD_AVRG = np.array([10., 20.])                    # already-averaged <polgrad>[a]


@pytest.fixture
def molsys(states) -> MolSystemData:
    props = MolPropsCollection([MolecularProperty(trivial_name='cff', vals=CFF, extra_data={})])
    return MolSystemData(name='toy', eigenvals={0: E0_eigval, 1: E1_eigval}, eigenvecs=None, mol_props=props, states=states)


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
    toy_term() plus its precalculated <polgrad>[a] tensor (see helpers.toy_term).
    """
    term = toy_term()
    avrg_key = PropsCollection([polprop(ops=(0, 1), inds='a')])   # equal by value to term.avrg_props
    pre = PrecalculatedData(avrg_tensors={avrg_key: POLGRAD_AVRG},
                            avrg_expr_tensor_mapping={avrg_key: avrg_key},
                            vibenedenoms_tensors=None)  # None -> harmonic denominator on the fly # type: ignore
    return term, pre


def test_evaluate_single_index_dict_multiplies_the_four_factors(term_and_precalc, molsys):
    term, pre = term_and_precalc

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1},
                                               molsys_data=molsys, precalculated_data=pre, zero_tol=1e-18)

    # a=0, b=1, c=1:  0.5 * cff[0,1,1] * <polgrad>[0] * 1/(E_01 - E_0) * 1/omega_0
    assert value == pytest.approx(0.5 * CFF[0, 1, 1] * POLGRAD_AVRG[0] / convNu2Ene(E01 - E0) / convNu2Ene(E0_eigval))
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
                        idx_summ=('b', 'c'), idx_nonsumm=('a',))

    from wilson_suite.wilson_intensities.amplitudes.averaging import (
        getGeneralPolarizationAveragingExpression,
    )
    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank=4,laser_pol=(1.,1.,1.))
    total, leaves = evaluate_term_coeff_sumover(term, fixed, precalculated_data=None, molsys_data=molsys, polarization_linear_comb=polarization_linear_comb)

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

    total, leaves = evaluate_term_coeff_sumover(term, {'a': 0}, precalculated_data=pre, molsys_data=molsys)

    per_leaf = {leaf: evaluate_full_index_dict(term, {k: leaf[k] for k in 'abc'}, molsys_data=molsys, precalculated_data=pre, zero_tol=1e-18)
                for leaf in leaves}
    assert len(leaves) == 4             # b, c in {0, 1}
    assert total == pytest.approx(sum(value for value, _ in per_leaf.values()))
    assert leaves == {leaf: contribs for leaf, (_, contribs) in per_leaf.items()}


def test_evaluate_term_coeffs_hand_computed_total(term_and_precalc, molsys):
    """
    a=0, sum over b, c of  0.5 * cff[0,b,c] * <polgrad>[0] * 1/(E_0b - E_0) * 1/omega_0 .
    c enters only cff, so the two c values are added first; b picks state '0,0' or '0,1'.
    """
    term, pre = term_and_precalc

    total, _ = evaluate_term_coeff_sumover(term, {'a': 0}, precalculated_data=pre, molsys_data=molsys)

    by_b = ((CFF[0, 0, 0] + CFF[0, 0, 1]) / convNu2Ene(E00 - E0)
            + (CFF[0, 1, 0] + CFF[0, 1, 1]) / convNu2Ene(E01 - E0))
    assert total == pytest.approx(0.5 * POLGRAD_AVRG[0] / convNu2Ene(E0_eigval) * by_b)


def test_evaluate_term_coeffs_on_the_fly_avrg_matches_precalculated(term_and_precalc, molsys):
    """
    No precalculated tensor -> <polgrad>[a] is built from polarization_linear_comb and the raw
    'polgrad' data:  <polgrad>[a] = sum over (i, j) of coeff_ij * polgrad[a, i, j] .
    polgrad is chosen so that this gives POLGRAD_AVRG again (4 + 2*3 = 10, 8 + 2*6 = 20);
    every other entry is 100, so reading a wrong cartesian slot changes the result.
    """
    term, pre = term_and_precalc
    polgrad = np.full((2, 3, 3), 100.)
    polgrad[:, 0, 0] = [4., 8.]
    polgrad[:, 1, 1] = [3., 6.]
    props = MolPropsCollection([*molsys.mol_props, MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={})])
    molsys = replace(molsys, mol_props=props)

    precalc_total, precalc_leaves = evaluate_term_coeff_sumover(term, {'a': 0}, molsys,
                                                                precalculated_data=pre)
    otf_total, otf_leaves = evaluate_term_coeff_sumover(term, {'a': 0}, molsys,
                                                        polarization_linear_comb={(0, 0): 1., (1, 1): 2.})

    assert otf_total == pytest.approx(precalc_total)
    assert set(otf_leaves) == set(precalc_leaves)
    assert all(otf_leaves[leaf]['AVRG'] == pytest.approx(POLGRAD_AVRG[0]) for leaf in otf_leaves)


@pytest.mark.parametrize('pre', [None, PrecalculatedData()], ids=['no precalculated data', 'avrg not in mapping'])
def test_evaluate_term_coeffs_needs_polarization_when_avrg_not_precalculated(pre, molsys):
    with pytest.raises(ValueError, match='polarization_vec is required'):
        evaluate_term_coeff_sumover(toy_term(), {'a': 0}, molsys, precalculated_data=pre)


def test_evaluate_term_coeffs_needs_eigenvals(term_and_precalc, molsys):
    """The number of modes to sum over is len(eigenvals); without eigenvals there is nothing to sum over."""
    term, pre = term_and_precalc

    with pytest.raises(ValueError, match='eigenvals'):
        evaluate_term_coeff_sumover(term, {'a': 0}, replace(molsys, eigenvals=None), precalculated_data=pre)


def test_evaluate_term_coeffs_mode_count_comes_from_eigenvals(term_and_precalc, molsys, monkeypatch):
    """Three eigenvalues -> each missing label (b, c) runs over {0, 1, 2}: 3 * 3 = 9 leaves."""
    monkeypatch.setattr(evaluate_mod, 'evaluate_full_index_dict', lambda term, idx, *_: (1., {}))
    term, pre = term_and_precalc
    three_modes = replace(molsys, eigenvals={0: E0_eigval, 1: E1_eigval, 2: 2000.})

    total, leaves = evaluate_term_coeff_sumover(term, {'a': 0}, three_modes, precalculated_data=pre)

    assert total == 9.
    assert {(leaf['b'], leaf['c']) for leaf in leaves} == {(b, c) for b in range(3) for c in range(3)}


def test_evaluate_term_coeffs_zero_avrg_leaves_are_kept(molsys):
    """
    <polgrad>[1] = 0 -> every leaf with a=1 stops right after AVRG. The leaves are still recorded,
    with only the AVRG entry. The states a=1 would need ('1,1' is missing) are never looked up.
    """
    term = toy_term()
    avrg_key = PropsCollection([polprop(ops=(0, 1), inds='a')])
    pre = PrecalculatedData(avrg_tensors={avrg_key: np.array([10., 0.])},
                            avrg_expr_tensor_mapping={avrg_key: avrg_key})

    total, leaves = evaluate_term_coeff_sumover(term, {'a': 1}, molsys, precalculated_data=pre)

    assert total == 0.
    assert len(leaves) == 4
    assert all(contribs == {'AVRG': 0.} for contribs in leaves.values())


def test_evaluate_term_coeffs_result_shape(term_and_precalc, molsys):
    """
    Shows the shape of the evaluate_term_coeff_sumover result (run with -s to see the print).
    Currently: {ParameterSet(idx_dict): (total, {ParameterSet(full idx): contribs})}, one key only.
    """
    term, pre = term_and_precalc
    ps = ParameterSet({'a': 0, 'b': 0})

    result = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 0}, molsys, precalculated_data=pre)

    from pprint import pprint
    pprint(result)

    total, leaves = result
    assert isinstance(total, float)
    assert set(leaves) == {ParameterSet({'a': 0, 'b': 0, 'c': c}) for c in (0, 1)}
    assert all(set(contribs) == {'NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM'} for contribs in leaves.values())


## coefficient_compute_loop ---------------------------------------------------

def test_coefficient_compute_loop_result_shape(term_and_precalc, molsys):
    """
    Shows the shape of one loop entry (run with -s to see the print).
    Currently the key appears twice: results[ps][ps] == (total, leaves).
    """
    term, pre = term_and_precalc
    ps = ParameterSet({'a': 0, 'b': 0})

    results = coefficient_compute_loop([{'a': 0, 'b': 0}], term, molsys, None, pre)

    from pprint import pprint
    pprint(results)

    entry = results[ps]
    _total, leaves = entry
    assert len(leaves) == 2              # c in {0, 1}


def test_get_motifs_feats():
    """
    The motifs and locations returned by get_motifs_locs are keyed by ParameterSet, one entry
    per index set. Each location is a ResLocPoint with axes and values in Eh.
    """
    motif = ResonanceMotif.from_tuples(MOTIF_AB)
    idx_sets = [{'a': 0, 'b': 0}, {'a': 1, 'b': 0}, {'a': 0, 'b': 1}]

    results = get_motifs_feats([motif], idx_sets, toy_states())

    assert set(results) == {ParameterSet({'a': 0, 'b': 0}), ParameterSet({'a': 1, 'b': 0}), ParameterSet({'a': 0, 'b': 1})}
    for ps in results:
        loc = results[ps]
        assert isinstance(loc, dict)

        # check that the location satisfies the resonance conditions
        res_motif = ResonanceMotif.from_tuples(MOTIF_AB)
        assert loc[res_motif].location.axes == ('A', 'B')

        residuals = resonance_residuals(res_motif, loc[res_motif].location, ps.to_dict(), toy_states())
        assert residuals == pytest.approx([0.] * len(res_motif))

    print()
    for r,v in results.items():
        print(f"{r}:")
        for m, loc in v.items():
            print(f"  {m}:")
            for ax, val in zip(loc.location.axes, loc.location.values):
                print(f"    {ax}: {val:.6f} Eh")
        print()
