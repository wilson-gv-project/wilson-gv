"""
evaluate.py — the numeric stage. Everything here is built by hand from tiny arrays;
no VibPerturbedTerm, no data files.
"""

import itertools
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.evaluate as evaluate_mod
import wilson_suite.wilson_intensities.refac_rsp_eval.lineshape as lineshape_mod
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    AvrgTensor,
    all_features_on_grid,
    boxed_features_on_grid,
    build_contributions,
    calculate_avrg_tensor_einsum,
    eval_avrg_per_indexdict,
    eval_feature_on_grid,
    eval_non_avrg_per_indexdict,
    eval_rescond_resloc,
    eval_resonance_factor,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    get_RHS_motif,
    harmonic_denom,
    included_modes,
    make_avrg_einsum_subscripts,
    otf_vibdiffdenom,
    solve_LSE_motif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    SpectralFeature,
    SpectralWindow,
    amplitude_half_widths,
    condition_states,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box, box_slices, make_mesh
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResCondKey,
    ResLocPoint,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    CFF,
    E0,
    E00,
    E01,
    E1,
    MOTIF_A,
    MOTIF_AB,
    MOTIF_AB_EXTRA,
    MOTIF_B_TWICE,
    MOTIF_MIXED,
    PS_00,
    PS_01,
    PS_10,
    PS_11,
    TOY_RECIPE,
    E0_eigval,
    E1_eigval,
    ab_term,
    flat_term,
    get_box_extent,
    make_func_to_compute_avrg,
    polprop,
    row,
    state,
    toy_term,
    vibdiff,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    VibDiff,
    VibStatesData,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

## Resonance location: shared motifs -------------------------------------------
# MOTIF_AB, MOTIF_A, MOTIF_MIXED and MOTIF_B_TWICE are explained in helpers.py.
# The fixtures `states`, `molsys` and `molsys_a1_zero` come from conftest.py.


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


## get_RHS_motif --------------------------------------------------------------
# One entry per condition: -(E_left - E_right), in Eh by default, cm-1 on request.

@pytest.mark.parametrize('motif, expected_cm1', [
    (MOTIF_AB,      [E0, -(E01 - E0)]),           # -(E_0 - E_a), -(E_ab - E_a)
    (MOTIF_A,       [-E0]),                       # -(E_a - E_0)
    (MOTIF_MIXED,   [E0, E0]),                    # -(E_0 - E_a)
    (MOTIF_B_TWICE, [E0, -(E1 - E0)]),
])
def test_get_RHS_motif_cm1(motif, expected_cm1, params_obj, states):
    rhs = get_RHS_motif(ResonanceMotif.from_tuples(motif), params_obj, states, unit='cm-1')

    assert rhs == pytest.approx(expected_cm1)


def test_get_RHS_motif_defaults_to_hartree(params_obj, states):
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    assert get_RHS_motif(motif, params_obj, states) == get_RHS_motif(motif, params_obj, states, unit='Eh')
    assert get_RHS_motif(motif, params_obj, states) == pytest.approx([convNu2Ene(E0), convNu2Ene(-(E01 - E0))])


def test_get_RHS_motif_follows_the_mode_assignment(states):
    """Same motif, a and b swapped -> different states resonate."""
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    swapped = get_RHS_motif(motif, ParameterSet({'a': 1, 'b': 0}), states, unit='cm-1')

    assert swapped == pytest.approx([E1, -(E01 - E1)])


@pytest.mark.parametrize('unit', ['cm', 'cm^-1', 'eh', 'au'])
def test_get_RHS_motif_unknown_unit_raises(unit, params_obj, states):
    """Only 'Eh' and 'cm-1': any other text raises, rather than being read as cm-1."""
    with pytest.raises(ValueError, match="unit must be 'Eh' or 'cm-1'"):
        get_RHS_motif(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, unit=unit)


def test_get_RHS_motif_unknown_state_raises(states):
    motif = ResonanceMotif.from_tuples(MOTIF_AB)

    with pytest.raises(ValueError):          # a=b=1 needs state '1,1', which `states` lacks
        get_RHS_motif(motif, ParameterSet({'a': 1, 'b': 1}), states)


## solve_LSE_motif ------------------------------------------------------------

@pytest.mark.parametrize('motif, expected', [
    (MOTIF_AB,    {'A': E0, 'B': E01 - E0}),     # -w_A = -E0 ; w_B = E01 - E0
    (MOTIF_A,     {'A': E0}),
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


def test_solve_LSE_motif_unknown_unit_raises(params_obj, states):
    with pytest.raises(ValueError, match="unit must be 'Eh' or 'cm-1'"):
        solve_LSE_motif(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, unit='cm')


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


def test_solve_LSE_motif_empty_motif_is_the_location_without_axes(params, states):
    """A term without resonance conditions is a flat background: no axes, nothing to solve."""
    assert solve_LSE_motif(ResonanceMotif(()), params, states) == ResLocPoint({})


## Evaluation kernels -------------------------------------------------------

POLGRAD_AVRG = np.array([0.2, 0.])     # <polgrad>[a] of the toy molecule: TOY_POLGRAD with TOY_RECIPE


def test_eval_non_avrg_reads_the_tensor_at_the_modes_of_its_labels(molsys):
    """cff[a,b,c] at a=1, b=0, c=1 is CFF[1, 0, 1] = 2; at a=1, b=0, c=0 it is 0 (mode 1 once: 0 by symmetry)."""
    expr = PropsCollection([polprop(inds='abc')])

    assert eval_non_avrg_per_indexdict(expr, {'a': 1, 'b': 0, 'c': 1}, molsys) == CFF[1, 0, 1] == 2.
    assert eval_non_avrg_per_indexdict(expr, {'a': 1, 'b': 0, 'c': 0}, molsys) == 0.


def test_eval_non_avrg_short_circuits_on_zero(molsys):
    molsys.mol_props['cff'].vals = np.zeros((2, 2, 2))
    expr = PropsCollection([polprop(inds='abc'), polprop(inds='abc')])

    assert eval_non_avrg_per_indexdict(expr, {'a': 0, 'b': 0, 'c': 0}, molsys) == 0.


def test_eval_non_avrg_missing_property_raises(molsys):
    expr = PropsCollection([polprop(inds='ab')])  # 'hess' is not in molsys

    with pytest.raises(KeyError):
        eval_non_avrg_per_indexdict(expr, {'a': 0, 'b': 0}, molsys)


## Energy denominators, on the fly ---------------------------------------------
# otf_vibdiffdenom: 1/(E_left - E_right) for the perturbed-wavefunction differences, from the state energies.
# harmonic_denom:   1/omega for the plain denominators, from the harmonic eigenvalues (not the state energies).

def test_otf_vibdiffdenom_is_product_of_inverse_au_differences(molsys):
    freqterms = FreqTermsCollection([vibdiff(sl='ab', sr='a', pert=True), vibdiff(sl='b', sr='', pert=True)])

    value = otf_vibdiffdenom(freqterms, {'a': 0, 'b': 1}, molsys, states_choice='anharmonic')

    assert value == pytest.approx(1. / convNu2Ene(E01 - E0) / convNu2Ene(E1))


def test_otf_vibdiffdenom_of_no_terms_is_one(molsys):
    assert otf_vibdiffdenom(FreqTermsCollection([]), {}, molsys, states_choice='anharmonic') == 1.


def test_otf_vibdiffdenom_zero_difference_raises(molsys):
    """a = b = 0: both sides are state '0', so the difference is 0."""
    with pytest.raises(ZeroDivisionError):
        otf_vibdiffdenom(FreqTermsCollection([vibdiff(sl='a', sr='b', pert=True)]), {'a': 0, 'b': 0}, molsys,
                         states_choice='anharmonic')


def test_harmonic_denom_uses_eigenvals_not_state_energies(molsys):
    """eigenvals are 1100 and 1580 cm-1; the states '0' and '1' are at 1000 and 1500 cm-1."""
    value = harmonic_denom(FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='b')]), {'a': 0, 'b': 1}, molsys)

    assert value == pytest.approx(1. / convNu2Ene(E0_eigval) / convNu2Ene(E1_eigval))


def test_harmonic_denom_of_no_terms_is_one(molsys):
    assert harmonic_denom(FreqTermsCollection([]), {}, molsys) == 1.


def test_harmonic_denom_needs_the_ground_state_on_the_right(molsys):
    with pytest.raises(ValueError, match='not a ground state'):
        harmonic_denom(FreqTermsCollection([vibdiff(sl='a', sr='b')]), {'a': 0, 'b': 1}, molsys)


def test_harmonic_denom_needs_one_mode_per_denominator(molsys):
    with pytest.raises(ValueError):
        harmonic_denom(FreqTermsCollection([vibdiff(sl='ab')]), {'a': 0, 'b': 1}, molsys)


## Averaged properties, on the fly ---------------------------------------------
# EVV terms average three properties over four cartesian slots, e.g. term 8 of evv.json (data_for_tests/translated_terms):
#     polgrad[b] (ops 0, 3) * dipgrad[a] (op 1) * dipgrad[c] (op 2)
# The op number is the slot in a polarization key: for key (i, j, k, l), polgrad reads the cartesian
# pair (i, l), the first dipgrad j, the second dipgrad k. The data has no symmetry, so reading a
# wrong slot or a wrong mode changes the result.

N_MODES = 3
_rng = np.random.default_rng(7)
POLGRAD_3 = _rng.normal(size=(N_MODES, 3, 3))      # polgrad[mode, i, j]
DIPGRAD_3 = _rng.normal(size=(N_MODES, 3))         # dipgrad[mode, i]
ISO4 = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=(1., 1., 1.))


def evv_props(polgrad=POLGRAD_3, dipgrad=DIPGRAD_3) -> MolPropsCollection:
    return MolPropsCollection([MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={}),
                               MolecularProperty(trivial_name='dipgrad', vals=dipgrad, extra_data={})])


def evv_avrg_expr(labels: str = 'bac') -> PropsCollection:
    """polgrad[labels[0]] (ops 0, 3) * dipgrad[labels[1]] (op 1) * dipgrad[labels[2]] (op 2)."""
    p, d1, d2 = labels
    return PropsCollection([polprop(ops=(0, 3), inds=p), polprop(ops=(1,), inds=d1), polprop(ops=(2,), inds=d2)])


def dense(pol: dict) -> np.ndarray:
    """A polarization dict as a dense 3x3x3x3 coefficient tensor C[i, j, k, l]."""
    C = np.zeros((3,) * 4)
    for key, coeff in pol.items():
        C[key] = coeff
    return C


def rotation(alpha: float, beta: float) -> np.ndarray:
    """Rotation by alpha about z, then by beta about x."""
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    Rz = np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])
    Rx = np.array([[1., 0., 0.], [0., cb, -sb], [0., sb, cb]])
    return Rx @ Rz


ALL_ABC = [dict(zip('abc', combo)) for combo in itertools.product(range(N_MODES), repeat=3)]


# -- make_func_to_compute_avrg: one value for one index set --------------------------

def test_avrg_func_reads_the_cartesian_slot_of_each_operator():
    """One polarization entry, key (2, 0, 1, 1): polgrad[b] reads (2, 1), dipgrad[a] reads 0, dipgrad[c] reads 1."""
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={(2, 0, 1, 1): 3.})

    value = func({'a': 0, 'b': 1, 'c': 2}, evv_props())

    assert value == pytest.approx(3. * POLGRAD_3[1, 2, 1] * DIPGRAD_3[0, 0] * DIPGRAD_3[2, 1])


def test_avrg_func_is_the_full_contraction_with_the_averaging_coefficients():
    """sum over (i, j, k, l) of C[i, j, k, l] * polgrad[b, i, l] * dipgrad[a, j] * dipgrad[c, k], written with einsum."""
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    for idx in ALL_ABC:
        expected = np.einsum('ijkl,il,j,k->', dense(ISO4), POLGRAD_3[idx['b']], DIPGRAD_3[idx['a']], DIPGRAD_3[idx['c']])
        assert func(idx, evv_props()) == pytest.approx(expected)


@pytest.mark.parametrize('laser_pol', [(1., 1., 1.), (1., 0., 0.), (0.3, -1.2, 2.)])
def test_avrg_func_does_not_change_when_the_molecule_is_rotated(laser_pol):
    """
    The orientational average cannot depend on how the molecule sits in its own axes: rotating every
    cartesian index of every property by the same R leaves the value unchanged. A single, non-averaged
    element does change, which shows the rotation is not trivial.
    """
    R = rotation(0.7, 1.9)
    rotated = evv_props(np.einsum('ip,jq,npq->nij', R, R, POLGRAD_3), np.einsum('ip,np->ni', R, DIPGRAD_3))
    average = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(),
                                         polarization_linear_comb=getGeneralPolarizationAveragingExpression(4, laser_pol))
    one_element = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={(0, 0, 0, 0): 1.})

    for idx in ALL_ABC:
        assert average(idx, rotated) == pytest.approx(average(idx, evv_props()))
    assert one_element(ALL_ABC[-1], rotated) != pytest.approx(one_element(ALL_ABC[-1], evv_props()))


def test_avrg_func_with_a_repeated_mode_label():
    """dipgrad[a] * dipgrad[a], rank 2: both factors read mode a."""
    expr = PropsCollection([polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='a')])
    func = make_func_to_compute_avrg(avrg_expression=expr, polarization_linear_comb={(0, 0): 1., (1, 2): 3.})

    assert func({'a': 2}, evv_props()) == pytest.approx(DIPGRAD_3[2, 0] ** 2 + 3. * DIPGRAD_3[2, 1] * DIPGRAD_3[2, 2])


def test_avrg_func_without_polarization_entries_is_zero():
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={})

    assert func({'a': 0, 'b': 1, 'c': 2}, evv_props()) == 0.


def test_avrg_func_needs_every_mode_label():
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    with pytest.raises(KeyError, match="'c'"):
        func({'a': 0, 'b': 1}, evv_props())


def test_avrg_func_needs_a_mol_props_collection():
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    with pytest.raises(TypeError, match='MolPropsCollection'):
        func({'a': 0, 'b': 1, 'c': 2}, {'polgrad': POLGRAD_3, 'dipgrad': DIPGRAD_3})  # type: ignore


def test_avrg_func_needs_every_property_in_the_data():
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)
    no_polgrad = MolPropsCollection([MolecularProperty(trivial_name='dipgrad', vals=DIPGRAD_3, extra_data={})])

    with pytest.raises(KeyError):
        func({'a': 0, 'b': 1, 'c': 2}, no_polgrad)


# -- eval_avrg_per_indexdict: the averaged part at one index set, from the term's tensor ---------

def one_value(v: float) -> AvrgTensor:
    """the averaged part v at a = 0"""
    return AvrgTensor(('a',), np.array([v]))


def test_eval_avrg_per_indexdict_reads_the_tensor_at_the_index_set():
    """labels the tensor does not have (c) are ignored"""
    tensor = AvrgTensor(('a', 'b'), np.array([[1., 2.], [3., 4.]]))

    assert eval_avrg_per_indexdict(tensor, {'a': 1, 'b': 0, 'c': 1}) == 3.


def test_eval_avrg_per_indexdict_turns_values_below_zero_tol_into_zero():
    def value(v):
        return eval_avrg_per_indexdict(one_value(v), {'a': 0}, zero_tol=1e-18)

    assert value(1e-20) == 0.
    assert value(-1e-20) == 0.
    assert value(1e-17) == 1e-17


@pytest.mark.parametrize('v, expected', [(1e-18, 0.), (-1e-18, 0.), (1.0000001e-18, 1.0000001e-18), (-2e-18, -2e-18)],
                         ids=['+zero_tol', '-zero_tol', 'just above', 'below -zero_tol'])
def test_eval_avrg_per_indexdict_zero_tol_itself_counts_as_zero(v, expected):
    """|v| <= zero_tol is 0, the same test as np.isclose(v, 0, atol=zero_tol, rtol=0)"""
    assert eval_avrg_per_indexdict(one_value(v), {'a': 0}, zero_tol=1e-18) == expected


@pytest.mark.parametrize('v, expected', [(1e-20, 0.), (-1e-20, 0.), (1e-18, 0.), (-1e-18, 0.),
                                         (2e-18, 2e-18), (-2e-18, -2e-18)])
def test_eval_non_avrg_turns_values_up_to_zero_tol_into_zero(v, expected, molsys):
    """|v| <= zero_tol is 0, the same test as np.isclose(v, 0, atol=zero_tol, rtol=0)"""
    molsys.mol_props['cff'].vals = np.full((2, 2, 2), v)

    assert eval_non_avrg_per_indexdict(PropsCollection([polprop(inds='abc')]), {'a': 0, 'b': 0, 'c': 0},
                                       molsys, zero_tol=1e-18) == expected


# -- make_avrg_einsum_subscripts: the symbolic averaged part as one einsum ---------------------
# One operand per property: its mode labels, then the slot letter of each Cartesian axis (I, J, K, L = slot 0, 1, 2, 3),
# as the data is laid out (modes first, then x, y, z). Then W: the slot letters in slot order.
# Output: the distinct mode labels, alphabetical. No data needed.

@pytest.mark.parametrize('expr, subscripts', [
    (evv_avrg_expr('bac'), 'bIL,aJ,cK,IJKL->abc'),
    (evv_avrg_expr('cab'), 'cIL,aJ,bK,IJKL->abc'),
    (evv_avrg_expr('bab'), 'bIL,aJ,bK,IJKL->ab'),
    (evv_avrg_expr('baa'), 'bIL,aJ,aK,IJKL->ab'),
    (PropsCollection([polprop(ops=(0, 3), inds='ab'), polprop(ops=(1,), inds='a'), polprop(ops=(2,), inds='b')]),
     'abIL,aJ,bK,IJKL->ab'),
    (PropsCollection([polprop(ops=(0, 1), inds='aa')]), 'aaIJ,IJ->a'),
    (PropsCollection([polprop(ops=(1, 0))]), 'JI,IJ->'),
], ids=['term 8: polgrad[b] dipgrad[a] dipgrad[c]',
        'terms 3, 6: polgrad[c] dipgrad[a] dipgrad[b]',
        'terms 2, 7, 9, 12: no c, so no c axis',
        'terms 10, 11: dipgrad[a] twice',
        'term 0: polhess[a, b] dipgrad[a] dipgrad[b]',
        'a label twice in one property: its diagonal',
        'no mode label; slots in reverse order'])
def test_avrg_einsum_subscripts_follow_the_expression(expr, subscripts):
    assert make_avrg_einsum_subscripts(expr) == subscripts


def test_avrg_einsum_subscripts_need_slots_0_to_rank_minus_1():
    with pytest.raises(ValueError, match=r'0 \.\. 1, each once; got \[0, 2\]'):
        make_avrg_einsum_subscripts(PropsCollection([polprop(ops=(0, 2), inds='a')]))


def test_avrg_einsum_subscripts_need_lowercase_mode_labels():
    """upper case is for the slots: a mode label 'I' would be summed with slot 0"""
    with pytest.raises(ValueError, match='one lowercase letter'):
        make_avrg_einsum_subscripts(PropsCollection([polprop(ops=(0, 1), inds='I')]))


# -- calculate_avrg_tensor_einsum: every index set at once = the on-the-fly loop -------------

# 5 expressions (labels) x 3 recipes (laser_pol) = 15 cases.
# Expected in each case: the einsum tensor equals the on-the-fly loop at all 27 index sets (a, b, c each over
# modes 0, 1, 2), to rel 1e-12, abs 0. The data is random (POLGRAD_3, DIPGRAD_3): no symmetry, no zeros. So
# reading a wrong mode, a wrong slot or the polgrad axes in the wrong order gives another number.
#
# labels -> evv_avrg_expr(labels) = polgrad[labels[0]] (slots 0, 3) * dipgrad[labels[1]] (slot 1) * dipgrad[labels[2]] (slot 2)
#   'bac'  terms 8, 13        expression order b, a, c; tensor axes a, b, c (alphabetical, tensor.labels).
#                             Right: read with tensor.labels, as AvrgTensor.at does:
#                                 {'a': 0, 'b': 1, 'c': 2} -> values[0, 1, 2]
#                             Wrong: read in expression order (b, a, c):
#                                 {'a': 0, 'b': 1, 'c': 2} -> values[1, 0, 2], the value of a=1, b=0, c=2.
#   'abc'  no EVV term        expression order = axis order: the plain case.
#   'cab'  terms 3, 6         polgrad read at c, the last axis.
#   'bab'  terms 2, 7, 9, 12  no c: a tensor with 2 axes (a, b). ALL_ABC still runs over c; einsum (AvrgTensor.at)
#                             and the loop both ignore it.
#   'baa'  terms 10, 11       both dipgrads read at mode a: one mode, two slot letters (J, K).
#   Not here: 'aab' (terms 4, 5); terms 0 and 1 (polhess, diphess). test_integration.py checks all 14 terms.
#
# laser_pol -> the rank-4 recipe (getGeneralPolarizationAveragingExpression). With e_k the polarization of slot k,
# the 3 numbers are ((e_0.e_1)(e_2.e_3), (e_0.e_2)(e_1.e_3), (e_0.e_3)(e_1.e_2)).
#   (1, 1, 1)         all four fields along x. This recipe treats every slot alike, so wrong slots give the same
#                     numbers: this case alone cannot catch a slot mix-up.
#   (1, 0, 0)         slots 0, 1 along one axis, slots 2, 3 along another, e.g. (x, x, y, y). Catches a swap of
#                     slots 1 and 2, or of the two polgrad axes; not a swap of slots 0 and 1 (the same pair).
#   (0.3, -1.2, 2.0)  no setup of unit vectors gives these numbers. Each pairing has its own weight, so every
#                     slot mix-up changes the value.
@pytest.mark.parametrize('laser_pol', [(1., 1., 1.), (1., 0., 0.), (0.3, -1.2, 2.)])
@pytest.mark.parametrize('labels', ['bac', 'abc', 'cab', 'bab', 'baa'])
def test_avrg_tensor_einsum_is_the_on_the_fly_value_for_every_index_set(labels, laser_pol):
    pol = getGeneralPolarizationAveragingExpression(4, laser_pol)
    func = make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(labels), polarization_linear_comb=pol)

    tensor = calculate_avrg_tensor_einsum(evv_avrg_expr(labels), evv_props(), pol)

    for idx in ALL_ABC:
        assert tensor.at(idx) == pytest.approx(func(idx, evv_props()), rel=1e-12, abs=0)


def test_avrg_tensor_einsum_term8_axes_are_a_b_c_not_the_expression_order():
    """
    Term 8, polgrad[b] dipgrad[a] dipgrad[c]: values[0, 1, 2] is a=0, b=1, c=2, so polgrad is read at mode 1 (b),
    the dipgrads at modes 0 (a) and 2 (c). The sum is written out here, without einsum and without the loop.
    """
    tensor = calculate_avrg_tensor_einsum(evv_avrg_expr('bac'), evv_props(), ISO4)

    W = dense(ISO4)
    by_hand = sum(W[i, j, k, l] * POLGRAD_3[1, i, l] * DIPGRAD_3[0, j] * DIPGRAD_3[2, k]
                  for i, j, k, l in itertools.product(range(3), repeat=4))
    assert tensor.labels == ('a', 'b', 'c')
    assert tensor.values.shape == (N_MODES,) * 3
    assert tensor.values[0, 1, 2] == pytest.approx(by_hand, rel=1e-12, abs=0)
    assert tensor.at({'a': 0, 'b': 1, 'c': 2}) == tensor.values[0, 1, 2]
    assert tensor.values[1, 0, 2] != pytest.approx(by_hand)      # read as (b, a, c): another number


def test_avrg_tensor_einsum_without_c_has_no_c_axis():
    """terms 2, 7, 9, 12: polgrad[b] dipgrad[a] dipgrad[b]: one value per (a, b), the same for every c"""
    tensor = calculate_avrg_tensor_einsum(evv_avrg_expr('bab'), evv_props(), ISO4)

    assert (tensor.labels, tensor.values.shape) == (('a', 'b'), (N_MODES, N_MODES))
    assert tensor.at({'a': 2, 'b': 1, 'c': 0}) == tensor.at({'a': 2, 'b': 1, 'c': 2}) == tensor.values[2, 1]


def test_avrg_tensor_einsum_reads_the_diagonal_for_a_label_twice_in_one_property():
    """polhess[a, a] (slots 0, 1): values[a] = sum over I, J of W[I, J] * polhess[a, a, I, J]"""
    polhess = np.random.default_rng(11).normal(size=(N_MODES, N_MODES, 3, 3))
    props = MolPropsCollection([MolecularProperty(trivial_name='polhess', vals=polhess, extra_data={})])
    expr = PropsCollection([polprop(ops=(0, 1), inds='aa')])
    pol = {(0, 0): 1., (1, 2): 3.}
    func = make_func_to_compute_avrg(avrg_expression=expr, polarization_linear_comb=pol)

    tensor = calculate_avrg_tensor_einsum(expr, props, pol)

    assert tensor.labels == ('a',)
    for a in range(N_MODES):
        assert tensor.at({'a': a}) == pytest.approx(polhess[a, a, 0, 0] + 3. * polhess[a, a, 1, 2], rel=1e-12, abs=0)
        assert tensor.at({'a': a}) == pytest.approx(func({'a': a}, props), rel=1e-12, abs=0)


def test_avrg_tensor_einsum_without_polarization_entries_is_zero():
    tensor = calculate_avrg_tensor_einsum(evv_avrg_expr(), evv_props(), {})

    assert tensor.values.shape == (N_MODES,) * 3
    assert not tensor.values.any()


def test_avrg_tensor_einsum_needs_one_recipe_entry_per_slot():
    """a rank-2 key for a rank-4 expression would fill a whole 3 x 3 block of W without an error"""
    with pytest.raises(ValueError, match=r'\(0, 0\) has length 2, the rank is 4'):
        calculate_avrg_tensor_einsum(evv_avrg_expr(), evv_props(), {(0, 0): 1.})


# -- one EVV-shaped term, every factor on the fly -------------------------------------

def evv_shaped_term() -> CompiledTerm:
    """
    0.25 * cff[a,b,c] * <polgrad[b] dipgrad[a] dipgrad[c]> * 1/(E_ab - E_c) * 1/(omega_a omega_b omega_c),
    the shape of an EVV term: resonance labels a, b (MOTIF_AB); c is summed.
    """
    return CompiledTerm(
        avrg_props=evv_avrg_expr('bac'),
        non_avrg_props=PropsCollection([polprop(inds='abc')]),
        cmp_resmotf=ResonanceMotif.from_tuples(MOTIF_AB),
        cmp_freqdenom=FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='b'), vibdiff(sl='c'),
                                           vibdiff(sl='ab', sr='c', pert=True)]),
        frac_factor=0.25,
        idx_summ=('c',),
        idx_nonsumm=('a', 'b'),
    )


def with_evv_props(molsys):
    """molsys (cff, 2 modes, toy states) plus polgrad and dipgrad."""
    return replace(molsys, mol_props=MolPropsCollection([*molsys.mol_props, *evv_props()]))


def test_evaluate_full_index_dict_on_the_fly_multiplies_the_four_factors(molsys):
    term = evv_shaped_term()
    tensor = calculate_avrg_tensor_einsum(term.avrg_props, evv_props(), ISO4)

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1}, with_evv_props(molsys), tensor,
                                               states_choice='anharmonic')

    # a=0, b=1, c=1: states ab = '0,1' and c = '1'; omegas from eigenvals
    avrg = np.einsum('ijkl,il,j,k->', dense(ISO4), POLGRAD_3[1], DIPGRAD_3[0], DIPGRAD_3[1])
    expected = {'NON_AVRG': CFF[0, 1, 1],
                'AVRG': avrg,
                'VIBDIFF_TERMS': 1. / convNu2Ene(E01 - E1),
                'VIBENE_DENOM': 1. / (convNu2Ene(E0_eigval) * convNu2Ene(E1_eigval) * convNu2Ene(E1_eigval))}
    assert contribs == pytest.approx(expected)
    assert value == pytest.approx(0.25 * np.prod(list(expected.values())))


def test_evaluate_full_index_dict_on_the_fly_zero_average_stops_before_the_states(molsys):
    """dipgrad[a=1] = 0 makes the average 0: the state '1,1' (missing from the toy states) is never looked up."""
    dipgrad = DIPGRAD_3.copy()
    dipgrad[1] = 0.
    otf_molsys = replace(molsys, mol_props=MolPropsCollection([*molsys.mol_props, *evv_props(dipgrad=dipgrad)]))
    term = evv_shaped_term()
    tensor = calculate_avrg_tensor_einsum(term.avrg_props, otf_molsys.mol_props, ISO4)

    value, contribs = evaluate_full_index_dict(term, {'a': 1, 'b': 1, 'c': 0}, otf_molsys, tensor,
                                               states_choice='anharmonic')

    assert (value, contribs) == (0., {'AVRG': 0.})


def test_build_contributions_makes_the_avrg_tensor_once_per_term(molsys, monkeypatch):
    """One einsum per term, passed to every index set of that term; not one einsum per index set."""
    made, passed = [], []
    einsum = evaluate_mod.calculate_avrg_tensor_einsum

    def counting_einsum(*args):
        made.append(einsum(*args))
        return made[-1]

    def stub_coeff(*_, avrg_tensor=None, **__):
        passed.append(avrg_tensor)
        return 0., None

    monkeypatch.setattr(evaluate_mod, 'calculate_avrg_tensor_einsum', counting_einsum)
    monkeypatch.setattr(evaluate_mod, 'evaluate_term_coeff_sumover', stub_coeff)

    build_contributions([evv_shaped_term(), evv_shaped_term()], with_evv_props(molsys), ISO4, states_choice='anharmonic')

    assert len(made) == 2
    assert len(passed) == 2 * 4                              # 2 terms x 4 index sets (a, b over 2 modes)
    assert all(t is made[0] for t in passed[:4]) and all(t is made[1] for t in passed[4:])


def test_build_contributions_splits_the_denominators_once_per_term(molsys_a1_zero, monkeypatch):
    """
    The split into 1/omega and energy-difference denominators is made once per term and kept
    (CompiledTerm.freq_vibenedenom, freq_pert_wf_diff), not once per index set. Each split builds a new
    FreqTermsCollection with a deepcopy: per index set, that was 11 of 22 s of build_contributions on ACDM.
    Here: per term, a = 0 with b, c over 2 modes gives 4 index sets that reach the denominators.
    """
    calls = {'get_vibenedenom': 0, 'get_pert_wf_diff': 0}
    for name in calls:
        original = getattr(FreqTermsCollection, name)

        def counting(self, _name=name, _original=original):
            calls[_name] += 1
            return _original(self)
        monkeypatch.setattr(FreqTermsCollection, name, counting)

    build_contributions([ab_term(), ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert calls == {'get_vibenedenom': 2, 'get_pert_wf_diff': 2}


def test_build_contributions_keeps_no_factors_by_index_set(molsys_a1_zero, monkeypatch):
    """The rows do not carry their factors per index set; FeatureResult.factors_by_index_set computes them again
    when asked (test_pipeline.py)."""
    asked = []
    sumover = evaluate_mod.evaluate_term_coeff_sumover

    def recording(*args, **kwargs):
        asked.append(kwargs['keep_factors'])
        return sumover(*args, **kwargs)
    monkeypatch.setattr(evaluate_mod, 'evaluate_term_coeff_sumover', recording)

    build_contributions([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert asked == [False] * 4                              # 4 index sets (a, b over 2 modes)


## One term end to end -------------------------------------------------------

@pytest.fixture
def term_and_avrg():
    """
    toy_term() and its averaged part <polgrad>[a] = POLGRAD_AVRG, given as the tensor (see helpers.toy_term).
    """
    return toy_term(), AvrgTensor(('a',), POLGRAD_AVRG)


def test_evaluate_single_index_dict_multiplies_the_four_factors(term_and_avrg, molsys):
    term, avrg = term_and_avrg

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1},
                                               molsys_data=molsys, avrg_tensor=avrg, zero_tol=1e-18,
                                               states_choice='anharmonic')

    # a=0, b=1, c=1:  0.5 * cff[0,1,1] * <polgrad>[0] * 1/(E_01 - E_0) * 1/omega_0
    assert value == pytest.approx(0.5 * CFF[0, 1, 1] * POLGRAD_AVRG[0] / convNu2Ene(E01 - E0) / convNu2Ene(E0_eigval))
    assert set(contribs) == {'NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM'}
    assert contribs['NON_AVRG'] == CFF[0, 1, 1]
    assert contribs['AVRG'] == POLGRAD_AVRG[0]


def test_evaluate_single_index_dict_zero_non_avrg_short_circuits(term_and_avrg, molsys):
    term, avrg = term_and_avrg
    molsys.mol_props['cff'].vals = np.zeros((2, 2, 2))

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1},
                                               molsys_data=molsys, avrg_tensor=avrg, zero_tol=1e-18,
                                               states_choice='anharmonic')

    assert value == 0.
    assert contribs['NON_AVRG'] == 0.


## evaluate_term_coeff_sumover: the sum over the summed labels -----------------------
# The tests of evaluate_term_coeff_sumover, grouped by what they check. Toy molecule with 2 modes unless noted.
#
# Which index sets the sum visits
#   test_evaluate_term_coeff_sumover_enumerates_missing_index_combinations      0, 1, 2 labels summed; stubbed values
#   test_evaluate_term_coeff_sumover_mode_count_comes_from_eigenvals            3 modes; stubbed values
#   test_evaluate_term_coeff_sumover_sums_only_over_the_given_modes             modes=(0,)
#   test_evaluate_term_coeff_sumover_summed_labels_in_any_alphabetical_place    summed before or between fixed ones
# The total
#   test_evaluate_term_coeff_sumover_hand_computed_total                        2 labels summed, by hand
#   test_evaluate_term_coeff_sumover_total_is_sum_of_full_index_dict_values     2 labels summed
#   test_evaluate_term_coeff_sumover_on_the_fly_sums_c_over_both_modes          1 label summed, rank-4 average
# Where the averaged part comes from
#   test_evaluate_term_coeff_sumover_reads_the_avrg_tensor_it_is_given
#   test_evaluate_term_coeff_sumover_avrg_from_the_recipe_is_the_given_tensor
# The factors per index set
#   test_evaluate_term_coeff_sumover_zero_avrg_index_sets_are_kept
#   test_evaluate_term_coeff_sumover_without_factors_gives_the_same_total
#   test_evaluate_term_coeff_sumover_result_shape
# Errors
#   test_evaluate_term_coeff_sumover_resonance_labels_must_be_fixed
#   test_evaluate_term_coeff_sumover_needs_polarization_without_an_avrg_tensor
#
# Through other functions
#   test_build_contributions_keeps_no_factors_by_index_set (above)              keep_factors=False
#   test_build_contributions_coeff_is_evaluate_term_coeff_sumover (below)       row coefficient = the sum
#   test_pipeline.py     test_compute_features_amplitude_adds_both_terms
#                        test_feature_result_factors_by_index_set_of_a_row_are_its_summed_index_sets
#   test_integration.py  test_evaluate_term                                     formaldehyde (6 modes), 1 label summed
#   test_dimensions.py   test_ir_absorption_polhess_term_by_hand                formaldehyde (6 modes), 1 label summed
#                        test_ir_absorption_cff_terms_are_one_sum_twice         formaldehyde (6 modes), 2 labels summed

@pytest.mark.parametrize('fixed, expected_index_sets, expected_total', [
    ({'a': 1, 'b': 0, 'c': 1}, {(1, 0, 1)},                                 101),  # nothing missing
    ({'a': 1, 'b': 0},         {(1, 0, 0), (1, 0, 1)},                       201),  # sum over c
    ({'a': 1},                 {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}, 422),  # sum over b and c
])
def test_evaluate_term_coeff_sumover_enumerates_missing_index_combinations(fixed,
                                                                    expected_index_sets, expected_total,
                                                                    molsys, monkeypatch):
    """
    Which full index sets the sum visits. Every label of the term left out of `fixed` runs over
    the modes 0 .. n_modes-1, and each full index set gets one evaluate_full_index_dict call.
    The term fixes a and sums b and c; fixing b or c as well leaves fewer labels to sum.

    evaluate_full_index_dict is stubbed to return 100a + 10b + c, so the value of each index set
    spells out its modes and the totals add up by hand, e.g. {'a': 1} -> 100 + 101 + 110 + 111 = 422.
    """
    # Stand-in for the per-index-set evaluation: value = 100a + 10b + c, so the sums above are checkable by eye.
    monkeypatch.setattr(evaluate_mod, 'evaluate_full_index_dict',
                        lambda term, idx, *_, **__: (100 * idx['a'] + 10 * idx['b'] + idx['c'], {}))
    # the stub term has no averaged props, so the avrg tensor is stubbed out too
    monkeypatch.setattr(evaluate_mod, 'calculate_avrg_tensor_einsum', lambda *_: None)
    # Only the index split is read from the term here; molsys has 2 modes, so each missing index runs over {0, 1}.
    term = CompiledTerm(PropsCollection([]), PropsCollection([]), ResonanceMotif(()), FreqTermsCollection([]), 1.,
                        idx_summ=('b', 'c'), idx_nonsumm=('a',))

    from wilson_suite.wilson_intensities.amplitudes.averaging import (
        getGeneralPolarizationAveragingExpression,
    )
    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank=4,laser_pol=(1.,1.,1.))
    total, factors = evaluate_term_coeff_sumover(term, fixed, molsys_data=molsys, polarization_linear_comb=polarization_linear_comb,
                                                 states_choice='anharmonic')

    assert total == expected_total
    assert {tuple(index_set[k] for k in 'abc') for index_set in factors} == expected_index_sets


def test_evaluate_term_coeff_sumover_mode_count_comes_from_eigenvals(term_and_avrg, molsys, monkeypatch):
    """Three eigenvalues -> each missing label (b, c) runs over {0, 1, 2}: 3 * 3 = 9 index sets."""
    monkeypatch.setattr(evaluate_mod, 'evaluate_full_index_dict', lambda term, idx, *_, **__: (1., {}))
    term, avrg = term_and_avrg
    three_modes = replace(molsys, eigenvals={0: E0_eigval, 1: E1_eigval, 2: 2000.})

    total, factors = evaluate_term_coeff_sumover(term, {'a': 0}, three_modes, avrg_tensor=avrg,
                                                 states_choice='anharmonic')

    assert total == 9.
    assert {(index_set['b'], index_set['c']) for index_set in factors} == {(b, c) for b in range(3) for c in range(3)}


def test_evaluate_term_coeff_sumover_sums_only_over_the_given_modes(term_and_avrg, molsys):
    """toy_term at a = 0 sums over b and c; with modes (0,) only b = c = 0 is left: one index set instead of four"""
    term, avrg = term_and_avrg

    total, factors = evaluate_term_coeff_sumover(term, {'a': 0}, molsys, avrg_tensor=avrg, states_choice='anharmonic',
                                                 modes=(0,))

    only, _ = evaluate_full_index_dict(term, {'a': 0, 'b': 0, 'c': 0}, molsys, avrg, states_choice='anharmonic')
    assert set(factors) == {ParameterSet({'a': 0, 'b': 0, 'c': 0})}
    assert total == only


def renamed_ab_term(motif, names: str) -> CompiledTerm:
    """
    ab_term(motif) with the labels a, b, c renamed to names[0], names[1], names[2], e.g. MOTIF_A and 'bac':
    0.5 * cff[b,a,c] * <polgrad>[b] * 1/(E_ba - E_b) * 1/omega_b with A = E_b - E_0, b fixed, a and c summed.
    """
    new = dict(zip('abc', names))

    def rename(labels):
        return tuple(new[label] for label in labels)

    renamed_motif = tuple(((rename(left), rename(right)), pf) for (left, right), pf in motif)
    return replace(ab_term(renamed_motif),
                   avrg_props=PropsCollection([polprop(ops=(0, 1), inds=new['a'])]),
                   non_avrg_props=PropsCollection([polprop(inds=names)]),
                   cmp_freqdenom=FreqTermsCollection([vibdiff(sl=new['a']),
                                                      vibdiff(sl=new['a'] + new['b'], sr=new['a'], pert=True)]))


@pytest.mark.parametrize('motif, fixed, names', [
    (MOTIF_A,  {'a': 0},         'bac'),    # b fixed; a, c summed: a summed label before the fixed one
    (MOTIF_A,  {'a': 0},         'cab'),    # c fixed; a, b summed: both summed labels before it
    (MOTIF_AB, {'a': 0, 'b': 1}, 'acb'),    # a, c fixed; b summed: a summed label between the fixed ones
])
def test_evaluate_term_coeff_sumover_summed_labels_in_any_alphabetical_place(motif, fixed, names, molsys_a1_zero):
    """
    The summed labels may come before or between the fixed ones in the alphabet. Renaming the labels
    changes nothing else: the renamed term gives the total of ab_term(motif), and the same factors on the
    same index sets under the new names.
    """
    new = dict(zip('abc', names))

    def renamed(index_set):
        return ParameterSet({new[label]: mode for label, mode in index_set.items()})

    total, factors = evaluate_term_coeff_sumover(ab_term(motif), fixed, molsys_a1_zero, TOY_RECIPE,
                                                 states_choice='anharmonic')
    renamed_total, renamed_factors = evaluate_term_coeff_sumover(renamed_ab_term(motif, names),
                                                                 renamed(fixed).to_dict(), molsys_a1_zero,
                                                                 TOY_RECIPE, states_choice='anharmonic')

    assert renamed_total == pytest.approx(total) != 0.
    assert set(renamed_factors) == {renamed(index_set) for index_set in factors}
    assert all(renamed_factors[renamed(index_set)] == pytest.approx(contribs)
               for index_set, contribs in factors.items())


def test_evaluate_term_coeff_sumover_hand_computed_total(term_and_avrg, molsys):
    """
    a=0, sum over b, c of  0.5 * cff[0,b,c] * <polgrad>[0] * 1/(E_0b - E_0) * 1/omega_0 .
    c enters only cff, so the two c values are added first; b picks state '0,0' or '0,1'.
    """
    term, avrg = term_and_avrg

    total, _ = evaluate_term_coeff_sumover(term, {'a': 0}, avrg_tensor=avrg, molsys_data=molsys,
                                           states_choice='anharmonic')

    by_b = ((CFF[0, 0, 0] + CFF[0, 0, 1]) / convNu2Ene(E00 - E0)
            + (CFF[0, 1, 0] + CFF[0, 1, 1]) / convNu2Ene(E01 - E0))
    assert total == pytest.approx(0.5 * POLGRAD_AVRG[0] / convNu2Ene(E0_eigval) * by_b)


def test_evaluate_term_coeff_sumover_total_is_sum_of_full_index_dict_values(term_and_avrg, molsys):
    """
    The total is the sum of evaluate_full_index_dict over the index sets, and the factors kept per
    index set are what that call returns: nothing dropped, counted twice or rescaled.
    toy_term() at a = 0 sums b and c over 2 modes: 4 index sets.
    index_set.to_dict() gives the plain dict that evaluate_full_index_dict takes.
    """
    term, avrg = term_and_avrg

    total, factors = evaluate_term_coeff_sumover(term, {'a': 0}, avrg_tensor=avrg, molsys_data=molsys,
                                                 states_choice='anharmonic')

    per_index_set = {
        index_set: evaluate_full_index_dict(term, index_set.to_dict(), molsys_data=molsys, avrg_tensor=avrg,
                                            zero_tol=1e-18, states_choice='anharmonic')
        for index_set in factors}
    assert len(factors) == 4             # b, c in {0, 1}
    assert total == pytest.approx(sum(value for value, _ in per_index_set.values()))
    assert factors == {index_set: contribs for index_set, (_, contribs) in per_index_set.items()}


def test_evaluate_term_coeff_sumover_on_the_fly_sums_c_over_both_modes(molsys):
    term = evv_shaped_term()
    otf_molsys = with_evv_props(molsys)
    tensor = calculate_avrg_tensor_einsum(term.avrg_props, evv_props(), ISO4)

    total, factors = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 1}, otf_molsys, polarization_linear_comb=ISO4,
                                                 states_choice='anharmonic')

    by_hand = sum(evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': c}, otf_molsys, tensor,
                                           states_choice='anharmonic')[0] for c in (0, 1))
    assert set(factors) == {ParameterSet({'a': 0, 'b': 1, 'c': c}) for c in (0, 1)}
    assert total == pytest.approx(by_hand) != 0.


def test_evaluate_term_coeff_sumover_reads_the_avrg_tensor_it_is_given(molsys):
    """No polarization: the average must come from the given tensor, not from a new einsum."""
    given = AvrgTensor(('a', 'b', 'c'), np.full((2, 2, 2), 7.))

    _, factors = evaluate_term_coeff_sumover(evv_shaped_term(), {'a': 0, 'b': 1}, with_evv_props(molsys), avrg_tensor=given,
                                             states_choice='anharmonic')

    assert [contribs['AVRG'] for contribs in factors.values()] == [7., 7.]


def test_evaluate_term_coeff_sumover_avrg_from_the_recipe_is_the_given_tensor(term_and_avrg, molsys_a1_zero):
    """
    No tensor given -> <polgrad>[a] is built from polarization_linear_comb and the raw 'polgrad' data:
    <polgrad>[a] = sum over (i, j) of coeff_ij * polgrad[a, i, j], with TOY_RECIPE the trace / 3 of
    TOY_POLGRAD[a]: (0.6 / 3, 0) = POLGRAD_AVRG, the given tensor. Both ways give the same total.
    """
    term, avrg = term_and_avrg

    given_total, given_factors = evaluate_term_coeff_sumover(term, {'a': 0}, molsys_a1_zero, avrg_tensor=avrg,
                                                             states_choice='anharmonic')
    recipe_total, recipe_factors = evaluate_term_coeff_sumover(term, {'a': 0}, molsys_a1_zero,
                                                               polarization_linear_comb=TOY_RECIPE,
                                                               states_choice='anharmonic')

    assert recipe_total == pytest.approx(given_total)
    assert set(recipe_factors) == set(given_factors)
    assert all(recipe_factors[index_set]['AVRG'] == pytest.approx(POLGRAD_AVRG[0]) for index_set in recipe_factors)


def test_evaluate_term_coeff_sumover_zero_avrg_index_sets_are_kept(molsys):
    """
    <polgrad>[1] = 0 -> every index set with a=1 stops right after AVRG. The index sets are still recorded,
    with only the AVRG entry. The states a=1 would need ('1,1' is missing) are never looked up.
    """
    total, factors = evaluate_term_coeff_sumover(toy_term(), {'a': 1}, molsys,
                                                 avrg_tensor=AvrgTensor(('a',), np.array([10., 0.])),
                                                 states_choice='anharmonic')

    assert total == 0.
    assert len(factors) == 4
    assert all(contribs == {'AVRG': 0.} for contribs in factors.values())


def test_evaluate_term_coeff_sumover_without_factors_gives_the_same_total(term_and_avrg, molsys):
    """keep_factors=False (build_contributions): no factors per index set, the same total to the last bit"""
    term, avrg = term_and_avrg

    total, factors = evaluate_term_coeff_sumover(term, {'a': 0}, molsys, avrg_tensor=avrg,
                                                 states_choice='anharmonic')
    total_only, no_factors = evaluate_term_coeff_sumover(term, {'a': 0}, molsys, avrg_tensor=avrg,
                                                         keep_factors=False, states_choice='anharmonic')

    assert no_factors is None
    assert len(factors) == 4
    assert total_only == total != 0.


def test_evaluate_term_coeff_sumover_result_shape(term_and_avrg, molsys):
    """
    Shows the shape of the evaluate_term_coeff_sumover result (run with -s to see the print):
    (total, {ParameterSet(full index set): factors}). cff[0, 0, 1] = 0 by symmetry, so c = 1 stops after
    NON_AVRG.
    """
    term, avrg = term_and_avrg

    result = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 0}, molsys, avrg_tensor=avrg,
                                         states_choice='anharmonic')

    from pprint import pprint
    pprint(result)

    total, factors = result
    assert isinstance(total, float)
    assert set(factors) == {ParameterSet({'a': 0, 'b': 0, 'c': c}) for c in (0, 1)}
    assert set(factors[ParameterSet({'a': 0, 'b': 0, 'c': 0})]) == {'NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM'}
    assert set(factors[ParameterSet({'a': 0, 'b': 0, 'c': 1})]) == {'NON_AVRG', 'AVRG'}


@pytest.mark.parametrize('fixed, not_fixed', [
    ({'a': 0},  ['b']),
    ({'b': 0},  ['a']),
    ({},        ['a', 'b']),
])
def test_evaluate_term_coeff_sumover_resonance_labels_must_be_fixed(fixed, not_fixed, molsys):
    """
    a and b are resonance labels: they decide where the peak sits. Summing over b would add
    peaks at different positions into one number, so the call raises instead.
    """
    term = CompiledTerm(PropsCollection([]), PropsCollection([]), ResonanceMotif(()), FreqTermsCollection([]), 1.,
                        idx_summ=('c',), idx_nonsumm=('a', 'b'))

    with pytest.raises(ValueError) as err:
        evaluate_term_coeff_sumover(term, fixed, molsys_data=molsys, states_choice='anharmonic')
    assert str(err.value) == f'resonance labels {not_fixed} must be fixed, not summed'


def test_evaluate_term_coeff_sumover_needs_polarization_without_an_avrg_tensor(molsys):
    with pytest.raises(ValueError, match='polarization_linear_comb is required when no avrg_tensor is given'):
        evaluate_term_coeff_sumover(toy_term(), {'a': 0}, molsys, states_choice='anharmonic')


## build_contributions --------------------------------------------------------
# One row per (term, index set) with a nonzero coefficient. ab_term fixes a, b (its motif's
# labels) and sums over c. molsys_a1_zero with TOY_RECIPE gives <polgrad>[a] = (0.2, 0), so every pair
# with a=1 has coeff 0: 2 rows (a=0) and 2 zero pairs (a=1). a=b=1 needs state '1,1', which `states`
# lacks, so solving that pair would raise: these tests also show that zero pairs never reach solve_LSE_motif.
# MOTIF_AB at a=0: A = E0 = 1000; B = E00 - E0 = 950 for b=0, E01 - E0 = 1400 for b=1.

def test_toy_polgrad_and_recipe_give_0_2_and_0(molsys_a1_zero):
    """
    The averaged part the build_contributions and compute_features tests use: trace / 3 of each mode's polgrad.
    TOY_RECIPE is the real rank-2 recipe.
    """
    tensor = calculate_avrg_tensor_einsum(ab_term().avrg_props, molsys_a1_zero.mol_props, TOY_RECIPE)

    assert TOY_RECIPE == pytest.approx(getGeneralPolarizationAveragingExpression(rank=2, laser_pol=(1.,)))
    assert tensor.labels == ('a',)
    np.testing.assert_allclose(tensor.values, [0.2, 0.], rtol=1e-15, atol=0)


# -- what becomes a row: rows, zero pairs ------------------------------------------

def test_build_contributions_one_row_per_nonzero_pair(molsys_a1_zero):
    rows, zero = build_contributions([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert {r.params for r in rows} == {PS_00, PS_01}
    assert all(r.term_id == 0 and r.motif == ResonanceMotif.from_tuples(MOTIF_AB) for r in rows)
    assert zero == [(0, PS_10), (0, PS_11)]
    assert len(rows) + len(zero) == 4          # every (a, b) pair exactly once


def test_build_contributions_term_built_by_hand_without_a_single_point_stops_the_run(molsys_a1_zero):
    """
    MOTIF_B_TWICE asks w_B = -E_a and w_B = E_b - E_a at once: no point for any a=0 pair.
    A compiled term cannot have this motif (CompiledTerm.from_VibPertTerm raises);
    a term built by hand can, and then the run stops instead of dropping the pair.
    """
    with pytest.raises(np.linalg.LinAlgError, match='inconsistent'):
        build_contributions([ab_term(MOTIF_B_TWICE)], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')


def test_build_contributions_without_terms_is_empty(molsys):
    rows, zero = build_contributions([], molsys, TOY_RECIPE, states_choice='anharmonic')

    assert (len(rows), zero) == (0, [])


def test_build_contributions_term_without_conditions_is_one_row_summing_every_label(molsys_a1_zero):
    """
    A flat background: no label to fix, so one index set, {}, at the location without axes.
    Its coefficient sums every (a, b, c): the rows and zero pairs of ab_term (the same summand) added up.
    """
    resonant, _ = build_contributions([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    rows, zero = build_contributions([flat_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    [flat] = rows
    assert zero == []
    assert (flat.term_id, flat.motif, flat.params, flat.location) == (0, ResonanceMotif(()), ParameterSet({}),
                                                                     ResLocPoint({}))
    assert flat.coeff == pytest.approx(sum(r.coeff for r in resonant), rel=1e-12)


# -- the numbers in a row: coeff and location --------------------------------------

def test_build_contributions_coeff_is_evaluate_term_coeff_sumover(molsys_a1_zero):
    term = ab_term()

    rows, _ = build_contributions([term], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    for r in rows:
        expected, _ = evaluate_term_coeff_sumover(term, {'a': r.params['a'], 'b': r.params['b']}, molsys_a1_zero,
                                                  TOY_RECIPE, states_choice='anharmonic')
        assert r.coeff == expected != 0.


def test_build_contributions_locations_are_in_cm1(molsys_a1_zero):
    rows, _ = build_contributions([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    location = {r.params: r.location for r in rows}
    assert location[PS_01].as_dict() == pytest.approx({'A': E0, 'B': E01 - E0})   # 1000, 1400 cm-1
    assert location[PS_00].as_dict() == pytest.approx({'A': E0, 'B': E00 - E0})   # 1000, 950 cm-1


# -- the location cache: one solve per (motif, params) -------------------------------

def test_build_contributions_one_solve_per_motif_and_params(molsys_a1_zero, monkeypatch):
    """Two terms with one motif: 2 solves, not 4, and the rows of both terms hold the same location object."""
    solved = []
    real_solve = evaluate_mod.solve_LSE_motif
    def counting_solve(motif, ps, *args, **kwargs):
        solved.append(ps)
        return real_solve(motif, ps, *args, **kwargs)
    monkeypatch.setattr(evaluate_mod, 'solve_LSE_motif', counting_solve)
    term = ab_term()

    rows, _ = build_contributions([term, replace(term, frac_factor=2.)], molsys_a1_zero, TOY_RECIPE,
                                  states_choice='anharmonic')

    assert Counter(solved) == Counter([PS_00, PS_01])           # each once, in any order
    row = {(r.term_id, r.params): r for r in rows}
    for ps in (PS_00, PS_01):
        assert row[(0, ps)].location is row[(1, ps)].location
        assert row[(1, ps)].coeff == pytest.approx(4 * row[(0, ps)].coeff)    # frac_factor 0.5 -> 2.


def test_build_contributions_cache_keeps_motifs_apart(molsys_a1_zero):
    """
    Same index sets, different motifs: each motif gets its own location. A cache keyed by params alone fails this.
    The second motif is the other EVV shape: B = E_b - E_a instead of E_ab - E_a.
    """
    other_evv = ((((), ('a',)), ('-A',)), ((('b',), ('a',)), ('B',)))
    rows, _ = build_contributions([ab_term(MOTIF_AB), ab_term(other_evv)], molsys_a1_zero, TOY_RECIPE,
                                  states_choice='anharmonic')

    location = {(r.term_id, r.params): r.location for r in rows}
    assert location[(0, PS_01)].as_dict() == pytest.approx({'A': E0, 'B': E01 - E0})   # 1000, 1400 cm-1
    assert location[(1, PS_01)].as_dict() == pytest.approx({'A': E0, 'B': E1 - E0})    # 1000, 500 cm-1


# -- input checks --------------------------------------------------------------------

def test_build_contributions_needs_every_prop_the_terms_use(molsys):
    """ab_term uses polgrad and cff, molsys has cff only: a ValueError before any row, not a KeyError deep inside"""
    with pytest.raises(ValueError, match=r"no values for \['polgrad'\]"):
        build_contributions([ab_term()], molsys, TOY_RECIPE, states_choice='anharmonic')


## exclude_modes: a mode left out of the evaluation ----------------------------
# The toy molecule has 2 modes, 0 and 1. An excluded mode is no value of a resonance label (no row, no zero
# pair) and no value of a summed label (included_modes).

def test_included_modes_without_exclusion_is_every_mode(molsys):
    assert included_modes(molsys) == (0, 1)


def test_included_modes_leaves_out_the_excluded_modes(molsys):
    assert included_modes(molsys, (1,)) == (0,)
    assert included_modes(molsys, [1, 1]) == (0,)


@pytest.mark.parametrize('exclude', [(2,), (-1,), ('1',)], ids=['beyond the last mode', 'negative', 'not a number'])
def test_included_modes_raises_for_a_mode_that_does_not_exist(exclude, molsys):
    """modes count from 0, as in the data: a 1-based list would name a mode beyond the last one"""
    with pytest.raises(ValueError, match=r'no such mode, the modes are 0 \.\. 1'):
        included_modes(molsys, exclude)


def test_included_modes_raises_if_no_mode_is_left(molsys):
    with pytest.raises(ValueError, match='removes all 2 modes'):
        included_modes(molsys, (0, 1))


def test_build_contributions_excluded_mode_is_left_out_of_rows_and_sums(molsys_a1_zero):
    """
    Exclude mode 1: toy_term's only index set is a = 0 (no row and no zero pair with a 1), and its sum over b, c
    keeps only b = c = 0. So its coefficient is that piece alone; without exclusion the b = c = 1 piece adds to it
    (cff[0, 1, 1] = 2; the pieces with b != c are 0, as cff[0, 0, 1] = 0 by symmetry).
    """
    term = toy_term()

    rows, zero = build_contributions([term], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic',
                                     exclude_modes=(1,))
    full_rows, _ = build_contributions([term], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    [row] = rows
    assert (row.params, zero) == (ParameterSet({'a': 0}), [])
    avrg = calculate_avrg_tensor_einsum(term.avrg_props, molsys_a1_zero.mol_props, TOY_RECIPE)
    c0_only, _ = evaluate_full_index_dict(term, {'a': 0, 'b': 0, 'c': 0}, molsys_a1_zero, avrg,
                                          states_choice='anharmonic')
    assert row.coeff == pytest.approx(c0_only, rel=1e-14, abs=0)
    full = {r.params: r.coeff for r in full_rows}[ParameterSet({'a': 0})]
    assert row.coeff != pytest.approx(full)


## Resonance factors on the grid -----------------------------------------------
# One condition is the denominator  E_left - E_right - sum_j s_j w_j - i*gamma  (derive's convention).
# eval_rescond_resloc and eval_resonance_factor take grids in Eh; eval_feature_on_grid takes cm-1.
# resonance_residuals (top of this file) writes  E_left - E_right - sum_j s_j w_j  out independently.
#
# The get_intensity test and the hand-worked values (eval_resonance_factor, eval_feature_on_grid) use the
# Lorentzian (test_lineshape.py tests it alone). The other tests swap in a recording factor (the fixture
# `recorded`), so they hold for any line shape.

G_CM = 5.                   # propagator_param parameter, cm-1
G_AU = convNu2Ene(G_CM)
CM1_PER_EH = 219474.6313632  # 1 Eh in cm-1 (CODATA), written out so the hand-worked values don't use convNu2Ene


def grid_around(location: dict[str, float], step: float) -> dict[str, np.ndarray]:
    """3 points per axis (v - step, v, v + step), ij meshgrid: index (1, 1, ...) is `location`."""
    axes = sorted(location)
    mesh = np.meshgrid(*(location[ax] + np.array([-step, 0., step]) for ax in axes), indexing='ij')
    return dict(zip(axes, mesh))


def to_au(grid: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {ax: convNu2Ene(v) for ax, v in grid.items()}


def feature_at_solved_location(motif, params: ParameterSet, states, coeffs=(0.3,), propagator_param=G_CM) -> SpectralFeature:
    """
    One feature, one row per coeff, at the location solve_LSE_motif finds (cm-1); width `propagator_param`.
    A propagator_param function also gets the conditions, as features_from_rows gives them.
    """
    res_motif = ResonanceMotif.from_tuples(motif)
    location = solve_LSE_motif(res_motif, params, states, unit='cm-1')
    rows = tuple(ContributionRow(term_id=i, motif=res_motif, params=params, location=location, coeff=c)
                 for i, c in enumerate(coeffs))
    conditions = condition_states(res_motif, params, states) if callable(propagator_param) else None
    return SpectralFeature(location=location, rows=rows, propagator_param=propagator_param, conditions=conditions)


@pytest.fixture
def recorded(monkeypatch) -> list:
    """
    Stands in for any line shape on the grid: lineshape.factor, swapped for one test, stores
    each (resloc, gamma) it gets and returns resloc + i*gamma (not a Lorentzian).
    Returns the list of calls, in order.
    """
    calls = []

    def factor(resloc, gamma):
        calls.append((resloc, gamma))
        return resloc + 1j * gamma

    monkeypatch.setattr(lineshape_mod, 'factor', factor)
    return calls


# -- eval_rescond_resloc --------------------------------------------------------

def test_eval_rescond_resloc_sign_is_E_minus_freq(states):
    """Condition ((a,), ()) on A, a=0: E = E0. At A = 0 the result is +E0, not -E0."""
    key = ResCondKey(diff=(('a',), ()), pf=('A',))
    vd = VibDiff.from_quanta(*key.diff, {'a': 0}, states)

    assert eval_rescond_resloc(key, vd, {'A': 0.}) == pytest.approx(convNu2Ene(E0))         # type: ignore
    assert eval_rescond_resloc(key, vd, {'A': convNu2Ene(E0)}) == pytest.approx(0.)          # type: ignore


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_rescond_resloc_matches_the_independent_residuals(motif, params_obj, states):
    """Every condition, on a 2D grid in Eh; '-B' enters the sum with -1 (MOTIF_MIXED)."""
    res_motif = ResonanceMotif.from_tuples(motif)
    grid = to_au(grid_around({'A': 300., 'B': -700.}, 100.))

    expected = resonance_residuals(res_motif, grid, params_obj.to_dict(), states)  # type: ignore

    for key, exp in zip(res_motif, expected):
        vd = VibDiff.from_quanta(*key.diff, params_obj.to_dict(), states)
        np.testing.assert_allclose(eval_rescond_resloc(key, vd, grid), exp)


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_rescond_resloc_is_zero_at_the_solved_location(motif, params_obj, states):
    """Same equation as solve_LSE_motif: every condition is met at the point it finds."""
    res_motif = ResonanceMotif.from_tuples(motif)
    location = solve_LSE_motif(res_motif, params_obj, states).as_dict()     # Eh

    for key in res_motif:
        vd = VibDiff.from_quanta(*key.diff, params_obj.to_dict(), states)
        assert eval_rescond_resloc(key, vd, location) == pytest.approx(0., abs=1e-15)  # type: ignore


# -- eval_resonance_factor ------------------------------------------------------

@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_resonance_factor_calls_the_line_shape_once_per_condition_with_E_minus_freq(motif, params_obj, states,
                                                                                         recorded):
    """
    e.g. MOTIF_AB, a=0, b=1 (in cm-1): two calls, with resloc A - 1000 ('-A') and 1400 - B.
    Checked on a 3x3 grid in Eh against resonance_residuals.
    """
    res_motif = ResonanceMotif.from_tuples(motif)
    grid = to_au(grid_around({'A': 300., 'B': -700.}, 100.))

    eval_resonance_factor(res_motif, params_obj, states, grid, lambda _vd: G_AU)

    residuals = resonance_residuals(res_motif, grid, params_obj.to_dict(), states)  # type: ignore
    assert len(recorded) == len(res_motif)
    for (resloc, _), expected in zip(recorded, residuals):
        np.testing.assert_allclose(resloc, expected)


def test_eval_resonance_factor_gives_each_condition_its_own_gamma(params_obj, states, recorded):
    """
    gamma(vd) is asked once per condition, so a width per pair of states is possible.
    Here gamma returns the condition's own E (cm-1). At freq = 0, resloc is E in Eh,
    so each call must get resloc == convNu2Ene(gamma).
    """
    eval_resonance_factor(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, {'A': 0., 'B': 0.},  # type: ignore
                          propagator_param=lambda vd: vd.energy_difference())

    assert sorted(g for _, g in recorded) == pytest.approx(sorted([-E0, E01 - E0]))
    for resloc, g in recorded:
        assert resloc == pytest.approx(convNu2Ene(g))


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_resonance_factor_multiplies_the_line_shape_factors(motif, params_obj, states, recorded):
    """e.g. MOTIF_AB: factor(resloc_1, gamma) * factor(resloc_2, gamma), at every grid point."""
    res_motif = ResonanceMotif.from_tuples(motif)
    grid = to_au(grid_around({'A': 300., 'B': -700.}, 100.))

    value = eval_resonance_factor(res_motif, params_obj, states, grid, lambda _vd: G_AU)

    np.testing.assert_allclose(value, np.prod([r + 1j * g for r, g in recorded], axis=0))


def test_eval_resonance_factor_lorentzian_values_with_a_width_per_condition(params_obj, states):
    """
    A width per condition. MOTIF_AB, a=0, b=1, width = |E| / 100: 10 cm-1 for the A condition (E = -1000),
    14 cm-1 for B (E = 1400).
        1 / ((A - 1000 - 10i) * (1400 - B - 14i))   in cm-1, times CM1_PER_EH**2
    Peak: (i/10) * (i/14) = -1/140. One width off the peak on either axis gives the same value,
        (1010, 1400):  (1 + i)/20 * i/14  =  (-1 + i) / 280
        (1000, 1386):  i/10 * (1 + i)/28  =  (-1 + i) / 280
    which only holds if each condition gets its own width (swapped widths give different values).
    """
    grid = to_au({'A': np.array([1000., 1010., 1000.]), 'B': np.array([1400., 1400., 1386.])})

    value = eval_resonance_factor(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, grid,
                                  lambda vd: abs(vd.energy_difference(au=True)) / 100) # type: ignore

    expected = [-1 / 140, (-1 + 1j) / 280, (-1 + 1j) / 280]
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


# -- eval_feature_on_grid -------------------------------------------------------

@pytest.mark.parametrize('ps', [PS_01, PS_10])
@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_feature_on_grid_gives_the_line_shape_resloc_and_gamma_in_eh(motif, ps, states, recorded):
    """
    Grid and propagator_param go in as cm-1. The line shape must get E_i - freq_i and gamma in Eh.
    The states come from the feature's own params: a=0, b=1 and a=1, b=0 resonate at different places.
    """
    f = feature_at_solved_location(motif, ps, states)
    mesh_cm = grid_around(f.location.as_dict(), G_CM)

    eval_feature_on_grid(f, states, mesh_cm)

    residuals = resonance_residuals(ResonanceMotif.from_tuples(motif), to_au(mesh_cm), ps.to_dict(), states)  # type: ignore
    assert len(recorded) == len(motif)
    for (resloc, gamma), expected in zip(recorded, residuals):
        np.testing.assert_allclose(resloc, expected, atol=1e-15)   # atol: the centre point is 0
        assert gamma == pytest.approx(G_AU)


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED, MOTIF_AB_EXTRA])
def test_eval_feature_on_grid_peak_matches_get_intensity(motif, params_obj, states):
    """
    get_intensity (features.py) and the grid take the line shape from lineshape.py, so the peak on
    the grid is the peak get_intensity gives (and sizes the boxes with): for the Lorentzian
    |amplitude / (-i*gamma)^N|^2, N the number of conditions, also with more conditions than
    axes (MOTIF_AB_EXTRA: N = 3 on 2 axes).
    """
    f = feature_at_solved_location(motif, params_obj, states)

    value = eval_feature_on_grid(f, states, f.location.as_dict())  # type: ignore

    assert abs(value) ** 2 == pytest.approx(f.get_intensity())


def test_eval_feature_on_grid_uses_the_summed_amplitude(params_obj, states):
    """Rows add up before the resonance factor, which is applied once per feature (not once per row)."""
    one = feature_at_solved_location(MOTIF_AB, params_obj, states, coeffs=(1.,))
    two = feature_at_solved_location(MOTIF_AB, params_obj, states, coeffs=(0.3, 0.5))
    grid = grid_around(one.location.as_dict(), G_CM)
    base = eval_feature_on_grid(one, states, grid)

    np.testing.assert_allclose(eval_feature_on_grid(two, states, grid), 0.8 * base)


# Hand-worked values, default Lorentzian, coeff 0.3, gamma 5 cm-1. One condition is
#     0.3 / (E - freq - i*gamma)  in Eh  =  CM1_PER_EH * 0.3 / (E - freq - i*gamma)  in cm-1,
# so a motif with N conditions picks up CM1_PER_EH**N. The numbers below are the cm-1 part.
# Grid centres are written out (not taken from solve_LSE_motif).

@pytest.mark.parametrize('ps, centre', [(PS_01, 1000.),     # E_a - E_0 = E0
                                        (PS_10, 1500.)])    # E1
def test_eval_feature_on_grid_values_one_condition(ps, centre, states):
    """
    MOTIF_A: 0.3 / (centre - A - 5i) at A = centre -10, -5, 0, +5, +10 cm-1, e.g.
        A = centre - 5:  0.3 / (5 - 5i) = 0.3 * (5 + 5i) / 50 = 0.03 + 0.03i
    At A = centre +- gamma, |value|^2 is half the peak (0.0018 vs 0.0036): full width 2*gamma.
    """
    f = feature_at_solved_location(MOTIF_A, ps, states)
    grid = {'A': centre + np.array([-10., -5., 0., 5., 10.])}

    value = eval_feature_on_grid(f, states, grid)

    expected = [0.024 + 0.012j, 0.03 + 0.03j, 0.06j, -0.03 + 0.03j, -0.024 + 0.012j]
    assert np.shape(value) == (5,)
    np.testing.assert_allclose(value, CM1_PER_EH * np.array(expected))


def test_eval_feature_on_grid_values_two_conditions(states):
    """
    MOTIF_AB, a=0, b=1: 0.3 / ((A - 1000 - 5i) * (1400 - B - 5i)), one factor per axis; A enters as
    A - 1000, since the A condition is on -A.
    3x3 grid, A = 1000 +- 5 (rows), B = 1400 +- 5 (columns). Per axis, 1 / (x - 5i) is
        x = +5: 0.1 + 0.1i      x = 0: 0.2i      x = -5: -0.1 + 0.1i
    with x = A - 1000 on A and x = 1400 - B on B. Each entry is 0.3 times the product of the two,
    e.g. the peak 0.3 * (0.2i)^2 = -0.012.
    """
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    grid = grid_around({'A': 1000., 'B': 1400.}, 5.)

    value = eval_feature_on_grid(f, states, grid)

    expected = [[-0.006,          -0.006 - 0.006j, -0.006j],
                [-0.006 + 0.006j, -0.012,          -0.006 - 0.006j],
                [0.006j,          -0.006 + 0.006j, -0.006]]
    assert np.shape(value) == (3, 3)
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


def test_eval_feature_on_grid_values_with_a_minus_axis(states):
    """
    MOTIF_MIXED, a=0: conditions on B and on A - B, both with E = 0 - E0 = -1000:
        0.3 / ((-1000 - B - 5i) * (-1000 - (A - B) - 5i))
    3x3 grid, A = -2000 +- 5 (rows), B = -1000 +- 5 (columns). B is in both conditions, so this
    is not one factor per axis, e.g.
        A = -2005, B = -995:  0.3 / ((-5 - 5i) * (10 - 5i)) = 0.3 / (-75 - 25i) = -0.0036 + 0.0012i
    On the diagonal (A and B move together) the A - B condition stays at resonance.
    """
    f = feature_at_solved_location(MOTIF_MIXED, PS_01, states)
    grid = grid_around({'A': -2000., 'B': -1000.}, 5.)

    value = eval_feature_on_grid(f, states, grid)

    expected = [[-0.006 + 0.006j,   -0.006 + 0.006j, -0.0036 + 0.0012j],
                [-0.006,            -0.012,          -0.006],
                [-0.0036 - 0.0012j, -0.006 - 0.006j, -0.006 - 0.006j]]
    assert np.shape(value) == (3, 3)
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


def test_eval_feature_on_grid_axis_outside_the_motif_changes_nothing(states):
    """MOTIF_A only has a condition on A. On an A x B grid every column is the 1D result: a ridge along B."""
    f = feature_at_solved_location(MOTIF_A, PS_01, states)
    grid = grid_around({'A': 1000., 'B': 0.}, 5.)

    value = eval_feature_on_grid(f, states, grid)

    expected = [[0.03 + 0.03j] * 3,
                [0.06j] * 3,
                [-0.03 + 0.03j] * 3]
    assert np.shape(value) == (3, 3)
    np.testing.assert_allclose(value, CM1_PER_EH * np.array(expected))


@pytest.mark.parametrize('motif, gamma, expected_cm1', [
    (MOTIF_A,     5.,  0.06j),      # 0.3 / (-5i)    =  0.3i / 5
    (MOTIF_A,     10., 0.03j),
    (MOTIF_AB,    5.,  -0.012),     # 0.3 / (-5i)^2  = -0.3 / 25
    (MOTIF_AB,    10., -0.003),
    (MOTIF_MIXED, 5.,  -0.012),
    (MOTIF_AB_EXTRA, 5., -0.0024j),   # 0.3 / (-5i)^3 = 0.3 * (i / 5)^3 = -0.3i / 125
])
def test_eval_feature_on_grid_peak_is_coeff_times_i_over_gamma_to_the_N(motif, gamma, expected_cm1, params_obj, states):
    """
    At resonance every condition gives 1 / (-i*gamma) = i / gamma, so the peak is 0.3 * (i / gamma)^N:
    on the imaginary axis for one condition, real and negative for two. Wider line -> lower peak.
    get_intensity only checks |peak|^2; this also checks the phase (the -i*gamma sign).
    """
    f = replace(feature_at_solved_location(motif, params_obj, states), propagator_param=gamma)

    value = eval_feature_on_grid(f, states, f.location.as_dict())  # type: ignore

    assert value == pytest.approx(CM1_PER_EH ** len(motif) * expected_cm1)


def test_eval_feature_on_grid_propagator_param_function_gives_each_condition_its_own_width(states):
    """
    Same points and widths as the eval_resonance_factor test above (10 cm-1 on A, 14 cm-1 on B), with the
    width function on the feature, in cm-1. Values are 0.3 times that test's.
    """
    f = feature_at_solved_location(MOTIF_AB, PS_01, states,
                                   propagator_param=lambda vd, freqs: abs(vd.energy_difference()) / 100)
    mesh_cm = {'A': np.array([1000., 1010., 1000.]), 'B': np.array([1400., 1400., 1386.])}

    value = eval_feature_on_grid(f, states, mesh_cm)

    expected = 0.3 * np.array([-1 / 140, (-1 + 1j) / 280, (-1 + 1j) / 280])
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


def test_eval_feature_on_grid_propagator_param_function_of_a_constant_equals_the_number(states):
    """lambda vd, freqs: 5. gives the same as propagator_param=5. MOTIF_AB, 3x3 grid around the peak."""
    number = feature_at_solved_location(MOTIF_AB, PS_01, states)
    function = feature_at_solved_location(MOTIF_AB, PS_01, states, propagator_param=lambda vd, freqs: 5.)
    grid = grid_around(number.location.as_dict(), G_CM)

    np.testing.assert_allclose(eval_feature_on_grid(function, states, grid), eval_feature_on_grid(number, states, grid))


def test_eval_feature_on_grid_propagator_param_function_can_depend_on_the_frequency(states):
    """
    The function gets the grid as freqs and returns one value per grid point.
    Here width = A - 995 cm-1: 5 at A = 1000, 10 at A = 1005. MOTIF_A, peak at 1000:
        A = 1000:  0.3 / (-5i)       = 0.06i
        A = 1005:  0.3 / (-5 - 10i)  = 0.3 * (-5 + 10i) / 125 = -0.012 + 0.024i
    """
    f = feature_at_solved_location(MOTIF_A, PS_01, states, propagator_param=lambda vd, freqs: freqs['A'] - 995.)

    value = eval_feature_on_grid(f, states, {'A': np.array([1000., 1005.])})

    np.testing.assert_allclose(value, CM1_PER_EH * np.array([0.06j, -0.012 + 0.024j]))


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_feature_on_grid_peak_matches_get_intensity_with_a_propagator_param_function(motif, params_obj, states):
    """
    The peak on the grid and get_intensity (so the box size) take one width. Here a width per condition,
    1 % of |E| (MOTIF_AB, a=0, b=1: 10 and 14 cm-1), plus a part that changes with A.
    abs(): E is -1000 on MOTIF_AB's A condition and on both conditions of MOTIF_MIXED.
    """
    width = lambda vd, freqs: 0.01 * abs(vd.energy_difference()) + 0.001 * np.abs(freqs['A'])
    f = feature_at_solved_location(motif, params_obj, states, propagator_param=width)

    value = eval_feature_on_grid(f, states, f.location.as_dict())  # type: ignore

    assert abs(value) ** 2 == pytest.approx(f.get_intensity())


def test_eval_feature_on_grid_needs_a_propagator_param(params_obj, states):
    f = replace(feature_at_solved_location(MOTIF_A, params_obj, states), propagator_param=None)

    with pytest.raises(ValueError, match='propagator_param'):
        eval_feature_on_grid(f, states, {'A': np.zeros(3)})


def test_eval_feature_on_grid_needs_rows(states):
    f = SpectralFeature(location=ResLocPoint({'A': 0.}), propagator_param=G_CM)

    with pytest.raises(ValueError, match='no motif'):
        eval_feature_on_grid(f, states, {'A': np.zeros(3)})


def flat_feature(coeff: float = 0.3) -> SpectralFeature:
    """A flat background: one row of a term without conditions, at the location without axes."""
    return SpectralFeature(location=ResLocPoint({}), rows=(row(coeff, location=ResLocPoint({})),),
                           propagator_param=G_CM)


@pytest.mark.parametrize('grid, shape', [({'A': np.zeros((4, 3)), 'B': np.ones((4, 3))}, (4, 3)),
                                         ({'A': 1600., 'B': 500.}, ())])
def test_eval_feature_on_grid_flat_feature_is_its_coefficient_on_every_point(grid, shape, states):
    value = eval_feature_on_grid(flat_feature(0.3), states, grid)

    assert np.shape(value) == shape
    np.testing.assert_array_equal(value, np.full(shape, 0.3 + 0j))


# -- all_features_on_grid -------------------------------------------------------------------
# Peaks with a=0, b=1 (PS_01): MOTIF_AB at A = 1000, B = 1400.  With a=1, b=0 (PS_10): MOTIF_A at A = 1500,
# a ridge along B.

def test_all_features_on_grid_is_the_sum_of_the_features_on_the_whole_grid(states):
    peak = feature_at_solved_location(MOTIF_AB, PS_01, states)
    ridge = feature_at_solved_location(MOTIF_A, PS_10, states)
    axis_points = {'A': np.array([1000., 1500.]), 'B': np.array([1395., 1400., 1405.])}
    mesh = make_mesh(axis_points)

    value = all_features_on_grid([peak, ridge], states, axis_points)

    expected = eval_feature_on_grid(peak, states, mesh) + eval_feature_on_grid(ridge, states, mesh)
    np.testing.assert_allclose(value, expected)


def test_all_features_on_grid_index_i_j_is_the_point_A_i_B_j(states):
    """ij indexing: value[i, j] belongs to A = axis_points['A'][i], B = axis_points['B'][j]; A has 5 points, B has 3."""
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    axis_points = {'A': 1000. + np.array([-10., -5., 0., 5., 10.]), 'B': 1400. + np.array([-5., 0., 5.])}

    value = all_features_on_grid([f], states, axis_points)

    assert value.shape == (5, 3)
    for i, a in enumerate(axis_points['A']):
        for j, b in enumerate(axis_points['B']):
            assert value[i, j] == pytest.approx(eval_feature_on_grid(f, states, {'A': a, 'B': b}))  # type: ignore


def test_all_features_on_grid_adds_amplitudes_so_opposite_features_cancel(states):
    """
    Same place, same shape, coefficients +0.3 and -0.3: the amplitudes cancel everywhere.
    Adding intensities instead would give twice the intensity of one feature.
    """
    plus = feature_at_solved_location(MOTIF_AB, PS_01, states, coeffs=(0.3,))
    minus = feature_at_solved_location(MOTIF_AB, PS_01, states, coeffs=(-0.3,))
    axis_points = {'A': np.array([995., 1000.]), 'B': np.array([1400., 1405.])}

    np.testing.assert_allclose(all_features_on_grid([plus, minus], states, axis_points), 0.)


def test_all_features_on_grid_without_features_is_zero_on_the_grid(states):
    value = all_features_on_grid([], states, {'A': np.zeros(4), 'B': np.zeros(2)})

    assert value.shape == (4, 2)
    assert value.dtype == complex
    assert not value.any()


# -- boxed_features_on_grid ----------------------------------------------------------------
# MOTIF_A (one condition, A = E_a) peaks: a=0 (PS_00, PS_01) at A = 1000; a=1 (PS_10) at 1500.
# box_states puts the fundamentals at 950 and 1100 cm-1 instead: a weak peak 150 cm-1 from a strong one.
# Width G_CM = 5 cm-1. Half-widths given by hand, except in the amplitude_half_widths tests.


@pytest.fixture
def box_states() -> VibStatesData:
    """Two fundamentals, at 950 and 1100 cm-1: MOTIF_A peaks at A = 950 (a=0) and 1100 (a=1)."""
    return VibStatesData(allstates=(state('0', 950.), state('1', 1100.)), harmonic_osc_states_labels=(0, 1))

def test_boxed_features_on_grid_one_feature_is_all_features_on_grid_inside_its_box_and_zero_outside(states):
    """
    MOTIF_AB peak at A = 1000, B = 1400, half-width 55: box (945, 1055) x (1345, 1455).
    Grid A = 900 ... 1100 (201 points), B = 1200 ... 1600 (401 points): the box covers a different part of each axis.
    """
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    window = SpectralWindow.from_features(Box({'A': (900., 1100.), 'B': (1200., 1600.)}), [f], [55.])
    axis_points = window.box.grid_points({'A': 201, 'B': 401})

    value = boxed_features_on_grid(window, states, axis_points)

    inside = np.zeros(value.shape, dtype=bool)
    inside[box_slices(window.full_features[0].feat_box, axis_points)] = True  # type: ignore
    np.testing.assert_allclose(value[inside], all_features_on_grid([f], states, axis_points)[inside])
    assert not value[~inside].any()


def test_boxed_features_on_grid_overlapping_boxes_add_their_amplitudes(box_states):
    """
    A weak peak next to a strong one: PS_00 at A = 950 (coeff 0.03), PS_10 at A = 1100 (coeff 0.3).
    Half-widths 550: boxes (400, 1500) and (550, 1650) both cover the grid A = 900 ... 1150, so nothing is cut
    off: the result is all_features_on_grid's, the strong peak's tail under the weak peak included.
    """
    weak = feature_at_solved_location(MOTIF_A, PS_00, box_states, coeffs=(0.03,))
    strong = feature_at_solved_location(MOTIF_A, PS_10, box_states)
    window = SpectralWindow.from_features(Box({'A': (900., 1150.)}), [weak, strong], [550., 550.])
    axis_points = window.box.grid_points({'A': 251})

    value = boxed_features_on_grid(window, box_states, axis_points)

    np.testing.assert_allclose(value, all_features_on_grid([weak, strong], box_states, axis_points))


def test_boxed_features_on_grid_puts_each_feature_on_its_own_box_only(box_states):
    """
    The strong peak's box (550) covers the grid, the weak peak's box is (940, 960). The weak peak is evaluated on
    its own box only, not on the strong peak's box around it.
    """
    weak = feature_at_solved_location(MOTIF_A, PS_00, box_states, coeffs=(0.03,))
    strong = feature_at_solved_location(MOTIF_A, PS_10, box_states)
    window = SpectralWindow.from_features(Box({'A': (900., 1150.)}), [weak, strong], [10., 550.])
    axis_points = window.box.grid_points({'A': 251})        # A = 900, 901, ..., 1150

    value = boxed_features_on_grid(window, box_states, axis_points)

    weak_box = (axis_points['A'] >= 940.) & (axis_points['A'] <= 960.)
    expected = all_features_on_grid([strong], box_states, axis_points)
    expected[weak_box] += all_features_on_grid([weak], box_states, {'A': axis_points['A'][weak_box]})
    np.testing.assert_allclose(value, expected)


def test_boxed_features_on_grid_differs_from_all_features_on_grid_only_by_the_cut_off_tails(states):
    """
    Peaks at A = 1500 and 1000, half-widths 55: boxes (1445, 1555) and (945, 1055).
    boxed_features_on_grid leaves each feature out beyond its box, all_features_on_grid does not. A Lorentzian only
    gets smaller further out, so the difference is at most the sum of the two features' values at their box edges.
    """
    features = [feature_at_solved_location(MOTIF_A, PS_10, states), feature_at_solved_location(MOTIF_A, PS_01, states)]
    window = SpectralWindow.from_features(Box({'A': (900., 1600.)}), features, [55., 55.])
    axis_points = window.box.grid_points({'A': 701})

    difference = boxed_features_on_grid(window, states, axis_points) - all_features_on_grid(features, states, axis_points)

    at_box_edge = [abs(eval_feature_on_grid(f, states, {'A': f.location['A'] + get_box_extent(f)}))  # type: ignore
                   for f in window.full_features]
    assert 0. < np.abs(difference).max() <= sum(at_box_edge)


def on_grid_with_amplitude_boxes(features, window: Box, n_points: int, states, dynrange: float):
    """
    (the window as used, intensity of boxed_features_on_grid, intensity of all_features_on_grid), boxes from
    amplitude_half_widths
    """
    spectral_window = SpectralWindow.from_features(window, features,
                                                   amplitude_half_widths(window, features, dynrange=dynrange))
    axis_points = spectral_window.box.grid_points({'A': n_points})
    return (spectral_window, abs(boxed_features_on_grid(spectral_window, states, axis_points)) ** 2,
            abs(all_features_on_grid(features, states, axis_points)) ** 2)


def test_boxed_features_on_grid_with_amplitude_boxes_cuts_less_than_eps_floor_next_to_a_strong_peak(box_states):
    """
    Strong peak at A = 1100 (coeff 0.3), weak at A = 950 (coeff 0.003). dynrange 101, eps 0.1 (default):
    the strong box covers the grid, the weak one is 100.9 cm-1. Error against all_features_on_grid: at most 0.1 floors
    (floor: max / dynrange).
    """
    strong = feature_at_solved_location(MOTIF_A, PS_10, box_states)
    weak = feature_at_solved_location(MOTIF_A, PS_00, box_states, coeffs=(0.003,))

    window, intensity, reference = on_grid_with_amplitude_boxes([strong, weak], Box({'A': (850., 1200.)}), 351,
                                                             box_states, dynrange=101.)

    floor = reference.max() / 101.
    assert get_box_extent(window.full_features[1]) == pytest.approx(100.9, abs=0.1)
    assert 0. < np.abs(intensity - reference).max() < 0.1 * floor


def test_boxed_features_on_grid_with_amplitude_boxes_includes_a_strong_peak_outside_the_window(box_states):
    """
    Window A = 850 ... 1050 holds the weak peak (A = 950, coeff 0.03); the strong one (A = 1100, coeff 0.3) is
    outside. Its box reaches in, so it is on the grid: error below 0.1 floors.
    """
    strong = feature_at_solved_location(MOTIF_A, PS_10, box_states)
    weak = feature_at_solved_location(MOTIF_A, PS_00, box_states, coeffs=(0.03,))

    window, intensity, reference = on_grid_with_amplitude_boxes([strong, weak], Box({'A': (850., 1050.)}), 201,
                                                             box_states, dynrange=101.)

    assert window.contrib_features == [strong]
    assert np.abs(intensity - reference).max() < 0.1 * reference.max() / 101.


def test_boxed_features_on_grid_skips_a_feature_whose_box_is_outside_the_grid(states):
    """The grid covers only A = 1000 ... 1200 of the window; the peak at 1500 (box 1445 ... 1555) adds nothing."""
    f = feature_at_solved_location(MOTIF_A, PS_10, states)
    window = SpectralWindow.from_features(Box({'A': (1000., 1700.)}), [f], [55.])

    value = boxed_features_on_grid(window, states, {'A': np.linspace(1000., 1200., 201)})

    assert value.shape == (201,)
    assert not value.any()


def test_boxed_features_on_grid_needs_feature_boxes(states):
    window = SpectralWindow(Box({'A': (1000., 1700.)}), full_features=[feature_at_solved_location(MOTIF_A, PS_01, states)])

    with pytest.raises(ValueError, match='no feat_box'):
        boxed_features_on_grid(window, states, {'A': np.linspace(1000., 1700., 11)})


def test_boxed_features_on_grid_without_features_is_zero_on_the_grid(states):
    value = boxed_features_on_grid(SpectralWindow(Box({'A': (0., 10.)})), states, {'A': np.linspace(0., 10., 11)})

    assert value.shape == (11,)
    assert value.dtype == complex
    assert not value.any()


def test_boxed_features_on_grid_adds_a_flat_feature_on_every_grid_point(states):
    """
    The peak of test_boxed_features_on_grid_one_feature_..., plus a flat background of 0.3: its box
    is the window, so the grid is the peak's grid plus 0.3 everywhere, outside the peak's box too.
    """
    peak, box = feature_at_solved_location(MOTIF_AB, PS_01, states), Box({'A': (900., 1100.), 'B': (1200., 1600.)})
    alone = SpectralWindow.from_features(box, [peak], [55.])
    with_flat = SpectralWindow.from_features(box, [peak, flat_feature(0.3)], [55., 0.])
    axis_points = alone.box.grid_points({'A': 201, 'B': 401})

    value = boxed_features_on_grid(with_flat, states, axis_points)

    np.testing.assert_allclose(value, boxed_features_on_grid(alone, states, axis_points) + 0.3)
