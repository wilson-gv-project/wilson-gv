"""
evaluate.py — the numeric stage. Everything here is built by hand from tiny arrays;
no VibPerturbedTerm, no data files.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.evaluate as evaluate_mod
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    PrecalculatedData,
    _get_ind_tuple_from_base,
    _make_func_to_compute_avrg,
    build_contributions,
    calculate_avrg_tensor,
    draw_all,
    eval_avrg_per_indexdict,
    eval_feature_on_grid,
    eval_non_avrg_per_indexdict,
    eval_rescond_resloc,
    eval_resonance_factor,
    eval_vibenedenom,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    generate_LHS_motif,
    get_RHS_motif,
    harmonic_denom,
    lorentzian_propagator,
    otf_vibdiffdenom,
    solve_LSE_motif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    SpectralFeature,
)
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
    MOTIF_B_TWICE,
    MOTIF_MIXED,
    PS_00,
    PS_01,
    PS_10,
    PS_11,
    E0_eigval,
    E1_eigval,
    ab_term,
    polprop,
    toy_term,
    vibdiff,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    VibDiff,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

## Resonance location: shared motifs -------------------------------------------
# MOTIF_AB, MOTIF_A, MOTIF_MIXED and MOTIF_B_TWICE are explained in helpers.py.
# The fixtures `states`, `molsys` and `pre_a1_zero` come from conftest.py.


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


@pytest.mark.parametrize('unit', ['cm', 'cm^-1', 'eh', 'au'])
def test_get_RHS_motif_unknown_unit_raises(unit, params_obj, states):
    """Before, every text except 'Eh' silently meant cm-1."""
    with pytest.raises(ValueError, match="unit must be 'Eh' or 'cm-1'"):
        get_RHS_motif(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, unit=unit)


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


def test_solve_LSE_motif_empty_motif_raises(params, states):
    with pytest.raises(ValueError, match='no resonance conditions'):
        solve_LSE_motif(ResonanceMotif(()), params, states)


## Evaluation kernels -------------------------------------------------------

POLGRAD_AVRG = np.array([10., 20.])                    # already-averaged <polgrad>[a]


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


## Energy denominators, on the fly ---------------------------------------------
# otf_vibdiffdenom: 1/(E_left - E_right) for the perturbed-wavefunction differences, from the state energies.
# harmonic_denom:   1/omega for the plain denominators, from the harmonic eigenvalues (not the state energies).

def test_otf_vibdiffdenom_is_product_of_inverse_au_differences(molsys):
    freqterms = FreqTermsCollection([vibdiff(sl='ab', sr='a', pert=True), vibdiff(sl='b', sr='', pert=True)])

    value = otf_vibdiffdenom(freqterms, {'a': 0, 'b': 1}, molsys)

    assert value == pytest.approx(1. / convNu2Ene(E01 - E0) / convNu2Ene(E1))


def test_otf_vibdiffdenom_of_no_terms_is_one(molsys):
    assert otf_vibdiffdenom(FreqTermsCollection([]), {}, molsys) == 1.


def test_otf_vibdiffdenom_zero_difference_raises(molsys):
    """a = b = 0: both sides are state '0', so the difference is 0."""
    with pytest.raises(ZeroDivisionError):
        otf_vibdiffdenom(FreqTermsCollection([vibdiff(sl='a', sr='b', pert=True)]), {'a': 0, 'b': 0}, molsys)


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


def test_harmonic_denom_needs_eigenvals(molsys):
    with pytest.raises(ValueError, match='eigenvals'):
        harmonic_denom(FreqTermsCollection([vibdiff(sl='a')]), {'a': 0}, replace(molsys, eigenvals=None))


def test_eval_vibenedenom_reads_precomputed_tensor():
    tensor = np.array([[1., 2.], [3., 4.]])
    pre = PrecalculatedData(avrg_tensors={}, avrg_expr_tensor_mapping={},
                            vibenedenoms_tensors={('a', 'b'): tensor})
    freqterms = FreqTermsCollection([vibdiff(sl='b'), vibdiff(sl='a')])

    assert eval_vibenedenom(freqterms, {'a': 1, 'b': 0}, pre) == 3.


## Averaged properties, on the fly ---------------------------------------------
# EVV terms average three properties over four cartesian slots, e.g. term 8 of test_terms.json:
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


# -- _make_func_to_compute_avrg: one value for one index set --------------------------

def test_avrg_func_reads_the_cartesian_slot_of_each_operator():
    """One polarization entry, key (2, 0, 1, 1): polgrad[b] reads (2, 1), dipgrad[a] reads 0, dipgrad[c] reads 1."""
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={(2, 0, 1, 1): 3.})

    value = func({'a': 0, 'b': 1, 'c': 2}, evv_props())

    assert value == pytest.approx(3. * POLGRAD_3[1, 2, 1] * DIPGRAD_3[0, 0] * DIPGRAD_3[2, 1])


def test_avrg_func_is_the_full_contraction_with_the_averaging_coefficients():
    """sum over (i, j, k, l) of C[i, j, k, l] * polgrad[b, i, l] * dipgrad[a, j] * dipgrad[c, k], written with einsum."""
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

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
    average = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(),
                                         polarization_linear_comb=getGeneralPolarizationAveragingExpression(4, laser_pol))
    one_element = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={(0, 0, 0, 0): 1.})

    for idx in ALL_ABC:
        assert average(idx, rotated) == pytest.approx(average(idx, evv_props()))
    assert one_element(ALL_ABC[-1], rotated) != pytest.approx(one_element(ALL_ABC[-1], evv_props()))


def test_avrg_func_with_a_repeated_mode_label():
    """dipgrad[a] * dipgrad[a], rank 2: both factors read mode a."""
    expr = PropsCollection([polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='a')])
    func = _make_func_to_compute_avrg(avrg_expression=expr, polarization_linear_comb={(0, 0): 1., (1, 2): 3.})

    assert func({'a': 2}, evv_props()) == pytest.approx(DIPGRAD_3[2, 0] ** 2 + 3. * DIPGRAD_3[2, 1] * DIPGRAD_3[2, 2])


def test_avrg_func_without_polarization_entries_is_zero():
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb={})

    assert func({'a': 0, 'b': 1, 'c': 2}, evv_props()) == 0.


def test_avrg_func_needs_every_mode_label():
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    with pytest.raises(KeyError, match="'c'"):
        func({'a': 0, 'b': 1}, evv_props())


def test_avrg_func_needs_a_mol_props_collection():
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    with pytest.raises(TypeError, match='MolPropsCollection'):
        func({'a': 0, 'b': 1, 'c': 2}, {'polgrad': POLGRAD_3, 'dipgrad': DIPGRAD_3})  # type: ignore


def test_avrg_func_needs_every_property_in_the_data():
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)
    no_polgrad = MolPropsCollection([MolecularProperty(trivial_name='dipgrad', vals=DIPGRAD_3, extra_data={})])

    with pytest.raises(KeyError):
        func({'a': 0, 'b': 1, 'c': 2}, no_polgrad)


# -- eval_avrg_per_indexdict: precalculated tensor first, otherwise on the fly ---------

AVRG_A = PropsCollection([polprop(ops=(0, 1), inds='a')])


def test_eval_avrg_per_indexdict_on_the_fly_is_the_avrg_func_value(molsys):
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)
    idx = {'a': 0, 'b': 1, 'c': 2}

    value = eval_avrg_per_indexdict(evv_avrg_expr(), idx, avrg_func=func, molsys_data=replace(molsys, mol_props=evv_props()))

    assert value == func(idx, evv_props())


def test_eval_avrg_per_indexdict_prefers_precalculated_data(molsys):
    def must_not_run(*_):
        raise AssertionError('avrg_func was called although the tensor is precalculated')
    pre = PrecalculatedData(avrg_tensors={AVRG_A: np.array([10., 20.])}, avrg_expr_tensor_mapping={AVRG_A: AVRG_A})

    assert eval_avrg_per_indexdict(AVRG_A, {'a': 1}, avrg_func=must_not_run, molsys_data=molsys,
                                   precalculated_data=pre) == 20.


def test_eval_avrg_per_indexdict_falls_back_to_avrg_func_when_expr_is_not_precalculated(molsys):
    assert eval_avrg_per_indexdict(AVRG_A, {'a': 1}, avrg_func=lambda idx, props: 7., molsys_data=molsys,
                                   precalculated_data=PrecalculatedData()) == 7.


def test_eval_avrg_per_indexdict_needs_precalculated_data_or_avrg_func():
    with pytest.raises(ValueError, match='no avrg_func'):
        eval_avrg_per_indexdict(AVRG_A, {'a': 0})


def test_eval_avrg_per_indexdict_turns_values_below_zero_tol_into_zero(molsys):
    def value(v):
        return eval_avrg_per_indexdict(AVRG_A, {'a': 0}, avrg_func=lambda idx, props: v, molsys_data=molsys, zero_tol=1e-18)

    assert value(1e-20) == 0.
    assert value(-1e-20) == 0.
    assert value(1e-17) == 1e-17


# -- calculate_avrg_tensor: the on-the-fly value for every index set ----------------

def test_calculate_avrg_tensor_holds_the_on_the_fly_values_on_alphabetical_axes():
    """Axes follow the labels alphabetically (a, b, c), not their order in the expression (b, a, c)."""
    func = _make_func_to_compute_avrg(avrg_expression=evv_avrg_expr(), polarization_linear_comb=ISO4)

    tensor = calculate_avrg_tensor(evv_avrg_expr(), evv_props(), N_MODES, ISO4)

    assert tensor.shape == (N_MODES,) * 3
    for idx in ALL_ABC:
        assert tensor[idx['a'], idx['b'], idx['c']] == pytest.approx(func(idx, evv_props()))


def test_calculate_avrg_tensor_has_one_axis_per_distinct_label():
    """polgrad[b] * dipgrad[a] * dipgrad[b] (term 2): labels a, b -> a 2-axis tensor."""
    assert calculate_avrg_tensor(evv_avrg_expr('bab'), evv_props(), N_MODES, ISO4).shape == (N_MODES, N_MODES)


def test_calculate_avrg_tensor_fills_only_modes_to_fill():
    full = calculate_avrg_tensor(evv_avrg_expr(), evv_props(), N_MODES, ISO4)

    part = calculate_avrg_tensor(evv_avrg_expr(), evv_props(), N_MODES, ISO4, modes_to_fill=[0, 2])

    filled = np.ix_([0, 2], [0, 2], [0, 2])
    np.testing.assert_allclose(part[filled], full[filled])
    assert np.count_nonzero(part) == 8                      # 2**3 filled entries, mode 1 stays 0


def test_calculate_avrg_tensor_rejects_modes_beyond_the_mode_count():
    with pytest.raises(ValueError, match='exceeding number_of_nmodes'):
        calculate_avrg_tensor(evv_avrg_expr(), evv_props(), 2, ISO4, modes_to_fill=[0, 2])


# -- both paths together: a tensor from calculate_avrg_tensor, read back as precalculated data --

@pytest.mark.parametrize('labels', [
    'abc',                                    # labels in alphabetical order
    'bab',                                    # term 2: a repeated label
    pytest.param('bac', marks=pytest.mark.xfail(strict=True, reason=(
        'term 8: calculate_avrg_tensor stores alphabetical axes (a, b, c), but _get_ind_tuple_from_base '
        'reads distinct labels in expression order (b, a, c)'))),
])
def test_precalculated_tensor_gives_the_on_the_fly_value(labels, molsys):
    expr = evv_avrg_expr(labels)
    func = _make_func_to_compute_avrg(avrg_expression=expr, polarization_linear_comb=ISO4)
    pre = PrecalculatedData(avrg_tensors={expr: calculate_avrg_tensor(expr, evv_props(), N_MODES, ISO4)},
                            avrg_expr_tensor_mapping={expr: expr})
    otf_molsys = replace(molsys, mol_props=evv_props())

    for idx in ALL_ABC:
        assert (eval_avrg_per_indexdict(expr, idx, precalculated_data=pre)
                == pytest.approx(eval_avrg_per_indexdict(expr, idx, avrg_func=func, molsys_data=otf_molsys)))


# -- one EVV-shaped term, every factor on the fly -------------------------------------

def term8_like() -> CompiledTerm:
    """
    0.25 * cff[a,b,c] * <polgrad[b] dipgrad[a] dipgrad[c]> * 1/(E_ab - E_c) * 1/(omega_a omega_b omega_c)
    The same shape as term 8 of test_terms.json; resonance labels a, b; c is summed.
    """
    return CompiledTerm(
        avrg_props=evv_avrg_expr('bac'),
        non_avrg_props=PropsCollection([polprop(inds='abc')]),
        cmp_resmotf=ResonanceMotif(()),
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
    term = term8_like()
    func = _make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=ISO4)

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 0}, with_evv_props(molsys), avrg_func=func)

    # a=0, b=1, c=0: states ab = '0,1' and c = '0'; omegas from eigenvals
    avrg = np.einsum('ijkl,il,j,k->', dense(ISO4), POLGRAD_3[1], DIPGRAD_3[0], DIPGRAD_3[0])
    expected = {'NON_AVRG': CFF[0, 1, 0],
                'AVRG': avrg,
                'VIBDIFF_TERMS': 1. / convNu2Ene(E01 - E0),
                'VIBENE_DENOM': 1. / (convNu2Ene(E0_eigval) * convNu2Ene(E1_eigval) * convNu2Ene(E0_eigval))}
    assert contribs == pytest.approx(expected)
    assert value == pytest.approx(0.25 * np.prod(list(expected.values())))


def test_evaluate_term_coeff_sumover_on_the_fly_sums_c_over_both_modes(molsys):
    term = term8_like()
    otf_molsys = with_evv_props(molsys)
    func = _make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=ISO4)

    total, leaves = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 1}, otf_molsys, polarization_linear_comb=ISO4)

    by_hand = sum(evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': c}, otf_molsys, avrg_func=func)[0] for c in (0, 1))
    assert set(leaves) == {ParameterSet({'a': 0, 'b': 1, 'c': c}) for c in (0, 1)}
    assert total == pytest.approx(by_hand) != 0.


def test_evaluate_full_index_dict_on_the_fly_zero_average_stops_before_the_states(molsys):
    """dipgrad[a=1] = 0 makes the average 0: the state '1,1' (missing from the toy states) is never looked up."""
    dipgrad = DIPGRAD_3.copy()
    dipgrad[1] = 0.
    otf_molsys = replace(molsys, mol_props=MolPropsCollection([*molsys.mol_props, *evv_props(dipgrad=dipgrad)]))
    term = term8_like()
    func = _make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=ISO4)

    value, contribs = evaluate_full_index_dict(term, {'a': 1, 'b': 1, 'c': 0}, otf_molsys, avrg_func=func)

    assert (value, contribs) == (0., {'AVRG': 0.})


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

    Which labels get summed is decided by what is missing from the partial dict. The term's
    idx_nonsumm labels must all be fixed (see test_evaluate_term_coeffs_resonance_labels_must_be_fixed);
    here idx_nonsumm is just ('a',), so b and c may be left out.

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


@pytest.mark.parametrize('fixed, not_fixed', [
    ({'a': 0},  ['b']),
    ({'b': 0},  ['a']),
    ({},        ['a', 'b']),
])
def test_evaluate_term_coeffs_resonance_labels_must_be_fixed(fixed, not_fixed, molsys):
    """
    a and b are resonance labels: they decide where the peak sits. Summing over b would add
    peaks at different positions into one number, so the call raises instead.
    """
    term = CompiledTerm(PropsCollection([]), PropsCollection([]), ResonanceMotif(()), FreqTermsCollection([]), 1.,
                        idx_summ=('c',), idx_nonsumm=('a', 'b'))

    with pytest.raises(ValueError) as err:
        evaluate_term_coeff_sumover(term, fixed, molsys_data=molsys)
    assert str(err.value) == f'resonance labels {not_fixed} must be fixed, not summed'


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

    result = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 0}, molsys, precalculated_data=pre)

    from pprint import pprint
    pprint(result)

    total, leaves = result
    assert isinstance(total, float)
    assert set(leaves) == {ParameterSet({'a': 0, 'b': 0, 'c': c}) for c in (0, 1)}
    assert all(set(contribs) == {'NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM'} for contribs in leaves.values())


## build_contributions --------------------------------------------------------
# One row per (term, index set) with a nonzero coefficient. The toy term fixes a, b (its motif's
# labels) and sums over c. <polgrad>[1] = 0, so every pair with a=1 has coeff 0: 2 rows (a=0) and
# 2 zero pairs (a=1). a=b=1 needs state '1,1', which `states` lacks, so solving any a=1 pair would
# raise: these tests also show that zero pairs never reach solve_LSE_motif.

# -- what becomes a row: rows, zero pairs, failed pairs --------------------------

def test_build_contributions_one_row_per_nonzero_pair(molsys, pre_a1_zero):
    rows, zero, failed = build_contributions([ab_term()], molsys, precalculated_data=pre_a1_zero)

    assert {r.params for r in rows} == {PS_00, PS_01}
    assert all(r.term_id == 0 and r.motif == ResonanceMotif.from_tuples(MOTIF_AB) for r in rows)
    assert zero == [(0, PS_10), (0, PS_11)]
    assert failed == []
    assert len(rows) + len(zero) + len(failed) == 4          # every (a, b) pair exactly once


def test_build_contributions_no_single_point_goes_to_failed(molsys, pre_a1_zero):
    """MOTIF_B_TWICE asks w_B = -E_a and w_B = E_b - E_a at once: no point for any a=0 pair."""
    rows, zero, failed = build_contributions([ab_term(MOTIF_B_TWICE)], molsys,
                                             precalculated_data=pre_a1_zero)

    assert len(rows) == 0
    assert [(term_id, ps) for term_id, ps, _ in failed] == [(0, PS_00), (0, PS_01)]
    assert all(coeff != 0. for _, _, coeff in failed)        # failed pairs would have added to the spectrum
    assert zero == [(0, PS_10), (0, PS_11)]


def test_build_contributions_failed_pairs_are_listed_per_term(molsys, pre_a1_zero):
    """A failed solve is not cached: the next term with the same motif solves again and gets its own entries."""
    term = ab_term(MOTIF_B_TWICE)

    rows, _, failed = build_contributions([term, replace(term, frac_factor=2.)], molsys,
                                          precalculated_data=pre_a1_zero)

    assert len(rows) == 0
    assert [(term_id, ps) for term_id, ps, _ in failed] == [(0, PS_00), (0, PS_01), (1, PS_00), (1, PS_01)]
    assert failed[2][2] == pytest.approx(4 * failed[0][2])   # each entry keeps its own term's coeff


def test_build_contributions_lets_other_solve_errors_through(molsys, pre_a1_zero, monkeypatch):
    """Only LinAlgError (no single point) goes to `failed`. Any other error means a bug or missing data: the run stops."""
    def broken_solve(*args, **kwargs):
        raise ValueError('not a LinAlgError')
    monkeypatch.setattr(evaluate_mod, 'solve_LSE_motif', broken_solve)

    with pytest.raises(ValueError, match='not a LinAlgError'):
        build_contributions([ab_term()], molsys, precalculated_data=pre_a1_zero)


def test_build_contributions_without_terms_is_empty(molsys):
    rows, zero, failed = build_contributions([], molsys)

    assert (len(rows), zero, failed) == (0, [], [])


# -- the numbers in a row: coeff and location --------------------------------------

def test_build_contributions_coeff_is_evaluate_term_coeff_sumover(molsys, pre_a1_zero):
    term = ab_term()

    rows, _, _ = build_contributions([term], molsys, precalculated_data=pre_a1_zero)

    for r in rows:
        expected, _ = evaluate_term_coeff_sumover(term, {'a': r.params['a'], 'b': r.params['b']}, molsys,
                                                  precalculated_data=pre_a1_zero)
        assert r.coeff == expected != 0.


def test_build_contributions_polarization_path_matches_precalculated(molsys, pre_a1_zero):
    """
    No precalculated data: <polgrad>[a] = polgrad[a,0,0] + 2 * polgrad[a,1,1] from polarization_linear_comb.
    polgrad gives (10, 0), the same as pre_a1_zero; every other entry is 100, so a wrong slot changes the result.
    """
    polgrad = np.full((2, 3, 3), 100.)
    polgrad[:, 0, 0] = [4., 0.]
    polgrad[:, 1, 1] = [3., 0.]
    props = MolPropsCollection([*molsys.mol_props, MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={})])
    with_polgrad = replace(molsys, mol_props=props)

    precalc_rows, precalc_zero, _ = build_contributions([ab_term()], molsys, precalculated_data=pre_a1_zero)
    otf_rows, otf_zero, _ = build_contributions([ab_term()], with_polgrad,
                                                polarization_linear_comb={(0, 0): 1., (1, 1): 2.})

    assert [(r.params, r.location) for r in otf_rows] == [(r.params, r.location) for r in precalc_rows]
    assert [r.coeff for r in otf_rows] == pytest.approx([r.coeff for r in precalc_rows])
    assert otf_zero == precalc_zero


def test_build_contributions_locations_are_in_cm1(molsys, pre_a1_zero):
    rows, _, _ = build_contributions([ab_term()], molsys, precalculated_data=pre_a1_zero)

    location = {r.params: r.location for r in rows}
    assert location[PS_01].as_dict() == pytest.approx({'A': E01 - E0, 'B': E1 - E0})   # 1600, 500 cm-1
    assert location[PS_00].as_dict() == pytest.approx({'A': E00 - E0, 'B': 0.})


# -- the location cache: one solve per (motif, params) -------------------------------

def test_build_contributions_one_solve_per_motif_and_params(molsys, pre_a1_zero, monkeypatch):
    """Two terms with one motif: 2 solves, not 4, and the rows of both terms hold the same location object."""
    solved = []
    real_solve = evaluate_mod.solve_LSE_motif
    def counting_solve(motif, ps, *args, **kwargs):
        solved.append(ps)
        return real_solve(motif, ps, *args, **kwargs)
    monkeypatch.setattr(evaluate_mod, 'solve_LSE_motif', counting_solve)
    term = ab_term()

    rows, _, _ = build_contributions([term, replace(term, frac_factor=2.)], molsys,
                                     precalculated_data=pre_a1_zero)

    assert sorted(solved) == [PS_00, PS_01]
    row = {(r.term_id, r.params): r for r in rows}
    for ps in (PS_00, PS_01):
        assert row[(0, ps)].location is row[(1, ps)].location
        assert row[(1, ps)].coeff == pytest.approx(4 * row[(0, ps)].coeff)    # frac_factor 0.5 -> 2.


def test_build_contributions_cache_keeps_motifs_apart(molsys, pre_a1_zero):
    """Same index sets, different motifs: each motif gets its own location. A cache keyed by params alone fails this."""
    rows, _, _ = build_contributions([ab_term(MOTIF_AB), ab_term(MOTIF_A)], molsys, precalculated_data=pre_a1_zero)

    location = {(r.term_id, r.params): r.location for r in rows}
    assert location[(0, PS_01)].as_dict() == pytest.approx({'A': E01 - E0, 'B': E1 - E0})   # MOTIF_AB: axes A, B
    assert location[(1, PS_01)].as_dict() == pytest.approx({'A': E01 - E0})                 # MOTIF_A: axis A only


# -- input checks --------------------------------------------------------------------

def test_build_contributions_needs_eigenvals(molsys, pre_a1_zero):
    """The number of modes comes from molsys_data.eigenvals."""
    with pytest.raises(ValueError, match='eigenvals'):
        build_contributions([ab_term()], replace(molsys, eigenvals=None), precalculated_data=pre_a1_zero)


## Resonance factors on the grid -----------------------------------------------
# One condition is the denominator  E_left - E_right - sum_j s_j w_j - i*gamma  (derive's convention).
# eval_rescond_resloc and eval_resonance_factor take grids in Eh; eval_feature_on_grid takes cm-1.
# resonance_residuals (top of this file) writes  E_left - E_right - sum_j s_j w_j  out independently.
#
# The lorentzian_propagator tests, the get_intensity test and the hand-worked values (eval_resonance_factor,
# eval_feature_on_grid) use the Lorentzian. The other tests pass a RecordingPropagator, so they hold for any propagator.

G_CM = 5.                   # lineshape parameter, cm-1
G_AU = convNu2Ene(G_CM)
CM1_PER_EH = 219474.6313632  # 1 Eh in cm-1 (CODATA), written out so the hand-worked values don't use convNu2Ene


def grid_around(location: dict[str, float], step: float) -> dict[str, np.ndarray]:
    """3 points per axis (v - step, v, v + step), ij meshgrid: index (1, 1, ...) is `location`."""
    axes = sorted(location)
    mesh = np.meshgrid(*(location[ax] + np.array([-step, 0., step]) for ax in axes), indexing='ij')
    return dict(zip(axes, mesh))


def to_au(grid: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {ax: convNu2Ene(v) for ax, v in grid.items()}


def feature_at_solved_location(motif, params: ParameterSet, states, coeffs=(0.3,)) -> SpectralFeature:
    """One feature, one row per coeff, at the location solve_LSE_motif finds (cm-1); width G_CM."""
    res_motif = ResonanceMotif.from_tuples(motif)
    location = solve_LSE_motif(res_motif, params, states, unit='cm-1')
    rows = tuple(ContributionRow(term_id=i, motif=res_motif, params=params, location=location, coeff=c)
                 for i, c in enumerate(coeffs))
    return SpectralFeature(location=location, rows=rows, lineshape_parameter=G_CM)


class RecordingPropagator:
    """Stands in for any propagator: stores each (resloc, gamma) it gets, returns resloc + i*gamma (not a Lorentzian)."""
    def __init__(self):
        self.calls = []

    def __call__(self, resloc, gamma):
        self.calls.append((resloc, gamma))
        return resloc + 1j * gamma


# -- lorentzian_propagator ------------------------------------------------------

def test_lorentzian_propagator_at_resonance_is_i_over_gamma():
    assert lorentzian_propagator(0., 0.5) == pytest.approx(1. / (-1j * 0.5))


def test_lorentzian_propagator_imaginary_part():
    """-i*gamma: Im = gamma / (x^2 + gamma^2) > 0, Re = x / (x^2 + gamma^2)"""
    x, gamma = np.linspace(-3., 3., 7), 0.5

    value = lorentzian_propagator(x, gamma)

    np.testing.assert_allclose(value.imag, gamma / (x**2 + gamma**2))  # type: ignore
    np.testing.assert_allclose(value.real, x / (x**2 + gamma**2))      # type: ignore


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
def test_eval_resonance_factor_calls_the_propagator_once_per_condition_with_E_minus_freq(motif, params_obj, states):
    """
    e.g. MOTIF_AB, a=0, b=1 (in cm-1): two calls, with resloc 1600 - A and 500 - B.
    Checked on a 3x3 grid in Eh against resonance_residuals.
    """
    res_motif = ResonanceMotif.from_tuples(motif)
    grid = to_au(grid_around({'A': 300., 'B': -700.}, 100.))
    propagator = RecordingPropagator()

    eval_resonance_factor(res_motif, params_obj, states, grid, propagator, lambda _vd: G_AU)

    residuals = resonance_residuals(res_motif, grid, params_obj.to_dict(), states)  # type: ignore
    assert len(propagator.calls) == len(res_motif)
    for (resloc, _), expected in zip(propagator.calls, residuals):
        np.testing.assert_allclose(resloc, expected)


def test_eval_resonance_factor_gives_each_condition_its_own_gamma(params_obj, states):
    """
    gamma(vd) is asked once per condition, so a width per pair of states is possible.
    Here gamma returns the condition's own E (cm-1). At freq = 0, resloc is E in Eh,
    so each call must get resloc == convNu2Ene(gamma).
    """
    propagator = RecordingPropagator()

    eval_resonance_factor(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, {'A': 0., 'B': 0.},  # type: ignore
                          propagator, propagator_param=lambda vd: vd.energy_difference())

    assert sorted(g for _, g in propagator.calls) == pytest.approx(sorted([E01 - E0, E1 - E0]))
    for resloc, g in propagator.calls:
        assert resloc == pytest.approx(convNu2Ene(g))


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_resonance_factor_multiplies_the_propagator_values(motif, params_obj, states):
    """e.g. MOTIF_AB: propagator(resloc_1, gamma) * propagator(resloc_2, gamma), at every grid point."""
    res_motif = ResonanceMotif.from_tuples(motif)
    grid = to_au(grid_around({'A': 300., 'B': -700.}, 100.))
    propagator = RecordingPropagator()

    value = eval_resonance_factor(res_motif, params_obj, states, grid, propagator, lambda _vd: G_AU)

    np.testing.assert_allclose(value, np.prod([r + 1j * g for r, g in propagator.calls], axis=0))


def test_eval_resonance_factor_lorentzian_values_with_a_width_per_condition(params_obj, states):
    """
    eval_feature_on_grid always gives every condition the same width; here they differ.
    MOTIF_AB, a=0, b=1, width = E / 100: 16 cm-1 for the A condition (E = 1600), 5 cm-1 for B (E = 500).
        1 / ((1600 - A - 16i) * (500 - B - 5i))   in cm-1, times CM1_PER_EH**2
    Peak: (i/16) * (i/5) = -1/80. One width off the peak on either axis gives the same value,
        (1584, 500):  (1 + i)/32 * i/5  =  (-1 + i) / 160
        (1600, 495):  i/16 * (1 + i)/10 =  (-1 + i) / 160
    which only holds if each condition gets its own width (swapped widths give different values).
    """
    grid = to_au({'A': np.array([1600., 1584., 1600.]), 'B': np.array([500., 500., 495.])})

    value = eval_resonance_factor(ResonanceMotif.from_tuples(MOTIF_AB), params_obj, states, grid,
                                  lorentzian_propagator, lambda vd: vd.energy_difference(au=True) / 100)

    expected = [-0.0125, -0.00625 + 0.00625j, -0.00625 + 0.00625j]
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


# -- eval_feature_on_grid -------------------------------------------------------

@pytest.mark.parametrize('ps', [PS_01, PS_10])
@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_feature_on_grid_gives_the_propagator_resloc_and_gamma_in_eh(motif, ps, states):
    """
    Grid and lineshape_parameter go in as cm-1. The propagator must get E_i - freq_i and gamma in Eh.
    The states come from the feature's own params: a=0, b=1 and a=1, b=0 resonate at different places.
    """
    f = feature_at_solved_location(motif, ps, states)
    grid_cm = grid_around(f.location.as_dict(), G_CM)
    propagator = RecordingPropagator()

    eval_feature_on_grid(f, states, grid_cm, propagator)

    residuals = resonance_residuals(ResonanceMotif.from_tuples(motif), to_au(grid_cm), ps.to_dict(), states)  # type: ignore
    assert len(propagator.calls) == len(motif)
    for (resloc, gamma), expected in zip(propagator.calls, residuals):
        np.testing.assert_allclose(resloc, expected, atol=1e-15)   # atol: the centre point is 0
        assert gamma == pytest.approx(G_AU)


@pytest.mark.parametrize('motif', [MOTIF_AB, MOTIF_A, MOTIF_MIXED])
def test_eval_feature_on_grid_peak_matches_get_intensity(motif, params_obj, states):
    """
    Lorentzian only. get_intensity (features.py) assumes the peak is |amplitude / (-i*gamma)^N|^2
    and sizes the feature boxes with it. The default propagator must give the same peak.
    """
    f = feature_at_solved_location(motif, params_obj, states)

    value = eval_feature_on_grid(f, states, f.location.as_dict())  # type: ignore

    assert abs(value) ** 2 == pytest.approx(f.get_intensity())


def test_eval_feature_on_grid_uses_the_summed_and_scaled_amplitude(params_obj, states):
    """Rows add up before the resonance factor, which is applied once per feature (not once per row,
    see C3 in todo.txt). scale (normalize_coeffs_to_max) multiplies too."""
    one = feature_at_solved_location(MOTIF_AB, params_obj, states, coeffs=(1.,))
    two = feature_at_solved_location(MOTIF_AB, params_obj, states, coeffs=(0.3, 0.5))
    grid = grid_around(one.location.as_dict(), G_CM)
    base = eval_feature_on_grid(one, states, grid)

    np.testing.assert_allclose(eval_feature_on_grid(two, states, grid), 0.8 * base)
    np.testing.assert_allclose(eval_feature_on_grid(replace(two, scale=0.5), states, grid), 0.4 * base)


# Hand-worked values, default Lorentzian, coeff 0.3, gamma 5 cm-1. One condition is
#     0.3 / (E - freq - i*gamma)  in Eh  =  CM1_PER_EH * 0.3 / (E - freq - i*gamma)  in cm-1,
# so a motif with N conditions picks up CM1_PER_EH**N. The numbers below are the cm-1 part.
# Grid centres are written out (not taken from solve_LSE_motif).

@pytest.mark.parametrize('ps, centre', [(PS_01, 1600.),     # E_ab - E_a = E01 - E0
                                        (PS_10, 1100.)])    # E01 - E1
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
    MOTIF_AB, a=0, b=1: 0.3 / ((1600 - A - 5i) * (500 - B - 5i)), one factor per axis.
    3x3 grid, A = 1600 +- 5 (rows), B = 500 +- 5 (columns). Per axis, 1 / (x - 5i) is
        x = +5: 0.1 + 0.1i      x = 0: 0.2i      x = -5: -0.1 + 0.1i
    and each entry is 0.3 times the product of the two, e.g. the peak 0.3 * (0.2i)^2 = -0.012.
    """
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    grid = grid_around({'A': 1600., 'B': 500.}, 5.)

    value = eval_feature_on_grid(f, states, grid)

    expected = [[0.006j,          -0.006 + 0.006j, -0.006],
                [-0.006 + 0.006j, -0.012,          -0.006 - 0.006j],
                [-0.006,          -0.006 - 0.006j, -0.006j]]
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
    grid = grid_around({'A': 1600., 'B': 0.}, 5.)

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
])
def test_eval_feature_on_grid_peak_is_coeff_times_i_over_gamma_to_the_N(motif, gamma, expected_cm1, params_obj, states):
    """
    At resonance every condition gives 1 / (-i*gamma) = i / gamma, so the peak is 0.3 * (i / gamma)^N:
    on the imaginary axis for one condition, real and negative for two. Wider line -> lower peak.
    get_intensity only checks |peak|^2; this also checks the phase (the -i*gamma sign).
    """
    f = replace(feature_at_solved_location(motif, params_obj, states), lineshape_parameter=gamma)

    value = eval_feature_on_grid(f, states, f.location.as_dict())  # type: ignore

    assert value == pytest.approx(CM1_PER_EH ** len(motif) * expected_cm1)


def test_eval_feature_on_grid_propagator_param_replaces_lineshape_parameter(states):
    """
    Same points and widths as the eval_resonance_factor test above (16 cm-1 on A, 5 cm-1 on B), but the
    width function is in cm-1 and the feature's own lineshape_parameter (5 cm-1) is not used.
    Values are 0.3 times that test's: -0.0125 -> -0.00375, (-1 + i)/160 -> -0.001875 + 0.001875i.
    """
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    grid_cm = {'A': np.array([1600., 1584., 1600.]), 'B': np.array([500., 500., 495.])}

    value = eval_feature_on_grid(f, states, grid_cm, propagator_param=lambda vd: vd.energy_difference() / 100)

    expected = [-0.00375, -0.001875 + 0.001875j, -0.001875 + 0.001875j]
    np.testing.assert_allclose(value, CM1_PER_EH**2 * np.array(expected))


def test_eval_feature_on_grid_propagator_param_does_not_need_lineshape_parameter(states):
    """lineshape_parameter=None is fine when propagator_param is given. MOTIF_A peak, 5 cm-1: 0.3 / (-5i) = 0.06i."""
    f = replace(feature_at_solved_location(MOTIF_A, PS_01, states), lineshape_parameter=None)

    value = eval_feature_on_grid(f, states, {'A': np.array([1600.])}, propagator_param=lambda _vd: 5.)

    np.testing.assert_allclose(value, CM1_PER_EH * np.array([0.06j]))


def test_eval_feature_on_grid_propagator_param_can_depend_on_the_grid(states):
    """
    propagator_param only gets the condition, but it can read the grid itself and return one value per
    grid point. Here width = A - 1595 cm-1: 5 at A = 1600, 10 at A = 1605. MOTIF_A, peak at 1600:
        A = 1600:  0.3 / (-5i)       = 0.06i
        A = 1605:  0.3 / (-5 - 10i)  = 0.3 * (-5 + 10i) / 125 = -0.012 + 0.024i
    """
    f = feature_at_solved_location(MOTIF_A, PS_01, states)
    grid_cm = {'A': np.array([1600., 1605.])}

    value = eval_feature_on_grid(f, states, grid_cm, propagator_param=lambda _vd: grid_cm['A'] - 1595.)

    np.testing.assert_allclose(value, CM1_PER_EH * np.array([0.06j, -0.012 + 0.024j]))


def test_eval_feature_on_grid_needs_a_lineshape_parameter_or_a_propagator_param(params_obj, states):
    f = replace(feature_at_solved_location(MOTIF_A, params_obj, states), lineshape_parameter=None)

    with pytest.raises(ValueError, match='lineshape_parameter'):
        eval_feature_on_grid(f, states, {'A': np.zeros(3)})


def test_eval_feature_on_grid_needs_rows(states):
    f = SpectralFeature(location=ResLocPoint({'A': 0.}), lineshape_parameter=G_CM)

    with pytest.raises(ValueError, match='no motif'):
        eval_feature_on_grid(f, states, {'A': np.zeros(3)})


# -- draw_all -------------------------------------------------------------------
# Peaks with a=0, b=1 (PS_01): MOTIF_AB at A = 1600, B = 500.  With a=1, b=0 (PS_10): MOTIF_A at A = 1100,
# a ridge along B.

def test_draw_all_is_the_sum_of_the_features_on_the_whole_grid(states):
    peak = feature_at_solved_location(MOTIF_AB, PS_01, states)
    ridge = feature_at_solved_location(MOTIF_A, PS_10, states)
    coords = {'A': np.array([1100., 1600.]), 'B': np.array([495., 500., 505.])}
    mesh = dict(zip(coords, np.meshgrid(*coords.values(), indexing='ij')))

    value = draw_all([peak, ridge], states, coords)

    expected = eval_feature_on_grid(peak, states, mesh) + eval_feature_on_grid(ridge, states, mesh)
    np.testing.assert_allclose(value, expected)


def test_draw_all_index_i_j_is_the_point_A_i_B_j(states):
    """ij indexing: value[i, j] belongs to A = coords['A'][i], B = coords['B'][j]; A has 5 points, B has 3."""
    f = feature_at_solved_location(MOTIF_AB, PS_01, states)
    coords = {'A': 1600. + np.array([-10., -5., 0., 5., 10.]), 'B': 500. + np.array([-5., 0., 5.])}

    value = draw_all([f], states, coords)

    assert value.shape == (5, 3)
    for i, a in enumerate(coords['A']):
        for j, b in enumerate(coords['B']):
            assert value[i, j] == pytest.approx(eval_feature_on_grid(f, states, {'A': a, 'B': b}))  # type: ignore


def test_draw_all_adds_amplitudes_so_opposite_features_cancel(states):
    """
    Same place, same shape, coefficients +0.3 and -0.3: the amplitudes cancel everywhere.
    Adding intensities instead would give twice the intensity of one feature.
    """
    plus = feature_at_solved_location(MOTIF_AB, PS_01, states, coeffs=(0.3,))
    minus = feature_at_solved_location(MOTIF_AB, PS_01, states, coeffs=(-0.3,))
    coords = {'A': np.array([1595., 1600.]), 'B': np.array([500., 505.])}

    np.testing.assert_allclose(draw_all([plus, minus], states, coords), 0.)


def test_draw_all_without_features_is_zero_on_the_grid(states):
    value = draw_all([], states, {'A': np.zeros(4), 'B': np.zeros(2)})

    assert value.shape == (4, 2)
    assert value.dtype == complex
    assert not value.any()
