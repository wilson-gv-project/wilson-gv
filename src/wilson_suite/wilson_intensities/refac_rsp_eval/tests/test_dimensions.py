"""
The evaluator with 1, 2, 3 and 4 axes. The other real-data tests are 2D only (EVV).

Mock data, 1D-4D. One toy term per dimensionality, its resonance conditions a chain (A ; A+B ; A+B+C ; A+B+C+D),
as derive writes them for pulses at different times, on a toy molecule with 2 modes and states up to
4 quanta (more than the Gaussian files have). The term is compiled by hand, so 4D needs no translation to axes
(derive's translation stops at 3 axes). Rank 2 averaging (TOY_RECIPE): odd ranks would need the slot order of
get_pol_tensor fixed first (test_averaging_pol_order.py).

Real data: derive's terms of three experiments, stored translated to the experiment's canonical axes
(stored_terms.py), data from formaldehyde's Gaussian file:
    1D  CARS: polgrad only
    1D  IR absorption: 3 of its 5 terms have no resonance conditions, a flat background
    3D  3 IR: two sets of resonance axes, {A, B} (14 terms) and {A, B, C} (2 terms), so one evaluation per set
"""

import itertools
from functools import partial

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import all_features_on_grid, eval_feature_on_grid
from wilson_suite.wilson_intensities.refac_rsp_eval.features import amplitude_half_widths
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box, box_slices
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    compute_features,
    evaluate_response_terms,
    prop_names,
    spectrum_on_grid,
    terms_by_axis_set,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ResonanceMotif,
    generate_LHS_motif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import make_evv_reference as ref
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    TOY_RECIPE,
    toy_term,
    vibdiff,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import StoredSetup, load_setup
from wilson_suite.wilson_system.system_data import MolecularProperty, MolPropsCollection, MolSystemData
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

G_CM = 20.          # peak width, cm-1


## mock data: 2 modes, states up to 4 quanta -----------------------------------

FUNDAMENTALS = {0: 1000., 1: 1500.}     # cm-1
EIGVALS = {0: 1100., 1: 1580.}          # nc_sqrt_eigval, cm-1


def mock_energy(modes) -> float:
    """the fundamentals added up, 10 cm-1 lower per extra quantum, e.g. (0, 1) -> 2490, (0, 0, 1) -> 3480"""
    return sum(FUNDAMENTALS[int(m)] for m in modes) - 10. * (len(modes) - 1)


def mock_data() -> dict:
    """
    A data dict the load checks take (check_data_dict): harmonic and anharmonic states up to 4 quanta, polgrad,
    a symmetric cff. Both modes are totally symmetric, so no entry is 0 by symmetry and no index set has
    coefficient 0: polgrad is diagonal, and with TOY_RECIPE <polgrad>[a] = trace / 3 = (0.2, 0.15).
    """
    all_modes = [modes for n in range(1, 5) for modes in itertools.combinations_with_replacement((0, 1), n)]
    harmonic = {tuple(str(m) for m in modes): sum(EIGVALS[m] for m in modes) for modes in all_modes}
    anharmonic = {tuple(str(m) for m in modes): mock_energy(modes) for modes in all_modes}
    polgrad = np.array([np.diag([0.1, 0.3, 0.2]), np.diag([0.05, 0.25, 0.15])])
    base = np.arange(8, dtype=float).reshape(2, 2, 2) + 1.
    cff = sum(np.transpose(base, p) for p in itertools.permutations(range(3))) / 6   # symmetric in the mode indices
    return {'nc_sqrt_eigval': EIGVALS, 'harmonic_states': harmonic, 'anharmonic_states': anharmonic,
            'polgrad': polgrad, 'cff': cff}


@pytest.fixture(scope='module')
def mock_molsys() -> MolSystemData:
    props = MolPropsCollection([MolecularProperty(trivial_name='polgrad'), MolecularProperty(trivial_name='cff')])
    return MolSystemData.from_datadict(props, mock_data())


## the chain term, 1D-4D --------------------------------------------------------

LABELS = 'abcd'


def chain_motif(ndim: int) -> ResonanceMotif:
    """condition k: E of the first k labels = the sum of the first k axes, e.g. ndim 2: E_a = A, E_ab = A + B"""
    return ResonanceMotif.from_tuples(tuple(((tuple(LABELS[:k]), ()), tuple(AXES[:k])) for k in range(1, ndim + 1)))


def chain_term(ndim: int) -> CompiledTerm:
    """
    toy_term (0.5 * cff[a,b,c] * <polgrad>[a] / omega_a / (E_ab - E_a)) with the chain motif of ndim conditions.
    The motif's labels are fixed by the index sets, the others summed: ndim 1 sums b and c, ndim 3 sums none.
    ndim 4 adds 1/omega_d, so d is in the term too.
    """
    term = toy_term()
    labels = LABELS if ndim == 4 else LABELS[:3]
    freqdenom = FreqTermsCollection([*term.cmp_freqdenom, vibdiff(sl='d')]) if ndim == 4 else term.cmp_freqdenom
    return CompiledTerm(avrg_props=term.avrg_props, non_avrg_props=term.non_avrg_props,
                        cmp_resmotf=chain_motif(ndim), cmp_freqdenom=freqdenom, frac_factor=term.frac_factor,
                        idx_summ=tuple(labels[ndim:]), idx_nonsumm=tuple(labels[:ndim]))


def chain_location(params: dict[str, int]) -> dict[str, float]:
    """where the chain term of these index values resonates, from mock_energy: A = E_a, B = E_ab - E_a, ..."""
    energies = [mock_energy([params[label] for label in LABELS[:k]]) for k in range(1, len(params) + 1)]
    return {ax: e - before for ax, e, before in zip(AXES, energies, [0., *energies])}


@pytest.fixture(scope='module', params=NDIMS, ids=[f'{n}D' for n in NDIMS])
def chain_result(request, mock_molsys) -> FeatureResult:
    return compute_features([chain_term(request.param)], mock_molsys, TOY_RECIPE, G_CM, states_choice='anharmonic')


@pytest.mark.parametrize('ndim', NDIMS)
def test_chain_motif_fixes_one_point_on_its_ndim_axes(ndim):
    """As many conditions as axes, all independent (rank), the check from_VibPertTerm runs on derive's terms."""
    lhs, axes = generate_LHS_motif(chain_motif(ndim))

    assert axes == tuple(AXES[:ndim])
    assert len(chain_motif(ndim)) == np.linalg.matrix_rank(lhs) == ndim


def test_compute_features_chain_term_one_row_and_one_feature_per_index_set(chain_result):
    """2 modes, ndim fixed labels: 2**ndim index sets, none with coefficient 0, each at its own point."""
    ndim = len(chain_result.compiled[0].idx_nonsumm)  # type: ignore

    assert len(chain_result.table) == len(chain_result.features) == 2 ** ndim
    assert chain_result.zero == []
    for r in chain_result.table:
        assert r.location.axes == tuple(AXES[:ndim])
        assert r.location.as_dict() == pytest.approx(chain_location(r.params.to_dict()))


def test_chain_feature_peak_is_amplitude_times_i_over_gamma_to_the_ndim(chain_result):
    """
    At its location every condition gives 1 / (-i*gamma) = i / gamma (gamma in Eh), so the peak is
    amplitude * (i / gamma)**ndim. get_intensity, which sizes the boxes, assumes the same.
    """
    for f in chain_result.features:
        value = eval_feature_on_grid(f, chain_result.states, f.location.as_dict())  # type: ignore

        assert value == pytest.approx(f.amplitude_coeff * (1j / convNu2Ene(G_CM)) ** f.location.dimensionality)
        assert abs(value) ** 2 == pytest.approx(f.get_intensity())


POINTS = {1: 2001, 2: 101, 3: 31, 4: 13}     # grid points per axis


def test_spectrum_on_grid_chain_term_puts_each_feature_on_its_own_box(chain_result):
    """
    Boxes of 3 gamma (60 cm-1) with cut_tails=True cut every feature: the spectrum is each feature's
    all_features_on_grid value inside its own box, 0 outside, added up, in every dimensionality. Window: the
    features +- 100 cm-1 on every axis.
    (amplitude_half_widths would give these features the whole window: they are of similar strength.)
    """
    locations = np.array([f.location.values for f in chain_result.features])
    ndim = locations.shape[1]
    window = Box({ax: (lo - 100., hi + 100.) for ax, lo, hi in zip(AXES, locations.min(0), locations.max(0))})

    spec = spectrum_on_grid(chain_result, window, {ax: POINTS[ndim] for ax in AXES[:ndim]},
                         box_rule=lambda window, features: [3 * G_CM] * len(features), cut_tails=True)

    expected = np.zeros((POINTS[ndim],) * ndim, dtype=complex)
    for f in [*spec.spectral_window.full_features, *spec.spectral_window.contrib_features]:
        inside = box_slices(f.feat_box, spec.axis_points)  # type: ignore
        expected[inside] += all_features_on_grid([f], chain_result.states, spec.axis_points)[inside]
    assert spec.amplitude.shape == expected.shape
    np.testing.assert_allclose(spec.amplitude, expected)
    every_point = all_features_on_grid(chain_result.features, chain_result.states, spec.axis_points)
    assert np.abs(spec.amplitude - every_point).max() > 0


## real data: stored setups, every pulse and the detector along x ------------------------------------------------
# Terms and settings from data_for_tests/translated_terms (stored_terms.py): derive's terms, translated to the
# experiment's canonical axes; laser_pol is the experiment's polarization_avg_vector.

def evaluate_on_formaldehyde(setup: StoredSetup, terms: list) -> FeatureResult:
    molsys = MolSystemData.load(ref.data_origin('formaldehyde'), prop_names(terms))
    return evaluate_response_terms(terms, molsys_data=molsys, states_choice='anharmonic',
                                   laser_pol=setup.laser_pol, propagator_param=10.)


## 1D, real data: CARS on formaldehyde -------------------------------------------
# The setup 'try: CARS': three UV pulses, the first two at the same time, filter +1 -2 +3. One axis, A = w1 - w2.

@pytest.fixture(scope='module')
def cars() -> tuple[StoredSetup, list, FeatureResult]:
    setup = load_setup('try: CARS')
    return setup, setup.terms, evaluate_on_formaldehyde(setup, setup.terms)


def test_cars_derive_gives_4_terms_on_A_with_polgrad_only(cars):
    setup, terms, _ = cars

    assert setup.axes == {'A': [(-2, 1)]}                                                  # A = w1 - w2
    assert setup.laser_pol == (1., 1., 1.)                                                 # rank 4
    assert len(terms) == 4
    assert prop_names(terms) == ['polgrad']
    assert terms_by_axis_set(terms) == {('A',): [0, 1, 2, 3]}


def test_cars_peaks_are_the_anharmonic_fundamentals_with_both_signs(cars):
    """
    The conditions E_a = A and -E_a = A: anti-Stokes and Stokes. 6 modes, 4 terms: 24 rows, 2 per feature,
    12 features at +- each anharmonic fundamental.
    """
    _, _, result = cars
    fundamentals = [result.states.get_state_by_label(str(m)).energy for m in range(6)]

    assert len(result.table) == 24
    assert result.zero == []
    assert all(len(f.rows) == 2 for f in result.features)
    assert sorted(f.location['A'] for f in result.features) == pytest.approx(
        sorted([*fundamentals, *(-e for e in fundamentals)]))


def test_cars_on_grid_within_eps_floor_of_all_features_on_grid(cars):
    """
    1D, the features +- 200 cm-1, 2001 points, cut_tails=True: at most 0.1 floors from all_features_on_grid.
    """
    _, _, result = cars
    locations = [f.location['A'] for f in result.features]
    window = Box({'A': (min(locations) - 200., max(locations) + 200.)})

    spec = spectrum_on_grid(result, window, {'A': 2001}, box_rule=partial(amplitude_half_widths, dynrange=500.),
                            cut_tails=True)

    reference = abs(all_features_on_grid(result.features, result.states, spec.axis_points)) ** 2
    assert 0. < np.abs(abs(spec.amplitude) ** 2 - reference).max() < 0.1 * reference.max() / 500.


## 1D, real data: IR absorption on formaldehyde, with a flat background ----------------------
# The setup '1D: IR absorption': one IR pulse, filter +1. One axis, A = w1; rank 2. derive gives 5 terms: 3 without
# resonance conditions (polhess, and polgrad * cff twice), a flat background, and 2 on A (dipgrad * dipgrad, the
# conditions E_a = A and -E_a = A). All 5 go into one call: the background interferes with the peaks.

CM_TO_EH = 4.556335e-6      # 1 cm-1 in Eh


@pytest.fixture(scope='module')
def ir_absorption() -> tuple[StoredSetup, list, FeatureResult]:
    setup = load_setup('1D: IR absorption')
    return setup, setup.terms, evaluate_on_formaldehyde(setup, setup.terms)


def test_ir_absorption_derive_gives_3_terms_without_conditions_and_2_on_A(ir_absorption):
    setup, terms, _ = ir_absorption

    assert setup.axes == {'A': [(1,)]}                                                     # A = w1
    assert setup.laser_pol == (1.,)                                                        # rank 2
    assert prop_names(terms) == ['polhess', 'polgrad', 'cff', 'dipgrad']
    assert terms_by_axis_set(terms) == {(): [0, 1, 2], ('A',): [3, 4]}


def test_ir_absorption_flat_terms_make_one_feature_next_to_12_peaks(ir_absorption):
    """
    Each term without conditions: one row (no label to fix) at the location without axes; the 3 rows add up in
    one flat feature. The 2 terms on A: 12 rows, one feature each, at +- each anharmonic fundamental.
    """
    _, _, result = ir_absorption
    fundamentals = [result.states.get_state_by_label(str(m)).energy for m in range(6)]
    flat_rows = [r for r in result.table if not r.location.axes]
    [flat] = [f for f in result.features if f.is_flat]
    peaks = [f for f in result.features if not f.is_flat]

    assert (len(result.table), result.zero) == (15, [])
    assert [(r.term_id, r.params.to_dict()) for r in flat_rows] == [(0, {}), (1, {}), (2, {})]
    assert flat.amplitude_coeff == pytest.approx(sum(r.coeff for r in flat_rows))
    assert sorted(f.location['A'] for f in peaks) == pytest.approx(
        sorted([*fundamentals, *(-e for e in fundamentals)]))


def test_ir_absorption_polhess_term_by_hand(ir_absorption):
    """
    Term 0, 1/4 * polhess_(a,a) / <a>, numpy only: 1/4 * sum over a of <polhess>[a, a] / w_a. Rank 2, both fields
    along x: <T_xx> = trace(T) / 3. w_a: the harmonic frequency (nc_sqrt_eigval, see Decisions), in Eh.
    """
    _, _, result = ir_absorption
    polhess, w_cm = np.asarray(result.molsys.mol_props['polhess'].vals), result.molsys.eigenvals
    [row] = [r for r in result.table if r.term_id == 0]

    hand = 0.25 * sum(np.trace(polhess[a, a]) / 3. / (w_cm[a] * CM_TO_EH) for a in range(6))

    assert row.coeff == pytest.approx(hand, rel=1e-6)


def test_ir_absorption_cff_terms_are_one_sum_twice(ir_absorption):
    """
    Term 1: -1/12 * polgrad_(b) * cff_(a,a,b) / (w_a w_b w_b).
    Term 2: -1/6 * polgrad_(a) * cff_(a,b,b) / (w_a w_a w_b), the same sum with a and b swapped (cff is
    symmetric) and twice the factor: its coefficient is twice term 1's.
    """
    _, _, result = ir_absorption
    coeff = {r.term_id: r.coeff for r in result.table if not r.location.axes}

    assert coeff[2] == pytest.approx(2. * coeff[1], rel=1e-12)


def test_ir_absorption_on_grid_with_the_background_within_eps_floor_of_all_features_on_grid(ir_absorption):
    """1D, the peaks +- 200 cm-1, 2001 points, amplitude boxes: the flat feature's box is the window."""
    _, _, result = ir_absorption
    locations = [f.location['A'] for f in result.features if not f.is_flat]
    window = Box({'A': (min(locations) - 200., max(locations) + 200.)})

    spec = spectrum_on_grid(result, window, {'A': 2001}, box_rule=partial(amplitude_half_widths, dynrange=500.))

    [flat] = [f for f in spec.spectral_window.full_features if f.is_flat]
    assert flat.feat_box == window
    reference = abs(all_features_on_grid(result.features, result.states, spec.axis_points)) ** 2
    assert np.abs(abs(spec.amplitude) ** 2 - reference).max() < 0.1 * reference.max() / 500.


## 3D, real data: 3 IR on formaldehyde -------------------------------------------
# The setup 'nb: 3 IR (3D)': three IR pulses at different times, filter -1 +2 +3, magnitude condition -w1 + w2 > 0.
# Axes A = -w1, B = w2, C = w3.

@pytest.fixture(scope='module')
def three_ir() -> tuple[StoredSetup, list, dict[tuple[str, ...], list[int]]]:
    setup = load_setup('nb: 3 IR (3D)')
    return setup, setup.terms, terms_by_axis_set(setup.terms)


def test_three_ir_terms_resonate_on_two_axis_sets(three_ir):
    """16 terms: 14 on {A, B}, 2 on {A, B, C}. One evaluation of all 16 raises before any row is computed."""
    setup, terms, groups = three_ir

    assert setup.axes == {'A': [(-1,)], 'B': [(2,)], 'C': [(3,)]}
    assert {axis_set: len(ids) for axis_set, ids in groups.items()} == {('A', 'B'): 14, ('A', 'B', 'C'): 2}
    with pytest.raises(ValueError, match='2 sets of axes'):
        evaluate_on_formaldehyde(setup, terms)


def test_three_ir_each_axis_set_runs_on_its_own(three_ir):
    """{A, B}: 504 pairs, as the 14 EVV terms (340 rows, 164 zero, 68 features). {A, B, C}: 72 rows, one per feature."""
    setup, terms, groups = three_ir

    ab = evaluate_on_formaldehyde(setup, [terms[i] for i in groups[('A', 'B')]])
    abc = evaluate_on_formaldehyde(setup, [terms[i] for i in groups[('A', 'B', 'C')]])

    assert (len(ab.table), len(ab.zero), len(ab.features)) == (340, 164, 68)
    assert (len(abc.table), len(abc.zero), len(abc.features)) == (72, 0, 72)
    assert all(f.location.axes == ('A', 'B', 'C') for f in abc.features)


def test_three_ir_abc_set_on_a_3d_grid_within_eps_floor_of_all_features_on_grid(three_ir):
    """
    The {A, B, C} features +- 100 cm-1, 41 points per axis, amplitude boxes (amplitude_half_widths), cut_tails=True: at most
    0.1 floors from all_features_on_grid.
    """
    setup, terms, groups = three_ir
    result = evaluate_on_formaldehyde(setup, [terms[i] for i in groups[('A', 'B', 'C')]])
    locations = np.array([f.location.values for f in result.features])
    window = Box({ax: (lo - 100., hi + 100.) for ax, lo, hi in zip('ABC', locations.min(0), locations.max(0))})

    spec = spectrum_on_grid(result, window, {ax: 41 for ax in 'ABC'},
                            box_rule=partial(amplitude_half_widths, dynrange=500.), cut_tails=True)

    reference = abs(all_features_on_grid(result.features, result.states, spec.axis_points)) ** 2
    assert spec.amplitude.shape == (41, 41, 41)
    assert 0. < np.abs(abs(spec.amplitude) ** 2 - reference).max() < 0.1 * reference.max() / 500.
