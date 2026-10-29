"""
A reference test for the whole evaluation: if any step changes the spectrum, a test here fails.

Each case runs the two public calls on real data, and compares the result, grid point by grid
point, with the numbers saved in grid_reference.npz:
    terms + data -> rows -> features              evaluate_response_terms
    features -> boxes -> the amplitude on a grid  spectrum_on_grid
So every step counts: the coefficients, the locations, the widths, the boxes, the line shape and
the sum on the grid. test_evv_reference.py checks the rows alone.

The test shows that the numbers did not change, not that they are right. Other tests check that:
the rows against a hand calculation, the grid formulas against hand-written Lorentzians, and the
boxed grid against all_features_on_grid.

The cases: formaldehyde, amplitude feat_boxes with dynrange 500. Each window_box holds several features,
so their peaks interfere, and features outside window_box reach in with their tails.
    evv        14 EVV terms, anharmonic states, gamma 4.7 cm-1, cut_tails=False.
               4 features inside, among them the strongest EVV feature.
    evv_width  14 EVV terms, harmonic states, a width function (width_cm), cut_tails=False.
               6 features inside.
    three_ir   3 IR, its terms on {A, B, C}, anharmonic states, gamma 10 cm-1, cut_tails=True.
               8 features inside. Its conditions use 1, 2 and 3 axes: A, A + B, A + B + C.
Together they cover both state sets, a number and a function as the width, both cut_tails
settings, and 2 and 3 axes.

grid_reference.npz holds, per case, e.g. for evv:
    evv__amplitude  the complex amplitude, au, one value per grid point, shape (56, 71):
                    array dimension k is the k-th axis, axes sorted (A, B)
    evv__A, evv__B  the grid points on each axis, cm-1

A failure message gives the size of the change, as a fraction of the case's largest amplitude,
and the grid point where the change is largest, e.g.
    changed by 2.0e-04 of the largest amplitude, most at {'A': 2682.0, 'B': 0.0} cm-1
About 1 or more: the shape of the spectrum changed. Much less than 1: a small change, e.g. a weak
feature, a cut tail or a slightly different constant.

A deliberate change of numbers:
    1. Run this file as a script. It prints, per case, how much the amplitude changed against
       the old file, then overwrites the file:
           python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.test_grid_reference
    2. Check that only the expected cases changed, by about the expected amount.
    3. Commit the new grid_reference.npz together with the change.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    PropagatorParam,
    amplitude_half_widths,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    EvalRspResult,
    evaluate_response_terms,
    prop_names,
    spectrum_on_grid,
    terms_by_axis_set,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import load_setup
from wilson_suite.wilson_system.system_data import MolSystemData

REFERENCE_FILE = Path(__file__).parent / 'grid_reference.npz'
# the box rule of every case: boxes by amplitude, for a plot with dynamic range 500
BOXES = partial(amplitude_half_widths, dynrange=500.)
# The allowed change, as a fraction of the case's largest amplitude. The same code and data give
# the same numbers up to float rounding, e.g. on another machine; a change of 1e-9 already fails.
REL = 1e-11


def width_cm(vd, freqs):
    """
    The width function of the evv_width case, cm-1: a width per condition, from its energy
    difference E, that also grows along A:
        3 + 0.002 |E| + 0.001 |A|
    For the box and the peak height, A is the feature's location; on the grid, A is an array of
    every grid point's A, so the width changes from point to point.
        e.g. the condition with E = 1534.5 at A = 1534.5: 3 + 3.07 + 1.53 = 7.6
    """
    return 3. + 0.002 * abs(vd.energy_difference()) + 0.001 * np.abs(freqs['A'])


@dataclass(frozen=True)
class Case:
    """
    One reference spectrum: which terms, with which settings, on which grid.
        setup      the stored setup the terms come from (stored_terms.load_setup), e.g. 'evv'
        axis_set   the setup's terms on these resonance axes (terms_by_axis_set): one
                   evaluation takes one set
        states     the state set, 'anharmonic' or 'harmonic'
        width      propagator_param, cm-1: a number or a function (features.PropagatorParam)
        window_box         the range of the spectrum, cm-1, one range per axis
        n_points_per_axis  grid points on each axis, both edges of window_box included
        cut_tails          False: each feature of spectral_window on the whole grid; True: on its own feat_box only
        n_inside           the number of features with their location in window_box
    """
    setup: str
    axis_set: tuple[str, ...]
    states: str
    width: PropagatorParam
    window_box: Box
    n_points_per_axis: dict[str, int]
    cut_tails: bool
    n_inside: int


# Each window_box sits on a group of features; their locations below in cm-1, rounded.
# The 2D cases have different point counts on A and B, so swapped axes also change the shape.
CASES = {
    # 2 cm-1 steps. Inside: A 2682.8 with B 0 (the strongest EVV feature) and 44;
    # A 2726.8 with B -44 and 0
    'evv': Case('evv', ('A', 'B'), 'anharmonic', 4.7,
                Box({'A': (2650., 2760.), 'B': (-70., 70.)}), {'A': 56, 'B': 71},
                cut_tails=False, n_inside=4),
    # harmonic locations, 5 cm-1 steps. Inside: A 1534.5 with B 1268.9, 1344.1, 1399.0, 1534.5;
    # A 1820.4 with B 1268.9 and 1534.5
    'evv_width': Case('evv', ('A', 'B'), 'harmonic', width_cm,
                      Box({'A': (1500., 1850.), 'B': (1250., 1560.)}), {'A': 71, 'B': 63},
                      cut_tails=False, n_inside=6),
    # 5 cm-1 steps on A and B, 10 on C. Inside: A -2726.8 and -2682.8, each with B 2682.8 and
    # 2726.8, C from 2618.5 to 2745.9
    'three_ir': Case('nb: 3 IR (3D)', ('A', 'B', 'C'), 'anharmonic', 10.,
                     Box({'A': (-2750., -2660.), 'B': (2660., 2750.), 'C': (2600., 2760.)}),
                     {'A': 19, 'B': 19, 'C': 17}, cut_tails=True, n_inside=8),
}


def case_terms(case: Case) -> tuple[list, tuple[float, ...]]:
    """
    The case's terms (the setup's terms on case.axis_set, in the setup's order) and the setup's
    laser_pol.
    """
    setup = load_setup(case.setup)
    return [setup.terms[i] for i in terms_by_axis_set(setup.terms)[case.axis_set]], setup.laser_pol


def load_formaldehyde() -> MolSystemData:
    """
    Formaldehyde's data with every property the cases use: polhess, dipgrad, polgrad, diphess,
    cff.
    """
    names = dict.fromkeys(name for case in CASES.values() for name in prop_names(case_terms(case)[0]))
    return MolSystemData.load(ref.data_origin('formaldehyde'), list(names))


def spectrum(case: Case, molsys: MolSystemData) -> EvalRspResult:
    """
    One case through the two public calls: evaluate_response_terms (rows and features), then
    spectrum_on_grid (the amplitude on the grid).
    """
    terms, laser_pol = case_terms(case)
    result = evaluate_response_terms(terms, molsys_data=molsys, states_choice=case.states,
                                     laser_pol=laser_pol, propagator_param=case.width)
    return spectrum_on_grid(result, window_box=case.window_box, n_points_per_axis=case.n_points_per_axis,
                            box_rule=BOXES, cut_tails=case.cut_tails)


def stored_arrays(name: str, spec: EvalRspResult) -> dict[str, np.ndarray]:
    """
    The arrays of one case, under their keys in grid_reference.npz: '<case>__amplitude', and
    '<case>__<axis>' for each axis, e.g. 'evv__amplitude', 'evv__A', 'evv__B'.
    """
    return {f'{name}__amplitude': spec.amplitude, **{f'{name}__{ax}': p for ax, p in spec.axis_points.items()}}


def largest_change(now: np.ndarray, before: np.ndarray) -> float:
    """
    The largest |now - before| over the grid, as a fraction of the largest |before|.
        e.g. 1e-9: no grid point changed by more than 1e-9 of the largest amplitude
    """
    return float(np.abs(now - before).max() / np.abs(before).max())


@pytest.fixture(scope='module')
def spectra() -> dict[str, EvalRspResult]:
    """Every case's spectrum, computed once for all tests here."""
    molsys = load_formaldehyde()
    return {name: spectrum(case, molsys) for name, case in CASES.items()}


@pytest.fixture(scope='module')
def reference() -> dict[str, np.ndarray]:
    """The saved arrays, {key: array}, keys as in stored_arrays."""
    with np.load(REFERENCE_FILE) as stored:
        return dict(stored)


def test_reference_file_has_every_case(reference):
    """
    grid_reference.npz holds exactly the arrays of CASES. Fails after a case is added, renamed or
    given other axes: then write the file again (see the top).
    """
    expected = {f'{name}__{key}' for name, case in CASES.items() for key in ('amplitude', *case.window_box.axes)}

    assert set(reference) == expected


@pytest.mark.parametrize('name', CASES)
def test_window_box_holds_several_features_and_features_outside_reach_in(spectra, name):
    """
    Each window_box holds n_inside features, and the feat_box of at least one feature outside reaches in:
    the setting each case is chosen for. Fails if the features moved, or if the box rule keeps
    other features; the spectrum then changes too.
    """
    spectral_window = spectra[name].spectral_window

    assert len(spectral_window.full_features) == CASES[name].n_inside
    assert spectral_window.contrib_features


@pytest.mark.parametrize('name', CASES)
def test_spectrum_matches_the_reference(spectra, reference, name):
    """
    The grid points, then the amplitude at every grid point, against the saved arrays, to REL of
    the largest amplitude. The message gives the size of the change and where it is largest.
    """
    spec = spectra[name]
    stored = reference[f'{name}__amplitude']

    for ax, points in spec.axis_points.items():
        np.testing.assert_allclose(points, reference[f'{name}__{ax}'], rtol=REL, err_msg=f'grid points on {ax}')
    assert spec.amplitude.shape == stored.shape
    # the grid point with the largest change, for the message
    diff = np.abs(spec.amplitude - stored)
    at = np.unravel_index(np.argmax(diff), diff.shape)
    point = {ax: float(points[i]) for (ax, points), i in zip(spec.axis_points.items(), at)}
    assert largest_change(spec.amplitude, stored) <= REL, (
        f'changed by {largest_change(spec.amplitude, stored):.1e} of the largest amplitude, most at {point} cm-1')


def main():
    """
    Writes grid_reference.npz from CASES. Before it overwrites the old file, it prints per case:
    the grid's shape, the number of features inside window_box and reaching in, and how much the
    amplitude changed against the old file ('new' if the old file lacks the case or has another
    grid shape).
    """
    before = {}
    if REFERENCE_FILE.exists():
        with np.load(REFERENCE_FILE) as stored:
            before = dict(stored)
    molsys = load_formaldehyde()
    arrays = {}
    for name, case in CASES.items():
        spec = spectrum(case, molsys)
        arrays.update(stored_arrays(name, spec))
        old = before.get(f'{name}__amplitude')
        change = ('new' if old is None or old.shape != spec.amplitude.shape
                  else f'{largest_change(spec.amplitude, old):.1e} of the largest amplitude')
        print(f'{name:10s} grid {spec.amplitude.shape}, {len(spec.spectral_window.full_features)} features inside, '
              f'{len(spec.spectral_window.contrib_features)} reach in; changed: {change}')
    np.savez_compressed(REFERENCE_FILE, **arrays)


if __name__ == '__main__':
    main()
