"""
The slot order of get_pol_tensor (amplitudes/averaging.py), shown on the evaluator's numbers. The tests it breaks
are strict xfails until it is fixed (one line in get_pol_tensor); after the fix, remove the marks. The cause, slot by
slot, and more setups: test_averaging_pol_order.py (strict xfail there too).

Slots. Slot 0 is the detected field, slot k is pulse k. VibExperiment lists its polarizations in that order
(all_polarizations = [detector, pulse 1, pulse 2, ...]) and turns them into polarization_avg_vector with
get_pol_laser. The evaluator takes that vector as laser_pol (make_polarization_linear_comb). The experiment is
not used here: get_pol_laser gets the list in slot order directly. make_translated_terms.py
(data_for_tests/translated_terms) checks the slot order and the vector for every stored setup.

Rank 3, the clearest case. Fields along e0, e1, e2 (slots 0, 1, 2) and a molecular tensor T with three Cartesian
indices. Averaged over all orientations of the molecule (Andrews and Thirunamachandran, J. Chem. Phys. 67, 5026 (1977)):
    <(e0 x e1 x e2) . T_lab> = det(e0, e1, e2) * (eps : T) / 6,    eps : T = sum over i, j, k of eps_ijk T[i, j, k]
For three transition dipoles, T = mu_a x mu_b x mu_c and eps : T = det(mu_a, mu_b, mu_c). That is nonzero only if the
three dipoles are not in one plane: a chiral molecule. Its mirror image has the opposite det, so the sign of the
average tells the two mirror images apart.
With e0, e1, e2 = x, y, z: det = +1, the average is +det(mu) / 6. The evaluator gives -det(mu) / 6, the value
of the mirror-image molecule: get_pol_laser gives [-1.0] instead of [+1.0].

Rank 5. The error is not always a sign. Beta-alpha CARS form (experiment_beta_alpha_cars in fixtures.py),
fields x, y, z, x, x: exact -0.963, the evaluator +0.598.

Rank 2 and rank 4 (the 14 EVV terms) are right: test_rank_2_and_4_do_not_see_the_order in test_averaging_pol_order.py.
"""

from fractions import Fraction

import numpy as np
import pytest

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    calculate_avrg_tensor_einsum,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    make_polarization_linear_comb,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import split_props
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import polprop
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.test_averaging_pol_order import (
    REVERSED,
    exact_average,
)
from wilson_suite.wilson_system.system_data import MolecularProperty, MolPropsCollection

X, Y, Z = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)

# dipgrad of modes 0, 1, 2: three dipoles not in one plane, det = +1. The mirror image (z -> -z) has det = -1.
MU = np.array([[1.0, 0.0, 0.0],
               [0.3, 1.0, 0.0],
               [0.2, 0.4, 1.0]])
MU_MIRROR = MU * np.array([1., 1., -1.])


def rank_3_laser_pol(pulse_1_pol, pulse_2_pol) -> list:
    """polarization_avg_vector as VibExperiment makes it: detector along x, then pulse 1, pulse 2 (slot order)"""
    return get_pol_laser([X, pulse_1_pol, pulse_2_pol])


def evaluator_average(term: VibPerturbedTerm, laser_pol, data: dict[str, np.ndarray]) -> float:
    """the averaged part of `term` at a=0, b=1, c=2, as the evaluator computes it from laser_pol"""
    recipe = make_polarization_linear_comb([term], laser_pol)
    props = MolPropsCollection([MolecularProperty(trivial_name=name, vals=vals, extra_data={})
                                for name, vals in data.items()])
    avrg_props, _ = split_props(term)
    return float(calculate_avrg_tensor_einsum(avrg_props, props, recipe).at({'a': 0, 'b': 1, 'c': 2}))


# dipgrad[a] (slot 0) * dipgrad[b] (slot 1) * dipgrad[c] (slot 2): three IR interactions, rank 3
THREE_DIPOLES = VibPerturbedTerm(Fraction(1), [polprop(ops=(0,), inds='a'), polprop(ops=(1,), inds='b'),
                                               polprop(ops=(2,), inds='c')], [], [])


## rank 3: the polarization vector, the reference, the evaluator ------------------------------------

@pytest.mark.parametrize('pulse_1_pol, pulse_2_pol, det', [(Y, Z, +1.), (Z, Y, -1.)],
                         ids=['x, y, z: det +1', 'x, z, y: det -1'])
@REVERSED
def test_polarization_vector_is_det_of_the_polarizations(pulse_1_pol, pulse_2_pol, det):
    """polarization_avg_vector at rank 3 is [det(detector, pulse 1, pulse 2)]. Both setups get the other sign."""
    laser_pol = rank_3_laser_pol(pulse_1_pol, pulse_2_pol)

    assert laser_pol == [det], (
        f'polarization_avg_vector {laser_pol}, right [{det}]: '
        f'the vector of the setup with the polarizations in reverse order')


@pytest.mark.parametrize('mu', [MU, MU_MIRROR], ids=['molecule', 'mirror image'])
def test_the_reference_det_over_6_is_the_exact_orientation_average(mu):
    """the checking side, without the evaluator: the formula = the average over all orientations (quadrature). Passes."""
    T = np.einsum('i,j,k->ijk', *mu)

    assert exact_average([X, Y, Z], T) == pytest.approx(np.linalg.det(mu) / 6, rel=1e-12, abs=0)


@pytest.mark.parametrize('mu', [MU, MU_MIRROR], ids=['molecule, det +1', 'mirror image, det -1'])
@REVERSED
def test_evaluator_three_dipoles_x_y_z_is_det_over_6(mu):
    """
    Experiment x, y, z -> polarization_avg_vector -> recipe -> averaged part of dipgrad[a] dipgrad[b] dipgrad[c].
    Right: +det(mu) / 6. The evaluator: -det(mu) / 6, so the molecule gets its mirror image's value and the other way round.
    """
    laser_pol = rank_3_laser_pol(Y, Z)

    value = evaluator_average(THREE_DIPOLES, laser_pol, {'dipgrad': mu})

    expected = np.linalg.det(mu) / 6
    assert value == pytest.approx(expected, rel=1e-12, abs=0), (
        f'evaluator {value:+.4f}, exact {expected:+.4f}: the value of the mirror image')


## rank 5: not only a sign ----------------------------------------------------------------------------

@REVERSED
def test_evaluator_beta_alpha_rank_5_is_the_exact_orientation_average():
    """
    hypgrad[a] (slots 0, 1, 2) * polgrad[b] (slots 3, 4), the form of experiment_beta_alpha_cars; fields x, y, z, x, x.
    Random molecular data, polgrad symmetric as a real polarizability. Right: the exact average (quadrature),
    -0.963. The evaluator: +0.598, the exact average of the reversed list x, x, z, y, x.
    laser_pol from get_pol_laser, as VibExperiment makes it.
    """
    rng = np.random.default_rng(15)
    hypgrad = rng.normal(size=(3, 3, 3, 3))
    polgrad = rng.normal(size=(3, 3, 3))
    polgrad = polgrad + polgrad.transpose(0, 2, 1)
    term = VibPerturbedTerm(Fraction(1), [polprop(ops=(0, 1, 2), inds='a'), polprop(ops=(3, 4), inds='b')], [], [])
    pol = [X, Y, Z, X, X]

    value = evaluator_average(term, get_pol_laser(pol), {'hypgrad': hypgrad, 'polgrad': polgrad})

    T = np.einsum('ijk,lm->ijklm', hypgrad[0], polgrad[1])      # T[slot 0, ..., slot 4] at a=0, b=1
    expected = exact_average(pol, T)
    assert value == pytest.approx(expected, rel=1e-10, abs=0), (
        f'evaluator {value:+.4f}, exact {expected:+.4f}; '
        f'exact average of the reversed list x, x, z, y, x: {exact_average(pol[::-1], T):+.4f}')
