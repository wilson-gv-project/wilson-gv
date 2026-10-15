"""
A15 (refac_rsp_eval/todo.txt): get_pol_laser reads the polarization vectors in reverse order.

Convention. A polarization property has one Cartesian slot per field interaction. derive numbers the slots
0 = the detected field, k = pulse k (op_omega_label_int = 0 and op_labels_int = [1, 2, ...] in
wilson_utils/common_labels.py; dressWithPulseInteractions in wilson_derive/response_terms.py).
VibExperiment lists all_polarizations in the same order: [detector, pulse 1, pulse 2, ...].
So slot k must get pol[k].

The issue. get_pol_tensor builds the product A = pol[0] x pol[1] x ... with np.kron, but takes the last vector
first. np.kron gives its first argument the first index, so the result is pol[n-1] x ... x pol[0]:
slot k gets pol[n-1-k]. e.g. pol = [x, y, z]: the only 1 sits at A[2, 1, 0], not at A[0, 1, 2].
get_pol_laser then sums A over the index tuples of each isotropic tensor f, so it reads the wrong entry:
    rank 3, [x, y, z]:        f = epsilon. A[2, 1, 0] is in epsilon's minus list -> [-1.0]. Right: [1.0].
    rank 5, [x, y, z, x, x]:  A[0, 0, 2, 1, 0] fits f element 3 (delta_14 eps_023) -> [0, 0, 0, -1, 0, 0].
                              Right: A[0, 1, 2, 0, 0] fits element 0 (delta_34 eps_012) -> [1, 0, 0, 0, 0, 0].
getGeneralPolarizationAveragingExpression multiplies that vector by M, so the whole recipe is off:
rank 3 flips its sign, rank 5 takes the weights of another row of M.

Effect. Reversal maps slot k to n-1-k. At rank 2 and 4 every f element maps onto itself (the pairs (0,1)(2,3)
become (3,2)(1,0)), so get_pol_laser is right there: the 14 EVV terms (rank 4) are not affected.
Rank 3: the sign flips. Rank 5, and rank 6 with mixed polarizations: wrong values. With the same polarization
on every slot, A is the same in any order, so every rank is right.

The checking side. exact_average is the orientation average computed directly: a weighted sum over 512 rotations
(Euler angles z-y-z; alpha and gamma 8 even steps, cos(beta) 8 Gauss-Legendre points). The sum is exact for
products of up to 7 rotation-matrix elements, so for ranks 2-6. It uses numpy only: no f, no M.
test_exact_average_gives_the_textbook_values checks it against two textbook results.

Marks. A strict xfail marks the right result, which the code misses today. The fix, one line in get_pol_tensor:
    return get_pol_tensor(np.kron(pol_tensor, np.array(pol[0])), pol[1:])
makes them pass. Strict turns an unexpected pass into a failure: after the fix, remove the marks.
Checked 2026-10-12 with the fix applied in memory: every test here passes.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.amplitudes.averaging import (
    get_pol_laser,
    get_pol_tensor,
    getGeneralPolarizationAveragingExpression,
)

X, Y, Z = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)
AXES = {'x': X, 'y': Y, 'z': Z}
LETTERS = 'abcdef'
REVERSED = pytest.mark.xfail(strict=True, reason='A15: get_pol_tensor builds pol[n-1] x ... x pol[0]')


## The checking side: the exact orientation average ------------------------------------------------

def _rotations(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    The rotations and weights of a quadrature rule over all orientations: n points per Euler angle (z-y-z),
    R = Rz(alpha) Ry(beta) Rz(gamma).
        alpha, gamma  n even steps over 360 degrees. The plain mean over these steps is exact for sin and cos
                      up to frequency n-1 (7 for n = 8).
        beta          cos(beta) at the n Gauss-Legendre points, with their weights: exact for polynomials in
                      cos(beta) up to degree 2n-1 (15). Every orientation counts the same, so the measure is
                      sin(beta) d(beta) = d cos(beta): that is why the points sit in cos(beta).
    A signal with r fields (exact_average) has frequencies up to r in alpha and gamma, and after the means over
    alpha and gamma it is a polynomial of degree up to r in cos(beta). So n = 8 is exact for r <= 7.
    Returns the rotations, shape (n**3, 3, 3), and the weights, which sum to 1.
    """
    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def ry(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

    cos_beta, w_beta = np.polynomial.legendre.leggauss(n)
    steps = 2 * np.pi * np.arange(n) / n
    rotations, weights = [], []
    for alpha in steps:
        for cb, wb in zip(cos_beta, w_beta):
            for gamma in steps:
                rotations.append(rz(alpha) @ ry(np.arccos(cb)) @ rz(gamma))
                weights.append(wb)
    weights = np.array(weights)
    return np.array(rotations), weights / weights.sum()


ROTATIONS, WEIGHTS = _rotations()


def exact_average(pol, T: np.ndarray) -> float:
    """
    The orientation average, computed the direct way: the signal of one molecule in each of 512 orientations,
    then the weighted mean. Uses numpy only: no f, no M, no wilson code.

    1. One orientation: a rotation matrix R. In the lab, field k points along pol[k] (slot k gets pol[k]).
       Seen from the molecule, field k points along R^T pol[k].
    2. The signal of the molecule in that orientation: T with each index contracted with one field direction,
           sum over i, j, ... of T[i, j, ...] * (R^T pol[0])[i] * (R^T pol[1])[j] * ...
       e.g. rank 2, T = mu x mu, both fields along Z: (mu . R^T Z)^2, the squared length of mu along Z.
       This is A . T_lab with A = pol[0] x pol[1] x ... and T_lab = (R x ... x R) T, written in the molecule frame.
    3. The average over all orientations: the weighted sum of step 2 over the rotations of _rotations.

    Why exact and not an estimate. The signal in step 2 is a polynomial in the entries of R, one R per field.
    _rotations is a quadrature rule that integrates such polynomials exactly up to 7 fields, as Simpson's rule
    integrates cubics exactly. So for ranks 1-6 the weighted sum is the integral over all orientations, up to
    rounding. Checked 2026-10-13: ranks 1-6, random T and pol, 512 rotations against 4096 (16 points per angle):
    differences at most 1.5e-14. And against textbook values: test_exact_average_gives_the_textbook_values,
    e.g. rank 2, <mu_Z mu_Z> = |mu|^2 / 3.
    """
    in_molecule = [np.einsum('rij,i->rj', ROTATIONS, np.asarray(p, dtype=float)) for p in pol]
    letters = LETTERS[:T.ndim]
    per_rotation = np.einsum(','.join('r' + c for c in letters) + ',' + letters + '->r', *in_molecule, T)
    return float(WEIGHTS @ per_rotation)


def code_average(pol, T: np.ndarray) -> float:
    """The code's average: the recipe from get_pol_laser and getGeneralPolarizationAveragingExpression, applied to T."""
    recipe = getGeneralPolarizationAveragingExpression(T.ndim, get_pol_laser(pol))
    return sum(coeff * T[key] for key, coeff in recipe.items())


def setup_vectors(setup: str, rank: int) -> list:
    """'xyz' -> [X, Y, Z]; 'random' -> rank random vectors (not unit length, fixed seed)."""
    if setup == 'random':
        return list(np.random.default_rng(rank).normal(size=(rank, 3)))
    return [AXES[c] for c in setup]


def test_exact_average_gives_the_textbook_values():
    """
    Checks the checking side, without wilson code.
        rank 2:  <mu_Z mu_Z> = |mu|^2 / 3
        rank 4, Raman, alpha symmetric:  <alpha_ZZ alpha_ZZ> = (45 a^2 + 4 gamma^2) / 45,  <alpha_ZX alpha_ZX> = gamma^2 / 15
                 a = mean polarizability, gamma^2 = anisotropy squared
    """
    rng = np.random.default_rng(5)
    mu = rng.normal(size=3)
    alpha = rng.normal(size=(3, 3))
    alpha = alpha + alpha.T
    a = np.trace(alpha) / 3
    d = np.diag(alpha)
    gamma2 = 0.5 * ((d[0] - d[1])**2 + (d[1] - d[2])**2 + (d[2] - d[0])**2
                    + 6 * (alpha[0, 1]**2 + alpha[1, 2]**2 + alpha[0, 2]**2))
    alpha_alpha = np.einsum('ij,kl->ijkl', alpha, alpha)

    assert exact_average([Z, Z], np.outer(mu, mu)) == pytest.approx(mu @ mu / 3, rel=1e-12, abs=0)
    assert exact_average([Z, Z, Z, Z], alpha_alpha) == pytest.approx((45 * a**2 + 4 * gamma2) / 45, rel=1e-12, abs=0)
    assert exact_average([Z, X, Z, X], alpha_alpha) == pytest.approx(gamma2 / 15, rel=1e-12, abs=0)


## get_pol_tensor: the product A, slot by slot ------------------------------------------------------

@REVERSED
def test_pol_tensor_of_x_y_z_has_its_one_at_0_1_2():
    """pol = [x, y, z]: slot 0 is x (0), slot 1 is y (1), slot 2 is z (2). Today the 1 sits at A[2, 1, 0]."""
    A = get_pol_tensor(1.0, [X, Y, Z]).reshape(3, 3, 3)

    assert np.argwhere(A != 0).tolist() == [[0, 1, 2]]


@REVERSED
@pytest.mark.parametrize('rank', [2, 3, 4, 5, 6])
def test_pol_tensor_puts_pol_k_on_slot_k(rank):
    """
    A[i0, i1, ...] = pol[0][i0] * pol[1][i1] * ...  Today A is wrong at every rank, rank 2 and 4 included:
    get_pol_laser hides the error there (test_rank_2_and_4_do_not_see_the_order).
    """
    pol = setup_vectors('random', rank)
    letters = LETTERS[:rank]
    expected = np.einsum(','.join(letters) + '->' + letters, *pol)

    A = get_pol_tensor(1.0, pol).reshape((3,) * rank)

    np.testing.assert_allclose(A, expected, rtol=1e-14, atol=0)


## get_pol_laser: the vector A . f --------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason='A15: today [-1.0]')
def test_pol_laser_rank_3_x_y_z_is_plus_one():
    """f = epsilon, so the value is det(pol[0], pol[1], pol[2]) = x . (y x z) = +1. Today z . (y x x) = -1."""
    assert get_pol_laser([X, Y, Z]) == [1.0]


@pytest.mark.xfail(strict=True, reason='A15: today [0, 0, 0, -1, 0, 0]')
def test_pol_laser_rank_5_x_y_z_x_x():
    """
    Element 0 of f, delta_34 eps_012: (pol[3] . pol[4]) det(pol[0], pol[1], pol[2]) = 1 * 1 = 1.
    Elements 1-5 pair x with y or z (delta_24, delta_23, delta_14, delta_13, delta_12): 0.
    Today element 3, delta_14 eps_023, reads x . x = 1 and det(x, z, y) = -1.
    """
    assert get_pol_laser([X, Y, Z, X, X]) == [1., 0., 0., 0., 0., 0.]


@pytest.mark.parametrize('rank', [2, 4])
def test_rank_2_and_4_do_not_see_the_order(rank):
    """
    Reversing the slots maps every rank-2 and rank-4 f element onto itself, e.g. delta_01 delta_23 -> delta_32 delta_10.
    So get_pol_laser gives the same vector for pol and for pol reversed, and the reversed A does no harm:
    the 14 EVV terms (rank 4) are right. Passes before and after the fix.
    """
    pol = setup_vectors('random', rank)

    assert get_pol_laser(pol[::-1]) == pytest.approx(get_pol_laser(pol), rel=1e-12, abs=0)


## The whole recipe against the exact average ---------------------------------------------------------

@pytest.mark.parametrize('rank, setup', [
    (2, 'xx'), (2, 'xy'), (2, 'random'),
    pytest.param(3, 'xyz', marks=REVERSED),
    pytest.param(3, 'random', marks=REVERSED),
    (4, 'xxxx'), (4, 'yyxx'), (4, 'xzxz'), (4, 'xzzx'), (4, 'random'),
    pytest.param(5, 'xyzxx', marks=REVERSED),
    pytest.param(5, 'random', marks=REVERSED),
    (6, 'xxxxxx'),
    (6, 'xxyyzz'),                                  # reversed: zzyyxx, the same pairs -> right by luck
    pytest.param(6, 'xyxyxx', marks=REVERSED),
    pytest.param(6, 'random', marks=REVERSED),
])
def test_recipe_gives_the_exact_orientation_average(rank, setup):
    """
    sum of recipe[k] * T[k] = the exact average of A . T_lab, for a random molecular tensor T.
    Today the code gives the exact average of the reversed list, e.g. rank 5, xyzxx: the exact average of xxzyx.
    Rank 2, rank 4 and slot-symmetric setups pass.
    T's entries are about 1, so the averages are 0.01-1; abs 1e-12 only matters where the average is 0 (2, 'xy').
    """
    pol = setup_vectors(setup, rank)
    T = np.random.default_rng(100 + rank).normal(size=(3,) * rank)

    assert code_average(pol, T) == pytest.approx(exact_average(pol, T), rel=1e-10, abs=1e-12)
