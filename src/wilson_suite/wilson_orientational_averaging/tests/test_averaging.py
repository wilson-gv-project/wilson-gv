import numpy as np
import pytest

from wilson_suite.wilson_orientational_averaging import averaging


def test_pol_laser():

    # Unit polarization vectors

    pol_x = [1.0, 0.0, 0.0]
    pol_y = [0.0, 1.0, 0.0]
    pol_z = [0.0, 0.0, 1.0]

    # Reference data from evaluating the A * f product in JCP 141, 204103 for the respective A
    # resulting from each situation

    # "VVVV" cases: All three isotropic tensors in f "activate" once
    pol_vvvv_a = [pol_x, pol_x, pol_x, pol_x]
    assert averaging.get_pol_laser(pol_vvvv_a) == [1.0, 1.0, 1.0]

    pol_vvvv_b = [pol_y, pol_y, pol_y, pol_y]
    assert averaging.get_pol_laser(pol_vvvv_b) == [1.0, 1.0, 1.0]

    pol_vvvv_c = [pol_z, pol_z, pol_z, pol_z]
    assert averaging.get_pol_laser(pol_vvvv_c) == [1.0, 1.0, 1.0]

    # Various "HHVV" cases: Only one of the three isotropic tensors "activate"

    # The first (d_(i1, i2) * d_(i3, i4)) activates
    pol_yyxx =  [pol_y, pol_y, pol_x, pol_x]
    assert averaging.get_pol_laser(pol_yyxx) == [1.0, 0.0, 0.0]

    # (Again), the first (d_(i1, i2) * d_(i3, i4)) activates
    pol_zzyy = [pol_z, pol_z, pol_y, pol_y]
    assert averaging.get_pol_laser(pol_zzyy) == [1.0, 0.0, 0.0]

    # The second (d_(i1, i3) * d_(i2, i4)) activates
    pol_xzxz = [pol_x, pol_z, pol_x, pol_z]
    assert averaging.get_pol_laser(pol_xzxz) == [0.0, 1.0, 0.0]

    # The third (d_(i1, i4) * d_(i2, i3)) activates
    pol_xzzx = [pol_x, pol_z, pol_z, pol_x]
    assert averaging.get_pol_laser(pol_xzzx) == [0.0, 0.0, 1.0]

    # Various other cases which all should result in zeros (none of the isotropic tensors "activate")

    pol_yyzx =  [pol_y, pol_y, pol_z, pol_x]
    assert averaging.get_pol_laser(pol_yyzx) == [0.0, 0.0, 0.0]

    pol_xxxz =  [pol_x, pol_x, pol_x, pol_z]
    assert averaging.get_pol_laser(pol_xxxz) == [0.0, 0.0, 0.0]

    pol_yxyy =  [pol_y, pol_x, pol_y, pol_y]
    assert averaging.get_pol_laser(pol_yxyy) == [0.0, 0.0, 0.0]

def test_getGeneralPolarizationAveragingExpression():

    # Currently testing only "VVVV" polarization: All polarization vectors linearly polarized along same direction

    pol_z = [0.0, 0.0, 1.0]

    pol_zzzz = [pol_z, pol_z, pol_z, pol_z]

    lin_comb_vvvv = averaging.getGeneralPolarizationAveragingExpression(4, averaging.get_pol_laser(pol_zzzz))

    tol = 1.0e-12

    # Testing against pen/paper derived elements
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:

            if (i == j):
                assert ((lin_comb_vvvv[(i, i, i, i)] - 3.0 / 15.0) ** 2) ** 0.5 < tol

            else:
                assert ((lin_comb_vvvv[(i, i, j, j)] - 1.0 / 15.0) ** 2) ** 0.5 < tol
                assert ((lin_comb_vvvv[(i, j, i, j)] - 1.0 / 15.0) ** 2) ** 0.5 < tol
                assert ((lin_comb_vvvv[(i, j, j, i)] - 1.0 / 15.0) ** 2) ** 0.5 < tol

    # Pop elements from dictionary and assert empty afterwards (test if no "surplus" elements)s
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:

            if (i == j):

                del lin_comb_vvvv[(i, i, i, i)]

            else:

                del lin_comb_vvvv[(i, i, j, j)]
                del lin_comb_vvvv[(i, j, i, j)]
                del lin_comb_vvvv[(i, j, j, i)]

    assert lin_comb_vvvv == {}

    # Test if VVVV polarization along a different (but same for all pulses/detection) direction gives same average (it should)
    # Perhaps somewhat redundant since a corresponding test is made in test_get_pol_laser
    pol_y = [0.0, 1.0, 0.0]

    pol_yyyy = [pol_y, pol_y, pol_y, pol_y]

    lin_comb_vvvv = averaging.getGeneralPolarizationAveragingExpression(4, averaging.get_pol_laser(pol_yyyy))

    # Testing the same elements
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:

            if (i == j):
                assert ((lin_comb_vvvv[(i, i, i, i)] - 3.0 / 15.0) ** 2) ** 0.5 < tol

            else:
                assert ((lin_comb_vvvv[(i, i, j, j)] - 1.0 / 15.0) ** 2) ** 0.5 < tol
                assert ((lin_comb_vvvv[(i, j, i, j)] - 1.0 / 15.0) ** 2) ** 0.5 < tol
                assert ((lin_comb_vvvv[(i, j, j, i)] - 1.0 / 15.0) ** 2) ** 0.5 < tol


def dense_recipe(recipe: dict) -> np.ndarray:
    """The recipe as a dense 3x3x3x3 coefficient tensor C[i, j, k, l]."""
    C = np.zeros((3,) * 4)
    for key, coeff in recipe.items():
        C[key] = coeff
    return C


def rotation(alpha: float, beta: float) -> np.ndarray:
    """Rotation by alpha about z, then by beta about x."""
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    Rz = np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])
    Rx = np.array([[1., 0., 0.], [0., cb, -sb], [0., sb, cb]])
    return Rx @ Rz


def rotate_every_axis(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """C with each of its four axes rotated by R."""
    return np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, C)


@pytest.mark.parametrize('laser_pol', [(1., 1., 1.), (1., 0., 0.), (0.3, -1.2, 2.)])
def test_rank4_recipe_does_not_change_when_the_molecule_is_rotated(laser_pol):
    """
    The orientational average cannot depend on how the molecule sits in its own axes. So the recipe, as a
    tensor over the molecular axes, stays the same when all four axes are rotated by the same R. A tensor
    with a single entry does change, which shows the rotation is not trivial.
    """
    R = rotation(0.7, 1.9)
    C = dense_recipe(averaging.getGeneralPolarizationAveragingExpression(4, laser_pol))
    one_entry = dense_recipe({(0, 0, 0, 0): 1.})

    assert np.allclose(rotate_every_axis(R, C), C)
    assert not np.allclose(rotate_every_axis(R, one_entry), one_entry)
