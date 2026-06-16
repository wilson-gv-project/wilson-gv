import wilson_suite.wilson_intensities.amplitudes.averaging as averaging


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
