import wilson_suite.wilson_intensities.amplitudes.averaging as averaging


def test_pol_laser():

    # Four waves of (unit) wavevector [0.0, 0.0, 1.0]

    pol_x = [1.0, 0.0, 0.0]
    pol_y = [0.0, 1.0, 0.0]
    pol_z = [0.0, 0.0, 1.0]

    pol_vvvv_a = [pol_x, pol_x, pol_x, pol_x]
    pol_vvvv_b = [pol_y, pol_y, pol_y, pol_y]
    pol_vvvv_c = [pol_z, pol_z, pol_z, pol_z]

    pol_hhvv = [pol_x, pol_x, pol_y, pol_y]
    pol_vhhv_a = [pol_y, pol_x, pol_x, pol_y]
    pol_vhhv_b = [pol_x, pol_y, pol_y, pol_x]
    pol_vhvh = [pol_y, pol_x, pol_y, pol_x]

    pol_vhhh = [pol_y, pol_x, pol_x, pol_x]
    pol_hhhv = [pol_y, pol_y, pol_y, pol_x]
    pol_xxzz = [pol_x, pol_x, pol_z, pol_z]

    print('vvvv_a', averaging.get_pol_laser(pol_vvvv_a))
    assert averaging.get_pol_laser(pol_vvvv_a) == [1.0, 1.0, 1.0]

    print('vvvv_b', averaging.get_pol_laser(pol_vvvv_b))
    assert averaging.get_pol_laser(pol_vvvv_b) == [1.0, 1.0, 1.0]

    print('vvvv_c', averaging.get_pol_laser(pol_vvvv_c))

    print('hhvv', averaging.get_pol_laser(pol_hhvv))


    print('vhhv_a', averaging.get_pol_laser(pol_vhhv_a))
    print('vhhv_b', averaging.get_pol_laser(pol_vhhv_b))
    print('vhvh', averaging.get_pol_laser(pol_vhvh))

    print('vhhh', averaging.get_pol_laser(pol_vhhh))
    print('hhhv', averaging.get_pol_laser(pol_hhhv))

    print('xxzz', averaging.get_pol_laser(pol_xxzz))

    assert averaging.get_pol_laser(pol_hhvv) == [0.0, 0.0, 1.0 ]

def test_getGeneralPolarizationAveragingExpression():

    pol_z = [0.0, 0.0, 1.0]

    pol_vvvv = [pol_z, pol_z, pol_z, pol_z]

    lin_comb_vvvv = averaging.getGeneralPolarizationAveragingExpression(4, averaging.get_pol_laser(pol_vvvv))