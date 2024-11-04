import numpy as np

from wilson.spectrum.spectrum2D import (convNu2Ene, avrg_abc_tensor,
                                        Spectrum2D, EvalTerm, get_AlphaBetaGammaDelta_indices)

def test_convNu2Ene():
    """Unit conversion: from wavenumber (cm-1) to energy unit (Hartree)
    1/cm = 100 (1/m)
    Reference https://www.colby.edu/chemistry/PChem/Hartree.html
    """
    reference = np.array([0.000228, 0.002278, 0.0045563, 0.011391, 0.01458])
    result = convNu2Ene(np.array([50, 500, 1000, 2500, 3200]))
    assert np.allclose(result, reference, atol=1e-6)


def test_Spectrum2DObj():
    omega1 = np.array([55., 70., 150., 420., 740., 1155.])
    omega2 = np.array([2055., 2070., 2150., 2420., 3140.])

    W1, W2 = np.meshgrid(omega1, omega2, indexing='ij')
    specObj = Spectrum2D(omega1, omega2)

    assert np.all(specObj.w1_mesh == W1)
    assert np.all(specObj.w2_mesh == W2)


def test_avrg_abc_tensor():
    """

    """
    specObj = Spectrum2D([1.], [2.])
    # result = avrg_abc_tensor(self.electric_avrg[i], self.deriv_data, self.gammaCompsAll)
    pass


def test_get_derived_terms_evv():
    specObj = Spectrum2D([1.], [2.])
    specObj.getDerivedTermsEVV()
    specObj
    assert False


def test_load_data():
    assert False


def test_set_spectrum_settings():
    assert False


def test_conversion2internal_units():
    assert False


def test_add_terms():
    assert False


def test_precalculate_parts():
    assert False


def test_get_total_gamma_sum_el():
    assert False


def test_get_total_gamma_sum_mech():
    assert False


def test_intensity_totals():
    assert False


def test_generate_resonances_functions():
    assert False


def test_calc_averaging():

    # gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
    # [alpha, beta, gamma, delta]
    gammaCompsAll = [[2, 2, 0, 0],
                     [1, 0, 0, 1],
                     [1, 2, 1, 2]]

    from DATA import mu_Q, mu_QQ, alpha_Q, alpha_QQ
    data = {'mu_Q': mu_Q, 'mu_QQ': mu_QQ, 'alpha_Q': alpha_Q, 'alpha_QQ': alpha_QQ}

    term_m = EvalTerm(avrg=[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))], prefac='abc')
    avrgF_m = term_m.calcAveraging(data=data, gammaCompsAll=gammaCompsAll)
    # nmodes = 4
    a,b,c = (1, 3, 0)
    # ((beta,), (alpha,delta), (gamma))
    ref_avrgF_m1 = (data['mu_Q'][a, 2] * data['alpha_Q'][b, 2, 0] * data['mu_Q'][c, 0]+
                    data['mu_Q'][a, 0] * data['alpha_Q'][b, 1, 1] * data['mu_Q'][c, 0]+
                    data['mu_Q'][a, 2] * data['alpha_Q'][b, 1, 2] * data['mu_Q'][c, 1])
    assert ref_avrgF_m1==avrgF_m[a,b,c]

    term_e = EvalTerm(avrg=[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))], prefac='ab')
    avrgF_e = term_e.calcAveraging(data=data, gammaCompsAll=gammaCompsAll)

    # final tensor will have dimensions (nmodes, nmodes) or (nmodes, nmodes, nmodes)
    #           [a, b] for el. and [a, b, c] for mech. (c=a or c=b sometimes)
    # element of the final tensor is a sum of products according to
    #                     the gammaCompsAll array of xyz-indices for each greek index: [alpha, beta, gamma, delta]
    # one product per one array from the gammaCompsAll array:
    #   derT1[nmodes..., beta_val]*derT2[nmodes..., alpha_val, delta_val]*derT3[nmodes..., gamma_val] where
    #         nmodes..., marks the derivative dimensions, and greek indices refer to cartesian dimensions

    print('\n')
    np.set_printoptions(precision=4)

    # print('\n', avrgF_m)
    # print('\n', avrgF_e)

np.array([[0, 0, 0, 0],
       [0, 0, 1, 1],
       [0, 0, 2, 2],
       [1, 1, 0, 0],
       [1, 1, 1, 1],
       [1, 1, 2, 2],
       [2, 2, 0, 0],
       [2, 2, 1, 1],
       [2, 2, 2, 2],

       [0, 0, 0, 0],
       [0, 1, 0, 1],
       [0, 2, 0, 2],
       [1, 0, 1, 0],
       [1, 1, 1, 1],
       [1, 2, 1, 2],
       [2, 0, 2, 0],
       [2, 1, 2, 1],
       [2, 2, 2, 2],

       [0, 0, 0, 0],
       [0, 1, 1, 0],
       [0, 2, 2, 0],
       [1, 0, 0, 1],
       [1, 1, 1, 1],
       [1, 2, 2, 1],
       [2, 0, 0, 2],
       [2, 1, 1, 2],
       [2, 2, 2, 2]], dtype=object)