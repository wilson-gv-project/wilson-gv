import copy

import numpy as np
np.set_printoptions(legacy='1.25')

from wilson.spectrum.spectrum2D import (convNu2Ene, avrg_abc_tensor,
                                        Spectrum2D, combinations_with_permutations)
from CQCParse.parsing import GaussianDataParser
from CQCParse.relay import DataVault

data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')

omega1 = np.arange(1130., 2050., 2.91)
omega2 = np.arange(1300., 5150., 2.91)

molecule = 'FORM' # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
method = 'B3LYP'
basis = 'cc_pVQZ'
Gamma_rc = 5.1
diag_margin_rc=3.
list2exclude = []
terms_selection = [0,1], [0,1]

datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
gParser = GaussianDataParser(datadict)

dictInputs = {'parserObject': gParser,
              'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

spectrumObj = Spectrum2D(omega1, omega2)
spectrumObj.load_data(dictInputs['parserObject'], vpt2=False)

spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=diag_margin_rc, vib_levels_harmonic=True)
# currently requires diag_margin_rc attribute to be set
spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
spectrumObj.precalculateParts(list2exclude=list2exclude)



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



def test_precalculate_parts():
    vib_ene_levels_harmonic = convNu2Ene(np.array([v for k, v in spectrumObj.fundamentals_harmonic.items()]))
    a, b, c = 0, 1, 1
    assert spectrumObj.prefac_2d[a,b] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b]
    assert spectrumObj.prefac_3d[a,b,c] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b] * vib_ene_levels_harmonic[c]
    a, b, c = 2, 1, 1
    assert spectrumObj.prefac_2d[a, b] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b]
    assert spectrumObj.prefac_3d[a,b,c] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b] * vib_ene_levels_harmonic[c]
    a, b, c = 1, 3, 2
    assert (spectrumObj.prefac_2d[a,b] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b]
            == vib_ene_levels_harmonic[b] * vib_ene_levels_harmonic[a])
    assert spectrumObj.prefac_3d[a,b,c] == vib_ene_levels_harmonic[a] * vib_ene_levels_harmonic[b] * vib_ene_levels_harmonic[c]

    enelev = spectrumObj.all_states_harmonic_Eh
    a, b = 0, 1
    assert spectrumObj.w_mn_dict['a+b,a'][a,b] == enelev[(str(a), str(b))] - enelev[(str(a),)]
    assert spectrumObj.w_mn_dict['b,a'][a,b] == enelev[(str(b),)] - enelev[(str(a),)]
    assert spectrumObj.w_mn_dict['zero,a'][a,b] == enelev[('zero',)] - enelev[(str(a),)]
    a, b = 2, 1
    assert spectrumObj.w_mn_dict['a+b,a'][a,b] == enelev[(str(b), str(a))] - enelev[(str(a),)]
    assert spectrumObj.w_mn_dict['b,a'][a,b] == enelev[(str(b),)] - enelev[(str(a),)]
    assert spectrumObj.w_mn_dict['zero,a'][a,b] == enelev[('zero',)] - enelev[(str(a),)]



def test_generate_resonances_functions():
    elterm = (('a+b,a', 'zero,a'), None)
    elavrg = (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)))
    func = spectrumObj.generate_resonances_functions(elterm[0], elterm[1])
    avrgT = avrg_abc_tensor(elavrg, spectrumObj.deriv_data, spectrumObj.gammaCompsAll)

    a, b = 0, 2
    w1, w2 = 1343., 2574.
    allLevels_Eh = spectrumObj.all_states_harmonic_Eh
    w_res_dict = {(-1, 2): np.array([w1-w2]) - 1j * spectrumObj.Gamma,
                  (-1,): np.array([w1]) - 1j * spectrumObj.Gamma}
    intensity = avrgT[a, b] * func(allLevels_Eh=allLevels_Eh,
                                   w_res_dict=w_res_dict, abctuple=(a,b),
                                   w1w2Condition=np.array([True])) / spectrumObj.prefac_2d[a, b] / 24.
    resonance = 1. / (spectrumObj.w_mn_dict['a+b,a'][a,b]+w_res_dict[(-1,2)]) / (spectrumObj.w_mn_dict['zero,a'][a,b]+w_res_dict[(-1,)])
    reference = avrgT[a, b] * resonance / spectrumObj.prefac_2d[a, b] / 24.
    assert np.allclose(reference, intensity, atol=1e-16)

    PQ = w_res_dict[(-1, 2)] * w_res_dict[(-1,)]
    pqs = {('a+b,a', 'zero,a'): spectrumObj.w_mn_dict['a+b,a'] * spectrumObj.w_mn_dict['zero,a'],
           ('b,a', 'zero,a'): spectrumObj.w_mn_dict['b,a'] * spectrumObj.w_mn_dict['zero,a']}
    pQ = spectrumObj.w_mn_dict['a+b,a'][a,b] * w_res_dict[(-1,)]
    qP = spectrumObj.w_mn_dict['zero,a'][a,b] * w_res_dict[(-1,2)]
    res_opt = 1. / (pqs[elterm[0]][a,b] + pQ + qP +PQ)
    assert np.allclose(res_opt, resonance, atol=1e-16)

    mechterm = (('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b'))
    mechavrg = (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.)
    # func2 = spectrumObj.generate_resonances_functions(mechterm[0], mechterm[1])
    avrgT2 = avrg_abc_tensor(mechavrg, spectrumObj.deriv_data, spectrumObj.gammaCompsAll)

    combfactor = spectrumObj.comb_fac_dict[(mechterm, mechavrg)][a,b]
    ref_combfac = np.zeros((spectrumObj.nmodes,))
    # careful here with spectrumObj.mode_indices
    total_cs = 0.
    for c in spectrumObj.mode_indices:
        abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
        ijk_indx = tuple([abc[j] for j in mechavrg[-2]])
        F1 = spectrumObj.deriv_data['F_abc'][ijk_indx]
        F2 = spectrumObj.deriv_data['F_abc'][a, b, c]
        assert F1==F2

        freqDiff = [i.split(',') for i in mechterm[1]]
        letters = ['a', 'b', 'c', 'zero']
        dictabc = dict(zip(letters, (a, b, c) + tuple(['zero'])))
        allLevels_Eh[('zero',)] = 0.

        wr_fr11 = tuple(sorted([str(i) for i in (a,b,c)], key=int))
        wr_fr21 = ('zero',)
        wr_fr12 = (str(c),)
        wr_fr22 = tuple(sorted([str(i) for i in (a,b)], key=int))

        w_fr11 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][0].split('+')], key=int))
        if 'zero' not in freqDiff[0][1]:
            w_fr21 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][1].split('+')], key=int))
        else:
            w_fr21 = tuple([freqDiff[0][1]])
        w_fr12 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][0].split('+')], key=int))
        if 'zero' not in freqDiff[1][1]:
            w_fr22 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][1].split('+')], key=int))
        else:
            w_fr22 = tuple([freqDiff[1][1]])

        assert wr_fr11==w_fr11
        assert wr_fr12==w_fr12
        assert wr_fr21==w_fr21
        assert wr_fr22==w_fr22

        t3 = allLevels_Eh[wr_fr11] - allLevels_Eh[wr_fr21]
        t4 = allLevels_Eh[wr_fr12] - allLevels_Eh[wr_fr22]
        sumfrac = (1 / t3 + 1 / t4)
        ref_combfac[c] = avrgT2[a,b,c] * F1 * sumfrac / spectrumObj.prefac_3d[a,b,c] / (-48.)
        total_cs += avrgT2[a,b,c] * F1 * sumfrac / spectrumObj.prefac_3d[a,b,c] / (-48.)
    # print(ref_combfac, np.sum(ref_combfac), total_cs, combfactor)
    assert combfactor==np.sum(ref_combfac)


def test_compute_mech_factors():

    data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    # [(('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b'))]
    # [(('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.0)]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)
    spectrumObj = Spectrum2D(np.arange(1130., 2050., 90.), np.arange(1300., 5150., 90.))
    spectrumObj.load_data(gParser, vpt2=False)
    spectrumObj.setSpectrumSettings(Gamma_rc=5., diag_margin_rc=3., vib_levels_harmonic=False)
    spectrumObj.addTerms([], [2])
    spectrumObj.precalculateParts(list2exclude=[])


    vib_ene_levels = spectrumObj.all_states_Eh
    vib_ene_levels_h = spectrumObj.all_states_harmonic_Eh
    averaging_tens = spectrumObj.avrg_tensors_dict[2]

    F = spectrumObj.deriv_data['F_abc']

    a, b = 2, 1
    reference = 0.
    prefac_ab = 1./(vib_ene_levels_h[(str(a),)]*vib_ene_levels_h[(str(b),)])
    assert 1./spectrumObj.prefac_2d[a, b] == prefac_ab

    for c in spectrumObj.mode_indices:
        # ('a+b+c,zero', 'c,a+b')
        assert np.isclose(1./spectrumObj.prefac_3d[a, b, c] , prefac_ab / vib_ene_levels_h[(str(c),)])

        freqDiff = (1./vib_ene_levels[tuple([str(i) for i in sorted([a,b,c])])]
                    + 1./(vib_ene_levels[(str(c),)] - vib_ene_levels[tuple([str(i) for i in sorted([a,b])])]))
        reference += averaging_tens[a,b,c] * F[a,b,c] * freqDiff * prefac_ab / vib_ene_levels_h[(str(c),)] / (-48)

    from_factors = spectrumObj.compute_mech_factors(a, b)
    assert np.isclose(reference , from_factors[0])

    a, b = 4, 5
    reference = 0.
    prefac_ab = 1./(vib_ene_levels_h[(str(a),)]*vib_ene_levels_h[(str(b),)])
    assert 1./spectrumObj.prefac_2d[a, b] == prefac_ab

    for c in spectrumObj.mode_indices:
        # ('a+b+c,zero', 'c,a+b')
        assert np.isclose(1./spectrumObj.prefac_3d[a, b, c] , prefac_ab / vib_ene_levels_h[(str(c),)])

        freqDiff = (1./vib_ene_levels[tuple([str(i) for i in sorted([a,b,c])])]
                    + 1./(vib_ene_levels[(str(c),)] - vib_ene_levels[tuple([str(i) for i in sorted([a,b])])]))
        reference += averaging_tens[a,b,c] * F[a,b,c] * freqDiff * prefac_ab / vib_ene_levels_h[(str(c),)] / (-48)

    from_factors = spectrumObj.compute_mech_factors(a, b)
    assert np.isclose(reference , from_factors[0])

    for ab in combinations_with_permutations(spectrumObj.mode_indices, 2):

        a, b = ab
        reference = 0.
        prefac_ab = 1./(vib_ene_levels_h[(str(a),)]*vib_ene_levels_h[(str(b),)])
        assert 1./spectrumObj.prefac_2d[a, b] == prefac_ab

        for c in spectrumObj.mode_indices:
            # ('a+b+c,zero', 'c,a+b')
            assert np.isclose(1./spectrumObj.prefac_3d[a, b, c] , prefac_ab / vib_ene_levels_h[(str(c),)])

            freqDiff = (1./vib_ene_levels[tuple([str(i) for i in sorted([a,b,c])])]
                        + 1./(vib_ene_levels[(str(c),)] - vib_ene_levels[tuple([str(i) for i in sorted([a,b])])]))
            reference += averaging_tens[a,b,c] * F[a,b,c] * freqDiff * prefac_ab / vib_ene_levels_h[(str(c),)] / (-48)

        # from_factors = spectrumObj.compute_mech_factors(a, b)[0]
        # print(from_factors)
        from_factors = spectrumObj.comb_fac_dict[((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                                                (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)),
                                                 'abc', 1.0))][a, b]
        # print(from_factors)

        assert np.isclose(reference , from_factors)

def test_get_gamma_mech():
    data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    # [(('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b'))]
    # [(('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.0)]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)
    spectrumObj = Spectrum2D(np.array([1130., 2050., 2190.]), np.array([1300., 3150., 4590.]))
    spectrumObj.load_data(gParser, vpt2=False)
    spectrumObj.setSpectrumSettings(Gamma_rc=5., diag_margin_rc=3., vib_levels_harmonic=False)
    spectrumObj.addTerms([], [0])
    spectrumObj.precalculateParts(list2exclude=[], preview=False, screenmodeswindow=True)

    vib_ene_levels = copy.deepcopy(spectrumObj.all_states_Eh)
    # vib_ene_levels_h = spectrumObj.all_states_harmonic_Eh

    selectionCond = np.ones(spectrumObj.w1w2Condition.shape, dtype=bool)
    condition = (spectrumObj.w1w2Condition & selectionCond)

    resonances_args = {}
    for typelist in [(-1, 2), (-1,)]:
        resonances_args[typelist] = (-1) * sum([np.sign(ix) * np.where(spectrumObj.w1w2Condition,
                                                                       spectrumObj.axes[abs(ix)], 0) for ix in
                                                typelist]) - 1j * spectrumObj.Gamma

    # for ab in [(0, 3,), (2, 2), (3, 1)]:
    for ab in combinations_with_permutations(spectrumObj.mode_indices, 2):
        a, b = ab
        # testing this
        # mechfactor = spectrumObj.compute_mech_factors(a, b)[0]
        mechfactor = spectrumObj.comb_fac_dict[((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                                                (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)),
                                                 'abc', 1.0))][a, b]
        # print(spectrumObj.w_mn_dict)
        wmnab1 = spectrumObj.w_mn_dict['a+b,a'][a,b]
        wmnab2 = spectrumObj.w_mn_dict['zero,a'][a,b]
        assert wmnab1 == vib_ene_levels[tuple([str(i) for i in sorted([a,b])])] - vib_ene_levels[(str(a),)]
        assert wmnab2 == - vib_ene_levels[(str(a),)]
        # testing this
        resonance = spectrumObj.m_funcs[0](allLevels_Eh=vib_ene_levels,
                                           w_res_dict=resonances_args, abctuple=(a, b),
                                           w1w2Condition=condition)
        # res2 = np.where(condition, 1. / (spectrumObj.axes[1] - spectrumObj.axes[2] +
        #         vib_ene_levels[tuple([str(i) for i in sorted([a,b])])] - vib_ene_levels[(str(a),)] - 1j * spectrumObj.Gamma)
        #                 / (spectrumObj.axes[1] - vib_ene_levels[(str(a),)] - 1j * spectrumObj.Gamma), 0.+0.j)
        res2 = np.where(condition, 1. / (spectrumObj.axes[1] - spectrumObj.axes[2] + wmnab1 - 1j * spectrumObj.Gamma)
                    / (spectrumObj.axes[1] + wmnab2 - 1j * spectrumObj.Gamma), 0. + 0.j)

        assert np.allclose(resonance, res2)

        full = np.where(condition, mechfactor * res2, 0.+0.j)

        spectrumObj.intensities_grid = np.zeros(spectrumObj.shape2d, dtype='complex64')
        spectrumObj.get_gamma_mech(a, b, condition)
        ints = spectrumObj.intensities_grid

        assert np.allclose(ints , full)


# def test_calc_averaging():
#
#     # gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
#     # [alpha, beta, gamma, delta]
#     gammaCompsAll = [[2, 2, 0, 0],
#                      [1, 0, 0, 1],
#                      [1, 2, 1, 2]]
#
#     from DATA import mu_Q, mu_QQ, alpha_Q, alpha_QQ
#     data = {'mu_Q': mu_Q, 'mu_QQ': mu_QQ, 'alpha_Q': alpha_Q, 'alpha_QQ': alpha_QQ}
#
#     term_m = EvalTerm(avrg=[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))], prefac='abc')
#     avrgF_m = term_m.calcAveraging(data=data, gammaCompsAll=gammaCompsAll)
#     # nmodes = 4
#     a,b,c = (1, 3, 0)
#     # ((beta,), (alpha,delta), (gamma))
#     ref_avrgF_m1 = (data['mu_Q'][a, 2] * data['alpha_Q'][b, 2, 0] * data['mu_Q'][c, 0]+
#                     data['mu_Q'][a, 0] * data['alpha_Q'][b, 1, 1] * data['mu_Q'][c, 0]+
#                     data['mu_Q'][a, 2] * data['alpha_Q'][b, 1, 2] * data['mu_Q'][c, 1])
#     assert ref_avrgF_m1==avrgF_m[a,b,c]
#
#     term_e = EvalTerm(avrg=[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))], prefac='ab')
#     avrgF_e = term_e.calcAveraging(data=data, gammaCompsAll=gammaCompsAll)
#
#     # final tensor will have dimensions (nmodes, nmodes) or (nmodes, nmodes, nmodes)
#     #           [a, b] for el. and [a, b, c] for mech. (c=a or c=b sometimes)
#     # element of the final tensor is a sum of products according to
#     #                     the gammaCompsAll array of xyz-indices for each greek index: [alpha, beta, gamma, delta]
#     # one product per one array from the gammaCompsAll array:
#     #   derT1[nmodes..., beta_val]*derT2[nmodes..., alpha_val, delta_val]*derT3[nmodes..., gamma_val] where
#     #         nmodes..., marks the derivative dimensions, and greek indices refer to cartesian dimensions
#
#     print('\n')
#     np.set_printoptions(precision=4)
#
#     # print('\n', avrgF_m)
#     # print('\n', avrgF_e)

