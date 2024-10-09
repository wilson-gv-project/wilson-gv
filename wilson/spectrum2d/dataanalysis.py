import numpy as np
from wilson import spectrum
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 2000)
pd.set_option('display.width', 5000)

def get_resonances_DF(computedSpectrum, rec_cm=True, vib_levels_harmonic=False):
    """
    Returns dataframes with columns: res (expression), a, b, [c], ω_a, ω_b, [ω_c],
                                     ω_2-ω_1, ω_1, ω_2, [FR1, FR2, F_abc], avrg_g

    """
    electrical_terms_dict = dict(zip(computedSpectrum.ee, computedSpectrum.electrical_terms))
    mechanical_terms_dict = dict(zip(computedSpectrum.mm, computedSpectrum.mechanical_terms))
    w_all = computedSpectrum.all_states_harmonic if vib_levels_harmonic else computedSpectrum.all_states
    combos = (computedSpectrum.coords_ab, computedSpectrum.coords_abc)
    # nfunds = len([i for i in w_all if len(i) == 1])
    w_all[('zero',)] = 0.

    dfs4terms_el = []
    dfs4terms_mech = []
    electrical_terms_avrg_dict = dict(zip(list(electrical_terms_dict.keys()), computedSpectrum.el_avrg_tensors))
    mechanical_terms_avrg_dict = dict(zip(list(mechanical_terms_dict.keys()), computedSpectrum.mech_avrg_tensors))

    for elTerm in electrical_terms_dict:
        subscripts = electrical_terms_dict[elTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]
        letters = ['a', 'b', 'zero']
        dict_df_term = {'res': '__'.join(electrical_terms_dict[elTerm]),
                        'a': [], 'b': [],
                        'ω_a': [], 'ω_b': [], 'ω_2-ω_1': [], 'ω_1': [], 'ω_2': [], 'avrg_g': []}
        for c in combos[0]:
            dictabc = dict(zip(letters, tuple(c) + tuple(['zero'])))

            wm1 = m1n1m2n2[0][0].split('+')
            wn1 = m1n1m2n2[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            firstres = 1*(w_all[key_m] - w_all[key_n])

            wm1 = m1n1m2n2[1][0].split('+')
            wn1 = m1n1m2n2[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            secondres = -1*(w_all[key_m] - w_all[key_n])

            dict_df_term['a'].append(c[0])
            dict_df_term['b'].append(c[1])

            if rec_cm:
                dict_df_term['ω_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['ω_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['ω_2-ω_1'].append(firstres)
                dict_df_term['ω_1'].append(secondres)
                dict_df_term['ω_2'].append(firstres+secondres)
            else:
                dict_df_term['ω_a'].append(spectrum.rec_cm2rec_s(w_all[tuple([str(c[0])])]))
                dict_df_term['ω_b'].append(spectrum.rec_cm2rec_s(w_all[tuple([str(c[1])])]))
                dict_df_term['ω_2-ω_1'].append(spectrum.rec_cm2rec_s(firstres))
                dict_df_term['ω_1'].append(spectrum.rec_cm2rec_s(secondres))
                dict_df_term['ω_2'].append(spectrum.rec_cm2rec_s(firstres+secondres))
            dict_df_term['avrg_g'].append(electrical_terms_avrg_dict[elTerm][*c])
        dfs4terms_el.append(dict_df_term)

    for ii in range(len(dfs4terms_el)):
        dd = pd.DataFrame(data=dfs4terms_el[ii])
        dfs4terms_el[ii] = dd

    for index, mechTerm in enumerate(mechanical_terms_dict):
        subscripts, fermi = mechanical_terms_dict[mechTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]
        fermi = [i.split(',') for i in fermi]
        letters = ['a', 'b', 'c', 'zero']
        dict_df_term = {'ii': index, 'res1': '__'.join(mechanical_terms_dict[mechTerm][0]),
                        'res2': '__'.join(mechanical_terms_dict[mechTerm][1]),
                        'a': [], 'b': [], 'c': [],
                        'ω_a': [], 'ω_b': [], 'ω_c': [], 'ω_2-ω_1': [], 'ω_1': [], 'ω_2': [],
                        'FR1': [], 'FR2': [], 'F_abc': [], 'avrg_g': []}
        for c in combos[1]:
            dictabc = dict(zip(letters, tuple(c) + tuple(['zero'])))

            wm1 = m1n1m2n2[0][0].split('+')
            wn1 = m1n1m2n2[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            firstres = 1*(w_all[key_m] - w_all[key_n])

            wm1 = m1n1m2n2[1][0].split('+')
            wn1 = m1n1m2n2[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            secondres = -1*(w_all[key_m] - w_all[key_n])

            wm1 = fermi[0][0].split('+')
            wn1 = fermi[0][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            thirdres = 1*(w_all[key_m] - w_all[key_n])

            wm1 = fermi[1][0].split('+')
            wn1 = fermi[1][1].split('+')
            key_m = tuple(sorted([str(dictabc[i]) for i in wm1], key=int)) if 'zero' not in wm1 else tuple(wm1)
            key_n = tuple(sorted([str(dictabc[i]) for i in wn1], key=int)) if 'zero' not in wn1 else tuple(wn1)

            fourthres = 1*(w_all[key_m] - w_all[key_n])

            dict_df_term['a'].append(c[0])
            dict_df_term['b'].append(c[1])
            dict_df_term['c'].append(c[2])

            if rec_cm:
                dict_df_term['ω_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['ω_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['ω_c'].append(w_all[tuple([str(c[2])])])
                dict_df_term['ω_2-ω_1'].append(firstres)
                dict_df_term['ω_1'].append(secondres)
                dict_df_term['ω_2'].append(firstres+secondres)
                dict_df_term['FR1'].append(thirdres)
                dict_df_term['FR2'].append(fourthres)
                dict_df_term['F_abc'].append(computedSpectrum.deriv_data['F_abc'][c[0], c[1], c[2]])

            else:
                dict_df_term['ω_a'].append(spectrum.rec_cm2rec_s(w_all[tuple([str(c[0])])]))
                dict_df_term['ω_b'].append(spectrum.rec_cm2rec_s(w_all[tuple([str(c[1])])]))
                dict_df_term['ω_c'].append(spectrum.rec_cm2rec_s(w_all[tuple([str(c[2])])]))
                dict_df_term['ω_2-ω_1'].append(spectrum.rec_cm2rec_s(firstres))
                dict_df_term['ω_1'].append(spectrum.rec_cm2rec_s(secondres))
                dict_df_term['ω_2'].append(spectrum.rec_cm2rec_s(firstres+secondres))
                dict_df_term['FR1'].append(spectrum.rec_cm2rec_s(thirdres))
                dict_df_term['FR2'].append(spectrum.rec_cm2rec_s(fourthres))
                dict_df_term['F_abc'].append(computedSpectrum.deriv_data['F_abc'][c[0], c[1], c[2]])
            dict_df_term['avrg_g'].append(mechanical_terms_avrg_dict[mechTerm][*c])
            # dict_df_term['finalI'].append(dict_df_term['avrg_g']*dict_df_term['F_abc']*(dict_df_term['FR1']+dict_df_term['FR2'])/dict_df_term['FR1']/dict_df_term['FR2'])

        dfs4terms_mech.append(dict_df_term)

    for ii in range(len(dfs4terms_mech)):
        dd = pd.DataFrame(data=dfs4terms_mech[ii])
        dfs4terms_mech[ii] = dd

    return dfs4terms_el, dfs4terms_mech

def get_El2Mech_ratio(datainput, settings, vibEL):
    """
    """
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]

    regions = {1: ((1180., 2050., 10.), (2309., 5350., 10.)),
               2: ((2810., 3210., 10.), (5510., 6050., 10.)),
               3: ((1961.318, 1981.318, 10.), (4931.662, 4951.662, 10.)),
               4: ((680., 1750., 10.), (1509., 4350., 10.)),
               5: ((1800, 1900, 20.), (2300, 2400, 20.))}

    region = settings['region']
    Gamma_rc = settings['Gamma_rc']
    Gamma = spectrum.rec_cm2rec_s(Gamma_rc)

    omega1 = np.arange(*regions[region][0])
    omega2 = np.arange(*regions[region][1])

    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datainput, vib_levels_harmonic=vibEL)
    computedSpectrum.addTerms(*terms_selection)

    el_gamma = computedSpectrum.intensity_electrical(Gamma)
    mech_gamma = computedSpectrum.intensity_mechanical(Gamma)

    total = abs(el_gamma+mech_gamma)**2
    print('\ntotal\n', total)
    print('\n|el_gamma|**2\n', abs(el_gamma)**2)
    print('\n|mech_gamma|**2\n', abs(mech_gamma)**2)
    # percent_el = abs(el_gamma)**2/total*100
    # percent_mech = abs(mech_gamma)**2/total*100

    print('\n|el_gamma|**2/|mech_gamma|**2\n', abs(el_gamma)**2/abs(mech_gamma)**2)

    return abs(el_gamma)**2/abs(mech_gamma)**2