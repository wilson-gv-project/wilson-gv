"""
TODO: to be cleaned up
"""
import copy

from wilson.spectrum import spectrum2D
from CQCParse.parsing import GaussianDataParser, CFOURdataParser
from CQCParse.relay import DataVault

import pandas as pd
import numpy as np
import plotly
import plotly.express as px

# todo: needs to be updated or removed
def get_resonances_DF(computedSpectrum: spectrum2D.Spectrum2D,
                      rec_cm: bool = True,
                      vib_levels_harmonic: bool = False) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Returns dataframes with columns: res (expression), a, b, [c], w_a, w_b, [w_c],
                                     w_2-w_1, w_1, w_2, [FR1, FR2, F_abc], avrg_g

    # Printing pandas options:
    import sys
    pd.set_option('display.max_rows', sys.maxsize)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.width', 5000)
    """
    electrical_terms_dict = dict(zip(computedSpectrum.e_selected, computedSpectrum.electrical_terms))
    mechanical_terms_dict = dict(zip(computedSpectrum.m_selected, computedSpectrum.mechanical_terms))
    w_all = copy.deepcopy(computedSpectrum.all_states_harmonic) if vib_levels_harmonic else copy.deepcopy(computedSpectrum.all_states)
    combos = (computedSpectrum.coords_ab, computedSpectrum.coords_abc)
    w_all[('zero',)] = 0.

    dfs4terms_el = []
    dfs4terms_mech = []
    electrical_terms_avrg_dict = dict(zip(list(electrical_terms_dict.keys()), computedSpectrum.el_avrg_tensors))
    mechanical_terms_avrg_dict = dict(zip(list(mechanical_terms_dict.keys()), computedSpectrum.mech_avrg_tensors))

    for elTerm in electrical_terms_dict:
        subscripts = electrical_terms_dict[elTerm][0]

        m1n1m2n2 = [i.split(',') for i in subscripts]
        letters = ['a', 'b', 'zero']
        dict_df_term = {'ii': elTerm, 'res': '__'.join(subscripts),
                        'a': [], 'b': [],
                        'w_a': [], 'w_b': [],
                        'w_2-w_1': [], 'w_1': [], 'w_2': [],
                        'avrg_g': []}
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
                dict_df_term['w_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['w_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['w_2-w_1'].append(firstres)
                dict_df_term['w_1'].append(secondres)
                dict_df_term['w_2'].append(firstres+secondres)
            else:
                dict_df_term['w_a'].append(spectrum2D.convNu2Ene(w_all[tuple([str(c[0])])]))
                dict_df_term['w_b'].append(spectrum2D.convNu2Ene(w_all[tuple([str(c[1])])]))
                dict_df_term['w_2-w_1'].append(spectrum2D.convNu2Ene(firstres))
                dict_df_term['w_1'].append(spectrum2D.convNu2Ene(secondres))
                dict_df_term['w_2'].append(spectrum2D.convNu2Ene(firstres + secondres))
            dict_df_term['avrg_g'].append(electrical_terms_avrg_dict[elTerm][(c[0], c[1])])
        dfs4terms_el.append(dict_df_term)

    for ii in range(len(dfs4terms_el)):
        dd = pd.DataFrame(data=dfs4terms_el[ii])
        dfs4terms_el[ii] = dd

    for mechTerm in mechanical_terms_dict:
        subscripts, fermi = mechanical_terms_dict[mechTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]
        fermi = [i.split(',') for i in fermi]
        letters = ['a', 'b', 'c', 'zero']
        dict_df_term = {'ii': mechTerm+2, 'res1': '__'.join(mechanical_terms_dict[mechTerm][0]),
                        'res2': '__'.join(mechanical_terms_dict[mechTerm][1]),
                        'a': [], 'b': [], 'c': [],
                        'w_a': [], 'w_b': [], 'w_c': [],
                        'w_2-w_1': [], 'w_1': [], 'w_2': [],
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
                dict_df_term['w_a'].append(w_all[tuple([str(c[0])])])
                dict_df_term['w_b'].append(w_all[tuple([str(c[1])])])
                dict_df_term['w_c'].append(w_all[tuple([str(c[2])])])
                dict_df_term['w_2-w_1'].append(firstres)
                dict_df_term['w_1'].append(secondres)
                dict_df_term['w_2'].append(firstres+secondres)
                dict_df_term['FR1'].append(thirdres)
                dict_df_term['FR2'].append(fourthres)
                dict_df_term['F_abc'].append(computedSpectrum.deriv_data['F_abc'][c[0], c[1], c[2]])

            else:
                dict_df_term['w_a'].append(spectrum2D.convNu2Ene(w_all[tuple([str(c[0])])]))
                dict_df_term['w_b'].append(spectrum2D.convNu2Ene(w_all[tuple([str(c[1])])]))
                dict_df_term['w_c'].append(spectrum2D.convNu2Ene(w_all[tuple([str(c[2])])]))
                dict_df_term['w_2-w_1'].append(spectrum2D.convNu2Ene(firstres))
                dict_df_term['w_1'].append(spectrum2D.convNu2Ene(secondres))
                dict_df_term['w_2'].append(spectrum2D.convNu2Ene(firstres + secondres))
                dict_df_term['FR1'].append(spectrum2D.convNu2Ene(thirdres))
                dict_df_term['FR2'].append(spectrum2D.convNu2Ene(fourthres))
                dict_df_term['F_abc'].append(computedSpectrum.deriv_data['F_abc'][c[0], c[1], c[2]])
            dict_df_term['avrg_g'].append(mechanical_terms_avrg_dict[mechTerm][(c[0], c[1], c[2])])
            # dict_df_term['finalI'].append(dict_df_term['avrg_g']*dict_df_term['F_abc']*(dict_df_term['FR1']+dict_df_term['FR2'])/dict_df_term['FR1']/dict_df_term['FR2'])

        dfs4terms_mech.append(dict_df_term)

    for ii in range(len(dfs4terms_mech)):
        dd = pd.DataFrame(data=dfs4terms_mech[ii])
        dfs4terms_mech[ii] = dd

    return dfs4terms_el, dfs4terms_mech



def analyse_mechanical_resonances(dataframe_mech_resonances, computedSpectrum, rec_cm=True) -> pd.DataFrame:
    frames = []

    for dfMech in dataframe_mech_resonances:
        # how many resonances_args there could be - depends on the number of combinations of a,b,c - combinations_number = Nmodes**3
        # print the expressions for resonances_args of this term
        # print('Resonances:', dfMech['res1'].iloc[0], '\nFormula:', dfMech['res2'].iloc[0], '\n')

        if rec_cm:
            # filters: non-zero F_abc; within the window selected above
            dfMech = dfMech[(dfMech['F_abc'] != 0.) & (dfMech['w_2'] > dfMech['w_1'])
                            & (dfMech['avrg_g'] > 1e-20)
                            ].copy()
        else:
            dfMech = dfMech[(dfMech['F_abc'] != 0.) & (dfMech['w_2'] > dfMech['w_1'])
                            & (dfMech['avrg_g'] > 1e-20)
                            ].copy()

        Gamma = spectrum2D.convNu2Ene(computedSpectrum.Gamma_rc)

        # adding a column for the sum term in product
        if rec_cm:
            dfMech['SoF'] = spectrum2D.convNu2Ene(dfMech['FR1'] + dfMech['FR2']) / spectrum2D.convNu2Ene(dfMech['FR1']) / spectrum2D.convNu2Ene(
                dfMech['FR2'])
            dfMech['Fermi'] = 1. / spectrum2D.convNu2Ene(dfMech['FR1']) / spectrum2D.convNu2Ene(dfMech['FR2'])
            dfMech['abs Fermi'] = abs(1. / spectrum2D.convNu2Ene(dfMech['FR1']) / spectrum2D.convNu2Ene(dfMech['FR2']))
            dfMech['DoR'] = dfMech['SoF'] * (1. / (- 1j * Gamma) / (- 1j * Gamma))
        else:
            # sum of fermi terms
            dfMech['SoF'] = (dfMech['FR1'] + dfMech['FR2']) / dfMech['FR1'] / dfMech['FR2']
            dfMech['Fermi'] = 1. / dfMech['FR1'] / dfMech['FR2']
            dfMech['abs Fermi'] = abs(1. / dfMech['FR1'] / dfMech['FR2'])
            # product of resonance terms
            dfMech['DoR'] = dfMech['SoF'] * (1. / (- 1j * Gamma) / (- 1j * Gamma))
        # dfMech['resonances_args'] is confirmed now, so the rest should be okay too; now confirmed!
        dfMech['gamma_mn'] = dfMech['avrg_g'] * dfMech['F_abc'] * dfMech['DoR'] / computedSpectrum.prefac_3d[
            dfMech['a'], dfMech['b'], dfMech['c']] * (-1.) / 48.
        dd = []
        for idx, row in dfMech.iterrows():
            d = {abs(row['avrg_g']): 'avrg_g',
                 abs(row['F_abc']): 'F_abc',
                 abs(row['DoR']) / abs(row['SoF']): 'res',
                 abs(row['SoF']): 'SoF',
                 computedSpectrum.prefac_3d[row['a'], row['b'], row['c']]: 'pref'}
            dd.append(d[max([abs(row['avrg_g']), abs(row['F_abc']), abs(row['DoR']) / abs(row['SoF']), abs(row['SoF']),
                             computedSpectrum.prefac_3d[row['a'], row['b'], row['c']]])])
        dfMech['distribution'] = dd
        frames.append(dfMech)

    result = pd.concat(frames)
    result['abs'] = abs(result['gamma_mn']) ** 2

    df = result[
        ['ii', 'a', 'b', 'c', 'w_1', 'w_2', 'FR1', 'FR2', 'SoF', 'distribution', 'DoR', 'avrg_g', 'F_abc', 'gamma_mn',
         'abs']]
    formatted_df = df.copy()
    formatted_df['SoF'] = df['SoF'].map('{:.1f}'.format)
    formatted_df['FR1'] = df['FR1'].map('{:.1f}'.format)
    formatted_df['FR2'] = df['FR2'].map('{:.1f}'.format)
    formatted_df['DoR'] = df['DoR'].map('{:.1e}'.format)
    formatted_df['avrg_g'] = df['avrg_g'].map('{:.1e}'.format)
    formatted_df['F_abc'] = df['F_abc'].map('{:.1e}'.format)

    return formatted_df


def analyse_electrical_resonances(dataframe_electric_resonances, computedSpectrum, rec_cm=True) -> pd.DataFrame:
    frames = []

    for dfElectric in dataframe_electric_resonances:
        # how many resonances_args there could be - depends on the number of combinations of a,b - combinations_number = Nmodes**2
        # print the expressions for resonances_args of this term
        # print('Resonance:', dfElectric['res'].iloc[0], '\n')

        if rec_cm:
            # filters: non-zero F_abc; within the window selected above
            dfElectric = dfElectric[(dfElectric['w_2'] > dfElectric['w_1'])
                                    & (dfElectric['avrg_g'] != 0.)
                                    ].copy()
        else:
            dfElectric = dfElectric[(dfElectric['w_2'] > dfElectric['w_1'])
                                    & (dfElectric['avrg_g'] != 0.)
                                    ].copy()

        Gamma = spectrum2D.convNu2Ene(computedSpectrum.Gamma_rc)

        # adding a column for the sum term in product
        if rec_cm:
            dfElectric['DoR'] = 1. / (- 1j * Gamma) / (- 1j * Gamma)
        # else:
        #     dfElectric['SoF'] = (dfElectric['FR1']+dfElectric['FR2'])/dfElectric['FR1']/dfElectric['FR2']
        #     dfElectric['Fermi'] = 1./dfElectric['FR1']/dfElectric['FR2']
        #     dfElectric['abs Fermi'] = abs( 1./dfElectric['FR1']/dfElectric['FR2'])
        #     dfElectric['DoR'] = dfElectric['SoF']*(1./(- 1j * Gamma)/(- 1j * Gamma))

        # dfElectric['resonances_args'] is not yet confirmed
        dfElectric['gamma_mn'] = dfElectric['avrg_g'] * dfElectric['DoR'] / computedSpectrum.prefac_2d[
            dfElectric['a'], dfElectric['b']] / 24.
        dd = []
        for idx, row in dfElectric.iterrows():
            d = {abs(row['avrg_g']): 'avrg_g',
                 # abs(row['DoR'])/abs(row['SoF']): 'res',
                 computedSpectrum.prefac_2d[row['a'], row['b']]: 'pref'}
            dd.append(d[max([abs(row['avrg_g']), computedSpectrum.prefac_2d[row['a'], row['b']]])])
        dfElectric['distribution'] = dd
        frames.append(dfElectric)

    result = pd.concat(frames)
    result['abs'] = abs(result['gamma_mn']) ** 2

    df = result[['ii', 'res', 'a', 'b', 'w_1', 'w_2', 'distribution', 'DoR', 'avrg_g', 'gamma_mn', 'abs']]
    formatted_df = df.copy()
    formatted_df['DoR'] = df['DoR'].map('{:.1e}'.format)
    formatted_df['avrg_g'] = df['avrg_g'].map('{:.1e}'.format)

    return formatted_df


def allResDF(computedSpectrum) -> [pd.DataFrame, spectrum2D.Spectrum2D]:

    # settings
    vib_levels_harmonic = False
    rec_cm = True
    # datain = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    #
    # # spectrum object
    # computedSpectrum = Spectrum2D([], [])
    # computedSpectrum.load_data(GaussianDataParser(datain))
    # computedSpectrum.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=10., vib_levels_harmonic=False)
    # computedSpectrum.add_terms(*terms_selection)
    # if molecule=="ACDM":
    #     computedSpectrum.exclude_modes([34, 35, 36, 37, 38, 39, 40, 41])
    # computedSpectrum.precalculate4fullspectrum()

    # dataframes with resonances_args for used terms in spectrum object
    dfs4terms_el, dfs4terms_mech = get_resonances_DF(computedSpectrum, rec_cm=rec_cm,
                                                              vib_levels_harmonic=vib_levels_harmonic)
    formatted_df2 = analyse_electrical_resonances(dfs4terms_el, computedSpectrum)

    result_later = (
        formatted_df2
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later['term'] = result_later['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later['final'] = abs(result_later['gamma_mn']) ** 2
    result_later['w_1'] = result_later['w_1'].map('{:.3f}'.format)
    result_later['w_2'] = result_later['w_2'].map('{:.3f}'.format)
    result_later['ab_tuples'] = result_later.apply(lambda row: str(list(zip(row['a'], row['b']))), axis=1)
    result_later = result_later.drop(columns=['a', 'b'])
    # max_int = max(result_later['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df1 = result_later.copy()

    formatted_df1['Terms'] = formatted_df1['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    formatted_df4 = analyse_mechanical_resonances(dfs4terms_mech, computedSpectrum)

    # smaller dataframe
    result_later1 = (
        formatted_df4
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'c': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later1['term'] = result_later1['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later1['final'] = abs(result_later1['gamma_mn']) ** 2
    result_later1['w_1'] = result_later1['w_1'].map('{:.3f}'.format)
    result_later1['w_2'] = result_later1['w_2'].map('{:.3f}'.format)

    result_later1['ab_tuples'] = result_later1.apply(lambda row: str(list(zip(row['a'], row['b'], row['c']))), axis=1)
    result_later1 = result_later1.drop(columns=['a', 'b', 'c'])
    # max_int = max(result_later1['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df3 = result_later1.copy()
    # power = float(np.log10(formatted_df3['final'].max())).__floor__()

    formatted_df3['Terms'] = formatted_df3['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    combined_df = pd.concat([formatted_df1, formatted_df3])
    combined_df['gamma_mn_list'] = combined_df['gamma_mn']
    combined_df['gamma_mn'] = np.real(np.copy(combined_df['gamma_mn']))

    result_dfF = (
        combined_df.groupby(['w_1', 'w_2'])
        .agg({
            'Terms': lambda x: list(x),
            'ab_tuples': lambda x: list(x),
            'gamma_mn': 'sum',
            'gamma_mn_list': lambda x: list(x)
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_dfF['ab_tuples'] = result_dfF['ab_tuples'].apply(tuple)
    result_dfF['Terms'] = result_dfF['Terms'].apply(tuple)
    def format_tuple_to_scientific(tpl):
        return tuple(f"{np.real(x):.3e}" for x in tpl)

    result_dfF['gamma_mn_list'] = result_dfF['gamma_mn_list'].apply(format_tuple_to_scientific)

    result_dfF['final_both'] = abs(result_dfF['gamma_mn']) ** 2

    return result_dfF, computedSpectrum

# todo: needs to be updated or removed
def allAddedres(molecule, method, basis, data_vault,
                terms_selection, Gamma_rc, plotHigherThan=1e5) -> [plotly.graph_objs.Figure,
                                                                   pd.DataFrame, spectrum2D.Spectrum2D]:
    # settings
    vib_levels_harmonic = False
    rec_cm = True
    datain = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')

    # spectrum object
    computedSpectrum = spectrum2D.Spectrum2D([], [])
    computedSpectrum.load_data(GaussianDataParser(datain))
    computedSpectrum.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=10., vib_levels_harmonic=False)
    computedSpectrum.add_terms(*terms_selection)
    if molecule=="ACDM":
        computedSpectrum.exclude_modes([34, 35, 36, 37, 38, 39, 40, 41])
    computedSpectrum.precalculate4fullspectrum()

    # dataframes with resonances_args for used terms in spectrum object
    dfs4terms_el, dfs4terms_mech = get_resonances_DF(computedSpectrum, rec_cm=rec_cm,
                                                              vib_levels_harmonic=vib_levels_harmonic)
    formatted_df2 = analyse_electrical_resonances(dfs4terms_el, computedSpectrum)

    result_later = (
        formatted_df2
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later['term'] = result_later['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later['final'] = abs(result_later['gamma_mn']) ** 2
    result_later['w_1'] = result_later['w_1'].map('{:.3f}'.format)
    result_later['w_2'] = result_later['w_2'].map('{:.3f}'.format)
    result_later['ab_tuples'] = result_later.apply(lambda row: str(list(zip(row['a'], row['b']))), axis=1)
    result_later = result_later.drop(columns=['a', 'b'])
    # max_int = max(result_later['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df1 = result_later.copy()

    formatted_df1['Terms'] = formatted_df1['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    formatted_df4 = analyse_mechanical_resonances(dfs4terms_mech, computedSpectrum)
    # print(formatted_df4, '\n-------------')

    # smaller dataframe
    result_later1 = (
        formatted_df4
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'c': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later1['term'] = result_later1['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later1['final'] = abs(result_later1['gamma_mn']) ** 2
    result_later1['w_1'] = result_later1['w_1'].map('{:.3f}'.format)
    result_later1['w_2'] = result_later1['w_2'].map('{:.3f}'.format)

    result_later1['ab_tuples'] = result_later1.apply(lambda row: str(list(zip(row['a'], row['b'], row['c']))), axis=1)
    result_later1 = result_later1.drop(columns=['a', 'b', 'c'])
    # max_int = max(result_later1['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df3 = result_later1.copy()
    # power = float(np.log10(formatted_df3['final'].max())).__floor__()

    formatted_df3['Terms'] = formatted_df3['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    combined_df = pd.concat([formatted_df1, formatted_df3])
    combined_df['gamma_mn_list'] = combined_df['gamma_mn']
    combined_df['gamma_mn'] = np.real(np.copy(combined_df['gamma_mn']))

    # print(combined_df)
    result_dfF = (
        combined_df.groupby(['w_1', 'w_2'])
        .agg({
            'Terms': lambda x: list(x),
            # 'term': lambda x: list(x),
            # 'final': lambda x: list(x),
            'ab_tuples': lambda x: list(x),
            # 'ii': lambda x: list(x),
            'gamma_mn': 'sum',
            'gamma_mn_list': lambda x: list(x)
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_dfF['ab_tuples'] = result_dfF['ab_tuples'].apply(tuple)
    result_dfF['Terms'] = result_dfF['Terms'].apply(tuple)
    # result_dfF['term'] = result_dfF['term'].apply(tuple)
    def format_tuple_to_scientific(tpl):
        # print([x for x in tpl if np.imag(x)!=0.])
        return tuple(f"{np.real(x):.3e}" for x in tpl)

    result_dfF['gamma_mn_list'] = result_dfF['gamma_mn_list'].apply(format_tuple_to_scientific)

    # result_dfF['final'] = result_dfF['final'].apply(tuple)
    result_dfF['final_both'] = abs(result_dfF['gamma_mn']) ** 2
    # print(result_dfF[result_dfF['final_both']>1e4])
    # ('Term2',) ('Term2', 'Term4') ('Term1', 'Term3') ('Term1',)
    symbol_dict = {
        ('Term1',): 'diamond',
        ('Term2',): 'circle',
        ('Term1', 'Term3'): 'star',
        ('Term2', 'Term4'): 'triangle-up'
    }

    color_discrete_map = {
        ('Term1',): 'fuchsia',
        ('Term1', 'Term3'): 'blue',
        ('Term2',): 'green',
        ('Term2', 'Term4'): 'darkorange',
    }

    plt_bgcolor = 'white'
    plt_gridcolor = 'lightgrey'

    import plotly.io as pio
    pio.templates.default = "ggplot2"

    fig1 = px.scatter(result_dfF[(result_dfF['final_both'] > plotHigherThan)
                     ],
                     x='w_1',
                     y='w_2',
                     color_continuous_scale='viridis',
                     title=f'{molecule}/{method}/{basis}: electrical {terms_selection[0]} anharmonicity contribution\nmechanical {terms_selection[1]}',
                     width=1300, height=800,
                     hover_data={'gamma_mn_list': True, 'ab_tuples': True, 'final_both': ':.3e'
                                 },
                    color_discrete_map=color_discrete_map,
                    symbol_map=symbol_dict,
                    color='Terms',
                    symbol='Terms',
                     range_color=(5e4, 0.4e7)
                     )

    fig1.update_layout(
        xaxis_title='w_1',
        yaxis_title='w_2',
        coloraxis_colorbar=dict(title='final'),
        plot_bgcolor=plt_bgcolor
    )

    fig1.update_layout(
        legend=dict(
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor="LightBlue"
        ),
    )

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_xaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
    fig1.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
    fig1.update_coloraxes(colorbar_tickformat='.2e'.format())

    fig1.update_layout(autotypenumbers='convert types')
    fig1.update_traces(marker_size=8)

    # print(result_dfF)
    return fig1, result_dfF, computedSpectrum


# def ELplotly_resonances(molecule, method, basis, terms_selection, plotHigherThan=1e5) -> [plotly.graph_objs.Figure, pd.DataFrame]:
#     # settings
#     datain = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
#     vib_levels_harmonic = False
#     rec_cm = True
#
#     # spectrum object
#     computedSpectrum = Spectrum2D([], [])
#     computedSpectrum.load_data(GaussianDataParser(datain))
#     computedSpectrum.set_spectrum_settings(Gamma_rc=10., diag_margin_rc=10., vib_levels_harmonic=False)
#     computedSpectrum.add_terms(*terms_selection)
#     # computedSpectrum.exclude_modes([34, 35, 36, 37, 38, 39, 40, 41])
#     computedSpectrum.precalculate4fullspectrum()
#
#     # dataframes with resonances_args for used terms in spectrum object
#     dfs4terms_el, dfs4terms_mech = analysis.get_resonances_DF(computedSpectrum, rec_cm=rec_cm,
#                                                               vib_levels_harmonic=vib_levels_harmonic)
#     initialDF = analyse_electrical_resonances(dfs4terms_el, computedSpectrum)
#
#     # smaller dataframe
#     condensed1 = (
#         initialDF
#         .groupby(['w_1', 'w_2'])
#         .agg({
#             'gamma_mn': 'sum',
#             'a': lambda x: list(x),
#             'b': lambda x: list(x),
#             'ii': lambda x: list(set(x))
#         })
#         .reset_index()
#     )
#     condensed1['term'] = condensed1['ii'].apply(lambda x: ', '.join(map(str, x)))
#     condensed1['final'] = abs(condensed1['gamma_mn']) ** 2
#     condensed1['w_1'] = condensed1['w_1'].map('{:.3f}'.format)
#     condensed1['w_2'] = condensed1['w_2'].map('{:.3f}'.format)
#     condensed1['ab_tuples'] = condensed1.apply(lambda row: str(list(zip(row['a'], row['b']))), axis=1)
#     condensed1 = condensed1.drop(columns=['a', 'b'])
#
#     condensedDF2 = condensed1.copy()
#     # power = float(np.log10(formatted_df1['final'].max())).__floor__()
#
#     # second contraction and labeling for the plot
#     condensedDF2['Terms'] = condensedDF2['term'].map({
#         '0': 'Term1',
#         '1': 'Term2',
#         '2': 'Term3',
#         '3': 'Term4',
#         '4': 'Term5',
#         '5': 'Term6'
#     })
#
#     symbol_dict = {
#         'Term1': 'triangle-left',
#         'Term2': 'diamond',
#         'Term3': 'triangle-right',
#         'Term4': 'circle',
#         'Term5': 'star',
#         'Term6': 'triangle-up'
#     }
#
#     color_discrete_map = {
#         'Term1': 'fuchsia',
#         'Term2': 'blue',
#         'Term3': 'green',
#         'Term4': 'darkorange',
#         'Term5': 'purple',
#         'Term6': 'pink'
#     }
#     import plotly.io as pio
#     pio.templates.default = "ggplot2"
#
#     fig1 = px.scatter(condensedDF2[(condensedDF2['final'] > plotHigherThan)
#                       ],
#                       x='w_1',
#                       y='w_2',
#                       color_continuous_scale='viridis',
#                       title=f'{molecule}/{method}/{basis}: electrical {terms_selection[0]} anharmonicity contribution\nmechanical {terms_selection[1]}',
#                       width=1300, height=800,
#                       hover_data={'final': ':.2e', 'ab_tuples': True,
#                                   },
#                       color_discrete_map=color_discrete_map,
#                       symbol_map=symbol_dict,
#                       color='Terms',
#                       symbol='Terms',
#                       range_color=(5e4, 0.4e7)
#                       )
#
#     # plot settings/layout
#     fig1.update_layout(
#         xaxis_title='w_1',
#         yaxis_title='w_2',
#         coloraxis_colorbar=dict(title='final'),
#         plot_bgcolor=plt_bgcolor
#     )
#
#     fig1.update_layout(
#         legend=dict(
#             x=0.5,
#             y=-0.1,
#             xanchor='center',
#             yanchor='top',
#             orientation='h',
#             bgcolor="LightBlue"
#         ),
#     )
#
#     fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#
#     fig1.update_xaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
#     fig1.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
#
#     fig1.update_coloraxes(colorbar_tickformat='.2e'.format())
#
#     fig1.update_layout(autotypenumbers='convert types')
#     fig1.update_traces(marker_size=8)
#
#     return fig1, condensedDF2
#
#
# def MECHplotly_resonances(molecule, method, basis, terms_selection, plotHigherThan=1e5) -> [plotly.graph_objs.Figure, pd.DataFrame]:
#     # settings
#     datain = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
#     vib_levels_harmonic = False
#     rec_cm = True
#
#     # spectrum object
#     computedSpectrum = Spectrum2D([], [])
#     computedSpectrum.load_data(GaussianDataParser(datain))
#     computedSpectrum.set_spectrum_settings(Gamma_rc=10., diag_margin_rc=10., vib_levels_harmonic=False)
#     computedSpectrum.add_terms(*terms_selection)
#     # computedSpectrum.exclude_modes([34, 35, 36, 37, 38, 39, 40, 41])
#     computedSpectrum.precalculate4fullspectrum()
#
#     # dataframes with resonances_args for used terms in spectrum object
#     dfs4terms_el, dfs4terms_mech = analysis.get_resonances_DF(computedSpectrum, rec_cm=rec_cm,
#                                                               vib_levels_harmonic=vib_levels_harmonic)
#     formatted_df2 = analyse_mechanical_resonances(dfs4terms_mech, computedSpectrum)
#
#     # smaller dataframe
#     result_later = (
#         formatted_df2
#         .groupby(['w_1', 'w_2'])
#         .agg({
#             'gamma_mn': 'sum',
#             'a': lambda x: list(x),
#             'b': lambda x: list(x),
#             'c': lambda x: list(x),
#             'ii': lambda x: list(set(x))
#         })
#         .reset_index()
#     )
#     result_later['term'] = result_later['ii'].apply(lambda x: ', '.join(map(str, x)))
#     result_later['final'] = abs(result_later['gamma_mn']) ** 2
#     result_later['w_1'] = result_later['w_1'].map('{:.3f}'.format)
#     result_later['w_2'] = result_later['w_2'].map('{:.3f}'.format)
#     result_later['abc_tuples'] = result_later.apply(lambda row: str(list(zip(row['a'], row['b'], row['c']))), axis=1)
#     result_later = result_later.drop(columns=['a', 'b', 'c'])
#     # max_int = max(result_later['final'])
#     # power = float(np.log10(max_int)).__floor__()
#
#     formatted_df1 = result_later.copy()  # needed??
#     # power = float(np.log10(formatted_df1['final'].max())).__floor__()
#
#     # second contraction and labeling for the plot
#     formatted_df1['Terms'] = formatted_df1['term'].map({
#         '0': 'Term1',
#         '1': 'Term2',
#         '2': 'Term3',
#         '3': 'Term4',
#         '4': 'Term5',
#         '5': 'Term6'
#     })
#
#     symbol_dict = {
#         'Term1': 'triangle-left',
#         'Term2': 'diamond',
#         'Term3': 'triangle-right',
#         'Term4': 'circle',
#         'Term5': 'star',
#         'Term6': 'triangle-up'
#     }
#
#     color_discrete_map = {
#         'Term1': 'fuchsia',
#         'Term2': 'blue',
#         'Term3': 'green',
#         'Term4': 'darkorange',
#         'Term5': 'purple',
#         'Term6': 'pink'
#     }
#
#     fig2 = px.scatter(formatted_df1[(formatted_df1['final'] > plotHigherThan)
#                       ],
#                       x='w_1',
#                       y='w_2',
#                       color_continuous_scale='viridis',
#                       title=f'{molecule}/{method}/{basis}: electrical {terms_selection[0]} anharmonicity contribution\nmechanical {terms_selection[1]}',
#                       width=1300, height=800,
#                       hover_data={'final': ':.2e', 'abc_tuples': True,  # 'a': True, 'b': True, 'c': True
#                                   },
#                       color_discrete_map=color_discrete_map,
#                       symbol_map=symbol_dict,
#                       color='Terms',
#                       symbol='Terms',
#                       range_color=(5e4, 1.5e11))
#
#     # plot settings/layout
#     fig2.update_layout(
#         xaxis_title='w_1',
#         yaxis_title='w_2',
#         coloraxis_colorbar=dict(title='final'),
#         plot_bgcolor=plt_bgcolor
#     )
#
#     fig2.update_layout(
#         legend=dict(
#             x=0.5,
#             y=-0.1,
#             xanchor='center',
#             yanchor='top',
#             orientation='h',
#             bgcolor="LightBlue"
#         ),
#     )
#
#     fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
#     fig2.update_xaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
#     fig2.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor=plt_gridcolor)
#     fig2.update_coloraxes(colorbar_tickformat='.2e'.format())
#     fig2.update_layout(autotypenumbers='convert types')
#     fig2.update_traces(marker_size=13)
#
#     return fig2, formatted_df1


def check_unhashable_columns(df):
    unhashable_columns = []
    for column in df.columns:
        try:
            # Attempt to convert the column to a set
            _ = set(df[column])
        except TypeError:
            # If there is a TypeError, the column contains unhashable types
            unhashable_columns.append(column)
    return unhashable_columns


def find_nearest_index(array, value):
    # if value<np.max(array) and value>np.min(array):
    idx = np.abs(array - value).argmin()
    return idx

def fill_subgrid(grid, seed, radius, grid_size):
    top = max(0, seed[0] - radius)
    bottom = min(grid_size[0], seed[0] + radius + 1)
    left = max(0, seed[1] - radius)
    right = min(grid_size[1], seed[1] + radius + 1)
    grid[top:bottom, left:right] += 1


def get_where_matrix(computedSpectrum, radius):

    df_edata, spec1 = allResDF(computedSpectrum)
    # print(df_edata)
    df_edata['list_points_grid'] = list(zip(round(df_edata['w_1'].astype(float)),
                                            round(df_edata['w_2'].astype(float))))
    df_edata['list_points'] = list(zip(df_edata['w_1'].astype(float), df_edata['w_2'].astype(float)))
    seeds_freq = [list(i) for i in df_edata['list_points_grid']]

    seeds_indices = [(find_nearest_index(spec1.w1, seed[0]),
                      find_nearest_index(spec1.w2, seed[1])) for seed in
                     seeds_freq]

    grid_size = (len(spec1.w1), len(spec1.w2))
    grid = np.zeros(grid_size, dtype=int)

    for seed in seeds_indices:
        fill_subgrid(grid, seed, radius, grid_size)

    return grid


def get_where_matrix1seed(computedSpectrum, radius, seed):

    seeds_indices = tuple([(find_nearest_index(computedSpectrum.w1, seed[0]),
                            find_nearest_index(computedSpectrum.w2, seed[1]))])

    grid_size = (len(computedSpectrum.w1), len(computedSpectrum.w2))
    grid = np.zeros(grid_size, dtype=int)

    for seed in seeds_indices:
        fill_subgrid(grid, seed, radius, grid_size)

    return grid

def get_abctuples_res(computedSpectrum):
    df_edata, spec1 = allResDF(computedSpectrum)
    print(df_edata)
    alltuples = []
    for i in df_edata['ab_tuples']:
        for j in i:
            count = j.count(')')
            if count==1:
                it = tuple([int(k) for k in j.strip('[()]').split(',')])
                if it not in alltuples:
                    alltuples.append(it)
            else:
                it = tuple([k.strip('()').strip(' (') for k in j.strip('[]').split('),') ]) # for y in k
                ti = tuple([int(g) for g in f.split(',')] for f in it)
                for r in ti:
                    if r not in alltuples:
                        alltuples.append(tuple(r))
    print(len(alltuples))
    print(alltuples)
