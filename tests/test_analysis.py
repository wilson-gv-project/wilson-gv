import numpy as np
from wilson.dataanalysis import *
from wilson import spectrum


def test_get_resonances_electrical():
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]

    omega1 = np.arange(2810., 3210., 10.)
    omega2 = np.arange(5510., 6050., 10.)
    datain = spectrum.make_DatainputDict('gaussian', ('FOAC', 'HF', 'cc_pVQZ'))
    vib_levels_harmonic = True
    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain,
                                            vib_levels_harmonic=vib_levels_harmonic)
    computedSpectrum.addTerms(*terms_selection)

    print('------------------------\n')
    dfs4terms_el, dfs4terms_mech = get_resonances(computedSpectrum, rec_cm=True,
                                                  vib_levels_harmonic=vib_levels_harmonic)
    import matplotlib.pyplot as plt
    for dfEL in dfs4terms_el:
        # print(dfEL)
        pass

def test_get_resonances_mechanical():
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]

    omega1 = np.arange(2810., 3210., 10.)
    omega2 = np.arange(5510., 6050., 10.)
    datain = spectrum.make_DatainputDict('gaussian', ('FOAC', 'HF', 'cc_pVQZ'))
    vib_levels_harmonic = True
    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain,
                                            vib_levels_harmonic=vib_levels_harmonic)
    computedSpectrum.addTerms(*terms_selection)

    print('------------------------\n')
    dfs4terms_el, dfs4terms_mech = get_resonances(computedSpectrum, rec_cm=True,
                                                  vib_levels_harmonic=vib_levels_harmonic)

    import matplotlib.pyplot as plt

    for dfMech in dfs4terms_mech:
        print('Original length', len(dfMech))
        print('Resonances:', dfMech['res1'].iloc[0], '\nSum of 2:', dfMech['res2'].iloc[0])
        # dfMech = dfMech[(dfMech['F_abc'] != 0.) & (abs(dfMech['ω_2']) > 1600.) & (
        #             abs(dfMech['ω_2']) < 4500.) & (abs(dfMech['ω_1']) > 750.) & (abs(dfMech['ω_1']) < 1650.)]
        # print(dfMech[(dfMech['F_abc'] != 0.) & (abs(dfMech['FR2']) < 1000.) & (abs(dfMech['ω_2']) > 1600.) & (
        #             abs(dfMech['ω_2']) < 4500.) & (abs(dfMech['ω_1']) > 750.) & (abs(dfMech['ω_1']) < 1650.)])
        dfMech['(sum/prod)_FR'] = abs((dfMech['FR1']+dfMech['FR2'])/dfMech['FR1']*dfMech['FR2'])
        dfMech['prod_FR'] = abs(dfMech['FR1']*dfMech['FR2'])
        dfMech['abs(FR1)'] = abs(dfMech['FR1'])
        dfMech['abs(FR2)'] = abs(dfMech['FR2'])
        # small1 = dfMech[(dfMech['F_abc'] != 0.)&(dfMech['(sum/prod)_FR']!=0.)].nsmallest(5, 'prod_FR')
        # small2 = dfMech[dfMech['F_abc'] != 0.].nsmallest(5, 'sum_FR')
        # small1 = dfMech[(dfMech['F_abc'] != 0.) & (dfMech['sum_FR'] == 0.)]
        # small2 = dfMech[(dfMech['F_abc'] != 0.) & (dfMech[dfMech['sum_FR'] == 0.])]
        # print(small1.loc[:, ~small1.columns.isin(['abs(FR1)', 'abs(FR2)', 'res1', 'res2'])])
        # print('New length', len(small1))

        # print(small2.loc[:, ~small2.columns.isin(['abs(FR1)', 'abs(FR2)'])])
        print(dfMech.drop(['abs(FR1)', 'abs(FR2)', 'res1', 'res2'], axis=1))
        print('---------------')
        # plt.plot(dfMech['(sum/prod)_FR'], 'bo')
        # plt.bar(range(len(dfMech['(sum/prod)_FR'])), dfMech['(sum/prod)_FR'], 1.)
        # plt.title(f'(sum/prod)_FR: Resonances: {dfMech['res1'].iloc[0]} | Sum of 2: {dfMech['res2'].iloc[0]}')
        # plt.show()

        # plt.bar(range(len(dfMech['avrg_g'])), dfMech['avrg_g'], 1.)
        # plt.title(f'avrg_g: Resonances: {dfMech['res1'].iloc[0]} | Sum of 2: {dfMech['res2'].iloc[0]}')
        # plt.show()

def test_get_avrg_tensors():
    """
    """
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]
    omega1 = np.arange(2810., 3210., 10.)
    omega2 = np.arange(5510., 6050., 10.)
    datain = spectrum.make_DatainputDict('gaussian', ('FOAC', 'HF', 'cc_pVQZ'))

    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain, vib_levels_harmonic=False)
    computedSpectrum.addTerms(*terms_selection)

    print('\n-------------------')
    for et in computedSpectrum.el_avrg_tensors:
        print(et.shape)
        print(et)
    print('-------------------')
    for mt in computedSpectrum.mech_avrg_tensors:
        print(mt.shape)
        print(mt)
    print('\ncubic force constants')
    print(computedSpectrum.deriv_data['F_abc'])


def test_get_El2Mech_ratio():
    method = ('FOAC', 'HF', 'cc_pVQZ')
    datain = spectrum.make_DatainputDict('gaussian', method)
    settings_here = {'electrical': None, 'mechanical': None,
                     'Gamma_rc': 10., 'region': 5,
                     'font_dict': {'size': 18}, 'figsize': (12, 15)}

    el_percnt, mech_percnt = get_El2Mech_ratio(datain, settings_here, vibEL=True)
    print('\n-------------')
    print(el_percnt)

    print('\n-------------')
    print(mech_percnt)