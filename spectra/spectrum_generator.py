#!/usr/bin/env python
import time
start_time_global = time.time()
import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=17)
import os

from wilson import spectrum

def one_spectrum_fig(el: bool, mech: bool, datain: dict, region: int = 3, gamma_rc: float = 10.):
    """
        # one_spectrum_fig(el=el, mech=mech, datain=datain, region=1, gamma_rc=broad_factor_rc)
    """
    omega1 = np.arange(*regions[region][0])
    omega2 = np.arange(*regions[region][1])

    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain)
    computedSpectrum.addTerms(*terms_selection)

    step1 = regions[region][0][-1]
    gamma = spectrum.rec_cm2rec_s(gamma_rc)
    gamma_str = f"{gamma_rc:.2f}".replace('.', 'p')
    step_str = f"{step1:.1f}".replace('.', 'p')

    tOld = 'tNew'

    method_name = 'B3LYP' if datain['source'] == 'gaussian' else 'CCSDT'

    name=f'./{tOld}_{method_name}_el{str(el)[0]}_mech{str(mech)[0]}_w1mw2{str(w1mw2)[0]}_log10{str(log10)[0]}_gamma{gamma_str}_reg{region}_step{step_str}.svg'
    # with open("output.txt", "a") as f:
    #     print(name, file=f)
    #     print('\n-----------------------------------------', file=f)
    #     # print(computedSpectrum.deriv_data['mu_Q'], file=f)
    #     # with np.printoptions(precision=12, suppress=True):
    #         # print(computedSpectrum.deriv_data['mu_QQ'], file=f)  # check precision for CFOUR for both ders
    #         # print(computedSpectrum.deriv_data['alpha_Q'], file=f) # good match
    #         # print(computedSpectrum.deriv_data['alpha_QQ'], file=f)
    #         # print(computedSpectrum.deriv_data['F_abc'], file=f)
    #     # print('\ncomputedSpectrum.all_states', computedSpectrum.all_states, file=f)
    #     # print('\ncomputedSpectrum.all_states_harmonic', computedSpectrum.all_states_harmonic, file=f)
    #     # print('\ncomputedSpectrum.fundamentals', computedSpectrum.fundamentals, file=f)
    #     # print('\ncomputedSpectrum.fundamentals_harmonic', computedSpectrum.fundamentals_harmonic, file=f)
    #     print('\n-----------------------------------------\n', file=f)

    Z, savedict = computedSpectrum.intensity(gamma, {}, el=el, mech=mech)

    computedSpectrum.plot2Dmatplotlib(Z, w1mw2=w1mw2, nametuple=(name, __file__, "B3LYP/cc-pVQZ"), Gamma=gamma, el=el, mech=mech, dpi=200, log10=log10)

def one_fig_Object(settings: dict, datainput: dict, directory: str = '.', vibEL: bool = True):
    """Making a figure for one input data set"""
    name = make_name(datainput, vibEL, settings, directory)
    print(name)
    if not os.path.isfile(name):
        region = settings['region']
        Gamma_rc = settings['Gamma_rc']
        el_bool = settings['electrical']
        mech_bool = settings['mechanical']

        omega1 = np.arange(*regions[region][0])
        omega2 = np.arange(*regions[region][1])

        computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datainput, vib_levels_harmonic=vibEL)
        computedSpectrum.addTerms(*terms_selection)

        Gamma = spectrum.rec_cm2rec_s(Gamma_rc)
        sec_hypol_data, savedict = computedSpectrum.intensity(Gamma, {}, el=el_bool, mech=mech_bool)

        artist = spectrum.SpectrumFigure(sec_hypol_data, computedSpectrum.w1_mesh, computedSpectrum.w2_mesh, settings)
        title_on_top, text_under_the_figure = make_texts4fig(datainput, computedSpectrum, artist, settings, directory)
        print(name)
        artist.plot2Dmatplotlib(nametuple=(name, __file__, title_on_top),
                                text_under_the_figure=text_under_the_figure)
    else:
        print('Spectrum file exists')

def make_texts4fig(input_data_info: dict, computedSpectrum, artist, settings: dict, directory: str = '.'):

    method_name = input_data_info['files']['method']
    basis_name = input_data_info['files']['basis']
    mol_code = input_data_info['files']['mol_code']

    Gamma_rc = settings['Gamma_rc']

    title_on_top = f"{method_name}/{f"{basis_name}".replace('_', '-')}"

    # here we prepare the text for the textbox
    part1 = f'{directory}\n\nMolecule: {mol_code}\nd_max = {'{:.4e}'.format(artist.d_max)}\n'
    part5 = f'Terms in the expressions: \n      electrical -    {terms_selection[0]}\n      mechanical - {terms_selection[1]}\n'
    part8 = f'Used vibrational energy levels: vib_levels_harmonic={computedSpectrum.vib_levels_harmonic}\n\n'
    values = list(computedSpectrum.fundamentals_harmonic.values())
    if len(values) < 10:
        part6 = f'Fundamentals (harmonic): \n   {sorted(values)}\n\n'
    else:
        sorted_values = sorted(values)
        chunks = [sorted_values[i:i + 9] for i in range(0, len(sorted_values), 9)]
        part6 = 'Fundamentals (harmonic):\n'
        for chunk in chunks:
            part6 += f"   {chunk}\n"
        part6 += '\n'
    values2 = list(computedSpectrum.fundamentals.values())
    if len(values2) < 10:
        part7 = f'Fundamentals (anharmonic): \n   {sorted(values2)}\n\n'
    else:
        sorted_values2 = sorted(values2)
        chunks2 = [sorted_values2[i:i + 9] for i in range(0, len(sorted_values2), 9)]
        part7 = 'Fundamentals (anharmonic):\n'
        for chunk in chunks2:
            part7 += f"   {chunk}\n"
        part7 += '\n'
    part2 = f'intensities.max() = {artist.intensities.max()} = {'{:.4e}'.format(artist.intensities.max())}\n'
    part3 = f'Gamma in cm-1 = {Gamma_rc}\n'
    settings_str = ['\nSettings dict:']
    for key, value in settings.items():
        settings_str.append(f"  {key}: {value}")
    part4 = "\n".join(settings_str)

    part9 = f"\nsettings['norm_min'] {artist.settings['norm_min']}\nsettings['norm_max'] {artist.settings['norm_max']}\n"

    text_under_the_figure = part1+part5+part8+part6+part7+part2+part3+part4+part9

    # return name, title_on_top, text_under_the_figure
    return title_on_top, text_under_the_figure

def make_name(input_data_info: dict, vib_levels_harmonic, settings: dict, directory: str = '.'):

    software = input_data_info['source']
    vibEneLevels = 'harmonicEL' if vib_levels_harmonic else 'anharmonicEL'
    prefix = f'figObj_{vibEneLevels}_{software}'
    method_name = input_data_info['files']['method']
    basis_name = input_data_info['files']['basis']
    mol_code = input_data_info['files']['mol_code']

    el_bool = settings['electrical']
    mech_bool = settings['mechanical']

    region = settings['region']
    Gamma_rc = settings['Gamma_rc']

    step1 = regions[region][0][-1]
    Gamma_str = f"{Gamma_rc:.2f}".replace('.', 'p')
    step_str = f"{step1:.1f}".replace('.', 'p')

    name = f'{directory}/{prefix}_{mol_code}_{method_name}_{basis_name}_el{str(el_bool)[0]}_mech{str(mech_bool)[0]}_w1mw2{str(w1mw2)[0]}_G{Gamma_str}_reg{region}_step{step_str}.svg'

    return name

def normalize_colorbars(list_of):
    pass




print(f"""Generated with: 
'getcwd  :      {os.getcwd()}
'__file__:      {__file__}\n\n""")

log10=True
w1mw2=False
broad_factor_rc=10.

# select terms of electrical and mechanical anharmonicities
terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]

regions = {1: ((1180., 2050., 10.), (2309., 5350., 10.)),
           2: ((2810., 3210., 10.), (5510., 6050., 10.)),
           3: ((1961.318, 1981.318, 10.), (4931.662, 4951.662, 10.)),
           4: ((680., 1750., 10.), (1509., 4350., 10.))}

settings_here = {'electrical': None, 'mechanical': None,
                 'Gamma_rc': broad_factor_rc, 'region': 4,
                 'font_dict': {'size': 18}, 'figsize': (12, 15)}

spectra_dict = {"software": [], "code": [], "method": [], "basis_set": [],
                "el+mech max": [], "el max": [], "mech max": []}

from wilson.spectrum import getting_files_DB

dataframe_gaussian = getting_files_DB("gaussian")
method_basis = dataframe_gaussian[(dataframe_gaussian['code'] == 'FOAC') & (dataframe_gaussian['method'] != 'PBE0')][["code", "method", "basis_set"]]
tuples_method_basis = [(row['code'], row['method'], row['basis_set']) for index, row in method_basis.iterrows()]
print(tuples_method_basis)


for method in tuples_method_basis:
    print('\n>>>>>>>>>   ', method)
    datain = spectrum.make_DatainputDict('gaussian', method)
    list_figs = [(True, False), (False, True), (True, True)]
    for s in list_figs:
        settings_here['electrical'] = s[0]
        settings_here['mechanical'] = s[1]
        # print('\n         NEXT FIGURE\n')
        one_fig_Object(settings=settings_here, datainput=datain, directory='./new_specs_FOAC_anharm', vibEL=False)

dataframe_gaussian = getting_files_DB("cfour")
method_basis = dataframe_gaussian[dataframe_gaussian['code'] == 'FOAC'][["code", "method", "basis_set"]]
tuples_method_basis = [(row['code'], row['method'], row['basis_set']) for index, row in method_basis.iterrows()]

for method in tuples_method_basis:
    print('\n>>>>>>>>>   ', method)
    datain = spectrum.make_DatainputDict('cfour', method)
    list_figs = [(True, False), (False, True), (True, True)]
    for s in list_figs:
        settings_here['electrical'] = s[0]
        settings_here['mechanical'] = s[1]
        # print('\n         NEXT FIGURE\n')
        one_fig_Object(settings=settings_here, datainput=datain, directory='./new_specs_FOAC_anharm', vibEL=False)
