import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=17)
# from wilson import spectrum
# from wilson import rendering
# import os
#
# def one_spectrum_fig(el: bool, mech: bool, datain: dict, other: dict, region: int = 3, gamma_rc: float = 10.):
#     """
#         # one_spectrum_fig(el=el, mech=mech, datain=datain, region=1, gamma_rc=broad_factor_rc)
#     """
#     regions = other['regions']
#     terms_selection = other['terms_selection']
#     w1mw2 = other['w1mw2']
#     log10 = other['log10']
#
#     omega1 = np.arange(*regions[region][0])
#     omega2 = np.arange(*regions[region][1])
#
#     computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain)
#     computedSpectrum.addTerms(*terms_selection)
#
#     step1 = regions[region][0][-1]
#     gamma = spectrum.convWnum2Freq(gamma_rc)
#     gamma_str = f"{gamma_rc:.2f}".replace('.', 'p')
#     step_str = f"{step1:.1f}".replace('.', 'p')
#
#     tOld = 'tNew'
#
#     method_name = 'B3LYP' if datain['source'] == 'gaussian' else 'CCSDT'
#
#     name=f'./{tOld}_{method_name}_el{str(el)[0]}_mech{str(mech)[0]}_w1mw2{str(w1mw2)[0]}_log10{str(log10)[0]}_gamma{gamma_str}_reg{region}_step{step_str}.svg'
#
#     Z, savedict = computedSpectrum.intensity(gamma, {}, el=el, mech=mech)
#
#     computedSpectrum.plot2Dmatplotlib(Z, w1mw2=w1mw2, nametuple=(name, __file__, "B3LYP/cc-pVQZ"), Gamma=gamma, el=el, mech=mech, dpi=200, log10=log10)
#
# def one_fig_Object(settings: dict, datainput: dict, other: dict, directory: str = '.', vibEL: bool = True):
#     """Making a figure for one input data set"""
#     name = make_name(datainput, vibEL, settings, directory)
#     print(name)
#     if not os.path.isfile(name):
#         region = settings['region']
#         Gamma_rc = settings['Gamma_rc']
#         el_bool = settings['electrical']
#         mech_bool = settings['mechanical']
#
#         regions = other['regions']
#         terms_selection = other['terms_selection']
#
#         omega1 = np.arange(*regions[region][0])
#         omega2 = np.arange(*regions[region][1])
#
#         computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datainput, vib_levels_harmonic=vibEL)
#         computedSpectrum.addTerms(*terms_selection)
#
#         Gamma = spectrum.convWnum2Freq(Gamma_rc)
#         sec_hypol_data, savedict = computedSpectrum.intensity(Gamma, {}, el=el_bool, mech=mech_bool)
#
#         artist = spectrum.SpectrumFigure(sec_hypol_data, computedSpectrum.w1_mesh, computedSpectrum.w2_mesh, settings)
#         title_on_top, text_under_the_figure = make_texts4fig(datainput, computedSpectrum, artist, settings, directory)
#         print(name)
#         artist.plot2Dmatplotlib(nametuple=(name, __file__, title_on_top),
#                                 text_under_the_figure=text_under_the_figure)
#     else:
#         print('Spectrum file exists')

def make_texts4fig(input_data_info: dict, computedSpectrum, artist,
                   settings: dict, other: dict, directory: str = '.') -> tuple[str, str]:
    """
    other = {'regions': regions, 'terms_selection': terms_selection, 'w1mw2': False, 'log10': True}

    """
    terms_selection = other['terms_selection']

    method_name = input_data_info['files']['method']
    basis_name = input_data_info['files']['basis']
    mol_code = input_data_info['files']['mol_code']

    Gamma_rc = settings['Gamma_rc']

    title_on_top = f"{method_name}/{f'{basis_name}'.replace('_', '-')}"

    # here we prepare the text for the textbox
    part1 = f'{directory}\n\nMolecule: {mol_code}\nd_max = {"{:.4e}".format(artist.d_max)}\n'
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
    part2 = f'intensities.max() = {artist.intensities.max()} = {"{:.4e}".format(artist.intensities.max())}\n'
    part3 = f'Gamma in cm-1 = {Gamma_rc}\n'
    settings_str = ['\nSettings dict:']
    for key, value in settings.items():
        settings_str.append(f"  {key}: {value}")
    part4 = "\n".join(settings_str)

    part9 = f"\nsettings['norm_min'] {'{:.2e}'.format(artist.settings['norm_min'])}\nsettings['norm_max'] {'{:.2e}'.format(artist.settings['norm_max'])}\n"

    text_under_the_figure = part1+part5+part8+part6+part7+part2+part3+part4+part9

    # return name, title_on_top, text_under_the_figure
    return title_on_top, text_under_the_figure

def make_name(input_data_info: dict, vib_levels_harmonic: bool,
              settings: dict, other: dict, directory: str = '.') -> str:
    """
    other = {'regions': regions, 'terms_selection': terms_selection, 'w1mw2': False, 'log10': True}

    """
    w1mw2 = other['w1mw2']
    regions = other['regions']

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

