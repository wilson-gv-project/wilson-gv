import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=17)


def make_texts4fig(input_data_info: dict, computedSpectrum, artist, directory: str = '.') -> tuple[str, str]:
    """
    other = {'regions': regions, 'terms_selection': terms_selection, 'w1mw2': False, 'log10': True}

    """
    terms_selection = (computedSpectrum.e_selected, computedSpectrum.m_selected)

    method_name = input_data_info['files']['method']
    basis_name = input_data_info['files']['basis']
    mol_code = input_data_info['files']['mol_code']

    Gamma_rc = computedSpectrum.Gamma_rc

    title_on_top = f"{method_name}/{f'{basis_name} {terms_selection}'.replace('_', '-')}"

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
    for key, value in artist.settings.items():
        settings_str.append(f"  {key}: {value}")
    part4 = "\n".join(settings_str)
    if artist.settings['norm_min'] is None:
        n_min_str = 'None'
    else:
        n_min_str = '{:.2e}'.format(artist.settings['norm_min'])
    if artist.settings['norm_max'] is None:
        n_max_str = 'None'
    else:
        n_max_str = '{:.2e}'.format(artist.settings['norm_max'])

    # part9 = f"\nsettings['norm_min'] {n_min_str}\nsettings['norm_max'] {n_max_str}\n"
    part9 = f"\nsettings['dynamic_range_n'] {artist.settings['dynamic_range_n']}\n"

    text_under_the_figure = part1+part5+part8+part6+part7+part2+part3+part4+part9

    return title_on_top, text_under_the_figure

def make_name(input_data_info: dict, computedSpectrum, artist, directory: str = '.', prefix: str = None) -> str:
    """
    other = {'regions': regions, 'terms_selection': terms_selection, 'w1mw2': False, 'log10': True}

    """
    w1mw2 = artist.settings['w1mw2']
    vib_levels_harmonic = computedSpectrum.vib_levels_harmonic

    software = input_data_info['source']
    vibEneLevels = 'harmonicEL' if vib_levels_harmonic else 'anharmonicEL'

    if prefix is None:
        prefix = f'figObj_{vibEneLevels}_{software}'
    else:
        prefix += f'_{vibEneLevels}_{software}'

    # prefix = f'figObj_{vibEneLevels}_{software}'
    method_name = input_data_info['files']['method']
    basis_name = input_data_info['files']['basis']
    mol_code = input_data_info['files']['mol_code']

    # el_bool = artist.settings['electrical']
    # mech_bool = artist.settings['mechanical']

    els_str = ''.join([str(i) for i in artist.settings['electrical']])
    mechs_str = ''.join([str(i) for i in artist.settings['mechanical']])

    Gamma_rc = artist.settings['Gamma_rc']
    Gamma_str = f"{Gamma_rc:.2f}".replace('.', 'p')

    # step1 = regions[region][0][-1]
    # step_str = f"{step1:.1f}".replace('.', 'p')

    name = f'{directory}/{prefix}_{mol_code}_{method_name}_{basis_name}_el{els_str}_mech{mechs_str}_w1mw2{str(w1mw2)[0]}_G{Gamma_str}.svg'

    return name

