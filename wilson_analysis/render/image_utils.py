import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=17)
from wilson.spectrum.spectrum2D import combinations_with_permutations

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
        part6 = f'\nFundamentals (harmonic): \n   {sorted(values)}\n\n'
    else:
        sorted_values = sorted(values)
        chunks = [sorted_values[i:i + 9] for i in range(0, len(sorted_values), 9)]
        part6 = '\nFundamentals (harmonic):\n'
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

    d = {k:v for k,v in computedSpectrum.fundamentals.items() if int(k) in computedSpectrum.mode_indices}
    if len(d)==2:
        part10 = f'\n{d}; {list(d.values())[1]-list(d.values())[0]}\n'
    else:
        part10 = f'\n{d}\n'

    print(computedSpectrum.fundamentals)
    part11 = ''

    for nm in d:
        d2 = {}
        for i in list(combinations_with_permutations(computedSpectrum.mode_indices, 2)):
            if nm in (str(i[0]), str(i[1])):
                d2[tuple(sorted(i))] = computedSpectrum.all_states[tuple([str(j) for j in sorted(i)])]-d[nm]
        part11+=f'\n{d[nm]} {d2}\n'
    # part11 = f'\n{d2}\n'
    text_under_the_figure = part1+part5+part8+part10+part11+part6+part7+part2+part3+part4+part9

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


def text_spec_fig(conditions):

    title_on_top = f'{conditions.molecule} {conditions.method} {conditions.basis}'

    part1 = f'\nGamma: {conditions.Gamma_rc}\n'
    part2 = f'\ndynamic_range_n: {conditions.dynamic_range_n}\n'
    part3 = f'\nomega1: {np.min(conditions.omega1), np.max(conditions.omega1)}, {conditions.omega1.shape}\n'
    part4 = f'\nomega2: {np.min(conditions.omega2), np.max(conditions.omega2)}, {conditions.omega2.shape}\n'
    part5 = f'\nel_terms_selected: {conditions.el_terms_selected}\n'
    part6 = f'\nmech_terms_selected: {conditions.mech_terms_selected}\n'

    text_under_the_figure = part1+part2+part3+part4+part5+part6

    # class Conditions:
    #     Gamma_rc: float
    #     diag_margin_rc: float
    #     dynamic_range_n: int|float
    #     omega1: np.ndarray
    #     omega2: np.ndarray
    #     program: str
    #     data_parser: CFOURdataParser|GaussianDataParser
    #     molecule: str
    #     method: str
    #     basis: str
    #     new_idx_dict : dict
    #     el_terms_selected: list
    #     mech_terms_selected: list
    #     list2exclude: list = None
    #     only_modes: list = None
    #     vpt2settings: dict = field(default_factory=lambda: {'anharmonic_type': 'GVPT2'})
    #     vib_levels_harmonic: bool = False
    #     preview: bool = False

    return title_on_top, text_under_the_figure
