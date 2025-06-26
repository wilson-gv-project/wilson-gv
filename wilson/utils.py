import numpy as np

import os
# Get the root directory of the package dynamically
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import CFOURParser, CFOUROutput, GaussianParser, GaussianOutput, DataStorage


def get_package_root():
    """Returns the absolute path to the package root."""
    return PACKAGE_ROOT

def run_experiment1(conditions, settings_figure, get_max=False, sparse=0.,
                    reference_intensity_plot: float = None, compute_intensity: bool = False, figmake: bool = True):
    """
    reference_intensity_plot - a value, float

               {'spectrum': self,
                'parsed_data': parsed_data,
                'chart': chart,
                'resonancesDF': spectrumDF,
                'mediumDF': medium,
                'artist': None, 'figure': None,
                'maximum_intensity': None, 'sec_hypol_dataALL_ref': None}
    """
    from wilson import rendering
    # from utils import pickle_objs, unpickle_objs
    from CQCParse.relay import DataVault
    data_vault = DataVault(
        "/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv"
    )
    dataframe_gaussian = data_vault.getting_files_DB("gaussian")
    dataframe_cfour = data_vault.getting_files_DB("cfour")

    # Gamma_rc, diag_margin_rc = conditions.Gamma_rc, conditions.diag_margin_rc
    dynamic_range_n = conditions.dynamic_range_n
    omega1, omega2 = conditions.omega1, conditions.omega2,
    program, data_parser = conditions.program, conditions.data_parser
    molecule, method, basis = conditions.molecule, conditions.method, conditions.basis
    # el_terms_selected, mech_terms_selected = conditions.el_terms_selected, conditions.mech_terms_selected

    print('Start', molecule)
    # # setup
    datadict1 = data_vault.make_DatainputDict(program, (molecule, method, basis), '')

    if program=='gaussian':
        aa = dataframe_gaussian[(dataframe_gaussian['code'] == molecule) & (dataframe_gaussian['method'] == method) & (
                    dataframe_gaussian['basis_set'] == basis)]['g16_3quanta_full']
        filename = aa.iloc[0]
        gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
        parser = GaussianParser(gout)
    elif program=='cfour':
        aa = dataframe_cfour[(dataframe_cfour['code'] == molecule) & (dataframe_cfour['method'] == method) & (
                    dataframe_cfour['basis_set'] == basis)]
        gout = CFOUROutput(molecule, method, basis, 'cfour',
                           aa['c4_out'].iloc[0], aa['molden'].iloc[0],
                           aa['c4_cubic'].iloc[0], aa['c4_quartic'].iloc[0],
                           aa['c4_dipolexyz'].iloc[0][:-1], aa['pkl_polar'].iloc[0])
        parser = CFOURParser(gout)
    else:
        print("This program isn't supported")
        return {}

    parser.load()
    parsed_data = parser.parse(linear_molecule=False)

    spectrumObj = Spectrum2D(omega1, omega2)
    dict0 = spectrumObj.launch_sequence1(parsed_data, conditions,
                                         print_level=0)

    if sparse!=0.:
        d1 = spectrumObj.find_all_grids(sparse)

        import json
        with open('convert.txt', 'w') as convert_file:
            convert_file.write(str(d1[list(d1.keys())[0]]))

        settings_figure['prefix_name'] = 'windows_'+settings_figure['prefix_name']

        print('         Number of grids:', len(d1))
        print('   ---> sparse')
        new_w1_mesh = np.zeros(spectrumObj.w1_mesh.shape, dtype='complex64')
        new_w2_mesh = np.zeros(spectrumObj.w2_mesh.shape, dtype='complex64')

        # placement
        for r in d1:
            new_w1_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][2]
            new_w2_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][3]

        spectrumObj.w1_mesh_Eh = new_w1_mesh
        spectrumObj.w2_mesh_Eh = new_w2_mesh
        allp = spectrumObj.w2_mesh.shape[0] * spectrumObj.w2_mesh.shape[1]
        print(np.count_nonzero(spectrumObj.w1_mesh_Eh), allp, np.count_nonzero(spectrumObj.w1_mesh_Eh) / allp)
        mask = spectrumObj.w1_mesh_Eh != 0.
    else:
        mask = None

    if compute_intensity:
        sec_hypol_dataALL_ref = spectrumObj.intensity_both(selectionCond=mask)
        nan_mask = np.isnan(sec_hypol_dataALL_ref)
        sec_hypol_dataALL_ref[nan_mask] = 0 + 0j

        if figmake:
            if settings_figure['max_int'] is None:
                max_ref_Molecule = np.max(abs(sec_hypol_dataALL_ref)**2)
                maxima_pickle = pickle_objs({
                    f'max_{molecule}_{method}{basis}_harm{str(conditions.vib_levels_harmonic)[0]}': max_ref_Molecule},
                                            f'spectra_maxima_{molecule}.pkl')
                print(f'\nMaximum here is - {max_ref_Molecule:.3e}')

                allmaxdict = unpickle_objs(maxima_pickle)

                # if saved max in a pickle file, get it
                if f'max_{molecule}' in allmaxdict:
                    max_Molecule = allmaxdict[f'max_{molecule}']
                    print(f'max_Molecule {max_Molecule:.3e}')
                    print({k: f'{v:.3e}' for k,v in allmaxdict.items() if molecule in k})

                # if no max in a pickle file,
                #       either find the maximum now, save it and use for this spectrum
                #       or use own maximum simply
                else:

                    if get_max:
                        keymax = max(allmaxdict, key=allmaxdict.get)

                        if f'max_{molecule}' not in allmaxdict:
                            print('\nRERUN FOR PROPPER NORMALIZATION!')

                        pickle_objs({f'max_{molecule}': allmaxdict[keymax]}, f'spectra_maxima_{molecule}.pkl')
                        max_Molecule = allmaxdict[keymax]
                        print(f'Maximum is in {keymax}')
                        print(allmaxdict)
                    else:

                        max_Molecule = max_ref_Molecule
                        print('\nUsing own maximum here')
            else:
                max_Molecule = settings_figure['max_int']

            settings_figure['dynamic_range_n'] = dynamic_range_n
            if reference_intensity_plot is None:
                np.set_printoptions(legacy=False)  # or fully reset with defaults
                from numpy.core.arrayprint import _format_options
                # print(_format_options)

                # print(sec_hypol_dataALL_ref)
                artist_ref = rendering.SpectrumFigure(sec_hypol_dataALL_ref, spectrumObj,
                                                      spectrumObj.w1_mesh,
                                                      spectrumObj.w2_mesh,
                                                      settings_figure)
            else:
                artist_ref = rendering.SpectrumFigure(reference_intensity_plot-sec_hypol_dataALL_ref, spectrumObj,
                                                      spectrumObj.w1_mesh,
                                                      spectrumObj.w2_mesh,
                                                      settings_figure)
            title_on_top, text_under_the_figure = rendering.make_texts4fig(datadict1, spectrumObj,
                                                                           artist_ref, directory=settings_figure['directory'])
            name = rendering.make_name(datadict1, spectrumObj, artist_ref,
                                       directory=settings_figure['directory'],
                                       prefix=settings_figure['prefix_name']+'_'+str(conditions.only_modes))

            print('max_Molecule', max_Molecule)

            fig, ax = artist_ref.plot2Dmatplotlib(nametuple=(name, os.path.join(os.path.dirname('__file__')), title_on_top),
                                              text_under_the_figure=text_under_the_figure, normalized=(0., max_Molecule), # (0., max_Molecule)
                                              log10=True, diagonal=False, to_save=True)

            dict0.update({'artist': artist_ref,
                          'figure': fig, 'ax': ax,
                          'maximum_intensity': max_Molecule,
                          'sec_hypol_dataALL_ref': sec_hypol_dataALL_ref,
                          'datadict': datadict1})
        else:
            max_ref_Molecule = np.max(abs(sec_hypol_dataALL_ref) ** 2)
            dict0.update({'maximum_intensity': max_ref_Molecule,
                          'sec_hypol_dataALL_ref': sec_hypol_dataALL_ref,
                          'datadict': datadict1})
    else:
        dict0.update({'artist': None, 'figure': None, 'ax': None,
                      'maximum_intensity': None, 'sec_hypol_dataALL_ref': None,
                      'datadict': datadict1})

    return dict0

import pickle

def pickle_objs(dictobjs, filename):
    try:
        with open(filename, 'rb') as file:
            existing_data = {}
            while True:
                try:
                    existing_data = pickle.load(file)
                except EOFError:
                    break
    except FileNotFoundError:
        existing_data = {}

    existing_data = existing_data | dictobjs

    with open(filename, 'wb') as f:
        pickle.dump(existing_data, f)

    return filename


def unpickle_objs(filename):
    with open(filename, 'rb') as f:
        dictobjs = pickle.load(f)
    return dictobjs

from CQCParse.parsing import GaussianDataParser, CFOURdataParser, ParsedData
from dataclasses import dataclass, field

@dataclass
class Conditions:
    Gamma_rc: float
    diag_margin_rc: float
    dynamic_range_n: int|float
    omega1: np.ndarray
    omega2: np.ndarray
    program: str
    data_parser: CFOURdataParser|GaussianDataParser
    molecule: str
    method: str
    basis: str
    new_idx_dict : dict
    el_terms_selected: list
    mech_terms_selected: list
    list2exclude: list = None
    only_modes: list = None
    vpt2settings: dict = field(default_factory=lambda: {'anharmonic_type': 'GVPT2'})
    vib_levels_harmonic: bool = False
    preview: bool = False


def prep_data_load(parsed_data: ParsedData):

    ddata = [parsed_data.derivatives.dipole_first_derivatives,
             parsed_data.derivatives.dipole_second_derivatives,
             parsed_data.derivatives.polarizability_first_derivatives,
             parsed_data.derivatives.polarizability_second_derivatives,
             parsed_data.derivatives.cubic_force_constants]
    deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))

    allstates = parsed_data.vib_states.anharmonic_states
    harmonic_states = parsed_data.vib_states.harmonic_states

    mode_indices = [i for i in np.arange(parsed_data.nmodes) if i not in parsed_data.list2exclude]

    return deriv_data, allstates, harmonic_states, mode_indices

def pairwise_differences(A, B):
    """
    chatgpt

    for vib levels diffs tensors
    """
    a = np.asarray(A)
    b = np.asarray(B)

    # Reshape a to (a₁, ..., aₙ, 1, ..., 1) with m trailing 1s
    a_broad = a.reshape(*a.shape, *([1] * b.ndim))

    # Reshape B to (1, ..., 1, b₁, ..., bₘ) with n leading 1s
    b_broad = b.reshape(*([1] * a.ndim), *b.shape)

    return a_broad - b_broad

def coolprint(text):
    from rich import print
    print(f"[italic yellow2]{text}[/italic yellow2]")