import pytest
import numpy as np
from CQCParse.relay import DataVault
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.utils import prep_data_load
from wilson.spectrum.termND import TermND
from wilson.spectrum.terms_collection import TermsEvaluator
from tests.test_config import SimulationConfig
from tests.testing_utils import debug_mode
# ---------------- Fixtures ----------------
@pytest.fixture(scope="module")
def dict_8terms():
    """
    Fixture to provide the dictionary of 8 terms for testing.
    """
    allterms_str = { 0:
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24},
                     1:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24},
                     2:
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+b+c,zero', 'c,a+b'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'b', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/48.},
                     3:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+c,b', 'b+c,a'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'b',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/48.},
                     4:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',))),
                          'non_averaged_props': ('F', ('b', 'c', 'c',)),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1/48.},
                     5:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1 / 48.},
                     6:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('b', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1 / 48.},
                     7:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          # 'CFF': ('F', ('a', 'c', 'c',), tuple()), # old
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1 / 48.}
                     }
    return allterms_str
@pytest.fixture(scope="module")
def FORM_setup_parser():
    """
    Fixture to set up the Gaussian parser for FORM/B3LYP/cc_pVQZ.
    """
    molecule, method, basis = 'FORM', 'B3LYP', 'cc_pVQZ'
    data_vault = DataVault("/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv")
    dataframe_gaussian = data_vault.getting_files_DB("gaussian")
    aa = dataframe_gaussian[
        (dataframe_gaussian['code'] == molecule) &
        (dataframe_gaussian['method'] == method) &
        (dataframe_gaussian['basis_set'] == basis)
    ]['g16_3quanta_full']
    filename = aa.iloc[0]
    gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
    parser = GaussianParser(gout)
    parser.load()
    return parser
@pytest.fixture(scope="module")
def spectrum_setup(avrg_xyz_indices):
    """
    Fixture to provide the simulation configuration.
    """
    w1 = np.linspace(850.0, 3150.0, 1050)
    w2 = np.linspace(500.0, 6550.0, 800)
    w1m, w2m = np.meshgrid(w1, w2)
    return SimulationConfig(
        gammaCompsAll=avrg_xyz_indices,
        molecule='FORM',
        method='B3LYP',
        basis='cc_pVQZ',
        Gamma=3.8,
        diag_margin=0.0,
        start1=850.0,
        end1=3150.0,
        step1=3.1,
        start2=500.0,
        end2=6550.0,
        step2=3.1,
        old_new_dict={3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5},
        elevels='anharm',
        enelvl=True,
        w1m=w1m,
        w2m=w2m,
    )
@pytest.fixture(scope="module")
def avrg_xyz_indices():
    """
    Fixture to compute averaging indices.
    """
    return get_AlphaBetaGammaDelta_indices(num_f=4)
@pytest.fixture
def setup_term(dict_8terms, FORM_setup_parser, spectrum_setup):
    """
    Factory fixture to set up a TermND instance with parsed data and loaded calculations.
    """
    def create_term(term_id):
        term = TermND(term_id, dict_8terms[term_id])
        parsed_data = FORM_setup_parser.parse(linear_molecule=False)
        parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
        parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
        deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)
        term.load_calc_data(
            properties_data=deriv_data,
            allstates=allstates,
            harmonic_states=harmonic_states,
            mode_indices=mode_indices,
            gammaCompsAll=spectrum_setup.gammaCompsAll
        )
        return term
    return create_term
@pytest.fixture
def data_for_precalc(setup_term, spectrum_setup):
    """
    Fixture to prepare data for precalculation.
    """
    term_with_data = setup_term(0)  # Create term 0
    Nnmodes = 6
    data = {
        (1, 1): term_with_data.properties_data['mu_Q'],
        (1, 2): term_with_data.properties_data['mu_QQ'],
        (2, 1): term_with_data.properties_data['alpha_Q'],
        (2, 2): term_with_data.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    w1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2)
    axes_dict = {1: w1m, 2: w2m}
    alldata = [Nnmodes, data, avrg_terms, axes_dict,
               term_with_data.states_arrays_Eh, term_with_data.harmonic_arrays_Eh]
    return alldata
@pytest.fixture
def terms_collection(data_for_precalc, setup_term):
    """
    Fixture to create a TermsEvaluator with precalculated data.
    """
    terms = [setup_term(i) for i in range(4)]  # Create terms 0 to 3
    te = TermsEvaluator(terms)
    te.identify_to_precalculate()
    big_dict = te.precalculate(data_for_precalc)
    return te, big_dict

###################################################################################################
from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import GaussianParser, GaussianOutput, CFOURParser, CFOUROutput
from CQCParse.parsing import GaussianDataParser, CFOURdataParser
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

# ---------------- Fixtures ----------------
@pytest.fixture
def conditions():
    """
    Fixture to provide the configuration for the experiment using the Conditions dataclass.
    """
    omega1 = np.linspace(850.0, 3150.0, 1050)
    omega2 = np.linspace(500.0, 6550.0, 800)
    program = 'gaussian'
    molecule = 'FORM'
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    new_idx_dict = {3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5}
    el_terms_selected = [0,1]
    mech_terms_selected = [2,3]
    data_parser = None

    return Conditions(
        Gamma_rc=3.8,
        diag_margin_rc=0.0,
        dynamic_range_n=100,
        omega1=omega1,
        omega2=omega2,
        program=program,
        data_parser=data_parser,
        molecule=molecule,
        method=method,
        basis=basis,
        new_idx_dict=new_idx_dict,
        el_terms_selected=el_terms_selected,
        mech_terms_selected=mech_terms_selected,
        list2exclude=None,
        only_modes=None,
        vpt2settings={'anharmonic_type': 'GVPT2'},
        vib_levels_harmonic=False,
        preview=False)

@pytest.fixture
def dataframe_gaussian():
    data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')
    dataframe_gaussian = data_vault.getting_files_DB("gaussian")
    return dataframe_gaussian
@pytest.fixture
def dataframe_cfour():
    data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')
    dataframe_cfour = data_vault.getting_files_DB("cfour")
    return dataframe_cfour

@pytest.fixture
def parsed_data(conditions, dataframe_gaussian, dataframe_cfour):
    """
    Fixture to parse data based on the program (Gaussian or CFOUR).
    """
    program = conditions.program
    molecule, method, basis = conditions.molecule, conditions.method, conditions.basis
    if program == 'gaussian':
        aa = dataframe_gaussian[
            (dataframe_gaussian['code'] == molecule) &
            (dataframe_gaussian['method'] == method) &
            (dataframe_gaussian['basis_set'] == basis)
        ]['g16_3quanta_full']
        filename = aa.iloc[0]
        gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
        parser = GaussianParser(gout)
    elif program == 'cfour':
        aa = dataframe_cfour[
            (dataframe_cfour['code'] == molecule) &
            (dataframe_cfour['method'] == method) &
            (dataframe_cfour['basis_set'] == basis)
        ]
        gout = CFOUROutput(
            molecule, method, basis, 'cfour',
            aa['c4_out'].iloc[0], aa['molden'].iloc[0],
            aa['c4_cubic'].iloc[0], aa['c4_quartic'].iloc[0],
            aa['c4_dipolexyz'].iloc[0][:-1], aa['pkl_polar'].iloc[0]
        )
        parser = CFOURParser(gout)
    else:
        raise ValueError("Unsupported program: {}".format(program))
    parser.load()
    return parser.parse(linear_molecule=False)
@pytest.fixture
def spectrum2d(parsed_data, conditions):
    """
    Fixture to set up a Spectrum2D object.
    """
    omega1, omega2 = conditions.omega1, conditions.omega2
    spectrum_obj = Spectrum2D(omega1, omega2)
    return spectrum_obj
@pytest.fixture
def spectrum_sequence(spectrum2d, parsed_data, conditions):
    """
    Fixture to launch the spectrum sequence and return the resulting dictionary.
    """
    return spectrum2d.launch_sequence1(parsed_data, conditions, print_level=0)
@pytest.fixture
def intensity_data(spectrum2d, spectrum_sequence):
    """
    Fixture to calculate intensity for the Spectrum2D object.
    """
    # if sparse != 0.:
    #     d1 = spectrum2d.find_all_grids(sparse)
    #     new_w1_mesh = np.zeros(spectrum2d.w1_mesh.shape, dtype='complex64')
    #     new_w2_mesh = np.zeros(spectrum2d.w2_mesh.shape, dtype='complex64')
    #     # Placement
    #     for r in d1:
    #         new_w1_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][2]
    #         new_w2_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][3]
    #     spectrum2d.w1_mesh_Eh = new_w1_mesh
    #     spectrum2d.w2_mesh_Eh = new_w2_mesh
    #     mask = spectrum2d.w1_mesh_Eh != 0.
    # else:
    mask = None
    sec_hypol_dataALL_ref = spectrum2d.intensity_both(selectionCond=mask)
    nan_mask = np.isnan(sec_hypol_dataALL_ref)

    has_nan = np.any(nan_mask)
    print(f"Are there any NaN values? {has_nan}")
    num_nan = np.sum(nan_mask)
    print(f"Number of NaN values: {num_nan}")

    sec_hypol_dataALL_ref[nan_mask] = 0 + 0j

    return sec_hypol_dataALL_ref

@pytest.fixture
def terms_amplitudes(terms_collection, spectrum_setup):
    """
    Fixture to calculate amplitudes using TermsEvaluator.
    """
    te, _ = terms_collection
    with debug_mode(0):
        amplitudes = sum(
            term.get_intensity(
                spectrum_setup.w1m, spectrum_setup.w2m,
                3.8, 0.0, debugprint=False, collect_all=False
            )
            for term in te.terms.values()
        )
    return amplitudes