import pytest
import numpy as np
from CQCParse.relay import DataVault
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.utils import prep_data_load
from wilson.spectrum.termND import TermND
from wilson.spectrum.termsEvaluator import TermsEvaluator
from wilson.utils.spectrum_utils import SimulationConfig
from wilson.spectrum import debug_mode

from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import GaussianParser, GaussianOutput, CFOURParser, CFOUROutput

from wilson.utils import Conditions

# ---------------- Fixtures ----------------
def convert_lists_to_tuples(data):
    if isinstance(data, list):
        return tuple(convert_lists_to_tuples(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_lists_to_tuples(value) for key, value in data.items()}
    else:
        return data

@pytest.fixture(scope='module')
def derived_terms_json():
    import json
    with open('/home/vlev/wilson-suite/wilson_intensities/tests/unit/terms.json') as json_file:
        list_terms = json.load(json_file)
    d = {i:t for i,t in enumerate(list_terms)}
    return convert_lists_to_tuples(d)
@pytest.fixture(scope="module")
def dict_8terms():
    """
    Fixture to provide the dictionary of 8 terms for testing.
    """
    allterms_str = { 0:
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('b',), ('A', 'D')),
                                             ('diphess', ('a', 'b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/4},
                     1:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polhess', ('a', 'b',), ('A', 'D')),
                                             ('dipgrad', ('b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/4},
                     2:
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+b+c,zero', 'c,a+b'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('b',), ('A', 'D')),
                                             ('dipgrad', ('c',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'b', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/8.},
                     3:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+c,b', 'b+c,a'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('c',), ('A', 'D')),
                                             ('dipgrad', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'b',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/8.},
                     4:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('b',), ('A', 'D')),
                                             ('dipgrad', ('a',), ('G',))),
                          'non_averaged_props': (('F', ('b', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1/8.},
                     5:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('b',), ('A', 'D')),
                                             ('dipgrad', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1/8.},
                     6:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('a',), ('A', 'D')),
                                             ('dipgrad', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('b', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1/8.},
                     7:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('dipgrad', ('a',), ('B',)),
                                             ('polgrad', ('b',), ('A', 'D')),
                                             ('dipgrad', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1/8.}
                     }
    return allterms_str
@pytest.fixture(scope="module")
def MOL_setup_parser(conditions):
    """
    Fixture to set up the Gaussian parser for MOL/B3LYP/cc_pVQZ.
    Molecule is taken from conditions
    """
    parsers = {}
    # print(conditions.keys())

    for mol,cond in conditions.items():
        # print(mol)
        molecule, method, basis = cond.molecule, 'B3LYP', 'cc_pVQZ'
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
        parsers[molecule] = parser
    return parsers
@pytest.fixture(scope="module")
def spectrum_setup(avrg_xyz_indices, conditions):
    """
    Fixture to provide the simulation configuration.
    """
    setupsdict = {}
    # print(conditions.keys())
    for mol,conds in conditions.items():
        print(mol)
        w1 = np.arange(850.0, 3150.0, 3.1)
        w2 = np.arange(500.0, 6550.0, 3.1)
        # w1 = np.linspace(850.0, 3150.0, 1050)
        # w2 = np.linspace(500.0, 6550.0, 800)
        w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
        # if mol=='FORM':
        #     new_idx_dict = {3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5} #FORM
        # else:
        #     new_idx_dict = None
        new_idx_dict = None
        setupsdict[mol] = SimulationConfig(
            gammaCompsAll=avrg_xyz_indices,
            molecule=mol,
            method='B3LYP',
            basis='cc_pVQZ',
            Gamma=3.8,
            diag_margin=1.0,
            start1=850.0,
            end1=3150.0,
            step1=3.1,
            start2=500.0,
            end2=6550.0,
            step2=3.1,
            old_new_dict=new_idx_dict,
            elevels='anharm',
            enelvl=True,
            w1m=w1m,
            w2m=w2m,
        )
    return setupsdict
@pytest.fixture(scope="module")
def avrg_xyz_indices():
    """
    Fixture to compute averaging indices.
    """
    return get_AlphaBetaGammaDelta_indices(num_f=4)
@pytest.fixture(scope="module")
def setup_term(dict_8terms, MOL_setup_parser, spectrum_setup): #! dict_8terms or derived_terms_json
    """
    Factory fixture to set up a TermND instance with parsed data and loaded calculations.
    """
    term_funcs = {}
    # print(spectrum_setup.keys())

    for mol,spec_setup in spectrum_setup.items():
        print(mol)
        def create_term(term_id):
            term = TermND(term_id, dict_8terms[term_id]) #! dict_8terms or derived_terms_json
            parsed_data = MOL_setup_parser[mol].parse(linear_molecule=False)
            parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
            if spectrum_setup[mol].old_new_dict is not None:
                parsed_data.upd_indices_several_parts(spectrum_setup[mol].old_new_dict)
            deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)
            term.load_calc_data(
                properties_data=deriv_data,
                allstates=allstates,
                harmonic_states=harmonic_states,
                mode_indices=mode_indices,
                gammaCompsAll=spectrum_setup[mol].gammaCompsAll
            )
            return term
        term_funcs[mol] = create_term
    return term_funcs
@pytest.fixture(scope="module")
def setup_term_derived(derived_terms_json, MOL_setup_parser, spectrum_setup): #! dict_8terms or derived_terms_json
    """
    Factory fixture to set up a TermND instance with parsed data and loaded calculations.
    """
    term_funcs = {}
    # print(spectrum_setup.keys())

    for mol,spec_setup in spectrum_setup.items():
        print(mol)
        def create_term(term_id):
            term = TermND(term_id, derived_terms_json[term_id]) #! dict_8terms or derived_terms_json
            parsed_data = MOL_setup_parser[mol].parse(linear_molecule=False)
            parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
            if spectrum_setup[mol].old_new_dict is not None:
                parsed_data.upd_indices_several_parts(spectrum_setup[mol].old_new_dict)
            deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)
            term.load_calc_data(
                properties_data=deriv_data,
                allstates=allstates,
                harmonic_states=harmonic_states,
                mode_indices=mode_indices,
                gammaCompsAll=spectrum_setup[mol].gammaCompsAll
            )
            return term
        term_funcs[mol] = create_term
    return term_funcs

@pytest.fixture(scope="module")
def data_for_precalc(setup_term, spectrum_setup):
    """
    Fixture to prepare data for precalculation.
    """
    precalcs = {}
    # print(spectrum_setup.keys())
    for mol,spec_setup in spectrum_setup.items():
        # print(mol)
        term_with_data = setup_term[mol](0)  # Create term 0
        Nnmodes = 6
        # now here keys change; fixme: it the change needed??
        props_data_ready = {
            'dipgrad': term_with_data.properties_data['dipgrad'],
            'diphess': term_with_data.properties_data['diphess'],
            'polgrad': term_with_data.properties_data['polgrad'],
            'polhess': term_with_data.properties_data['polhess'],
        }
        avrg_terms = spectrum_setup[mol].gammaCompsAll
        w1 = np.arange(spectrum_setup[mol].start1,
                       spectrum_setup[mol].end1, spectrum_setup[mol].step1)
        w2 = np.arange(spectrum_setup[mol].start2,
                       spectrum_setup[mol].end2, spectrum_setup[mol].step2)
        w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
        axes_dict = {1: w1m, 2: w2m}

        # from rich import print as rprint
        # rprint('\n[deep_pink3]term_with_data.states_arrays_Eh[/deep_pink3]')
        # rprint(term_with_data.states_arrays_Eh)
        # rprint('\n[deep_pink3]term_with_data.harmonic_arrays_Eh[/deep_pink3]')
        # rprint(term_with_data.harmonic_arrays_Eh)

        from wilson.spectrum import DataForPrecalc
        alldata = DataForPrecalc(Nnmodes=Nnmodes,
                                 props_data=props_data_ready,
                                 avrg_terms=avrg_terms,
                                 axes_dict=axes_dict,
                                 states_arrays_Eh=term_with_data.states_arrays_Eh,
                                 harmonic_arrays_Eh=term_with_data.harmonic_arrays_Eh)
        # print('term_with_data.harmonic_arrays_Eh')
        # print(term_with_data.harmonic_arrays_Eh)
        precalcs[mol] = alldata
    return precalcs
@pytest.fixture(scope="module")
def data_for_precalc_derived(setup_term_derived, spectrum_setup):
    """
    Fixture to prepare data for precalculation.
    """
    precalcs = {}
    # print(spectrum_setup.keys())
    for mol,spec_setup in spectrum_setup.items():
        # print(mol)
        term_with_data = setup_term_derived[mol](0)  # Create term 0
        Nnmodes = 6
        props_data_ready = {
            'dipgrad': term_with_data.properties_data['dipgrad'],
            'diphess': term_with_data.properties_data['diphess'],
            'polgrad': term_with_data.properties_data['polgrad'],
            'polhess': term_with_data.properties_data['polhess'],
        }
        avrg_terms = spectrum_setup[mol].gammaCompsAll
        w1 = np.arange(spectrum_setup[mol].start1,
                       spectrum_setup[mol].end1, spectrum_setup[mol].step1)
        w2 = np.arange(spectrum_setup[mol].start2,
                       spectrum_setup[mol].end2, spectrum_setup[mol].step2)
        w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
        axes_dict = {1: w1m, 2: w2m}

        # from rich import print as rprint
        # rprint('\n[deep_pink3]term_with_data.states_arrays_Eh[/deep_pink3]')
        # rprint(term_with_data.states_arrays_Eh)
        # rprint('\n[deep_pink3]term_with_data.harmonic_arrays_Eh[/deep_pink3]')
        # rprint(term_with_data.harmonic_arrays_Eh)

        from wilson.spectrum import DataForPrecalc
        alldata = DataForPrecalc(Nnmodes=Nnmodes,
                                 props_data=props_data_ready,
                                 avrg_terms=avrg_terms,
                                 axes_dict=axes_dict,
                                 states_arrays_Eh=term_with_data.states_arrays_Eh,
                                 harmonic_arrays_Eh=term_with_data.harmonic_arrays_Eh)
        # print('term_with_data.harmonic_arrays_Eh')
        # print(term_with_data.harmonic_arrays_Eh)
        precalcs[mol] = alldata
    return precalcs
@pytest.fixture(scope="module")
def terms_collection(data_for_precalc, setup_term, dict_8terms): #! dict_8terms or derived_terms_json
    """
    Fixture to create a TermsEvaluator with precalculated data.
    """
    terms_cols = {}
    # print(setup_term.keys())

    for mol,term_setup in setup_term.items():
        # print(mol)
        terms = [term_setup(i) for i in range(len(dict_8terms))] #! dict_8terms or derived_terms_json
        te = TermsEvaluator(terms)
        te.identify_to_precalculate()
        precalc_dict = te.precalculate(data_for_precalc[mol])
        terms_cols[mol] = (te, precalc_dict)
    return terms_cols
@pytest.fixture(scope="module")
def terms_collection_derived(data_for_precalc_derived, setup_term_derived, derived_terms_json): #! dict_8terms or derived_terms_json
    """
    Fixture to create a TermsEvaluator with precalculated data.
    """
    terms_cols = {}
    # print(setup_term_derived.keys())

    for mol,term_setup in setup_term_derived.items():
        # print(mol)
        terms = [term_setup(i) for i in range(len(derived_terms_json))] #! dict_8terms or derived_terms_json
        te = TermsEvaluator(terms)
        te.identify_to_precalculate()
        precalc_dict = te.precalculate(data_for_precalc_derived[mol])
        terms_cols[mol] = (te, precalc_dict)
    return terms_cols


# ---------------- Fixtures ----------------
# @pytest.fixture(scope="module",params=["FORM", "OXAC2"])
@pytest.fixture(scope="module")
def conditions():
    """
    Fixture to provide the configuration for the experiment using the Conditions dataclass.
    """

    resdict = {}

    # for mol in ["FORM", "OXAC2"]:
    for mol in ["FORM"]:
        # print(mol)
        # omega1 = np.linspace(850.0, 3150.0, 1050)
        # omega2 = np.linspace(500.0, 6550.0, 800)
        omega1 = np.arange(850.0, 3150.0, 3.1)
        omega2 = np.arange(500.0, 6550.0, 3.1)
        program = 'gaussian'
        molecule = mol
        method = 'B3LYP'
        basis = 'cc_pVQZ'
        if mol=='FORM':
            # new_idx_dict = {3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5} #FORM
            new_idx_dict = None #FORM
        else:
            new_idx_dict = None
        el_terms_selected = [0,1]
        mech_terms_selected = [2,3,4,5,6,7]

        resdict[mol] = Conditions(
                            Gamma_rc=3.8,
                            diag_margin_rc=1.0,
                            dynamic_range_n=4500,
                            omega1=omega1,
                            omega2=omega2,
                            program=program,
                            data_parser=None,
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
    return resdict

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
    parsed_data_dict = {}
    # print(conditions.keys())

    for mol,cond in conditions.items():
        # print(mol)
        program = cond.program
        molecule, method, basis = mol, cond.method, cond.basis
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
        parsed_data_dict[mol] = parser.parse(linear_molecule=False)
    return parsed_data_dict
@pytest.fixture
def spectrum2d(conditions):
    """
    Fixture to set up a Spectrum2D object.
    """
    spectrum_objects = {}
    # print(conditions.keys())

    for mol,cond in conditions.items():
        # print(mol)
        omega1, omega2 = cond.omega1, cond.omega2
        spectrum_obj = Spectrum2D(omega1, omega2)
        spectrum_objects[mol] = spectrum_obj
    return spectrum_objects
@pytest.fixture
def spectrum_sequence(spectrum2d, parsed_data, conditions):
    """
    Fixture to launch the spectrum sequence and return the resulting dictionary.
    """
    preps = {}
    # print(conditions.keys())

    for mol,cond in conditions.items():
        # print(mol)
        preps[mol] = spectrum2d[mol].launch_sequence1(parsed_data[mol],
                                                      cond, print_level=0)
    return preps
@pytest.fixture
def intensity_data(spectrum2d, spectrum_sequence):
    """
    Fixture to calculate intensity for the Spectrum2D object.
    """

    sec_hypol_data_dict = {}
    # print(spectrum_sequence.keys())

    for mol,spec_preps in spectrum_sequence.items():
        # print(mol)
        mask = None
        sec_hypol_dataALL_ref = spectrum2d[mol].intensity_both(selectionCond=mask)
        nan_mask = np.isnan(sec_hypol_dataALL_ref)

        has_nan = np.any(nan_mask)
        print(f"Are there any NaN values? {has_nan}")
        num_nan = np.sum(nan_mask)
        print(f"Number of NaN values: {num_nan}")

        sec_hypol_dataALL_ref[nan_mask] = 0 + 0j

        sec_hypol_data_dict[mol] = sec_hypol_dataALL_ref
    return sec_hypol_data_dict

@pytest.fixture
def terms_amplitudes(terms_collection, spectrum_setup):
    """
    Fixture to calculate amplitudes using TermsEvaluator.
    """

    ampls = {}
    # print(spectrum_setup.keys())

    for mol,spec_setup in spectrum_setup.items():
        # print(mol)
        te, _ = terms_collection[mol]
        with debug_mode(0):
            amplitudes = sum(
                term.get_amplitudes(
                    spec_setup.w1m, spec_setup.w2m,
                    3.8, 1.0, debugprint=False, collect_all=False
                )
                for term in te.terms.values()
            )
        ampls[mol] = amplitudes
    return ampls