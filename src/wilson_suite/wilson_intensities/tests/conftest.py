"""
Fixtures for pytests.

Each of fixtures returns a dictionary where keys are molecular code strings.
So for each molecule there could be a setup. Method is B3LYP and basis set cc_pVQZ.
Initial use of list_of_molucules is in conditions()
"""
import os
import pytest
import numpy as np
import pandas as pd
from CQCParse.relay import DataVault
from ..spectrum.averaging import get_AlphaBetaGammaDelta_indices
from ..utils.utils import prep_data_load, get_package_root
from ..spectrum.termND import TermND
from ..utils import DataForPrecalc
from ..spectrum.termsEvaluator import TermsEvaluator
from ..utils.spectrum_utils import SimulationConfig
from ..utils import debug_mode

from CQCParse.parsing import GaussianParser, GaussianOutput, CFOURParser, CFOUROutput

from ..utils.utils import Conditions

# list of molucules to set up fixtures for
list_of_molecules = ["FORM"]
# minidatabase_csv = get_package_root()+ '/tests/test_database/mini_files_database.csv'
minidatabase_csv = '/home/vlev/sprint/calculations/calculations.csv'
terms_json = ''
directory = os.path.dirname(os.path.abspath(__file__))

# ---------------- Fixtures ----------------
def convert_lists_to_tuples(data: list|dict) -> tuple|dict:
    if isinstance(data, list):
        return tuple(convert_lists_to_tuples(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_lists_to_tuples(value) for key, value in data.items()}
    else:
        return data

@pytest.fixture(scope='module')
def derived_terms_json() -> dict:
    import os
    directory = os.path.dirname(os.path.abspath(__file__))
    import json
    with open(directory+'/unit/terms.json') as json_file:
        list_terms = json.load(json_file)
    d = {i:t for i,t in enumerate(list_terms)}
    return convert_lists_to_tuples(d)
@pytest.fixture(scope="module")
def dict_8terms() -> dict:
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
def MOL_setup_parser(conditions: dict[str,Conditions]) -> dict:
    """
    Fixture to set up the Gaussian parser for MOL/B3LYP/cc_pVQZ.
    Molecule is taken from conditions
    """
    parsers = {}

    for mol,cond in conditions.items():
        molecule, method, basis = cond.molecule, 'B3LYP', 'cc-pVQZ'
        data_vault = DataVault(minidatabase_csv)
        # dataframe_gaussian = data_vault.getting_files_DB("gaussian")
        dataframe_gaussian = data_vault.filter_database("gaussian")

        aa = dataframe_gaussian[
            (dataframe_gaussian['Name'] == molecule) &
            (dataframe_gaussian['Method'] == method) &
            (dataframe_gaussian['Basis'] == basis)
        ]['file_location']
        # filename = directory+aa.iloc[0]
        filename = aa.iloc[0]

        gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
        parser = GaussianParser(gout)
        parser.load()
        parsers[molecule] = parser
    return parsers
@pytest.fixture(scope="module")
def spectrum_setup(avrg_xyz_indices: tuple[list|np.ndarray, float], conditions: dict) -> dict:
    """
    Fixture to provide the simulation configuration.
    """
    setupsdict = {}
    for mol,conds in conditions.items():
        w1 = np.arange(850.0, 3150.0, 3.1)
        w2 = np.arange(500.0, 6550.0, 3.1)
        w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
        new_idx_dict = None
        setupsdict[mol] = SimulationConfig(
            gammaCompsAll=avrg_xyz_indices,
            molecule=mol,
            method='B3LYP',
            basis='cc-pVQZ',
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
def avrg_xyz_indices() -> tuple[np.ndarray|list, float]:
    """
    Fixture to compute averaging indices.
    """
    return get_AlphaBetaGammaDelta_indices(num_f=4), 1/15.
@pytest.fixture(scope="module")
def setup_term(dict_8terms: dict, MOL_setup_parser: dict, spectrum_setup: dict) -> dict: #! dict_8terms or derived_terms_json
    """
    Factory fixture to set up TermND instances with parsed data and loaded calculations.
    Uses a hardcoded dictionary of terms from EVV pen-and-paper derivations
    """
    term_funcs = {}

    for mol,spec_setup in spectrum_setup.items():
        def create_term(term_id: int|str) -> TermND:
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
def setup_term_derived(derived_terms_json: dict, 
                       MOL_setup_parser: dict, 
                       spectrum_setup: dict) -> dict: #! dict_8terms or derived_terms_json
    """
    Factory fixture to set up TermND instances with parsed data and loaded calculations.
    Uses terms derived with wilson_derive and saved into a json file 
    then retrieved from it as a dictionary of terms like in the previous function

    """
    term_funcs = {}

    for mol,spec_setup in spectrum_setup.items():
        def create_term(term_id: int|str) -> TermND:
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
def data_for_precalc(setup_term: dict, spectrum_setup: dict) -> dict:
    """
    Fixture to prepare data for precalculation.
    Based on pen-and-paper derived terms. See setup_term above
    """
    precalcs = {}
    for mol,spec_setup in spectrum_setup.items():
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
        # axes_dict = {1: w1m, 2: w2m}
        axes_dict = {'w1': w1m, 'w2': w2m}

        from ..spectrum import DataForPrecalc
        alldata = DataForPrecalc(Nnmodes=Nnmodes,
                                 props_data=props_data_ready,
                                 avrg_terms=avrg_terms,
                                 axes_dict=axes_dict,
                                 states_arrays_Eh=term_with_data.states_arrays_Eh,
                                 harmonic_arrays_Eh=term_with_data.harmonic_arrays_Eh)
        precalcs[mol] = alldata
    return precalcs
@pytest.fixture(scope="module")
def data_for_precalc_derived(setup_term_derived: dict, spectrum_setup: dict) -> dict:
    """
    Fixture to prepare data for precalculation.
    Based on terms derived with wilson_derived. See setup_term_derived above
    """
    precalcs = {}
    for mol,spec_setup in spectrum_setup.items():
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
        # axes_dict = {1: w1m, 2: w2m}
        axes_dict = {'w1': w1m, 'w2': w2m}

        alldata = DataForPrecalc(Nnmodes=Nnmodes,
                                 props_data=props_data_ready,
                                 avrg_terms=avrg_terms,
                                 axes_dict=axes_dict,
                                 states_arrays_Eh=term_with_data.states_arrays_Eh,
                                 harmonic_arrays_Eh=term_with_data.harmonic_arrays_Eh)
        precalcs[mol] = alldata
    return precalcs
@pytest.fixture(scope="module")
def terms_collection(data_for_precalc: dict, setup_term: dict, dict_8terms: dict) -> dict: #! dict_8terms or derived_terms_json
    """
    Fixture to create a TermsEvaluator with precalculated data.
    Based on pen-and-paper derived terms. See setup_term above
    """
    terms_cols = {}

    for mol,term_setup in setup_term.items():
        terms = [term_setup(i) for i in range(len(dict_8terms))] #! dict_8terms or derived_terms_json
        te = TermsEvaluator(terms)
        te.identify_to_precalculate()
        precalc_dict = te.precalculate(data_for_precalc[mol])
        terms_cols[mol] = (te, precalc_dict)
    return terms_cols
@pytest.fixture(scope="module")
def terms_collection_derived(data_for_precalc_derived: dict, setup_term_derived: dict, 
                             derived_terms_json: dict) -> dict: #! dict_8terms or derived_terms_json
    """
    Fixture to create a TermsEvaluator with precalculated data.
    Based on terms derived with wilson_derived. See setup_term_derived above
    """
    terms_cols = {}

    for mol,term_setup in setup_term_derived.items():
        terms = [term_setup(i) for i in range(len(derived_terms_json))] #! dict_8terms or derived_terms_json
        te = TermsEvaluator(terms)
        te.identify_to_precalculate()
        precalc_dict = te.precalculate(data_for_precalc_derived[mol])
        terms_cols[mol] = (te, precalc_dict)
    return terms_cols


# ---------------- Fixtures ----------------
# @pytest.fixture(scope="module",params=["FORM", "OXAC2"])
@pytest.fixture(scope="module")
def conditions() -> dict[str: Conditions]:
    """
    Fixture to provide the configuration for the experiment using the Conditions dataclass.
    """

    resdict = {}

    # for mol in ["FORM", "OXAC2"]:
    for mol in list_of_molecules:
        omega1 = np.arange(850.0, 3150.0, 3.1)
        omega2 = np.arange(500.0, 6550.0, 3.1)
        program = 'gaussian'
        molecule = mol
        method = 'B3LYP'
        basis = 'cc-pVQZ'
        if mol=='FORM':
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
def dataframe_gaussian() -> pd.DataFrame:
    data_vault = DataVault(minidatabase_csv)
    dataframe_gaussian = data_vault.filter_database("gaussian")
    return dataframe_gaussian
@pytest.fixture
def dataframe_cfour() -> pd.DataFrame:
    data_vault = DataVault(minidatabase_csv)
    dataframe_cfour = data_vault.filter_database("cfour")
    return dataframe_cfour

@pytest.fixture
def parsed_data(conditions: dict, 
                dataframe_gaussian: pd.DataFrame, dataframe_cfour: pd.DataFrame) -> dict:
    """
    Fixture to parse data based on the program (Gaussian or CFOUR).
    """
    parsed_data_dict = {}

    for mol,cond in conditions.items():
        program = cond.program
        molecule, method, basis = mol, cond.method, cond.basis
        if program == 'gaussian':
            aa = dataframe_gaussian[
                (dataframe_gaussian['Name'] == molecule) &
                (dataframe_gaussian['Method'] == method) &
                (dataframe_gaussian['Basis'] == basis)
            ]['file_location']
            # filename = directory+aa.iloc[0]
            filename = aa.iloc[0]

            gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
            parser = GaussianParser(gout)
        elif program == 'cfour':
            aa = dataframe_cfour[
                (dataframe_cfour['Name'] == molecule) &
                (dataframe_cfour['Method'] == method) &
                (dataframe_cfour['Basis'] == basis)
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
def terms_amplitudes(terms_collection: dict, spectrum_setup: dict) -> dict:
    """
    Fixture to calculate amplitudes using TermsEvaluator.
    """

    ampls = {}

    for mol,spec_setup in spectrum_setup.items():
        te, precalc_data = terms_collection[mol]
        for tid in te.terms:
            te.terms[tid].precalc_data = precalc_data
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