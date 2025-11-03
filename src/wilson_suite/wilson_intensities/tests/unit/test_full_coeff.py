import wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff as fac
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, PropsCollection, EvaluationDataAndConfigs
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
from wilson_suite.wilson_main.abstractions import MolPropsCollection, MolecularProperty
import pytest
import numpy as np

from wilson_suite.wilson_main.abstractions import VibState

def test_identify_precalc_unique_coeff_parts():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    
    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)
    
    print(need_to_precalc)
    print('\n')
    for piece in need_to_precalc:
        print('\n', piece)
        for sp in need_to_precalc[piece]:
            print(sp)


def generate_props_data4modes():
    return {'dipgrad': np.zeros((4, 3)), 
            'diphess': np.zeros((4, 4, 3)),
            'polgrad': np.zeros((4, 3, 3)), 
            'polhess': np.zeros((4, 4, 3, 3)),
            'cff':     np.zeros((4, 4, 4)),
            'qff':     np.zeros((4, 4, 4, 4)),
            }

def test_precalculate_unique_coeff_parts():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1, -1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15

    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={'0':1.}, state_label='0', energy=964.),
                                       VibState(harm_quanta_coeffs={'1':1.}, state_label='1', energy=1234.),
                                       VibState(harm_quanta_coeffs={'2':1.}, state_label='2', energy=3644.)),
                                   harmonic_osc_states_labels=(0, 1, 2))
    
    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)
    
    settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
                                        vibstates_data=vibdata,
                                        polarization='ZZZZ',
                                        number_of_nmodes=4)
    
    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=settings)
    print(results)



def test_evaluate_term_coeffs():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select: list[VibPerturbedTerm] = [terms_fuller_flat[tID] for tID in t_inds]
    
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7

    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
                                        vibstates_data=vibdata,
                                        polarization='ZZZZ',
                                        number_of_nmodes=4)

    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=settings)

    terms_select[0].props = PropsCollection(terms_select[0].props).sort().props
    terms_select[-1].props = PropsCollection(terms_select[-1].props).sort().props
    terms_select[-2].props = PropsCollection(terms_select[-2].props).sort().props
    terms_select[-3].props = PropsCollection(terms_select[-3].props).sort().props

    print('-----------')
    print(terms_select[0].to_latex())
    print(terms_select[-1].to_latex())
    print('-----------')


    term0_coeff = fac.evaluate_term_coeffs(term=terms_select[0], 
                                           relevant_indices=[{'a': 0, 'b': 0}], 
                                           necessary_data=results)
    print('\nterm0_coeff', term0_coeff, '\n')
    print('======================')
    

    # indices are not fully defined : missing 'c' value here
    with pytest.raises(ValueError, match="index_dict - {'a': 0, 'b': 0} - is missing values for some indices"):
        fac.evaluate_term_coeffs(term=terms_select[-1], relevant_indices=[{'a': 0, 'b': 0}], necessary_data=results)
    
    
    term12_coeff = fac.evaluate_term_coeffs(term=terms_select[-1], 
                                           relevant_indices=[{'a': 0, 'b': 0, 'c': 1}], 
                                           necessary_data=results)
    print('\nterm12_coeff', term12_coeff, '\n')
    print('======================')

    term13_coeff = fac.evaluate_term_coeffs(term=terms_select[-3], 
                                           relevant_indices=[{'a': 0, 'b': 1, 'c': 1}], 
                                           necessary_data=results)
    print('\nterm13_coeff', term13_coeff, '\n')
    print('======================')