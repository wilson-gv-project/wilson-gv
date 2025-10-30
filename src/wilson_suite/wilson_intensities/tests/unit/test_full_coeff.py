import wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff as fac
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, VibState, PropsCollection
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def test_identify_precalc_unique_coeff_parts():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    
    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)
    
    # for i in need_to_precalc['avrg_tensors']:
    #     print(i)
    print(need_to_precalc)
    print('\n')
    for piece in need_to_precalc:
        print('\n', piece)
        for sp in need_to_precalc[piece]:
            print(sp)


def generate_props_data4modes():
    import numpy as np
    return {'dipgrad': np.zeros((4, 3)), 
            'diphess': np.zeros((4, 4, 3)),
            'polgrad': np.zeros((4, 3, 3)), 
            'polhess': np.zeros((4, 4, 3, 3))}

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

    vibdata = VibStatesData(allstates=(VibState(s={'0':1.}, state_label='0', e=964.),
                                       VibState(s={'1':1.}, state_label='1', e=1234.),
                                       VibState(s={'2':1.}, state_label='2', e=3644.)),
                                   harmonic_osc_states_labels=(0, 1, 2))
    
    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)
    settings = {'polarization': 'ZZZZ', 
                'number_of_nmodes': 4, 
                'props_data': props_data,
                'vibstates_data': vibdata}

    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  settings=settings)
    # print('\n', results['avrg_expr_tensor_mapping'], len(results['avrg_expr_tensor_mapping']), '\n')
    # print(need_to_precalc['avrg_expr_tensor_mapping'])
    
    for k in results:
        print(k)
        if k=='avrg_expr_tensor_mapping':
            print(results[k])
        for i in results[k]:
            print(i)
            print(results[k][i])

# [dipgrad['a'][1] * dipgrad['b'][2] * polhess['a', 'b'][0, 3], 
#  polgrad['b'][0, 3] * dipgrad['a'][1] * diphess['a', 'b'][2], 
#  polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2], 
#  polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['b'][2], 
#  polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['a'][2]]

# dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1 * polhess['a', 'b'][0, 3]_d2: dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1 * polhess['a', 'b'][0, 3]_d2, 
# polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * diphess['a', 'b'][2]_d2: polgrad['a'][0, 3]_d1 * dipgrad['b'][1]_d1 * diphess['b', 'a'][2]_d2, 
# polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['c'][2]_d1:      polgrad['a'][0, 3]_d1 * dipgrad['b'][1]_d1 * dipgrad['c'][2]_d1, 
# polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1:      polgrad['a'][0, 3]_d1 * dipgrad['b'][1]_d1 * dipgrad['a'][2]_d1, 
# polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['a'][2]_d1:      polgrad['a'][0, 3]_d1 * dipgrad['b'][1]_d1 * dipgrad['b'][2]_d1

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
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15

    vibdata = VibStatesData(allstates=(VibState(s={'0':1.}, state_label='0', e=964.),
                                       VibState(s={'1':1.}, state_label='1', e=1234.),
                                       VibState(s={'2':1.}, state_label='2', e=3644.)),
                                   harmonic_osc_states_labels=(0, 1, 2))

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)
    print("need_to_precalc['avrg_tensors']\n", need_to_precalc['avrg_tensors'])
    settings = {'polarization': 'ZZZZ', 
                'number_of_nmodes': 4, 
                'props_data': props_data, 
                'vibstates_data': vibdata}

    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  settings=settings)
    # print('>>> results.keys():\n')
    print(">>> results['avrg_tensors']:")
    for k in results['avrg_tensors'].keys():
        print(k)
    print('\n')
    # print(">>> results['avrg_tensors']:", results['avrg_tensors'].keys(), '\n')

    # print('>>> terms_select[0].props:', PropsCollection(terms_select[0].props), '\n')
    # print('>>> terms_select[0]:', terms_select[0], '\n')

    terms_select[0].props = PropsCollection(terms_select[0].props).sort().props
    terms_select[-1].props = PropsCollection(terms_select[-1].props).sort().props

    print('-----------')
    print(terms_select[0].to_latex())
    print(terms_select[-1].to_latex())
    print('-----------')

    # term0_coeff = fac.evaluate_term_coeffs(term=terms_select[0], 
    #                                        relevant_indices=[{'a': 0, 'b': 0}], 
    #                                        pre_eval_data=results)
    # print('\nterm0_coeff', term0_coeff)
    
    
    # print(terms_select[-1], '\n')
    # term1_coeff = fac.evaluate_term_coeffs(term=terms_select[-1], 
    #                                        relevant_indices=[{'a': 0, 'b': 0}], 
    #                                        pre_eval_data=results)
    # print('\nterm1_coeff', term1_coeff)

    term1_coeff = fac.evaluate_term_coeffs(term=terms_select[-1], 
                                           relevant_indices=[{'a': 0, 'b': 0, 'c': 1}], 
                                           pre_eval_data=results)
    print('\nterm1_coeff', term1_coeff)