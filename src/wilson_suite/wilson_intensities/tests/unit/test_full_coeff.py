import wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff as fac
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, PropsCollection, EvaluationDataAndConfigs
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_main.abstractions import MolPropsCollection, MolecularProperty
import pytest
import numpy as np

from wilson_suite.wilson_main.abstractions import VibState

def test_identify_precalc_unique_coeff_parts():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()
    
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
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()
    
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
                                   harmonic_osc_states_labels=(0, 1, 2), number_of_nmodes=3)
    
    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
                                        vibstates_data=vibdata,
                                        pulse_polarization_vector=[1., 1., 1.],
                                        number_of_nmodes=4, 
                                        nm_inds_choices=[0, 1, 2, 3])
    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=settings)
    print(results)



def test_evaluate_term_coeffs_single_c_ind_contrib():
    """
    testing coefficient evaluation for terms - with single contribution of sum over 'c' index in mech terms

    [x] computed per term coefficient for mech terms has also a sum over index 'c'
    [x] computed per term coefficient for el terms is a product of orient.avrg and vibene denominator
    but need to investigate orient avrg tensors and how values are computed and used
    """
    print()
    from wilson_suite.fixtures import get_terms_from_json
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops

    terms_fuller_flat = get_terms_from_json()
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select: list[VibPerturbedTerm] = [terms_fuller_flat[tID] for tID in t_inds]
    
    # though in this test really using 3 out of these 4 modes for simplicity
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = props_data['cff'][1, 0, 1] = props_data['cff'][1, 1, 0] = 0.7

    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                                       VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3722., harmonic_WF=False),
                                       VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=6127., harmonic_WF=False),
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2), number_of_nmodes=3)

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
                                        vibstates_data=vibdata,
                                        pulse_polarization_vector=[1., 1., 1.],
                                        number_of_nmodes=3, 
                                        nm_inds_choices=[0, 1, 2])

    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=settings)

    terms_select[0].props = PropsCollection(terms_select[0].props).sort().props
    terms_select[-1].props = PropsCollection(terms_select[-1].props).sort().props
    terms_select[-2].props = PropsCollection(terms_select[-2].props).sort().props
    terms_select[-3].props = PropsCollection(terms_select[-3].props).sort().props


    ### term 0 -- el
    term0_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[0], 
                                           relevant_indices=[{'a': 0, 'b': 0}], 
                                           necessary_data=(settings, results))
    terms_select[0].anharmonicity = (1, 0)
    avrg_expressions_t0 = avrgprops.PropsCollection(props=terms_select[0].props).get_averaged_props().sort()
    
    t0_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t0]]
    
    
    # based on data above - frac * vibene denom * orient avrg
    ref_term0_coeff = -1./4 * 1./convNu2Ene(964.)/convNu2Ene(964.) * t0_avrg_tensor[0, 0]
    term0_coeff = list(term0_coeff_dict.values())[0] # list with single element

    assert term0_coeff == ref_term0_coeff

    ### term 1 -- el
    term1_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[1], 
                                           relevant_indices=[{'a': 0, 'b': 0}], 
                                           necessary_data=(settings, results))
    terms_select[1].anharmonicity = (1, 0)
    avrg_expressions_t1 = avrgprops.PropsCollection(props=terms_select[1].props).get_averaged_props().sort()
    
    t1_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t1]]
    assert avrg_expressions_t1 != avrg_expressions_t0
    
    # based on data above - frac * vibene denom * orient avrg
    ref_term1_coeff = -1./4 * 1./convNu2Ene(964.)/convNu2Ene(964.) * t1_avrg_tensor[0, 0]
    term1_coeff = list(term1_coeff_dict.values())[0] # list with single element
    assert term1_coeff == ref_term1_coeff
    
    ### term 2 -- mech
    term2_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[2], 
                                           relevant_indices=[{'a': 1, 'b': 1}], 
                                           necessary_data=(settings, results))
    terms_select[2].anharmonicity = (0, 1)
    avrg_expressions_t2 = avrgprops.PropsCollection(props=terms_select[2].props).get_averaged_props().sort()
    
    t2_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t2]]
    assert avrg_expressions_t2 != avrg_expressions_t1 != avrg_expressions_t0

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    assert props_data['cff'][1, 1, 0] != 0.
    assert t2_avrg_tensor[1, 1, 0] != 0.
    assert np.count_nonzero(props_data['cff'][1, 1, :3]) == 1
    assert np.count_nonzero(t2_avrg_tensor[1, 1, :]) == 1

    # based on data above - frac * vibene denom * orient avrg * CFF 
    # avrg_expressions_t2 ---- polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['c'][2]_d1
    ref_term2_coeff = 1./8 * 1./convNu2Ene(1234.)/convNu2Ene(1234.)/convNu2Ene(c_ene) * t2_avrg_tensor[1, 1, :] * props_data['cff'][1, 1, :3] / convNu2Ene(c_vibdiff)

    term2_coeff = list(term2_coeff_dict.values())[0] # list with single element
    assert np.isclose(term2_coeff, np.sum(ref_term2_coeff))

    ### term 3 -- mech
    term3_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[3], 
                                           relevant_indices=[{'a': 0, 'b': 1}], 
                                           necessary_data=(settings, results))
    terms_select[3].anharmonicity = (0, 1)
    avrg_expressions_t3 = avrgprops.PropsCollection(props=terms_select[3].props).get_averaged_props().sort()
    
    t3_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t3]]
    assert avrg_expressions_t3 != avrg_expressions_t2 != avrg_expressions_t1 != avrg_expressions_t0

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    assert props_data['cff'][0, 1, 1] != 0.
    assert np.count_nonzero(props_data['cff'][0, :3, :3]) == 1
    assert t3_avrg_tensor[1, 0, 1] != 0.

    # based on data above - frac * vibene denom * orient avrg * CFF 
    # 'a': 0, 'b': 1
    # avrg_expressions_t3 ---- polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1
    # CFF --- acc
    # t3_avrg_tensor[1, 0, 1] because it goes b,a,c
    ref_term3_coeff = 1./16 * 1./convNu2Ene(964.)/convNu2Ene(1234.)/convNu2Ene(c_ene) * t3_avrg_tensor[1, 0, 1] * props_data['cff'][0, :3, :3] / convNu2Ene(964.)

    term3_coeff = list(term3_coeff_dict.values())[0] # list with single element
    assert np.isclose(term3_coeff, np.sum(ref_term3_coeff))

    ### term 4 -- mech
    term4_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[4], 
                                           relevant_indices=[{'a': 0, 'b': 0}], 
                                           necessary_data=(settings, results))
    terms_select[4].anharmonicity = (0, 1)
    avrg_expressions_t4 = avrgprops.PropsCollection(props=terms_select[4].props).get_averaged_props().sort()
    
    t4_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t4]]
    assert avrg_expressions_t4 != avrg_expressions_t3 != avrg_expressions_t2 != avrg_expressions_t1 != avrg_expressions_t0

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    assert props_data['cff'][0, 1, 1] != 0.
    assert np.count_nonzero(props_data['cff'][0, :3, :3]) == 1
    assert t4_avrg_tensor[0, 0, 0] != 0

    # based on data above - frac * vibene denom * orient avrg * CFF 
    ref_term4_coeff = 1./16 * 1./convNu2Ene(964.)/convNu2Ene(964.)/convNu2Ene(c_ene) * t4_avrg_tensor[0, 0, 0] * props_data['cff'][0, :3, :3] / convNu2Ene(964.)
    term4_coeff = list(term4_coeff_dict.values())[0] # list with single element
    assert np.isclose(term4_coeff, np.sum(ref_term4_coeff))



def test_evaluate_term_coeffs_multi_c_ind_contrib():
    """
    testing coefficient evaluation for terms - with 2 contribution of sum over 'c' index in mech terms
    
    [x] computed per term coefficient for mech terms has also a sum over index 'c'
    [x] computed per term coefficient for el terms is a product of orient.avrg and vibene denominator
    but need to investigate orient avrg tensors and how values are computed and used
    """
    print()
    from wilson_suite.fixtures import get_terms_from_json
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops

    terms_fuller_flat = get_terms_from_json()
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select: list[VibPerturbedTerm] = [terms_fuller_flat[tID] for tID in t_inds]
    
    # though in this test really using 3 out of these 4 modes for simplicity
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['dipgrad'][1, 1] = -0.15
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = props_data['cff'][1, 0, 1] = props_data['cff'][1, 1, 0] = 0.7
    props_data['cff'][0, 0, 1] = props_data['cff'][0, 1, 0] = props_data['cff'][1, 0, 0] = 0.4
    props_data['cff'][1, 1, 1] = 0.2

    
    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                                       VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3722., harmonic_WF=False),
                                       VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=6127., harmonic_WF=False),
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2), number_of_nmodes=3)

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    settings = EvaluationDataAndConfigs(props_data=MolPropsCollection(props),
                                        vibstates_data=vibdata,
                                        pulse_polarization_vector=[1., 1., 1.],
                                        number_of_nmodes=3, 
                                        nm_inds_choices=[0, 1, 2])

    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=settings)

    terms_select[0].props = PropsCollection(terms_select[0].props).sort().props
    terms_select[-1].props = PropsCollection(terms_select[-1].props).sort().props
    terms_select[-2].props = PropsCollection(terms_select[-2].props).sort().props
    terms_select[-3].props = PropsCollection(terms_select[-3].props).sort().props

    
    ### term 2 -- mech
    term2_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[2], 
                                           relevant_indices=[{'a': 1, 'b': 1}], 
                                           necessary_data=(settings, results))
    terms_select[2].anharmonicity = (0, 1)
    avrg_expressions_t2 = avrgprops.PropsCollection(props=terms_select[2].props).get_averaged_props().sort()
    
    t2_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t2]]

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    assert props_data['cff'][1, 1, 0] != 0.
    assert props_data['cff'][1, 1, 1] != 0.
    assert t2_avrg_tensor[1, 1, 0] != 0.
    assert t2_avrg_tensor[1, 1, 1] != 0.
    assert np.count_nonzero(props_data['cff'][1, 1, :3]) == 2
    assert np.count_nonzero(t2_avrg_tensor[1, 1, :]) == 2

    # based on data above - frac * vibene denom * orient avrg * CFF 
    # avrg_expressions_t2 ---- polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['c'][2]_d1 --- bac
    # CFF ---- abc
    ref_term2_coeff = 1./8 * 1./convNu2Ene(1234.)/convNu2Ene(1234.)/convNu2Ene(c_ene) * t2_avrg_tensor[1, 1, :] * props_data['cff'][1, 1, :3] / convNu2Ene(c_vibdiff)

    term2_coeff = list(term2_coeff_dict.values())[0] # list with single element
    assert np.isclose(term2_coeff, np.sum(ref_term2_coeff))


    ### term 3 -- mech
    # data prep
    props_data['cff'][0, 1, 1] = props_data['cff'][1, 0, 1] = props_data['cff'][1, 1, 0] = 0.2
    props_data['cff'][0, 0, 1] = props_data['cff'][0, 1, 0] = props_data['cff'][1, 0, 0] = 0.
    props_data['cff'][1, 1, 1] = 0.0
    props_data['cff'][0, 0, 0] = 0.4

    term3_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[3], 
                                           relevant_indices=[{'a': 0, 'b': 1}], 
                                           necessary_data=(settings, results))
    terms_select[3].anharmonicity = (0, 1)
    avrg_expressions_t3 = avrgprops.PropsCollection(props=terms_select[3].props).get_averaged_props().sort()
    
    t3_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t3]]
    assert avrg_expressions_t3 != avrg_expressions_t2

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    
    assert props_data['cff'][0, 0, 0] != 0.
    assert props_data['cff'][0, 1, 1] != 0.
    assert np.count_nonzero(props_data['cff'][0, :3, :3]) == 2
    assert t3_avrg_tensor[1, 0, 1] != 0.

    
    # NON_AVRG zero -  cff['a', 'c', 'c'][]_d3 {'a': 0, 'b': 1, 'c': 0} --- should take 

    # based on data above - frac * vibene denom * orient avrg * CFF 
    # 'a': 0, 'b': 1
    # avrg_expressions_t3 ---- polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1   -- bab
    # CFF --- acc
    # t3_avrg_tensor[1, 0, 1] because it goes b,a,c in general expression
    ref_term3_coeff = 1./16 * 1./convNu2Ene(964.)/convNu2Ene(1234.)/convNu2Ene(c_ene) * t3_avrg_tensor[1, 0, 1] * props_data['cff'][0, :3, :3] / convNu2Ene(964.)
    term3_coeff = list(term3_coeff_dict.values())[0] # list with single element

    assert np.isclose(term3_coeff, np.sum(ref_term3_coeff))




    ### term 4 -- mech
    term4_coeff_dict = fac.evaluate_term_coeffs(term=terms_select[4], 
                                           relevant_indices=[{'a': 0, 'b': 0}], 
                                           necessary_data=(settings, results))
    terms_select[4].anharmonicity = (0, 1)
    avrg_expressions_t4 = avrgprops.PropsCollection(props=terms_select[4].props).get_averaged_props().sort()
    
    t4_avrg_tensor = results.avrg_tensors[results.avrg_expr_tensor_mapping[avrg_expressions_t4]]
    assert avrg_expressions_t4 != avrg_expressions_t3 != avrg_expressions_t2

    c_ene = np.array([964., 1234., 3644.])

    # 1,1,0; 1,1,1; 1,1,2
    c_vibdiff = np.array([3318., 3722., 6127.])

    assert props_data['cff'][0, 0, 0] != 0.
    assert props_data['cff'][0, 1, 1] != 0.
    assert np.count_nonzero(props_data['cff'][0, :3, :3]) == 2
    assert t4_avrg_tensor[0, 0, 0] != 0

    # based on data above - frac * vibene denom * orient avrg * CFF 
    ref_term4_coeff = 1./16 * 1./convNu2Ene(964.)/convNu2Ene(964.)/convNu2Ene(c_ene) * t4_avrg_tensor[0, 0, 0] * props_data['cff'][0, :3, :3] / convNu2Ene(964.)
    term4_coeff = list(term4_coeff_dict.values())[0] # list with single element

    assert np.isclose(term4_coeff, np.sum(ref_term4_coeff))


