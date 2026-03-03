"""
SpectralFeature.amplitude_coeff evaluation tests
- [ ] el term coeff
- [ ] 2 el terms coeff
- [ ] mech term coeff
- [ ] 2 mech terms coeff
- [ ] mech and el terms in one feat coeff

- [ ] evaluate_single_index_dict() -- in src/wilson_suite/wilson_intensities/amplitudes/full_amplitude_coeff.py
- [ ] eval_avrg_per_indexdict() -- in src/wilson_suite/wilson_intensities/amplitudes/full_amplitude_coeff.py

averaging tests:
- test_getGeneralPolarizationAveragingExpression() in src/wilson_suite/wilson_intensities/tests/unit/test_averaging.py
- test_precalculate_avrg_tensor() in src/wilson_suite/wilson_intensities/tests/unit/test_averaged_props.py
"""
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

from .testutils import MakeObjects
import wilson_suite as ws
import numpy as np
np.set_printoptions(linewidth=180, precision=3)


def test_ampl_coeff_sum_3_terms():
    """
    [x] feature coefficient is a sum of all contributing terms with same resmotif

    
    there are tests in src/wilson_suite/wilson_intensities/tests/unit/test_full_coeff.py::test_evaluate_term_coeffs_single_c_ind_contrib
    and test_evaluate_term_coeffs_multi_c_ind_contrib:
    - products of components
    - vibene denom calculation
    - single contribution in the sum over index 'c'

    need more tests for orient. avrg tensors - evaluation and use
    """
    from .testutils import get_from_pkl_features
    from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import (process_resonance_motifs, 
                                                                          evaluate_terms_coeffs, 
                                                                          precalculate_unique_coeff_parts, 
                                                                          identify_precalc_unique_coeff_parts,
                                                                          get_features_to_draw)
    from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_coeff_for_feat

    # getting all propper terms here and realistic combinations of contributing terms
    feats, hashmap_terms = get_from_pkl_features('data_for_tests/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl', 10.)
    terms = list(hashmap_terms.values())

    # selecting some terms - all (2) el and 4 mech [2 of each resmotif]
    # each feature (found res location) would have 3 terms contribution to the coefficient
    terms_selection = [0, 1, 3, 5, 8, 9]
    terms = [terms[s] for s in terms_selection]
    
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    terms_json = VibPerturbedTerm.load_many_from_json(filepath='tests/tests_dft_vs_cc_paper/terms_selection.json')
    assert sorted([t.h() for t in terms]) == sorted([t.h() for t in terms_json])

    p_names = ['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff']
    props = [MakeObjects.mk_prop_with_vals(pr_name) for pr_name in p_names]
    
    states = MakeObjects.mk_vibstates_states()
    include = [0, 1, 2]

    data_configs = MakeObjects.mk_data_for_eval(list_of_states=states,
                                                include_states_list=include,
                                                list_of_props=props,
                                                pulse_polarization_vector=[1., 1., 1.])

    # terms define expressions for coefficients
    # terms + data => coefficients
    # data could be orginized as prepared with a function...
    
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()

    motif_locs, terms_for_motifs = process_resonance_motifs(terms,
                                             data_configs.vibstates_data,
                                             vibdiff_cache)

    need_precalc = identify_precalc_unique_coeff_parts(terms=terms)
    precalculated = precalculate_unique_coeff_parts(need_precalc, 
                                                    data_and_configs=data_configs)
    coefficients = evaluate_terms_coeffs(terms,
                                         motif_locs, # this is given in feature.term_contributions: motif to resloc dict
                                         data_configs,
                                         precalculated)

    gamma = 10.2
    # these feature will now also have coefficients computed above
    features_to_draw = get_features_to_draw(motif_res_loc=motif_locs, 
                                            terms_for_motifs=terms_for_motifs,
                                            term_coeffs_per_index=coefficients,
                                            lineshape_parameter=gamma) # in cm-1 in features

    # taking 2 features for testing
    # It is in a list that has no fixed order of features, so the same index in different run might refer to different feature
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject
    selected_features = [f for f in features_to_draw 
                         if f.location == ResLocGeoObject({'A':1119.5, 'B': 744.5}) 
                         or f.location == ResLocGeoObject({'A':964., 'B': 270.})]
    ws.intensities.amplitudes.spectrum_composition.SpectralFeature.print_list_features(selected_features)

    # evaluating features coefficients based on features (a wrapper function that is using evaluate_terms_coeffs func)
    # feature 0
    feat0_coeff = evaluate_coeff_for_feat(selected_features[0],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat0_coeff = {k.h(): v for k,v in feat0_coeff.items()}

    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[0].amplitude_coeff == sum([c for i in result_feat0_coeff.values() for c in i.values()])

    # feature 1
    feat1_coeff = evaluate_coeff_for_feat(selected_features[1],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat1_coeff = {k.h(): v for k,v in feat1_coeff.items()}

    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[1].amplitude_coeff == sum([c for i in result_feat1_coeff.values() for c in i.values()])



def test_ampl_coeff_1mech_terms():
    """
    Calculation of coeffs for 2 features with 1 contribution of mech term (in 2 feats - 2 different terms contributions)
    """
    from .testutils import get_from_pkl_features
    from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import (process_resonance_motifs, 
                                                                          evaluate_terms_coeffs, 
                                                                          precalculate_unique_coeff_parts, 
                                                                          identify_precalc_unique_coeff_parts,
                                                                          get_features_to_draw)
    from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_coeff_for_feat

    # getting all propper terms here and realistic combinations of contributing terms
    feats, hashmap_terms = get_from_pkl_features('data_for_tests/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl', 10.)
    terms = list(hashmap_terms.values())

    # selecting some terms - 2 mech [1 of each resmotif]
    # each feature (found res location) would have 3 terms contribution to the coefficient
    terms_selection = [3, 9]
    terms = [terms[s] for s in terms_selection]
    
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    terms_json = VibPerturbedTerm.load_many_from_json(filepath='tests/tests_dft_vs_cc_paper/terms_selection.json')

    p_names = ['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff']
    props = [MakeObjects.mk_prop_with_vals(pr_name) for pr_name in p_names]
    
    states = MakeObjects.mk_vibstates_states()
    include = [0, 1, 2]

    data_configs = MakeObjects.mk_data_for_eval(list_of_states=states,
                                                include_states_list=include,
                                                list_of_props=props,
                                                pulse_polarization_vector=[1., 1., 1.])

    # terms define expressions for coefficients
    # terms + data => coefficients
    # data could be orginized as prepared with a function...
    
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()

    motif_locs, terms_for_motifs = process_resonance_motifs(terms,
                                             data_configs.vibstates_data,
                                             vibdiff_cache)

    need_precalc = identify_precalc_unique_coeff_parts(terms=terms)
    precalculated = precalculate_unique_coeff_parts(need_precalc, 
                                                    data_and_configs=data_configs)
    coefficients = evaluate_terms_coeffs(terms,
                                         motif_locs, # this is given in feature.term_contributions: motif to resloc dict
                                         data_configs,
                                         precalculated)

    gamma = 10.2
    # these feature will now also have coefficients computed above
    features_to_draw = get_features_to_draw(motif_res_loc=motif_locs, 
                                            terms_for_motifs=terms_for_motifs,
                                            term_coeffs_per_index=coefficients,
                                            lineshape_parameter=gamma) # in cm-1 in features

    # taking 2 features for testing
    # It is in a list that has no fixed order of features, so the same index in different run might refer to different feature
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject
    selected_features = []
    
    # now features list will be with this order of features
    for f in features_to_draw:
        if f.location == ResLocGeoObject({'A':1119.5, 'B': 744.5}):
            selected_features.append(f)

    for f in features_to_draw:
        if f.location == ResLocGeoObject({'A':964., 'B': 270.}):
            selected_features.append(f)

    ws.intensities.amplitudes.spectrum_composition.SpectralFeature.print_list_features(selected_features)

    # evaluating features coefficients based on features (a wrapper function that is using evaluate_terms_coeffs func)
    # feature 0
    feat0_coeff = evaluate_coeff_for_feat(selected_features[0],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat0_coeff = {k.h(): v for k,v in feat0_coeff.items()}
    assert len(result_feat0_coeff) == 1
    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[0].amplitude_coeff == sum([c for i in result_feat0_coeff.values() for c in i.values()])

    from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_term_coeffs
    # only single term contrib - mech term
    t_feat0 = hashmap_terms[selected_features[0].term_contributions[0].term_ids[0]]

    res_t_feat0 = evaluate_term_coeffs(term=t_feat0, 
                         relevant_indices=[selected_features[0].term_contributions[0].states_parameters[0].to_dict()],
                         necessary_data=(data_configs, precalculated))
    # single feature computed coeff here
    assert list(res_t_feat0.values())[0] == selected_features[0].amplitude_coeff

    # feature 1
    feat1_coeff = evaluate_coeff_for_feat(selected_features[1],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat1_coeff = {k.h(): v for k,v in feat1_coeff.items()}
    assert len(result_feat1_coeff) == 1
    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[1].amplitude_coeff == sum([c for i in result_feat1_coeff.values() for c in i.values()])

    # only single term contrib - mech term
    t_feat1 = hashmap_terms[selected_features[1].term_contributions[0].term_ids[0]]
    res_t_feat1 = evaluate_term_coeffs(term=t_feat1, 
                         relevant_indices=[selected_features[1].term_contributions[0].states_parameters[0].to_dict()],
                         necessary_data=(data_configs, precalculated))
    assert list(res_t_feat1.values())[0] == selected_features[1].amplitude_coeff


    t_feat0_a = selected_features[0].term_contributions[0].states_parameters[0].to_dict()['a']
    t_feat0_b = selected_features[0].term_contributions[0].states_parameters[0].to_dict()['b']
    assert t_feat0_a == 0
    assert t_feat0_b == 0

    t_feat1_a = selected_features[1].term_contributions[0].states_parameters[0].to_dict()['a']
    t_feat1_b = selected_features[1].term_contributions[0].states_parameters[0].to_dict()['b']
    assert t_feat1_a == 1
    assert t_feat1_b == 2

    c_ene = np.array([1119.5, 964., 1234.])

    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
    avrg_expressions_t_feat0 = avrgprops.PropsCollection(props=t_feat0.props).get_averaged_props().sort()
    avrg_expressions_t_feat1 = avrgprops.PropsCollection(props=t_feat1.props).get_averaged_props().sort()
    
    avrg_f0 = precalculated.avrg_tensors[precalculated.avrg_expr_tensor_mapping[avrg_expressions_t_feat0]]
    avrg_f1 = precalculated.avrg_tensors[precalculated.avrg_expr_tensor_mapping[avrg_expressions_t_feat1]]
    assert np.allclose(avrg_f0, avrg_f1)
    
    
    cff_f0_val = data_configs.props_data.get('cff').vals[0, :3, :3]
    avrg_f0_val = avrg_f0[t_feat0_b, t_feat0_a, t_feat0_b] # bab

    # single feature computed coeff here (reference value) from expression using data
    ref_res_t_feat0 = -1./16 * cff_f0_val * avrg_f0_val * 1./convNu2Ene(1119.5)/convNu2Ene(1119.5)/convNu2Ene(c_ene) / convNu2Ene(1864.-1119.5)
    assert np.allclose(np.sum(ref_res_t_feat0), selected_features[0].amplitude_coeff)

    
    cff_f1_val = data_configs.props_data.get('cff').vals[1, 2, :3]
    avrg_f1_val = avrg_f1[:3, t_feat1_a, t_feat1_b] # cab

    b_p_c = np.array([2274., 2360., 2362.])
    
    # single feature computed coeff here (reference value) from expression using data
    ref_res_t_feat1 = 1./8  * cff_f1_val * avrg_f1_val * 1./convNu2Ene(964.)/convNu2Ene(1234.)/convNu2Ene(c_ene) / convNu2Ene(b_p_c-964.)
    assert np.allclose(np.sum(ref_res_t_feat1), selected_features[1].amplitude_coeff)



def test_ampl_coeff_1el_terms():
    """
    Calculation of coeffs for 2 features with 1 contribution of mech term (in 2 feats - 2 different terms contributions)
    """
    from .testutils import get_from_pkl_features
    from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import (process_resonance_motifs, 
                                                                          evaluate_terms_coeffs, 
                                                                          precalculate_unique_coeff_parts, 
                                                                          identify_precalc_unique_coeff_parts,
                                                                          get_features_to_draw)
    from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_coeff_for_feat

    # getting all propper terms here and realistic combinations of contributing terms
    feats, hashmap_terms = get_from_pkl_features('data_for_tests/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl', 10.)
    terms = list(hashmap_terms.values())

    # selecting some terms - 2 mech [1 of each resmotif]
    # each feature (found res location) would have 3 terms contribution to the coefficient
    terms_selection = [0, 1]
    terms = [terms[s] for s in terms_selection]
    
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    terms_json = VibPerturbedTerm.load_many_from_json(filepath='tests/tests_dft_vs_cc_paper/terms_selection.json')

    p_names = ['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff']
    props = [MakeObjects.mk_prop_with_vals(pr_name) for pr_name in p_names]
    
    states = MakeObjects.mk_vibstates_states()
    include = [0, 1, 2]

    data_configs = MakeObjects.mk_data_for_eval(list_of_states=states,
                                                include_states_list=include,
                                                list_of_props=props,
                                                pulse_polarization_vector=[1., 1., 1.])

    # terms define expressions for coefficients
    # terms + data => coefficients
    # data could be orginized as prepared with a function...
    
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()

    motif_locs, terms_for_motifs = process_resonance_motifs(terms,
                                             data_configs.vibstates_data,
                                             vibdiff_cache)

    need_precalc = identify_precalc_unique_coeff_parts(terms=terms)
    precalculated = precalculate_unique_coeff_parts(need_precalc, 
                                                    data_and_configs=data_configs)
    coefficients = evaluate_terms_coeffs(terms,
                                         motif_locs, # this is given in feature.term_contributions: motif to resloc dict
                                         data_configs,
                                         precalculated)

    gamma = 10.2
    # these feature will now also have coefficients computed above
    features_to_draw = get_features_to_draw(motif_res_loc=motif_locs, 
                                            terms_for_motifs=terms_for_motifs,
                                            term_coeffs_per_index=coefficients,
                                            lineshape_parameter=gamma) # in cm-1 in features

    # taking 2 features for testing
    # It is in a list that has no fixed order of features, so the same index in different run might refer to different feature
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject
    selected_features = []
    
    # now features list will be with this order of features
    for f in features_to_draw:
        if f.location == ResLocGeoObject({'A':1119.5, 'B': 744.5}):
            selected_features.append(f)

    for f in features_to_draw:
        if f.location == ResLocGeoObject({'A':1119.5, 'B': 0.}):
            selected_features.append(f)

    ws.intensities.amplitudes.spectrum_composition.SpectralFeature.print_list_features(selected_features)

    # evaluating features coefficients based on features (a wrapper function that is using evaluate_terms_coeffs func)
    # feature 0
    feat0_coeff = evaluate_coeff_for_feat(selected_features[0],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat0_coeff = {k.h(): v for k,v in feat0_coeff.items()}
    assert len(result_feat0_coeff) == 1
    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[0].amplitude_coeff == sum([c for i in result_feat0_coeff.values() for c in i.values()])

    from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_term_coeffs
    # only single term contrib - mech term
    t_feat0 = hashmap_terms[selected_features[0].term_contributions[0].term_ids[0]]

    res_t_feat0 = evaluate_term_coeffs(term=t_feat0, 
                         relevant_indices=[selected_features[0].term_contributions[0].states_parameters[0].to_dict()],
                         necessary_data=(data_configs, precalculated))
    # single feature computed coeff here
    assert list(res_t_feat0.values())[0] == selected_features[0].amplitude_coeff


    # feature 1
    feat1_coeff = evaluate_coeff_for_feat(selected_features[1],
                                         {t.h(): t for t in terms_json},
                                         data_configs,
                                         precalculated)
    result_feat1_coeff = {k.h(): v for k,v in feat1_coeff.items()}
    assert len(result_feat1_coeff) == 1
    # testing that feature coefficient is the sum of computed coeffs per term
    assert selected_features[1].amplitude_coeff == sum([c for i in result_feat1_coeff.values() for c in i.values()])

    # only single term contrib - mech term
    t_feat1 = hashmap_terms[selected_features[1].term_contributions[0].term_ids[0]]
    res_t_feat1 = evaluate_term_coeffs(term=t_feat1, 
                         relevant_indices=[selected_features[1].term_contributions[0].states_parameters[0].to_dict()],
                         necessary_data=(data_configs, precalculated))
    assert list(res_t_feat1.values())[0] == selected_features[1].amplitude_coeff


    t_feat0_a = selected_features[0].term_contributions[0].states_parameters[0].to_dict()['a']
    t_feat0_b = selected_features[0].term_contributions[0].states_parameters[0].to_dict()['b']
    assert t_feat0_a == 0
    assert t_feat0_b == 0

    t_feat1_a = selected_features[1].term_contributions[0].states_parameters[0].to_dict()['a']
    t_feat1_b = selected_features[1].term_contributions[0].states_parameters[0].to_dict()['b']
    assert t_feat1_a == 0
    assert t_feat1_b == 0

    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
    avrg_expressions_t_feat0 = avrgprops.PropsCollection(props=t_feat0.props).get_averaged_props().sort()
    avrg_expressions_t_feat1 = avrgprops.PropsCollection(props=t_feat1.props).get_averaged_props().sort()
    assert avrg_expressions_t_feat0 != avrg_expressions_t_feat1

    # polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * diphess['a', 'b'][2]_d2
    avrg_f0 = precalculated.avrg_tensors[precalculated.avrg_expr_tensor_mapping[avrg_expressions_t_feat0]]
    # polhess['a', 'b'][0, 3]_d2 * dipgrad['a'][1]_d1 * dipgrad['b'][2]_d1
    avrg_f1 = precalculated.avrg_tensors[precalculated.avrg_expr_tensor_mapping[avrg_expressions_t_feat1]]
    
    avrg_f0_val = avrg_f0[t_feat0_a, t_feat0_b] # baab

    # single feature computed coeff here (reference value) from expression using data
    ref_res_t_feat0 = -1./4 * avrg_f0_val * 1./convNu2Ene(1119.5)/convNu2Ene(1119.5)

    assert np.allclose(ref_res_t_feat0, selected_features[0].amplitude_coeff)

    avrg_f1_val = avrg_f1[t_feat0_a, t_feat0_b] # abab
    
    # single feature computed coeff here (reference value) from expression using data
    ref_res_t_feat1 = -1./4  * avrg_f1_val * 1./convNu2Ene(1119.5)/convNu2Ene(1119.5)
    assert np.allclose(np.sum(ref_res_t_feat1), selected_features[1].amplitude_coeff)



########## ORIENTATIONAL AVERAGING TESTS
# --- Reference implementation ---
def reference_rank4_vvvv(p1, p2, p3, p4):
    """
    p1, p2, p3, p4: arrays of shape (n_modes, 3)
    Returns: (n_modes, n_modes, n_modes, n_modes) tensor

        Claude Opus 4.6 extended

    """
    n = p1.shape[0]
    result = np.zeros((n, n, n, n))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    val = 0.0
                    for i in range(3):
                        for j in range(3):
                            if i == j:
                                # (i,i,i,i) -> 3/15, plus three (i,i,j,j) patterns with j=i -> contributes to 3/15
                                val += (3/15) * p1[a,i] * p2[b,i] * p3[c,i] * p4[d,i]
                            else:
                                val += (1/15) * p1[a,i] * p2[b,i] * p3[c,j] * p4[d,j]
                                val += (1/15) * p1[a,i] * p2[b,j] * p3[c,i] * p4[d,j]
                                val += (1/15) * p1[a,i] * p2[b,j] * p3[c,j] * p4[d,i]
                    result[a,b,c,d] = val
    return result


# --- Reference implementation ---
def reference_rank4_vvvv_einsum(p1, p2, p3, p4):
    """
    Independent check using Kronecker delta decomposition.
    
    Claude Opus 4.6 extended

    """
    # Dot product matrices
    d12 = np.einsum('ia,ja->ij', p1, p2)
    d34 = np.einsum('ia,ja->ij', p3, p4)
    d13 = np.einsum('ia,ja->ij', p1, p3)
    d24 = np.einsum('ia,ja->ij', p2, p4)
    d14 = np.einsum('ia,ja->ij', p1, p4)
    d23 = np.einsum('ia,ja->ij', p2, p3)

    return (1.0/15.0) * (
        np.einsum('ab,cd->abcd', d12, d34) +
        np.einsum('ac,bd->abcd', d13, d24) +
        np.einsum('ad,bc->abcd', d14, d23)
    )


def test_reference_implementations_agree():
    """
    Claude Opus 4.6 extended
    
    """
    rng = np.random.default_rng(42)
    n_modes = 4
    p1 = rng.standard_normal((n_modes, 3))
    p2 = rng.standard_normal((n_modes, 3))
    p3 = rng.standard_normal((n_modes, 3))
    p4 = rng.standard_normal((n_modes, 3))

    result_loops = reference_rank4_vvvv(p1, p2, p3, p4)
    result_einsum = reference_rank4_vvvv_einsum(p1, p2, p3, p4)

    np.testing.assert_allclose(result_loops, result_einsum, atol=1e-14)


def test_calculate_avrg_tensor_rank4_vvvv():
    """
    Test rank-4 VVVV averaging of 4 dipole-gradient-like properties
    against analytical reference (Kronecker delta decomposition).

    Claude Opus 4.6 extended
    """
    import numpy as np
    # adjust import paths to your package
    from wilson_suite.wilson_utils.prop_trivname import prop_trivname

    import wilson_suite.wilson_intensities.amplitudes.term_parts as term_abst
    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
    from wilson_suite.wilson_main.abstractions import MolPropsCollection
    import wilson_suite.wilson_derive.abstractions as wd_abst
    

    # --- Synthetic data ---
    rng = np.random.default_rng(42)
    n_modes = 3
    dipgrad_data = rng.standard_normal((n_modes, 3))

    # --- Build expression: 4 rank-1 properties, one per pulse ---
    p1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
    p1.inds = ['a']
    p2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    p2.inds = ['b']
    p3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    p3.inds = ['c']
    p4 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
    p4.inds = ['d']

    avrg_expr = term_abst.PropsCollection(props=[p1, p2, p3, p4])

    # --- Build MolPropsCollection ---
    # All 4 properties share same trivname since they all have ord_el=1, ord_geo=1
    trname = prop_trivname(ord_el=1, ord_geo=1)
    props_data = {trname: dipgrad_data}
    from wilson_suite.wilson_intensities.tests.unit.test_averaged_props import dict_to_proplist
    props = dict_to_proplist(props_data)
    mol_props = MolPropsCollection(props)

    # --- VVVV polarization ---
    # [1.0, 1.0, 1.0] is the laser_pol for all-parallel rank-4
    # (same as get_pol_laser([[0,0,1]]*4))
    result = avrgprops.calculate_avrg_tensor(
        avrg_expression=avrg_expr,
        pulse_polarization_vector=[1.0, 1.0, 1.0],
        props_data=mol_props,
        number_of_nmodes=n_modes,
        nm_inds_choices=list(range(n_modes))
    )

    # --- Reference: same data for all 4 properties ---
    expected = reference_rank4_vvvv_einsum(
        dipgrad_data, dipgrad_data, dipgrad_data, dipgrad_data
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_precalc_retrieval_matches_direct():
    """
    Verify that the grouping/mapping/retrieval pipeline
    gives the same results as direct computation.
    """
    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
    import wilson_suite.wilson_derive.abstractions as wd_abst
    import wilson_suite.wilson_intensities.amplitudes.term_parts as term_abst

    # --- Setup: create several expressions with different index patterns ---
    
    # Expression 1: all different indices (a, b, c, d)
    p1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
    p1.inds = ['a']
    p2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    p2.inds = ['b']
    p3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    p3.inds = ['c']
    p4 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
    p4.inds = ['d']
    expr_abcd = term_abst.PropsCollection(props=[p1, p2, p3, p4])

    # Expression 2: repeated index (a, a, b, c) — different repetition pattern
    q1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
    q1.inds = ['a']
    q2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    q2.inds = ['a']
    q3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    q3.inds = ['b']
    q4 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
    q4.inds = ['c']
    expr_aabc = term_abst.PropsCollection(props=[q1, q2, q3, q4])

    # Expression 3: same pattern as expr_aabc but different labels (b, b, c, d)
    r1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
    r1.inds = ['b']
    r2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    r2.inds = ['b']
    r3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    r3.inds = ['c']
    r4 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
    r4.inds = ['d']
    expr_bbcd = term_abst.PropsCollection(props=[r1, r2, r3, r4])

    all_expressions = [expr_abcd, expr_aabc, expr_bbcd]

    # --- Get mapping ---
    mapping = avrgprops.make_unique_avrg_tensors_mapping(all_expressions)

    print()
    for k,v in mapping.items():
        print(k, '---', v)


    # Test 1: expr_aabc and expr_bbcd should map to the same unique tensor
    assert mapping[expr_aabc] is mapping[expr_bbcd], \
        "Same repetition pattern should map to same unique tensor"
    """

    # Test 2: expr_abcd should map to a different tensor
    assert mapping[expr_abcd] is not mapping[expr_aabc], \
        "Different repetition pattern should map to different tensor"

    # --- Synthetic data ---
    rng = np.random.default_rng(42)
    n_modes = 3
    from wilson_suite.wilson_utils.prop_trivname import prop_trivname
    trname = prop_trivname(ord_el=1, ord_geo=1)
    dipgrad_data = rng.standard_normal((n_modes, 3))
    props = dict_to_proplist({trname: dipgrad_data})
    mol_props = MolPropsCollection(props)

    pol_vec = [1.0, 1.0, 1.0]  # VVVV

    # --- Test 3: Direct computation matches retrieval for each expression ---
    for expr in all_expressions:
        # Direct computation
        direct = avrgprops.calculate_avrg_tensor(
            avrg_expression=expr,
            pulse_polarization_vector=pol_vec,
            props_data=mol_props,
            number_of_nmodes=n_modes,
            nm_inds_choices=list(range(n_modes))
        )

        # Via mapping: compute the unique tensor, then extract with index relabeling
        unique_expr = mapping[expr]
        unique_tensor = avrgprops.calculate_avrg_tensor(
            avrg_expression=unique_expr,
            pulse_polarization_vector=pol_vec,
            props_data=mol_props,
            number_of_nmodes=n_modes,
            nm_inds_choices=list(range(n_modes))
        )

        # The unique tensor has axes ordered by sorted unique index labels.
        # For expr_aabc (indices a,a,b,c -> unique: a,b,c -> 3D tensor),
        # the direct result is also 3D.
        # They should match if the mapping is correct.
        np.testing.assert_allclose(direct, unique_tensor, atol=1e-12,
            err_msg=f"Mismatch for expression with indices "
                    f"{[p.inds for p in expr]}")

    """



def test_precalc_and_retrieval_correctness():
    """
    Verify: for each actual expression, retrieving from the 
    precomputed unique tensor with remapped indices gives the
    same result as computing the actual expression directly.

    Claude Opus 4.6 extended
    """

    import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
    from wilson_suite.wilson_main.abstractions import MolPropsCollection
    from wilson_suite.wilson_intensities.tests.unit.test_averaged_props import dict_to_proplist
    from wilson_suite.wilson_intensities.tests.unit.test_averaged_props import make_avrg_expr
    from wilson_suite.wilson_utils.prop_trivname import prop_trivname
    import itertools

    # --- Setup expressions (use your make_avrg_expr helper) ---
    avrg_expr_abcd = make_avrg_expr([((0,), ('a',)), 
                                      ((1,), ('b',)),
                                      ((2,), ('c',)),
                                      ((3,), ('d',)),])
    avrg_expr_aabc = make_avrg_expr([((0,), ('a',)), 
                                      ((1,), ('a',)),
                                      ((2,), ('b',)),
                                      ((3,), ('c',)),])
    avrg_expr_abbc = make_avrg_expr([((0,), ('a',)), 
                                      ((1,), ('b',)),
                                      ((2,), ('b',)),
                                      ((3,), ('c',)),])

    props_colls = [avrg_expr_abcd, avrg_expr_aabc, avrg_expr_abbc]

    # --- Synthetic data ---
    rng = np.random.default_rng(42)
    n_modes = 3
    trname = prop_trivname(ord_el=1, ord_geo=1)
    dipgrad_data = rng.standard_normal((n_modes, 3))
    props = dict_to_proplist({trname: dipgrad_data})
    mol_props = MolPropsCollection(props)
    pol_vec = [1.0, 1.0, 1.0]
    nm_inds = list(range(n_modes))

    # --- Get mapping and precompute unique tensors ---
    mapping = avrgprops.make_unique_avrg_tensors_mapping(props_colls)
    unique_exprs = set(mapping.values())
    unique_tensors = {}
    for uexpr in unique_exprs:
        unique_tensors[uexpr] = avrgprops.calculate_avrg_tensor(
            avrg_expression=uexpr,
            pulse_polarization_vector=pol_vec,
            props_data=mol_props,
            number_of_nmodes=n_modes,
            nm_inds_choices=nm_inds
        )

    # --- For each actual expression, test retrieval ---
    for expr in props_colls:
        # Direct computation
        direct = avrgprops.calculate_avrg_tensor(
            avrg_expression=expr,
            pulse_polarization_vector=pol_vec,
            props_data=mol_props,
            number_of_nmodes=n_modes,
            nm_inds_choices=nm_inds
        )

        # Retrieval from unique tensor
        unique_expr = mapping[expr]
        unique_tensor = unique_tensors[unique_expr]

        # Build index mapping:
        # actual expr indices e.g. ['a', 'a', 'b', 'c']
        # unique expr indices e.g. ['a', 'b', 'c', 'd']
        actual_inds = []
        for prop in expr:
            actual_inds.extend(prop.inds)
        
        unique_inds = []
        for prop in unique_expr:
            unique_inds.extend(prop.inds)

        actual_unique = sorted(set(actual_inds))
        unique_unique = sorted(set(unique_inds))

        # For every combination of actual mode indices,
        # map to unique tensor indexing
        for combo in itertools.product(range(n_modes), repeat=len(actual_unique)):
            # combo assigns values to sorted unique actual indices
            actual_assignment = dict(zip(actual_unique, combo))

            # Index into direct tensor: axes = actual_unique (sorted)
            direct_val = direct[combo]

            # Index into unique tensor: 
            # each position in unique_inds gets a value from actual_assignment
            # via the correspondence actual_inds[i] -> unique_inds[i]
            #
            # e.g. actual ['a','a','b','c'] -> unique ['a','b','c','d']
            # means unique 'a' <-> actual 'a', unique 'b' <-> actual 'a',
            #        unique 'c' <-> actual 'b', unique 'd' <-> actual 'c'
            ind_map = {}
            for ai, ui in zip(actual_inds, unique_inds):
                ind_map[ui] = ai

            unique_idx = tuple(
                actual_assignment[ind_map[ui]] for ui in unique_unique
            )
            unique_val = unique_tensor[unique_idx]
            print(unique_val)
            print(direct_val)

            assert abs(direct_val - unique_val) < 1e-12, \
                f"Mismatch for {expr}:\n" \
                f"  actual_inds={actual_inds}, unique_inds={unique_inds}\n" \
                f"  combo={combo} -> direct={direct_val}\n" \
                f"  unique_idx={unique_idx} -> unique={unique_val}"

