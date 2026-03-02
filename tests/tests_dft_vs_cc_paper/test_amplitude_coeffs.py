"""
SpectralFeature.amplitude_coeff evaluation tests
- [ ] el term coeff
- [ ] 2 el terms coeff
- [ ] mech term coeff
- [ ] 2 mech terms coeff
- [ ] mech and el terms in one feat coeff

- [ ] evaluate_single_index_dict() -- in src/wilson_suite/wilson_intensities/amplitudes/full_amplitude_coeff.py
- [ ] eval_avrg_per_indexdict() -- in src/wilson_suite/wilson_intensities/amplitudes/full_amplitude_coeff.py
"""
from .testutils import MakeObjects
import wilson_suite as ws
import numpy as np
np.set_printoptions(linewidth=180, precision=3)


def test_ampl_coeff_one_el_term():
    """
    [x] feature coefficient is a sum of all contributing terms with same resmotif
    [ ] computed per term coefficient for mech terms has also a sum over index 'c'
    [ ] computed per term coefficient for el terms is a product of orient.avrg and vibene denominator

    
    there are tests in src/wilson_suite/wilson_intensities/tests/unit/test_full_coeff.py::test_evaluate_term_coeffs_single_c_ind_contrib
    and test_evaluate_term_coeffs_multi_c_ind_contrib:
    - products of components
    - vibene denom calculation
    - single contribution in the sum over index 'c'

    need more tests for orient. avrg tensors - evaluation and use
    """
    print()
    
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
    print()

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

    for term_hash, param_to_coeff in result_feat1_coeff.items():
        print()
    print(result_feat1_coeff)
    