from .testutils import MakeObjects, get_from_pkl_features
import wilson_suite as ws

def test_feat_boxX():
    print()
    f = MakeObjects.mk_feature_single()
    print(f)
    print('\n-----------\n')
    f = MakeObjects.mk_features_non_ovrl()
    print(f)

    print('\n-----------\n')
    f = MakeObjects.mk_features_ovrl()
    print(f)


def test_pkl_data():
    print()
    feats = get_from_pkl_features('data_for_tests/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl', 10.)
    ws.intensities.amplitudes.spectrum_composition.SpectralFeature.print_list_features(feats)



"""
def test_pkl_data():
    print()
    from wilson_suite.wilson_utils.serialization import unpickle_smth_from
    unpickled = unpickle_smth_from('data_for_tests/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl')
    print(unpickled)
    print(type(unpickled))
    print(list(unpickled.keys()))
    print(unpickled['anharmonic_states'])
    
    list_vibsstates = fillStatesData(unpickled)

    from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData, VibDiffCache
    include_list = tuple([0, 1, 2])
    vibstates_data = VibStatesData(allstates=tuple(list_vibsstates),
                                   harmonic_osc_states_labels=include_list)
    vibdiff_cache = VibDiffCache()

    from wilson_suite.wilson_intensities.amplitudes.evaluators import get_features_from_terms_for_eval
    from wilson_suite.fixtures import evv_experiment
    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    features = get_features_from_terms_for_eval(derived_terms=terms,
                                                vibstates_data=vibstates_data,
                                                vibdiff_cache=vibdiff_cache, 
                                                lineshape_parameter=lineshape_parameter)
    
"""