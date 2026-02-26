from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralFeature, ResLocGeoObject, SpectralWindow
from wilson_suite.wilson_intensities.amplitudes.term_parts import TermParametersChoice, ResonanceMotif, ResonanceCondition, ParameterSet

class MakeObjects:
    
    @staticmethod
    def mk_feature_single() -> SpectralFeature:
        """
        with random term_contributions for now
        """
        # if 
        rcs = [ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)),
               ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('B',))]
        res_motif = ResonanceMotif(rcs)
        
        param_set = ParameterSet({'a': 0, 'b': 1})

        term_contrib = TermParametersChoice(res_motif=res_motif,
                                            states_parameters=(param_set,),
                                            term_ids=tuple([hash(res_motif)]))
        
        feat = SpectralFeature(location=ResLocGeoObject({'A': 1119.5, 'B': 2921.}), 
                               lineshape_parameter=4.5, 
                               amplitude_coeff=-1.12e-06, 
                               term_contributions=(term_contrib,))
        features = SpectralFeature.dress_these_with_boxes([feat],
                                                          max_intensity=feat.get_intensity()+1., 
                                                          min_intensity=feat.get_intensity()-1.0e3,
                                                          box_range_safety_margin=0.1,
                                                          scale_wrt_max_intensity=True,
                                                          minimum_box_padding=30.0,
                                                          )
        return features[0]
    
    @staticmethod
    def mk_features_non_ovrl() -> list[SpectralFeature]:

        rcs1 = [ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)),
               ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('B',))]
        res_motif1 = ResonanceMotif(rcs1)

        param_set1 = ParameterSet({'a': 0, 'b': 1})
        term_contrib1 = TermParametersChoice(res_motif=res_motif1,
                                            states_parameters=(param_set1,),
                                            term_ids=tuple([hash(res_motif1)]))

        rcs2 = [ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)),
               ResonanceCondition.make_from_tuples(left_state=('b',), right_state=('a',), pert_freqs=('B',))]
        res_motif2 = ResonanceMotif(rcs2)

        param_set2 = ParameterSet({'a': 1, 'b': 2})
        term_contrib2 = TermParametersChoice(res_motif=res_motif2,
                                            states_parameters=(param_set2,),
                                            term_ids=tuple([hash(res_motif2)]))
        
        feat1 = SpectralFeature(location=ResLocGeoObject({'A': 1119.5, 'B': 2921.}), 
                                lineshape_parameter=4.5, 
                                amplitude_coeff=-4.32e-06, 
                                term_contributions=(term_contrib1,))
        feat2 = SpectralFeature(location=ResLocGeoObject({'A': 964., 'B': 270.}), 
                                lineshape_parameter=4.5, 
                                amplitude_coeff=-1.12e-05, 
                                term_contributions=(term_contrib2,))
        
        features = SpectralFeature.dress_these_with_boxes([feat1, feat2],
                                                          max_intensity=feat2.get_intensity()+1., 
                                                          min_intensity=feat1.get_intensity()-1.0e3,
                                                          box_range_safety_margin=0.1,
                                                          scale_wrt_max_intensity=True,
                                                          minimum_box_padding=30.0,
                                                          )
        return features

    @staticmethod
    def mk_features_ovrl() -> list[SpectralFeature]:

        rcs1 = [ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)),
               ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('B',))]
        res_motif1 = ResonanceMotif(rcs1)

        param_set1 = ParameterSet({'a': 0, 'b': 1})
        term_contrib1 = TermParametersChoice(res_motif=res_motif1,
                                            states_parameters=(param_set1,),
                                            term_ids=tuple([hash(res_motif1)]))

        rcs2 = [ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)),
                ResonanceCondition.make_from_tuples(left_state=('b',), right_state=('a',), pert_freqs=('B',))]
        res_motif2 = ResonanceMotif(rcs2)

        param_set2 = ParameterSet({'a': 0, 'b': 1})
        term_contrib2 = TermParametersChoice(res_motif=res_motif2,
                                            states_parameters=(param_set2,),
                                            term_ids=tuple([hash(res_motif2)]))
        
        feat1 = SpectralFeature(location=ResLocGeoObject({'A': 1000., 'B': 520.}), 
                                lineshape_parameter=4.5, 
                                amplitude_coeff=-4.32e-06, 
                                term_contributions=(term_contrib1,))
        feat2 = SpectralFeature(location=ResLocGeoObject({'A': 1000., 'B': 500.}), 
                                lineshape_parameter=4.5, 
                                amplitude_coeff=-1.12e-05, 
                                term_contributions=(term_contrib2,))
        features = SpectralFeature.dress_these_with_boxes([feat1, feat2],
                                                          max_intensity=feat2.get_intensity()+1., 
                                                          min_intensity=feat1.get_intensity()-1.0e3,
                                                          box_range_safety_margin=0.1,
                                                          scale_wrt_max_intensity=True,
                                                          minimum_box_padding=30.0,
                                                          )
        assert features[0].feat_box.overlaps(features[1].feat_box)
        return features
    
    @staticmethod
    def mk_gridregion_for_specwindow(specwindow: SpectralWindow, coords: dict):
        """

        coords is like, e.g.:
        {'A': array([[1089.5, 1089.5, 1089.5, 1089.5, 1089.5],
                     [1103.9, 1103.9, 1103.9, 1103.9, 1103.9],
                     [1118.3, 1118.3, 1118.3, 1118.3, 1118.3],
                     [1132.7, 1132.7, 1132.7, 1132.7, 1132.7],
                     [1147.1, 1147.1, 1147.1, 1147.1, 1147.1]]), 
        'B': array([[2891. , 2905.4, 2919.8, 2934.2, 2948.6],
                    [2891. , 2905.4, 2919.8, 2934.2, 2948.6],
                    [2891. , 2905.4, 2919.8, 2934.2, 2948.6],
                    [2891. , 2905.4, 2919.8, 2934.2, 2948.6]])}
        """
        from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridRegion, RectangularDomain
        domain = RectangularDomain(box=specwindow.box,
                                   full_features=specwindow.full_features,
                                   contrib_features=specwindow.contrib_features)

        return GridRegion(domain=domain,
                          coords=coords)
    

    @staticmethod
    def mk_gridregion_for_feature(feature, coords):
        return


    @staticmethod
    def mk_vibstates_states(closer=False):
        """
        rcs = [ResonanceCondition.make_from_tuples(left_state=('',), right_state=('a',), pert_freqs=('-A',)),
               ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('-B',))]
        res_motif = ResonanceMotif(rcs)
        
        param_set = ParameterSet({'a': 0, 'b': 1})

            'A': 1119.5, 'B': 2921.
        1119.5 = a 0
        2921 = 0,1-1
        """
        from wilson_suite.wilson_main import abstractions as wm_abst
        if not closer:
            states = (
                wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=1119.5, harmonic_WF=True),
                wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=964., harmonic_WF=True),
                wm_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=1234., harmonic_WF=True),
                wm_abst.VibState(harm_quanta_coeffs={(0, 0):1.}, state_label='0,0', energy=1864., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=3885., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(1, 1):1.}, state_label='1,1', energy=2368., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(1, 2):1.}, state_label='1,2', energy=2360., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(2, 2):1.}, state_label='2,2', energy=2362., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 2):1.}, state_label='0,2', energy=2274., harmonic_WF=False),

                wm_abst.VibState(harm_quanta_coeffs={(0, 0, 0):1.}, state_label='0,0,0', energy=2685., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3581., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(2, 2, 2):1.}, state_label='2,2,2', energy=3690., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 1, 2):1.}, state_label='0,1,2', energy=3742., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 2, 2):1.}, state_label='0,2,2', energy=3680., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 0, 1):1.}, state_label='0,0,1', energy=3155., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(0, 0, 2):1.}, state_label='0,0,2', energy=3498., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=3594., harmonic_WF=False),
                wm_abst.VibState(harm_quanta_coeffs={(1, 2, 2):1.}, state_label='1,2,2', energy=3642., harmonic_WF=False),
            )
        else:
            states = (
                wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=1000., harmonic_WF=True),
                wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1500., harmonic_WF=True),
                wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=2020., harmonic_WF=False),
            )

        return states