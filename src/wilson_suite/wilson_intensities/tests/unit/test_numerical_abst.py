from wilson_suite.wilson_intensities.amplitudes.numerical_abstractions import (
    NumericalResonanceMotif,
    compile_resonance_motif,
    CompiledTermGroup,
    compile_feature
)
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import (
    ResonanceMotif,
    SpectralFeature
)
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet
from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiff, VibDiffCache
from wilson_suite.wilson_derive.abstractions import ResonanceCondition

from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

def test_compile_resonance_motif():
    """Test compile_resonance_motif function"""
    print()

    # Setup mock objects and data
    from .test_evaluators import prep_vibanasetup_with_degen_states
    vib_ana_setup = prep_vibanasetup_with_degen_states()

    rcs = [ResonanceCondition.make_from_tuples(left_state=('a',), right_state=('b',), pert_freqs=('A', '-B')),
           ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('B',))]
    
    res_motif = ResonanceMotif(rcs)
    param_set = ParameterSet({'a': 0, 'b': 1})
    vib_data = VibStatesData(vib_ana_setup.states)
    vibdiff_cache = VibDiffCache()


    numerical_motif = compile_resonance_motif(
        res_motif,
        param_set,
        vib_data,
        vibdiff_cache
    )

    vib_diff_01 = VibDiff(vib_data.get_state_by_label('0'), vib_data.get_state_by_label('1')).energy_difference()
    vib_diff_01p1 = VibDiff(vib_data.get_state_by_label('0,1'), vib_data.get_state_by_label('1')).energy_difference()
    all_vds = [cond.vib_energy_diff for cond in numerical_motif.res_conds]
    
    assert convNu2Ene(vib_diff_01) in all_vds
    assert convNu2Ene(vib_diff_01p1) in all_vds


    assert isinstance(numerical_motif, NumericalResonanceMotif)
    assert len(numerical_motif.res_conds) == len(res_motif)


def test_compile_feature():
    """Test compile_feature function"""
    # Setup mock objects and data
    feature = SpectralFeature(...)  # Fill with appropriate test data
    vib_data = VibStatesData(...)   # Fill with appropriate test data
    vibdiff_cache = VibDiffCache()  # Initialize empty cache

    # Call the function to test
    compiled_groups = compile_feature(
        feature,
        vib_data,
        vibdiff_cache
    )

    # Assertions to verify correctness
    assert isinstance(compiled_groups, list)
    for group in compiled_groups:
        assert isinstance(group, CompiledTermGroup)
        for motif in group.resonance_motifs:
            assert isinstance(motif, NumericalResonanceMotif)
    # Add more specific assertions based on expected values


# from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

# def test_compile_resonance_motif():
#     """Test compile_resonance_motif function"""
#     # Setup mock objects and data
#     # (In a real test, you would create actual instances with meaningful data)
#     res_motif = ResonanceMotif(...)  # Fill with appropriate test data
#     param_set = ParameterSet(...)    # Fill with appropriate test data
#     vib_data = VibStatesData(...)    # Fill with appropriate test data
#     vibdiff_cache = VibDiffCache()   # Initialize empty cache

#     # Call the function to test
#     numerical_motif = compile_resonance_motif(
#         res_motif,
#         param_set,
#         vib_data,
#         vibdiff_cache
#     )

#     # Assertions to verify correctness
#     assert isinstance(numerical_motif, NumericalResonanceMotif)
#     assert len(numerical_motif.res_conds) == len(res_motif)
#     # Add more specific assertions based on expected values