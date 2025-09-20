"""
Integration tests for full procedure with figure rendering using TermND functionality.
"""
import numpy as np
from ..testing_utils import require_asserts
from ...utils import debug_mode
from ....wilson_analysis.render.simple_plot import render_spectrum

from ....wilson_utils import printing as debug
import CQCParse.debug as cqc_debug

debug.level = 0
cqc_debug.level = 0


np.set_printoptions(precision=4,suppress=False)


def compare_amplitudes(amplitudes1: np.ndarray, amplitudes2: np.ndarray) -> None:
    """
    Helper function to compare two sets of amplitudes.
    """
    assert amplitudes1.shape == amplitudes2.shape, "Amplitude shapes do not match"
    diff = np.abs(amplitudes1 - amplitudes2)
    max_diff = np.max(diff)
    print(f"Maximum difference between amplitudes: {max_diff:.2e}")
    assert max_diff < 1e-6, "Amplitudes differ significantly"

# FUNCTIONALITY IS BROKEN NOW, THESE TESTS SHOULD BE REWRITTEN

# @require_asserts
# def test_terms_collection_calculation(terms_collection: dict, spectrum_setup: dict, conditions: dict) -> None:
#     """
#     get a spectrum figure
#     """
    
#     # terms_collection is a fixture
#     te, precalc_dict = terms_collection['FORM']
#     spectrum_setup = spectrum_setup['FORM']
#     conditions = conditions['FORM']

#     expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
#     for key in expected_keys:
#         assert key in precalc_dict, f"Key '{key}' missing in precalculated data"
#     assert precalc_dict['vibene_denoms'], "vibene_denoms data is empty"

    
#     # can run with debug prints if > 0
#     with debug_mode(0):
#         amplitudes = 0.0
#         for id, term in te.terms.items():
#             term.precalc_data = precalc_dict
            
#             # can run with debug prints if > 0
#             with debug_mode(0):
#                 intensity = term.get_amplitudes(spectrum_setup.w1m,
#                                                 spectrum_setup.w2m,
#                                                 3.8, 1.0, debugprint=True, collect_all=False)
#             amplitudes += intensity

#     print(amplitudes.shape)

#     assert np.isfinite(amplitudes).all(), "Amplitudes contain non-finite values (NaN or Inf)"
#     max_amplitude = np.max(np.abs(amplitudes))
#     print(f"Maximum amplitude: {max_amplitude:.2e}")
#     print(f"Maximum intensity: {np.max(np.abs(amplitudes)**2):.3e}")

#     np.set_printoptions(precision=4)

#     intensities = np.abs(amplitudes)**2

#     hist, bin_edges = np.histogram(intensities, bins=10)
#     print('\nintensities from test:')
#     print("Histogram counts:", hist)
#     print("Bin edges:", bin_edges)

#     render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
#                     filename=f'yo_terms_{spectrum_setup.molecule}.svg',
#                     dynamic_range=conditions.dynamic_range_n,
#                     nicetitle='TermsEvaluator')


# @require_asserts
# def test_terms_collection_calculation_derived(terms_collection_derived: dict, 
#                                               spectrum_setup: dict, conditions: dict) -> None:
#     """
#     get a spectrum figure
#     """
#     # terms_collection is a fixture
#     te, precalc_dict = terms_collection_derived['FORM']
#     spectrum_setup = spectrum_setup['FORM']
#     conditions = conditions['FORM']

#     expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
#     for key in expected_keys:
#         assert key in precalc_dict, f"Key '{key}' missing in precalculated data"
#     assert precalc_dict['vibene_denoms'], "vibene_denoms data is empty"

#     # can run with debug prints if > 0
#     with debug_mode(0):
#         amplitudes = 0.0
#         for id, term in te.terms.items():
#             term.precalc_data = precalc_dict
#             # with np.printoptions(precision=2,legacy='1.25'):
#                 # formatted_resonances = {
#                     # key: (f"{value[0]:.2f}", f"{value[1]:.2f}")
#                     # for key, value in term.get_all_resonances(w2mw1=True).items()
#                 # }
#                 # print(formatted_resonances)
#             # can run with debug prints if > 0
#             with debug_mode(0):
#                 intensity = term.get_amplitudes(spectrum_setup.w1m,
#                                                 spectrum_setup.w2m,
#                                                 3.8, 1.0, debugprint=True, collect_all=False)
#             amplitudes += intensity

#     print(amplitudes.shape)

#     assert np.isfinite(amplitudes).all(), "Amplitudes contain non-finite values (NaN or Inf)"
#     max_amplitude = np.max(np.abs(amplitudes))
#     print(f"Maximum amplitude: {max_amplitude:.2e}")
#     print(f"Maximum intensity: {np.max(np.abs(amplitudes)**2):.3e}")

#     np.set_printoptions(precision=4)

#     intensities = np.abs(amplitudes)**2

#     hist, bin_edges = np.histogram(intensities, bins=10)
#     print('\n')
#     print("Histogram counts:", hist)
#     print("Bin edges:", bin_edges)

#     render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
#                     filename=f'yo_terms_derived_{spectrum_setup.molecule}.svg',
#                     dynamic_range=conditions.dynamic_range_n,
#                     nicetitle='TermsEvaluator')
