from ..analysis.analyses import get_features_report, get_resonances_report
from wilson_suite.fixtures import evv_terms
from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData

def test_get_resonances_report_no_ax_choice():
    print()
    der_terms = evv_terms()
    get_resonances_report(der_terms)


def test_get_resonances_report():
    der_terms = evv_terms()

    from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
    axes_choice = make_SpectralAxisSet({'A': [1], 'B': [-1,2]}) # this makes A and B > 0
    get_resonances_report(der_terms, axis_set_choice=axes_choice)

def test_get_resonances_report_f():
    der_terms = evv_terms()

    from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
    axes_choice = make_SpectralAxisSet({'A': [1], 'B': [-1,2]}) # this makes A and B > 0

    from wilson_suite.wilson_utils.paths import SUITE_ROOT
    from wilson_suite.wilson_utils.serialization import unpickle_smth_from
    vib_states_data = unpickle_smth_from(SUITE_ROOT+'/../examples/workflows/VibStatesData.pkl')

    get_resonances_report(der_terms, axis_set_choice=axes_choice, vib_states_data=vib_states_data)
