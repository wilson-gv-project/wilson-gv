import wilson_suite.wilson_intensities.amplitudes.evaluators as evaluators
import numpy as np


def test_terms_evaluator_general_compilation():
    print()
    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()

    np.set_printoptions(linewidth=180, precision=3)

    r = evaluators.terms_evaluator_general_compilation(**datadict)
    print(r['result'])

