"""

"""
from wilson_suite.wilson_analysis.render.render import render_spectrum
from wilson_suite.wilson_main.spectrum_abstractions import SpecEvalSetup, RenderingInfo, EvaluationInfo

import numpy as np
import pytest
np.set_printoptions(linewidth=280, precision=1)


def test_render_spectrum():
    do_diagn = True
    
    start, stop, n_values = -5, 5, 11
    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)
    spec = np.sqrt(X**2 + Y**2)
    print('\n', spec)

    rndinfo = RenderingInfo(intensity_normalization_type=None, spec_data_operations='none', nlevels=6)
    spec_eval_setup = SpecEvalSetup(rnd_info=rndinfo)
    assert not spec_eval_setup.is_ready_render

    context = dict(spec_data=spec, spec_eval_setup=spec_eval_setup, do_diagn=do_diagn)
    with pytest.raises(ValueError) as excinfo:
        render_spectrum(**context)
        assert excinfo.value.message == 'spec_eval_setup does not have all rendering configs'
    
    spec_eval_setup.grid = {'A': X, 
                            'B': Y}
    context = dict(spec_data=spec, spec_eval_setup=spec_eval_setup, do_diagn=do_diagn)
    with pytest.raises(ValueError) as excinfo:
        render_spectrum(**context)
        assert excinfo.value.message == 'spec_eval_setup does not have all rendering configs'
    
    spec_eval_setup.ev_info = EvaluationInfo(dynamic_range=200.)
    r, diagn = render_spectrum(**context)
    print(diagn['renderer'].rnd_info.nlevels)
    print(diagn['renderer'].levels)
    print(diagn['renderer'].labels)