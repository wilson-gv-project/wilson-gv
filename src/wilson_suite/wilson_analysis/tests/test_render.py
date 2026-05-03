"""

"""
from wilson_suite.wilson_analysis.render.render import render_spectrum
from wilson_suite.wilson_main.spectrum_abstractions import SpecEvalSetup, RenderingInfo, EvaluationInfo
import os
import numpy as np
import pytest
import copy
np.set_printoptions(linewidth=280, precision=4)


def test_render_spectrum_simplecontour():
    do_diagn = True
    
    start, stop, n_values = -4, 4, 9
    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)
    spec = X**2 + Y**2

    rndinfo = RenderingInfo(spec_data_operations='none', nlevels=6,
                            filename='simple_contour.svg')
    spec_eval_setup = SpecEvalSetup(rnd_info=rndinfo)
    assert not spec_eval_setup.is_ready_render

    # test1
    with pytest.raises(TypeError) as excinfo:
        render_spectrum(**dict(spec_data='invalid', spec_eval_setup=None, do_diagn=None))
    assert str(excinfo.value) == 'spec_data should be a np.ndarray'

    # test2
    with pytest.raises(TypeError) as excinfo:
        render_spectrum(**dict(spec_data=spec, spec_eval_setup='invalid', do_diagn=None))
    assert str(excinfo.value) == 'spec_eval_setup should be a SpecEvalSetup instance'

    context = dict(spec_data=spec, spec_eval_setup=spec_eval_setup, do_diagn=do_diagn)

    # test3
    with pytest.raises(ValueError) as excinfo:
        render_spectrum(**context)
    assert str(excinfo.value) == 'spec_eval_setup does not have all rendering configs'
    
    spec_eval_setup.grid = {'A': X, 'B': Y}
    spec_eval_setup.ev_info = EvaluationInfo(dynamic_range=20.)

    # test4
    with pytest.raises(NotImplementedError) as excinfo:
        render_spectrum(**dict(spec_data=np.array([1., 2.]), spec_eval_setup=spec_eval_setup, do_diagn=None))
    assert str(excinfo.value) == 'only 2D contour plots can be made - input spectrum data is not 2D'

    # test5
    with pytest.raises(NotImplementedError) as excinfo:
        cp_se = copy.deepcopy(spec_eval_setup)
        cp_se.rnd_info.backend = 'rc'
        render_spectrum(**dict(spec_data=spec, spec_eval_setup=cp_se, do_diagn=None))
    assert str(excinfo.value) == 'Only matplotlib backend is currently supported'

    # test6 - success
    r, diagn = render_spectrum(**context)
    assert np.allclose(diagn['renderer'].levels, np.array([ 1.6,  2.9129,  5.3031,  9.6547, 17.577 , 32.]))
    assert diagn['renderer'].labels == ['$1.6e+00$', '$2.9e+00$', '$5.3e+00$', '$9.7e+00$', '$1.8e+01$', '$3.2e+01$']

    # test7
    cntx = copy.deepcopy(context)
    cntx['spec_data'] = np.array([])

    with pytest.raises(ValueError) as excinfo:
        r, diagn = render_spectrum(**cntx)
    assert str(excinfo.value) == 'Empty spec_data array'
    os.remove('simple_contour.svg')

def test_render_spectrum_simplecontour_sq():

    do_diagn = True
    
    start, stop, n_values = -4, 4, 9
    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)
    spec = np.sqrt(X**2 + Y**2)

    rndinfo = RenderingInfo(spec_data_operations='abs()**2', nlevels=6,
                            filename='simple_contour_sq.svg')
    spec_eval_setup = SpecEvalSetup(rnd_info=rndinfo)
    assert not spec_eval_setup.is_ready_render

    context = dict(spec_data=spec, spec_eval_setup=spec_eval_setup, do_diagn=do_diagn)
    spec_eval_setup.grid = {'A': X, 'B': Y}
    spec_eval_setup.ev_info = EvaluationInfo(dynamic_range=20.)
    
    # test - success
    r, diagn = render_spectrum(**context)
    
    assert np.allclose(diagn['renderer'].levels, np.array([ 1.6,  2.9129,  5.3031,  9.6547, 17.577 , 32.]))
    assert diagn['renderer'].labels == ['$1.6e+00$', '$2.9e+00$', '$5.3e+00$', '$9.7e+00$', '$1.8e+01$', '$3.2e+01$']
    os.remove('simple_contour_sq.svg')
