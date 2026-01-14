import numpy as np
from wilson_suite.wilson_analysis.render.matplotlib_renderer import MatplotlibRenderer
from wilson_suite.wilson_main.spectrum_abstractions import EvaluationInfo, RenderingInfo
import pytest

import logging
logger = logging.getLogger(__name__)

def tests_MatplotlibRenderer():

    spec_data = []
    spec_grid = {'x': np.array([]), 'y': np.array([])}
    ev_info = EvaluationInfo()
    rnd_info = RenderingInfo()

    with pytest.raises(TypeError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                                    spec_grid=spec_grid, 
                                    ev_info=ev_info, rnd_info=rnd_info, 
                                    do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_data should be a np.ndarray"

    spec_data = np.array([])
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                                    spec_grid=spec_grid, 
                                    ev_info=ev_info, rnd_info=rnd_info, 
                                    do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_data array should not be empty"


    spec_data = np.array([[1000, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 827,   1]])
    
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_grid['x'] is an empty array"

    spec_grid = {'x': np.array([1, 2, 3]), 'y': np.array([])}
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_grid['y'] is an empty array"

    spec_grid = {'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])}
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert "X,Y and intensities data do not match in shape:" in str(error.value)
    
    x,y = np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 8, 9])
    X,Y = np.meshgrid(x,y,indexing='xy')

    spec_grid = {'x': X, 'y': Y}
    MatplotlibRenderer(spec_data=spec_data, 
                spec_grid=spec_grid, 
                ev_info=ev_info, rnd_info=rnd_info, 
                do_diagn=True).render('f0.svg')