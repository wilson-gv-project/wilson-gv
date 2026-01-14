import numpy as np
from wilson_suite.wilson_analysis.render.matplotlib_renderer import MatplotlibRenderer
from wilson_suite.wilson_main.spectrum_abstractions import EvaluationInfo, RenderingInfo

import logging
logger = logging.getLogger(__name__)

def tests_MatplotlibRenderer():

    spec_data = np.array([])
    spec_grid = {'A': np.array([]), 'B': np.array([])}
    ev_info = EvaluationInfo()
    rnd_info = RenderingInfo()

    renderer = MatplotlibRenderer(spec_data=spec_data, 
                                  spec_grid=spec_grid, 
                                  ev_info=ev_info, rnd_info=rnd_info, 
                                  do_diagn=True)

    renderer.render('f0.svg')

    spec_data = np.array([[1000, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 827,   1]])