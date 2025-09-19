from ..render.render import render_spectrum
from ..render.spectrum_renderer import NormalizationType
from ...wilson_main.abstractions import (SpecEvalSetup, SpectralAxis, SpectralGrid, 
                                        RenderingInfo, EvaluationInfo, PlotConfig,
                                        MolecularSystem, VibExperiment)
from ...fixtures import evv_experiment
import numpy as np
import logging
logger = logging.getLogger(__name__)

def run():
    """
    run function to use logger
    """
    x = np.linspace(-6, 5, 300)
    y = np.linspace(-5, 6, 300)
    X1, X2 = np.meshgrid(x, y)

    def Z(x1, x2):
        """
        | Peak # | Location `(x1, x2)` |
        | ------ | ------------------- |
        | 1      | `(2.3, 3.1)`        |
        | 2      | `(-3.8, -0.7)`      |
        | 3      | `(0.6, -2.9)`       |
        | 4      | `(-1.7, 4.0)`       |

        For reference - to check peaks in figure `filename.svg`
        """
        z = (
            1.2 * np.exp(-((x1 - 2.3)**2 / 0.1 + (x2 - 3.1)**2 / 0.2)) +   
            0.9 * np.exp(-((x1 + 3.8)**2 / 0.2 + (x2 + 0.7)**2 / 0.15)) + 
            1.1 * np.exp(-((x1 - 0.6)**2 / 0.15 + (x2 + 2.9)**2 / 0.1)) + 
            0.6 * np.exp(-((x1 + 1.7)**2 / 0.1 + (x2 - 4.0)**2 / 0.05))   
        )
        return z

    z_vals = Z(X1, X2)
    Y = X2 - X1

    logger.debug("X1 (X-axis):")
    logger.debug(X1)
    logger.debug("\nY (Y-axis):")
    logger.debug(Y)
    logger.debug("\nZ (Contour values):")
    logger.debug(z_vals)
    return z_vals, X1, X2

# choice for Y axis values
w1_minus_w2 = False

def test_plt_NOw1_minus_w2():
    """
    A qualitative test with figure inspection.

    no specific asserts
    """
    from ...wilson_utils.logger import setup_logger
    setup_logger(__name__, level=logging.DEBUG)

    z_vals, X1, X2 = run()
    eval_vars_meshgrids = {'w1': X1, 'w2': X2}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})
    spec_grid = SpectralGrid({'x': axis1, 'y': axis2}, range_style='custom')
    
    evi = EvaluationInfo(**{'freq_variables': eval_vars_meshgrids,
                            'Gamma': 4.7, 'Gamma_unit': 'cm-1'})
    rndi = RenderingInfo(**{'intensity_normalization_type': NormalizationType.DECIBEL, 'dynamic_range': 1e3, 
                            'num_levels': 20,  'reference_max': None, 'spec_data_operations': 'abs', 
                            'projection': '2d', 'to_save': True, 
                            'style_config': PlotConfig(other_colors=False,
                                                       colormap='magma',
                                                       tick_step=1)})
    spec_eval_setup = SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)
    
    system = MolecularSystem(name='mock', natoms=4)
    experiment = evv_experiment()

    context = dict(spec_data=z_vals, system=system, experiment=experiment,
                    diagn={}, name='name',
                    spec_eval_setup=spec_eval_setup, do_diagn=False)
    render_spectrum(**context)
    import os
    assert os.path.exists("filename1.svg")


def test_plt_w1_minus_w2():
    from ...wilson_utils.logger import setup_logger
    setup_logger(__name__, level=logging.DEBUG)

    z_vals, X1, X2 = run()
    eval_vars_meshgrids = {'w1': X1, 'w2': X2}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w1': 1, 'w2': -1})
    spec_grid = SpectralGrid({'x': axis1, 'y': axis2}, range_style='custom')
    
    evi = EvaluationInfo(**{'freq_variables': eval_vars_meshgrids,
                            'Gamma': 4.7, 'Gamma_unit': 'cm-1'})
    rndi = RenderingInfo(**{'intensity_normalization_type': NormalizationType.DECIBEL, 'dynamic_range': 1e3, 
                            'num_levels': 20,  'reference_max': None, 'spec_data_operations': 'abs', 
                            'projection': '2d', 'to_save': True, 
                            'style_config': PlotConfig(other_colors=False,
                                                       colormap='magma',
                                                       tick_step=1)})
    spec_eval_setup = SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)
    
    system = MolecularSystem(name='mock', natoms=4)
    experiment = evv_experiment()

    context = dict(spec_data=z_vals, system=system, experiment=experiment,
                    diagn={}, name='name',
                    spec_eval_setup=spec_eval_setup, do_diagn=False)
    render_spectrum(**context)
    
    import os
    assert os.path.exists("filename2.svg")