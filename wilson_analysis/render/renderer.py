"""
Use in WilsonSimulation:
self.rendering = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)
self.rendering, self.diagn = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)

Input:
self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup; 

Steps: 
1. choice of projection - how to present spectrum data: 2d or 3d or 1d ... (figure dimension setup)
2. assignment of data to axes
3. dynamic range settings (for contour - colorbar, for 1d - threshold on y values?)
. style:
    a. axes titles
    b. axes ticks
    c. axes ticks labels
    d. axes labels
    e. title if any
. finalize figure styling: tight layout, etc...
. Save figure (where, filename)
"""
import numpy as np
import wilson_main.abstractions as wm_abst
import wilson_experiment.abstractions as we_abst
import wilson_derive.abstractions as wd_abst

import logging
# wilson. - for hierarchy of loggers
logger = logging.getLogger("wilson."+__name__)

def renderer(spec: np.ndarray, system: wm_abst.MolecularSystem, 
             exp: we_abst.VibExperiment, terms: list[wd_abst.VibPerturbedTerm],
             diagn: dict, name: str, spec_eval_setup: wm_abst.SpecEvalSetup, 
             rnd_choice: str = 'matplotlib'):
    """
    spec_grid = ws.main.abstractions.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
                                                    start=start, end=end, spacer=spacer)

    evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
    rndi = {'num_level_ticks': 15}
    eval_setup = ws.main.abstractions.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)

    """
    dimensionality = exp.dim # 2 for EVV
    spec_grid = spec_eval_setup.grid
    
    # Gamma = spec_eval_setup.ev_info_class.Gamma
    # assert spec_eval_setup.ev_info_class.Gamma_unit == 'cm-1'

    # dynrange = spec_eval_setup.rnd_info_class.dynrange
    # maxmax = spec_eval_setup.rnd_info_class.maxPeak
    # num_level_ticks = spec_eval_setup.rnd_info_class.num_level_ticks

    # indp_vars = spec_eval_setup.ev_info_class.freq_variables
    # fixed_vars = spec_eval_setup.ev_info_class.fixed_variables

    # # 1 - x, 2 - y figure axes
    # specAxes = spec_eval_setup.grid.axes

    if rnd_choice == 'matplotlib':
        from .matplotlib_renderer import MatplotlibRenderer
        return MatplotlibRenderer(spec_grid=spec_grid, 
                                  rnd_info_class=spec_eval_setup.rnd_info_class, 
                                  ev_info_class=spec_eval_setup.ev_info_class).render()
    
    else:
        raise NotImplementedError('Other than `matplotlib` renderers are not implemented. Custom renderer is not supported yet')
    return 

