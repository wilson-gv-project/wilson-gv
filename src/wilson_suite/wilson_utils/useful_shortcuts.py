"""

"""
import pickle
import os.path
from rich.pretty import pprint

import wilson_suite.wilson_derive.response_terms
from ..wilson_derive import abstractions as wd_abst
from ..wilson_experiment import experiment_abstractions as we_abst

from ..wilson_derive.derive import get_fully_enhanced_terms
from ..wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts

from ..wilson_analysis.render.render_utils import PlotConfig, NormalizationType
from wilson_suite.fixtures import evv_experiment, get_eval_ready_evv_terms

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..wilson_main.abstractions import (VibAnaSetup, DataOriginInfo, MolecularSystem)
    from ..wilson_main.spectrum_abstractions import SpecEvalSetup
    from wilson_suite.wilson_main.abstractions import VibAnaSetup

import logging
logger = logging.getLogger("wilson."+__name__)

def get_EVV_derived_terms():
    """
    
    """
    derived_terms_pkl = '/home/vlev/wilson-suite/notebooks/derived_terms.pkl'

    if os.path.isfile(derived_terms_pkl):
        with open(derived_terms_pkl, 'rb') as handle:
            derived_terms = pickle.load(handle)

        flat_derived_terms = derived_terms_dict_to_dicts(derived_terms=derived_terms, tolistonly=True)

        str_dict = derived_terms_dict_to_dicts(derived_terms=derived_terms, tolistonly=False)

        pprint(flat_derived_terms)

        pprint([i for i in dir(wilson_suite.wilson_derive.response_terms.VibPerturbedTerm) if '__' not in i])

        pprint(flat_derived_terms[0].__dict__)


        for t in flat_derived_terms:
            for i in t.res:
                pprint(f'ResonanceCondition instance: {i.__dict__}')
                pprint(i.diff.__dict__)

        pprint(str_dict)

        return derived_terms, flat_derived_terms, str_dict
    else:
        raise ValueError('No pkl terms file')
    

def evv_experiment() -> we_abst.VibExperiment:
    """
    Returns VibExperiment instance for EVV experiment
    """
    return wilson_suite.fixtures.evv_experiment()

def evv_terms() -> list[wilson_suite.wilson_derive.response_terms.VibPerturbedTerm]:
    """
    Returns EVV terms derived with wilson_derive
    """
    from ..wilson_main import workflow_abstractions as wf_abst
    sim = wf_abst.WilsonSimulation()
    sim.addExperiment(experiment=evv_experiment())
    sim.getTerms(deriver=get_fully_enhanced_terms)
    return sim.terms


def bare_wsim_for_EVVpGVPT2(vib_ana_setup:"VibAnaSetup", 
                            eval_uniform:"DataOriginInfo", 
                            system:"MolecularSystem",
                            project_directory:str='.',
                            silent=False):
    """
    EVV experiment with internally computed GVPT2 vibrational states

    vib_ana_setup=ws.main.abstractions.VibAnaSetup(regime='GVPT2', vibana_own_analysis='anharm')
    calc_setup=ws.main.abstractions.DataOriginInfo(program='gaussian', lvl_theory='B3LYP', basis='cc-pVQZ')
    """
    from ..wilson_main import abstractions as wm_abst
    from ..wilson_main import workflow_abstractions as wf_abst

    vib_ana_setup = wm_abst.VibAnaSetup(regime=vib_ana_setup.regime, vibana_own_analysis=vib_ana_setup.vibana_own_analysis)
    # -------------------------
    sim = wf_abst.WilsonSimulation()
    # adding EVV experiment
    sim.addExperiment(experiment=evv_experiment())

    # deriving EVV terms
    terms_file_path = os.path.join(project_directory, 'derived_terms.pkl')

    if os.path.isfile(terms_file_path):
        with open(terms_file_path, 'rb') as handle:
            derived_terms = pickle.load(handle)
        sim.addTerms(terms=derived_terms)
    else:
        sim.addTerms(terms=get_eval_ready_evv_terms())
        with open(terms_file_path, 'wb') as handle:
            pickle.dump(sim.terms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # -------------------------
    sim.addSystem(system=system) # maybe should enforce this same system for all other things
    # -------------------------
    vib_ana_setup.system = system
    vib_ana_setup.upd_exclude_modes()

    # do GVPT2 internally
    sim.addVibAnaSetup(vib_ana_setup=vib_ana_setup)
    assert vib_ana_setup.system==sim.vib_ana_setup.system
    assert vib_ana_setup.system==system
    assert vib_ana_setup.system==sim.system
    # -------------------------
    sim.setPropsAndMaxStateLvl()

    needed_props = {i.trivial_name: None for i in sim.props}

    if not silent:
        pprint('To compute a spectrum with this experiment you need:')
        pprint(f'Vibrational states up to {sim.max_state_lvl} quant(um/a).')
        pprint('Molecular properties:')
        pprint([(i.trivial_name, i.prop_spec) for i in sim.props])

        pprint('Needed properties:')
        pprint(needed_props)
    
    sim.addPropEvalSetup(eval_uniform=eval_uniform)
    sim.dressPropsWithSetup()
    return sim

def makeSpecSetup2D(start, end, spacer, axes: dict, configs: dict) -> "SpecEvalSetup":
    """ 
    axis1 = ws.main.abstractions.SpectralAxis({'w1': 1})
    axis2 = ws.main.abstractions.SpectralAxis({'w1': 1, 'w2': -1})

    axes={'x': axis1, 'y': axis2}
    """
    from ..wilson_main import spectrum_abstractions as spc_abst

    spec_grid = spc_abst.SpectralGrid(axes=axes, 
                                     range_style='uniform',
                                     start=start, end=end, spacer=spacer)

    w1var = spc_abst.EvaluationVariable(range_style='uniform', 
                                       start=start['x'], 
                                       end=end['x'], 
                                       spacer=spacer['x'])
    w2var = spc_abst.EvaluationVariable(range_style='uniform', 
                                       start=start['y'], 
                                       end=end['y'], 
                                       spacer=spacer['y'])
    eval_vars = {'w1': w1var.range,
                 'w2': w2var.range}
    # logger.warning(w1var)
    # logger.warning('------')
    # logger.warning(w2var)
    import numpy as np
    meshgrids = np.meshgrid(*eval_vars.values(), indexing='ij')

    eval_vars_meshgrids = {}
    for i, key in enumerate(eval_vars.keys()):
        eval_vars_meshgrids[key] = meshgrids[i]
    

    style_config = make_plot_config()

    evi = spc_abst.EvaluationInfo(**{'freq_variables': eval_vars_meshgrids,
                                                 'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                 'margins': {'w1': 10., 'w2': 10.}})
    rndi = spc_abst.RenderingInfo(**{'intensity_normalization_type': NormalizationType.LOG_SCALE,
                                                 'dynamic_range': configs.get('dynamic_range', None),
                                                 'num_levels': 15, 
                                                 'reference_max': None,
                                                 'spec_data_operations': 'abs()**2',
                                                 'projection': '2d', 
                                                 'filename': 'smth.svg',
                                                 'backend': 'matplotlib',
                                                 'to_save': True,
                                                 'style_config': style_config})
    
    eval_setup = spc_abst.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)
    
    return eval_setup

_CUSTOM_PlotConfig = dict(
    figsize=(35, 45),
    label_fontsize=30,
    font_dict={'size': 24},
    colormap='hot_r',
    saturation_color='#FF00FF',
    dpi=350,
    tick_step=200.0,
    equal_aspect=True,
    no_data_color='#E0E0E0',
    below_range_color='#F8F8F8',
    data_edge_color='black',
    data_edge_width=0.75,
    # y_min=0,
    # y_max=4500,
    colorbar_main_label="Intensity",
    colorbar_padding=0.02,
    show_top_ticks=True,
    show_right_ticks=True,
    x_tick_rotation=45,
    colormap_spacing='log',
    colormap_power=0.5,
)


def make_plot_config(**overrides) -> PlotConfig:
    """
    Return a PlotConfig instance using custom defaults, with optional overrides.
    """
    params = {**_CUSTOM_PlotConfig, **overrides}
    return PlotConfig(**params)


def make_VibStatesData(vib_ana_setup: 'VibAnaSetup', save_to_pkl: str = None):
    from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData
    from wilson_suite.wilson_utils.serialization import pickle_this_to
    
    vibstates_data = VibStatesData(allstates=tuple(vib_ana_setup.states), 
                                    harmonic_osc_states_labels=vib_ana_setup.include_list,
                                    number_of_nmodes=vib_ana_setup.number_of_modes)
    if save_to_pkl is not None:
        pickle_this_to(vibstates_data, filenamepkl=save_to_pkl)
    return vibstates_data