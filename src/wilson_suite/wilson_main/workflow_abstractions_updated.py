"""
1. VibExperiment object  -> 
2. derive Terms -> 
3. configure necessary settings (eval/render) -> 
4. request Data for evaluation [necessary for intensities eval, vib info - max state level only, nothing about vib analysis] -> 
5. get DataResults

"""
from typing import Any, Callable
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .spectrum_abstractions import SpecEvalSetup
	from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationInputsExtended

from .main_functions import find_props_and_max_state_lvl
from .abstractions import (VibAnaSetup, MolecularProperty,
						   MolecularSystem, DataOriginInfo)
from ..wilson_derive.response_terms import VibPerturbedTerm
from ..wilson_derive.term_var_translate import translate_terms_to_axis_variables
from wilson_suite.wilson_experiment.experiment_abstractions import VibExperiment
from wilson_suite.wilson_main.abstractions import VibState

import numpy as np

import logging

from ..wilson_experiment.indep_vars_and_axes import SpectralAxisSet

logger = logging.getLogger("wilson")




class ConfiguredSimulation:
	"""
	Class to hold up to a full set of information for a Wilson run and carry out operations related to the run
	workflow.
	"""

	def __init__(self, 
			  	 experiment: VibExperiment,
				 terms: dict[int, dict[tuple, VibPerturbedTerm]],
				 system: MolecularSystem,
				 eval_setup: 'SpecEvalSetup',
				 render_setup,
				 ):
		"""
		experiment: VibExperiment instance: The experiment to which this simulation pertains
		terms: list of VibPerturbedTerm instances: The terms (working expressions) to be evaluated over the spectral range in this simulation ---??? so the terms must follow the spectral range setting??

		spec_eval_setup: SpecEvalSetup instance: Setup information for evaluation and rendering
		system: MolecularSystem instance: The system under consideration in this simulation

		import_from: string: File reference from which to import attributes of the present instance of this class

		====

        # .ev_info.Gamma, .ev_info.Gamma_unit, 
		# .ev_info.dynamic_range, .ev_info.box_range_safety_margin, 
		# .ev_info.scale_wrt_max_intensity, .ev_info.minimum_box_padding, 
		# .ev_info.exp_magn_conditions, .ev_info.magn_conditions_margin, 
		# .ev_info.spectral_window, .ev_info.grid_resolution, 
		# vib_ana_setup.states, vib_ana_setup.include_list,
		# vib_ana_setup.number_of_modes, vib_ana_setup.nc_sqrt_eigval
		# experiment.polarization_avg_vector
		# system.Nnmodes

		spec_eval_setup = simulation.spec_eval_setup  
        
		# MR: Here assuming that spectral axes were set, so changed to use translated terms
        terms = simulation.terms_in_axis_choice
        number_of_modes = simulation.system.Nnmodes
        props = simulation.props
        vib_ana_setup = simulation.vib_ana_setup
        pulse_polarization_vector = tuple(simulation.exp.polarization_avg_vector)
		"""

		self.experiment = experiment
		self.terms = terms

		self.system = system
		
		self.eval_setup = eval_setup
		self.render_setup = render_setup


	@property
	def terms_collection(self):
		from wilson_suite.wilson_derive.response_terms import VibPertTermsCollection
		return VibPertTermsCollection(self.terms, self.experiment)
	

	def request_data(self) -> dict[str, None]:
		"""
		returns dictionary of trivial names and empty values, also max_st_lvl value 
		"""
		max_state_lvl, props = self.terms_collection.required_data()
		req_data = {'max_state_lvl': max_state_lvl}

		for p in props:
			req_data[p.trivial_name] = None
		
		return req_data

	
	def prep_eval_data(self, requested_data_in):
		from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationInputsExtended, EvaluationWorkflow
		# prepare data for input to EvaluationWorkflow
		# eval_inputs = make_evaluation_inputs(simulation=self)
		
		return EvaluationInputsExtended(
										terms=self.terms, # terms_in_axis_choice
										number_of_modes=self.system.Nnmodes,
										props=requested_data_in['props'],
										Gamma=self.eval_setup.Gamma,
										Gamma_unit=self.eval_setup.Gamma_unit,
										dynamic_range=self.render_setup.dynamic_range,
										box_range_safety_margin=self.eval_setup.box_range_safety_margin,
										scale_wrt_max_intensity=self.eval_setup.scale_wrt_max_intensity,
										minimum_box_padding=self.eval_setup.minimum_box_padding,
										exp_magn_conditions=self.eval_setup.exp_magn_conditions,
										magn_conditions_margin=self.eval_setup.magn_conditions_margin,
										spectral_window=self.eval_setup.spectral_window,
										grid_resolution=self.eval_setup.grid_resolution,
										states=requested_data_in['states'],
										include_list=self.eval_setup.include_list,
										nc_sqrt_eigval=requested_data_in['nc_sqrt_eigval'],
										pulse_polarization_vector=tuple(self.experiment.polarization_avg_vector),
										)

	def evaluate(self, eval_inputs: 'EvaluationInputsExtended'):
		"""
		returning evaluation results object for further analysis or rendering
		"""
		from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow
		# prepare data for input to EvaluationWorkflow
		
		workflow = EvaluationWorkflow(inputs=eval_inputs)
		self._workflow = workflow
		wf_result = workflow.run()

		# TODO: this is a temporary fix? can be organized better?
		self.spec_eval_setup.grid = {'A': wf_result['A'], 'B': wf_result['B']}
		self.spec = wf_result['result']

		return EvaluatedResult()


class SimulationBuilder:
	"""
	sim = (
    SimulationBuilder(EVV_EXPERIMENT)
    .with_terms(DERIVED_EVV_TERMS)
    .with_system(molecular_system)
    .with_eval_setup(calc_setup)
    .with_render_setup(render_setup)
    .build()
	)

	# only now does execution begin
	data = sim.request_data()
	results = sim.evaluate(data)
	results.render()
	
	"""
	def __init__(self, experiment: VibExperiment):
		self._experiment = experiment
		self._terms = None
		self._system = None
		self._eval_setup = None
		self._render_setup = None

	def with_terms(self, terms):
		self._terms = terms
		return self  # enables chaining

	def with_system(self, system):
		self._system = system
		return self

	def with_eval_setup(self, setup):
		self._eval_setup = setup
		return self

	def with_render_setup(self, setup):
		self._render_setup = setup
		return self

	def build(self) -> ConfiguredSimulation:
		# validation happens here, once, before anything executes
		if self._terms is None:
			raise ValueError("terms required")
		if self._system is None:
			raise ValueError("system required")
		if self._eval_setup is None:
			raise ValueError("eval setup required")

		return ConfiguredSimulation(
			experiment=self._experiment,
			terms=self._terms,
			system=self._system,
			eval_setup=self._eval_setup,
			render_setup=self._render_setup,
		)
	
	def build_from(self):
		raise NotImplementedError('cannot build from files yet')


@dataclass(frozen=True)
class EvaluatedResult:
	spec: np.ndarray
	grid: dict
	spec_eval_setup: SpecEvalSetup

	def render_spectrum(self, do_diagn: bool = False):
		from wilson_suite.wilson_analysis.render.matplotlib_renderer import MatplotlibRenderer

		filename = self.spec_eval_setup.rnd_info.filename
		backend = self.spec_eval_setup.rnd_info.backend

		if backend == 'matplotlib':
			renderer_class=MatplotlibRenderer
		else:
			raise NotImplementedError('Only matplotlib backend is currently supported')

		renderer = renderer_class(spec_data=self.spec, 
							spec_grid=self.spec_eval_setup.grid,
							ev_info=self.spec_eval_setup.ev_info, 
							rnd_info=self.spec_eval_setup.rnd_info,
							do_diagn=do_diagn)
		fig, ax, contour, cbar = renderer.render(filename)

