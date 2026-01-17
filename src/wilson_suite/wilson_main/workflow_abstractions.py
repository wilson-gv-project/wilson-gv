from typing import Any, Callable
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .spectrum_abstractions import SpecEvalSetup

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

class WilsonSimulation:
	"""
	Class to hold up to a full set of information for a Wilson run and carry out operations related to the run
	workflow.
	"""

	def __init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: 'SpecEvalSetup'=None, system: MolecularSystem=None,
				 props: list[MolecularProperty]=None, residual_vib_info: list=None,
				 spec: Any=None, diagn: dict=None, rendering: Any=None, import_from: str=None, name: str=None):
		"""
		FIXME: terms: list[VibPerturbedTerm]=[] - that's not true
		FIXME: Consider changing ordering of these arguments
		exp: VibExperiment instance: The experiment to which this simulation pertains
		terms: list of VibPerturbedTerm instances: The terms (working expressions) to be evaluated over the spectral
		range in this simulation
		vib_ana_setup: VibAnaSetup instance: Setup and storage of results concerning the vibrational states
		spec_eval_setup: SpecEvalSetup instance: Setup information for evaluation and rendering
		system: MolecularSystem instance: The system under consideration in this simulation
		eval_uniform: DataOriginInfo instance: If all properties are (to be) evaluated under the same external
		setup, then providing an DataOriginInfo as this argument signifies that
		eval_by_prop_name: dictionary {trivial name: DataOriginInfo}: If specific properties are (to be) evaluated
		under specific setups, then signify that with this argument. All properties needed but not referred to in this
		way will instead be assumed to be requested under the setup specified in eval_uniform. If no properties
		are missing specification in eval_by_prop_name, then eval_uniform can be excluded.
		props: List of MolecularProperty instances: Properties needed for evaluation
		spec: Type for now unspecified (likely numpy ndarray): Spectral evaluation results in tensor form
		diagn: Dictionary {string (attribute name): value}: Diagnostics information accumulated during evaluation
		and/or rendering
		rendering: Type for now unspecified: Rendered spectra
		import_from: string: File reference from which to import attributes of the present instance of this class
		# TODO: Implement hdf5 store/fetch routines
		name: string: The name of this simulation for reference purposes
		"""

		if import_from is None:

			self.exp = exp
			self.terms = terms
			self.vib_ana_setup = vib_ana_setup
			self.spec_eval_setup = spec_eval_setup
			self.system = system

			self.props = props
			self.residual_vib_info = residual_vib_info

			self.spec = spec
			self.diagn = diagn
			self.rendering = rendering

			self.name=name

			self.axis_choice = None
			self.terms_in_axis_choice = None

		else:

			# TODO: Implement functionality to set up class instance from file
			pass

	# ==================== Simple State Checks ====================

	@property
	def is_configured(self) -> bool:
		"""Check if basic configuration is complete"""
		return all([
			self.exp is not None,
			self.system is not None,
			self.terms is not None,
			self.vib_ana_setup is not None,
			self.spec_eval_setup is not None
		])

	@property
	def is_ready(self) -> bool:
		"""Check if ready to evaluate (has properties with data)"""
		# print(f'self.is_configured {self.is_configured}\nself.props is not None {self.props is not None}\nlen(self.props) > 0 {len(self.props) > 0}')
		# print(f"for p in self.props: { {p.trivial_name: p.vals is not None for p in self.props} }") 
		# print(f'self.vib_ana_setup.isAllSet {self.vib_ana_setup.isAllSet}')       
		conds = {'not self.is_configured': self.is_configured , 
				'not self.props is not None': self.props is not None , 
				'not len(self.props) > 0': len(self.props) > 0 ,
				f'p.trivial_name: p.vals is not None for p in self.props: {{p.trivial_name: p.vals is not None for p in self.props}}': all(p.vals is not None for p in self.props) ,
				'not self.vib_ana_setup.isAllSet': self.vib_ana_setup.isAllSet}
		final = True

		for c in conds:
			final &= conds[c]
			if not conds[c]:
				print(c)
				return final

		return final

	@property
	def has_spectrum(self) -> bool:
		"""Check if spectrum has been evaluated"""
		return self.spec is not None


	def addExperiment(self, experiment: VibExperiment):
		"""
		Add an experiment

		experiment: VibExperiment instance: The experiment to be added
		"""

		self.exp = experiment

	def addSystem(self, system: MolecularSystem):
		"""
		Add a system

		system: MolecularSystem instance: The experiment to be added
		"""

		self.system = system

	def addTerms(self, terms: dict, extend: bool=False):
		"""
		Add terms

		terms: List of VibPerturbedTerm instances: The terms to be added
		extend: Boolean: Add this to (possibly already existing) terms or (default) set up this
		attribute afresh (possibly overwriting existing terms)?
		"""

		if not extend:
			self.terms = terms

		else:
			self.terms.extend(terms)

	def setAxisChoiceAndTranslateTerms(self, axis_choice: SpectralAxisSet):
		"""
		Set an axis choice and translate self.terms to be given in terms of this axis choice
		"""
		self.axis_choice = axis_choice
		if self.terms is None:
			raise ValueError('No terms to translate to axis choice were found')
		self.terms_in_axis_choice = translate_terms_to_axis_variables(self.terms, self.axis_choice)


	def addVibAnaSetup(self, vib_ana_setup: VibAnaSetup):
		"""
		Add a vibrational analysis setup

		vib_ana_setup: VibAnaSetup instance: The vibrational analysis to be added
		"""
		self.vib_ana_setup = vib_ana_setup

	def addPropEvalSetup(self, eval_uniform: DataOriginInfo=None,
						 eval_by_prop_name: dict[str: DataOriginInfo]=None,):
		"""
		Add a property evaluation setup

		See argument explanation of __init__ method of this class for explanation of these arguments

		VL: What if done after self.props is filled? 
		then can check if all props have calculation setup specified in parameters here.
		Also can warn user about the use of eval_uniform for props not mentioned in eval_by_prop_name
		"""

		self.eval_uniform = eval_uniform
		self.eval_by_prop_name = eval_by_prop_name

	def addSpecEvalSetup(self, spec_eval_setup: 'SpecEvalSetup'):
		"""
		Add a spectral evaluation/rendering setup

		spec_eval_setup: SpecEvalSetup instance: The setup to be added
		"""

		self.spec_eval_setup = spec_eval_setup

	def updDiagnostics(self, upd_dict: dict):
		"""
		add info to self.diagn dictionary
		"""
		if self.diagn is None:
			self.diagn = {}
		
		self.diagn.update(upd_dict)

	def setPropsAndMaxStateLvl(self, freqs: str='static'):
		"""
		freqs: String: For terms involving properties that may be frequency dependent, use
		experiment information ('exp') or use the static ('static') properties?
		"""

		if self.terms is None:
			raise AssertionError('There must be terms present to determine needed properties')
		if self.vib_ana_setup is None:
			raise AssertionError('There must be a vibrational analysis setup present to get properties needed there, if any')

		self.props, self.residual_vib_info, self.vib_ana_setup.max_state_lvl = \
			find_props_and_max_state_lvl(self.terms, self.vib_ana_setup, freqs)


	def dressPropsWithSetup(self):
		"""
		Dress my self.properties with computational setups according to how they are specified in
		self.eval_uniform or self.eval_by_prop_name

		Run to reset values of residual_vib_info data - replace values with calc_setup
		"""
		if not self.props:
			logger.warning('There are no properties to be dressed')

		for i in self.props:

			dressed = False

			# See if property specifically mentioned and if so use that
			if self.eval_by_prop_name is not None:

				if i.trivial_name is not None:
					if i.trivial_name in self.eval_by_prop_name:
						i.addCalcSetup(self.eval_by_prop_name[i.trivial_name])
						dressed=True

				else:
					logger.warning('Warning: Property without trivial name encountered but eval_by_prop_name was specified.')

			# Otherwise, use uniform eval argument
			# if both are not None, this will overide previous setup if dressed before
			if self.eval_uniform is not None and not dressed:

				i.addCalcSetup(self.eval_uniform)
				dressed = True

			if not dressed:
				raise AssertionError(f'Unable to determine calculation setup for property: {i}')

		for i in self.residual_vib_info:
			
			dressed = False
			
			if self.eval_by_prop_name is not None:

				if self.eval_by_prop_name.get(i, None) is not None:
					self.residual_vib_info[i] = self.eval_by_prop_name[i]
					dressed=True

				else:
					logger.warning('Warning: _ encountered but eval_by_prop_name was specified.')

			# Otherwise, use uniform eval argument
			# if both are not None, this will overide previous setup if dressed before
			if self.eval_uniform is not None and not dressed:
				self.residual_vib_info[i] = self.eval_uniform
				dressed = True


	def fillResults(self, data_dict: dict):
		"""
		loading data into self.props (and optionally to self.vib_ana_setup)
		
		data_dict: dict - {data_name: values}

		"""
		from .main_functions import fill_props_results, fill_residual_vib_info_results
		fill_props_results(self.props, data_dict)

		fill_residual_vib_info_results(self.vib_ana_setup, self.residual_vib_info, data_dict)


	def requestData(self) -> dict:
		"""
		data_dict: dict - {data_name: DataOriginInfo}
		"""
		data_dict = {}
		from .main_functions import request_props, request_residual_vib_info

		if not all(isinstance(p.calc_setup, DataOriginInfo) for p in self.props):
			raise ValueError("Run WilsonSimulation.dressPropsWithSetup() to reset props values")
		request_props(self.props, data_dict)

		if not all(isinstance(i, DataOriginInfo) for i in self.residual_vib_info.values()):
			raise ValueError("Run WilsonSimulation.dressPropsWithSetup() to reset residual_vib_info values")
		request_residual_vib_info(self.residual_vib_info, data_dict)

		return data_dict
	
	def getResults(self, obtainer: Callable[[dict[str,DataOriginInfo]], dict],
					save_to_filename: str = None):
		"""
		obtainer must return : a dictionary:
		 	keys: trivial_name for properties or residual_vib_info keys
			values: values
		
		# todo: default obtainer??
		"""
		data_dict = obtainer(self.requestData())
		
		# FIXME should it be a separate function with saving option??
		if save_to_filename is not None:
			if not hasattr(self, '_run_dir'):
				raise ValueError("Project directory for saving files was not initialized")
			
			if '.' not in save_to_filename:
				raise ValueError("Provide save_to_filename with file extention specified")
			format = save_to_filename.split('.')[1]

			from wilson_suite.wilson_utils import save_obtained_data
			save_obtained_data(data_dict, format=format, filename=save_to_filename, save_to_dir=self._run_dir)

		self.fillResults(data_dict=data_dict)



	def attempt_setup_fill_with_defaults(self):
		"""
		If possible, attempt to complete remaining pieces of setup with default choices

		Here add handling for making canonical axis choice (and translating terms to same) if none selected
		Can also have defaults for spectral window, resolution, damping and other related information
		Can also add other "wrap-up" parts (e.g. translate terms if axes choice made but terms not translated yet)

		"""

		pass

	def evaluate(self, save_evalinputs_pkl: str = None):
		"""
		Evaluating method, using EvaluationWorkflow
		"""
		from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow, make_evaluation_inputs
		if self.axis_choice is None:
			self.setAxisChoiceAndTranslateTerms(self.exp.canonical_axes)

		# prepare data for input to EvaluationWorkflow
		eval_inputs = make_evaluation_inputs(simulation=self)
		
		# save EvaluationInputs data optionally to a pickle file
		if save_evalinputs_pkl is not None:
			if not hasattr(self, '_run_dir'):
				raise ValueError("Project directory for saving files was not initialized")
			
			from wilson_suite.wilson_utils.serialization import pickle_this_to
			pickle_this_to(eval_inputs, filenamepkl='EvaluationInputs.pkl', save_to=self._run_dir)

		workflow = EvaluationWorkflow(inputs=eval_inputs)
		self._workflow = workflow
		wf_result = workflow.run()

		if self.diagn is None:
			self.diagn = {}
		self.diagn.update({'artifacts': workflow.artifacts})

		# TODO: this is a temporary fix? can be organized better?
		self.spec_eval_setup.grid = {'A': wf_result['A'], 'B': wf_result['B']}
		self.spec = wf_result['result']

	def evaluate_with_default_setup_fill(self):
		"""
		Attempt to fill remaining setup with default and if successful, evaluate spectrum
		"""

		self.attempt_setup_fill_with_defaults()
		self.evaluate()

	def evaluateSpectrum(self,
                         evaluator: Callable[[
								   MolecularSystem, VibExperiment, list[VibPerturbedTerm], list[MolecularProperty],
								   'SpecEvalSetup', VibAnaSetup, bool], tuple[np.ndarray, dict]],
                         do_diagn: bool=False):
		"""
		Evaluate the spectrum

		! unused now, there is no generalized evaluator function now; should be removed?
		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, an experiment, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""
		# TODO - checks like in VibAnaSetup.doAnharmonicAnalysis
		if not self.vib_ana_setup.isAllSet:
			raise AssertionError('VibAnaSetup is not ready for evaluateSpectrum()')

		# NOTE 260106: Could now use self.terms_in_axis_choice and self.axis_choice
		# To discuss: Handling here (canonical axes plus translate) if no choice made already?


		context = dict(system=self.system, experiment=self.exp, derived_terms=self.terms, props=self.props,
				 spec_eval_setup=self.spec_eval_setup, vib_ana_setup=self.vib_ana_setup, 
				 do_diagn=do_diagn)
	
		if do_diagn:
			self.spec, diagn = evaluator(**context)
			self.updDiagnostics(upd_dict=diagn)
			
			if not isinstance(self.diagn, dict):
				raise AssertionError('Diagnostics result must be dictionary')
		else:
			self.spec, _ = evaluator(**context)

		# if not isinstance(self.spec, np.ndarray):
		# 	raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')


	def render(self, renderer: Callable[[np.ndarray, MolecularSystem, VibExperiment,
														dict, str, 'SpecEvalSetup'], tuple[Any, dict]],
														do_diagn: bool=False):
		"""
		Render the spectral data

		renderer: Callable: A function to carry out the rendering. As in render but must additionally return a
		dictionary of diagnostics information.
		"""
		assert self.spec is not None, 'No spectrum data, there is nothing to render'
		assert self.spec_eval_setup is not None, 'Setup information for evaluation and rendering is not provided'
		
		if self.diagn is None:
			self.diagn = {}

		# TODO also - self.system, self.exp, self.name
		# generate self.name?
		
		context = dict(spec_data=self.spec,
					   spec_eval_setup=self.spec_eval_setup, 
					   do_diagn=do_diagn)
		
		logger.debug('context')
		logger.debug(context)

		# Consider extending arguments to provide even more info to renderer
		if do_diagn:
			self.rendering, diagn = renderer(**context)
			self.updDiagnostics(upd_dict=diagn)
			
			if not isinstance(self.diagn, dict):
				raise AssertionError('Diagnostics result must be dictionary')
		else:
			self.rendering, _ = renderer(**context)

	def render_spectrum(self, do_diagn: bool):
		"""
		
		"""
		if self.spec is None:
			raise ValueError("No spectrum data to render")
		
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

	def to_dict(self):
		"""
		__init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
				 eval_uniform: DataOriginInfo=None, eval_by_prop_name: dict[str: DataOriginInfo]=None,
				 props: list[MolecularProperty]=None, calc_batches: dict[int: CalculationBatch]=None,
				 spec: Any=None, diagn: dict=None, rendering: Any=None, import_from: str=None, name: str=None):
		
		redundancies:

		"""
		attributes = ['exp', 'vib_ana_setup', 'spec_eval_setup', 
				'system', 'eval_uniform', 'eval_by_prop_name', 'props', 
				'calc_batches', 
				'diagn', 'rendering', 'name'] # skipping spec for now
		
		# redundancy
		# calc_batches
		
		result = {}
		for key in attributes:
			value = getattr(self, key)
			if is_dataclass(value):
				try:
					result[key] = asdict(value)
				except TypeError:
					result[key] = value.to_dict()
			elif hasattr(value, "to_dict"):  # If it's a custom object
				result[key] = value.to_dict()
			elif isinstance(value, (list, tuple)):
				result[key] = [
					item.to_dict() if hasattr(item, "to_dict") else item for item in value
				]
			elif isinstance(value, dict):
				result[key] = {
					k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in value.items()
				}
			else:
				result[key] = value
		from ..wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
		result['terms'] = derived_terms_dict_to_dicts(getattr(self, 'terms'))
		return result

	
	@classmethod
	def from_dict(cls, data):
		"""
		__init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
				 eval_uniform: DataOriginInfo=None, eval_by_prop_name: dict[str: DataOriginInfo]=None,
				 props: list[MolecularProperty]=None, calc_batches: dict[int: CalculationBatch]=None,
				 spec: Any=None, diagn: dict=None, rendering: Any=None, import_from: str=None, name: str=None):
		
		VibExperiment - dataclass
		terms - list[VibPerturbedTerm] - handle this or maybe no need?
		VibAnaSetup - not a dataclass

		SpecEvalSetup - dataclass
		MolecularSystem - dataclass
		DataOriginInfo - dataclass

		props - list[MolecularProperty] - not a dataclass
		calc_batches - dict[int: CalculationBatch] - not a dataclass
		"""
		return cls(
            exp=data['exp'].from_dict(),
            terms=data['terms'],
            vib_ana_setup=data['vib_ana_setup']
        )
	
	def writeToJsonFile(self, filename: str = "WilsonSimulation.json"):
		import json
		with open(filename, "w") as f:
			json.dump(self.to_dict(), f, indent=4)
		logger.info(f'WilsonSimulation instance is saved to file {filename}')

	# TODO: status_report() method

	def save_to_pkl(self, configs_only: bool = False, filename: str = 'WilsonSimulation_instance.pkl'):
		"""
		
		:param self: Description
		:param configs_only: Description
		"""
		if not hasattr(self, '_run_dir'):
			raise ValueError("Project directory for saving files was not initialized")
			# self.make_proj_dir()
		
		if not configs_only:
			from wilson_suite.wilson_utils.serialization import pickle_this_to
			pickle_this_to(self, filename, self._run_dir)
		else:
			self.save_configs(filename)


	from pathlib import Path
	def make_proj_dir(self, base_dir: Path = None) -> Path:
		"""
		base_dir = Path("workflows")
		run_dir = sim.make_proj_dir(base_dir)
		data_dir = run_dir / "data"

		:param base_dir: optional
		:return: Description
		"""
		if base_dir is None:
			from wilson_suite.wilson_utils.paths import WORKFLOW_BASE_DIR
			base_dir = WORKFLOW_BASE_DIR

		base_dir.mkdir(parents=True, exist_ok=True)
		from datetime import datetime
		
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		self._run_dir = WORKFLOW_BASE_DIR / f"run_{timestamp}"
		self._run_dir.mkdir()
		(self._run_dir / "figures").mkdir()

	def get_configs(self) -> dict:
		"""
		TODO: have those objects pruned (exp, vib_ana_setup) to only settings(setup) info
		"""
		return {'system': self.system,
		  		'experiment': self.exp,
				'spec_eval_setup': self.spec_eval_setup, 
		  		'vib_ana_setup': self.vib_ana_setup}

	def save_configs(self, filename: str = 'WilsonSimulation_configs.pkl'):
		from wilson_suite.wilson_utils.serialization import pickle_this_to
		pickle_this_to(self.get_configs(), filename, self._run_dir)


# simply copying old sketch for now
class WilsonSimulations:
	"""
	Class to hold collective jobs instructions (i.e. instructions for collections of jobs, not for individual jobs)
	# FIXME: Not implemented and form not yet settled, skipping further documentation for now
	"""

	def __init__(self, jobs=None, coll_instructions=None):

		self.jobs = jobs
		self.coll_instructions = None

	# A job is a WilsonSimulation instance
	def addJob(self, job):
	
		pass
		
	# Make calculation batches needed to fulfill tasks
	def makeCalculationBatches(self):
	
		# Walk through simulations and collect into calculation batches for each setup
		# Implement union of these batches
	
		pass

	def evaluate(self):

			#spectra = wilsonpart2.evaluate(self.exp.get_terms(), self.properties, self.eval_setup.visualization_setup)
			pass

	def render(self):

			pass
