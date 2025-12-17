from typing import Any, Callable
from dataclasses import dataclass, field, asdict, is_dataclass

from .spectrum_abstractions import SpecEvalSetup
from .main_functions import find_props_and_max_state_lvl
from .abstractions import (VibAnaSetup, MolecularProperty,
						   MolecularSystem, DataOriginInfo)
from ..wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_experiment.experiment_abstractions import VibExperiment
from wilson_suite.wilson_main.abstractions import VibState

import numpy as np

import logging
logger = logging.getLogger("wilson")

class WilsonSimulation:
	"""
	Class to hold up to a full set of information for a Wilson run and carry out operations related to the run
	workflow.
	"""

	def __init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
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

		else:

			# TODO: Implement functionality to set up class instance from file
			pass


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
		FIXME, not a list
		Add terms

		terms: List of VibPerturbedTerm instances: The terms to be added
		extend: Boolean: Add this to (possibly already existing) terms or (default) set up this
		attribute afresh (possibly overwriting existing terms)?
		"""

		if not extend:
			self.terms = terms

		else:
			self.terms.extend(terms)

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

	def addSpecEvalSetup(self, spec_eval_setup: SpecEvalSetup):
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

		for p in self.props:
			p.addValues(data_dict.get(p.trivial_name))

		for k in self.residual_vib_info:
			if k in ['anharmonic_states', 'harmonic_states']:
				states_list = []
				states_dict: dict = data_dict.get(k)

				for state, energy in states_dict.items():
					states_list.append(VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=','.join(state)))

				self.vib_ana_setup.setStates(states=states_list)
				self.residual_vib_info[k] = data_dict.get(k)

			else:
				self.residual_vib_info[k] = data_dict.get(k)
				setattr(self.vib_ana_setup, k, data_dict.get(k))


	def requestData(self) -> dict:
		"""
		data_dict: dict - {data_name: DataOriginInfo}
		"""
		data_dict = {}
		for p in self.props:
			data_dict[p.trivial_name] = p.calc_setup
		
		for k, v in self.residual_vib_info.items():
			data_dict[k] = v
		
		return data_dict
	
	def getResults(self, obtainer: Callable[[dict[str,DataOriginInfo]], dict]):
		"""
		obtainer must return : a dictionary:
		 	keys: trivial_name for properties or residual_vib_info keys
			values: values
		"""
		self.fillResults(data_dict=obtainer(self.requestData()))


	def evaluateSpectrum(self,
                         evaluator: Callable[[
								   MolecularSystem, VibExperiment, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup, bool], tuple[np.ndarray, dict]],
                         do_diagn: bool=False):
		"""
		Evaluate the spectrum

		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, an experiment, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""
		# TODO - checks like in VibAnaSetup.doAnharmonicAnalysis 
		if not self.vib_ana_setup.isAllSet:
			raise AssertionError('VibAnaSetup is not ready for evaluateSpectrum()')

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
														dict, str, SpecEvalSetup], tuple[Any, dict]],
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
		
		context = dict(spec_data=self.spec, system=self.system, experiment=self.exp,
					   diagn=self.diagn, name=self.name, 
					   spec_eval_setup=self.spec_eval_setup, do_diagn=do_diagn)
		
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
