from dataclasses import dataclass, field, asdict, is_dataclass, InitVar
from typing import Callable, Any, Optional
import json
import numpy as np
import copy

from ..wilson_utils.prop_trivname import prop_trivname
from ..wilson_derive.abstractions import VibPerturbedTerm
from ..wilson_experiment.abstractions import VibExperiment
from ..wilson_utils.abstractions import VibState
from ..wilson_analysis.render.spectrum_renderer import PlotConfig, NormalizationType

import logging
logger = logging.getLogger("wilson."+__name__)
# wilson.wilson_main.abstractions
namelogger = f'"wilson."+__name__: {"wilson."+__name__}'
logger.info(namelogger)

# A system is here only the system name, molecular geometry and atoms (masses for isotopes?)
@dataclass
class MolecularSystem:
	"""
	name: String: System name

	geo: Currently not fully fixed form but must specify system geometry in a form
	[[atom name, [x, y, z](Ångström)], ...], where atom_name must be and element name and where
	default isotopes are assumed

	natoms: integer: Number of atoms

	geo_extra: Currently not fixed form but reserved for situations where extra information is needed
	Speculated example format: # [[atom name, [x, y, z] (Ångström), (# protons, # neutrons)], ...]
	and possibly more attributes if needed
	"""
	name: str
	natoms: int = None
	geo: Any = None
	geo_extra: Any = None
	linear: bool = False
	conformer: str = 'conf1'

	@property
	def Nnmodes(self):
		if self.linear:
			return 3*self.natoms-5
		else:
			return 3*self.natoms-6

	def __post_init__(self):
		if (self.geo is not None) and (self.geo_extra is not None):
			raise AssertionError('Ambigious definition: Both default form geometry geo and extended form geometry geo_extra was defined')
		if (self.geo is None) and (self.geo_extra is None):
			logger.info('Note: Molecular system was instantiated without geometry information')
		if self.natoms is None and self.geo is None:
			raise ValueError('Incomplete definition: Either the number of atoms (natoms) or the geometry (geo) of the MolecularSystem is required')
		
		if self.natoms < 3:
			self.linear = True

	def __hash__(self):
		# FIXME: Influence hash with other attributes as well
		return hash(self.name)



@dataclass(frozen=True)
class DataOriginInfo:
	"""
	Class to represent computational setups for properties obtained external to Wilson
	Does not need to pertain to an actual program and could also be used for "get from no specific calculation"/
	"get from file"

	----
	source_type: String: Options: gaussian, cfour, wilson
	lvl_theory: String: Level of theory
	basis_set: String: Basis set
	base_file_loc: String: path to the base file
	"""
	# Strings
	source_type: str = ''
	
	lvl_theory: str = ''
	basis_set: str = ''

	base_file_loc: str = ''


	def __hash__(self):
		return hash((self.source_type, self.lvl_theory, self.basis_set, self.base_file_loc))

	def __eq__(self, other):
		if not isinstance(other, DataOriginInfo):
			return False
		
		return (
            self.source_type == other.source_type and
            self.lvl_theory == other.lvl_theory and
            self.basis_set == other.basis_set and
			self.base_file_loc == other.base_file_loc
		)


@dataclass
class MolecularProperty:
	"""
	Class to represent a molecular (energy derivative or similar) property
	Can both be used "head only" (only prop_spec, target_basis, target_units) to specify only the concept of a property
    and "full" (system, calc_setup) for a particular realization (optional with/without values)
	
	----
	prop_spec: Dictionary {'attr name': val, ...}: Info like perturbing operators, frequencies etc. (all values must be hashable)
	triv_name: String: Trivial name For simplified reference
	vals: Form not specified: Values of properties - could be array or dictionary
	system: MolecularSystem instance: For which system?
	calc_setup: DataOriginInfo instance: For which calculation setup?

	see more in test_main_dataclasses.py::test_MolecularProperty
	"""
	# FIXME: Improve on prop_spec name; settle more consistently what the attributes will be and what must be default
	prop_spec: dict
	trivial_name: str=None
	vals: InitVar[Any] = field(default=None, repr=False)
	calc_setup: DataOriginInfo = None

	def to_dict(self):
		return {
			"prop_spec": self.prop_spec,
			"trivial_name": self.trivial_name,
		}

	# FIXME: Complete/update this
	def h(self, htype: int) -> int:
		"""
		Hashing function with four hash types

		htype: integer: Hash type: Valid choices are

		1: "head only" information (only hash(prop_spec))
		2: hash involves attributes from 1) but also tgt basis, tgt units
		3: hash involves attributes from 2) but also system, calc_setup
		4: hash involves attributes from 3) but also in_basis, in_units
		# TODO: Check for adequate property specification and values format when known
		# TODO: Consider enforcing specification of units and basis when values are provided

		Returns an integer hash value
		"""

		hlist = []

		if (htype < 1) or (htype > 4):

			raise AssertionError('Property hash must be requested with type argument (1-4)')

		if htype >= 1:

			for i in self.prop_spec:

				hlist.append(i)
				hlist.append(self.prop_spec[i])

		if htype >= 3:

			hlist.append(self.system.h())
			hlist.append(self.calc_setup.h())


		return hash(tuple(hlist))

	def addSystem(self, system: MolecularSystem):
		"""
		Associate a MolecularSystem instance

		system:	MolecularSystem instance: The system to be attached
		"""

		self.system = system

	def addCalcSetup(self, calc_setup):
		"""
		Associate an DataOriginInfo instance

		calc_setup: DataOriginInfo instance: The setup to be attached
		"""

		self.calc_setup = calc_setup
	
	# Add values (usually scalars or a numPy array)
	def addValues(self, values: Any):
		"""
		Associate values to this property

		values: Undetermined form: The values to be added
		"""

		self.vals = values


# FIXMEs: Improved handling of mode exclusion; possibly methods changes
@dataclass
class VibAnaSetup:
	"""
    Class for setup for vibrational analysis and storage of the resulting information
	
	----
	regime: string: Vibrational analysis regime (e.g. "harmonic", "GVPT2", "VPT2")
	system: MolecularSystem instance: System to which this instance pertains
	regime_subinfo: dictionary: Extra configuration info for vibrational regime (e.g. skip rotational effects)
	max_state_lvl: integer: Maximum number of vibrational quanta per harmonic state involved in states
	states: List of VibState instances: Specification of each vibrational state in scope
	nc_sqrt_eigval: dictionary {mode index: value}: Harmonic vibrational energy levels
	nc_eigvec: dictionary {mode index: [values]}: Normal mode displacements (canonically in Cartesian basis)
	vibana_own_analysis: String: Which kinds of properties will I need to actually carry out the vibrational analysis? 
		Choices: 
			"full": I need properties for both harmonic and (if chosen) anharmonic analysis,
			"anharm": I only need properties to carry out an anharmonic analysis [I will or have already gotten the harmonic data], 
			"none": I don't need any properties [I will or have already gotten both harmonic and anharmonic data]
	exclude_modes: list: Tells which modes (if any) to exclude in this vibrational analysis
	"""
	regime: str=None
	system: MolecularSystem=None
	regime_subinfo: dict=None
	max_state_lvl: int=None
	states: list[VibState]=field(default=None, repr=False)

	# Dictionary: {nm index: w}
	nc_sqrt_eigval: dict=None
	nc_eigvec: dict=None

	# 'full': Will need properties for harmonic (and, if requested, anharmonic) analysis
	# 'anharm': Will only need props. for anharmonic analysis (harmonic results will be provided from external source)
	# 'none': All results will be provided by external source
	vibana_own_analysis: str='full'

	# TODO: MODE EXCLUSION, REGISTERING OF FERMI RESONANCES (TO BE PASSED TO EVALUATOR)
	exclude_modes: list = None
	diagn: dict = None

	def __post_init__(self):
		if self.exclude_modes is None:
			if self.system is not None:
				self.exclude_modes = []
		else:
			if self.system is None:
				logger.warning('VibAnaSetup().exclude_modes attribute is not meaningfull without having set the VibAnaSetup().system attribute')

	@property
	def modes_indices(self):
		"""
		Automatically set up based on number of modes in the system (3*N-5 or 3*N-6) and exclude_modes list, 
			which is an empty list by default
		Returns empty list if no system attribute
		"""
		if self.system is None:
			raise AttributeError('VibAnaSetup().modes_indices attribute cannot be created without having set the VibAnaSetup().system attribute')

		import numpy as np
		return [int(i) for i in np.arange(self.system.Nnmodes) if i not in self.exclude_modes]
	
	# FIXME: Possibly return to this later
	@property
	def serial_states(self):
		return [{'s': vibst.serial_s, 
		   'e': vibst.e, 'd': vibst.d} for vibst in getattr(self, 'states')]

	@property
	def isAllSet(self):
		"""
		Checking status of VibAna data.
		"""
		if self.nc_sqrt_eigval is not None and self.states is not None:
			return True
		return False
	
	def setStates(self, states: list[VibState]):
		"""
		Set vibrational states

		states: List of VibState instances: The states to be set
		"""

		self.states = states

	def upd_exclude_modes(self, upd_exclude_modes: list = None):
		if self.exclude_modes is None:
			if self.system is not None:
				self.exclude_modes = []
		elif upd_exclude_modes is not None:
			self.exclude_modes = upd_exclude_modes
		else:
			if self.system is None:
				logger.warning('VibAnaSetup().exclude_modes attribute is not meaningfull without having set the VibAnaSetup().system attribute')


def tell_needed_props_for_vib_analysis(vib_ana: VibAnaSetup):
	
	"""
	Tell which MolecularProperty instances are required for a specific vibrational analysis

	Returns a list of MolecularProperty instances detailing which properties are required for curent state of instance
	"""

	needed_props = []

	if vib_ana.vibana_own_analysis == 'none':
		if vib_ana.isAllSet():
			return needed_props
		else:
			needed_props.append({'nc_sqrt_eigvec': None})
			needed_props.append({'states': None})
	
	if (vib_ana.vibana_own_analysis == 'full'):
		# should have share the same setting of origin as nc_sqrt_eigval and nc_eigvec
		# in harmonic analysis procedure it will go from "hess" Property to nc_sqrt_eigval and nc_eigvec

		# FIXME: Not sure about target units
		needed_props.append(MolecularProperty(
			{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
			trivial_name=prop_trivname(ord_geo=2),
			target_basis='cart',
			target_units='au')
		)

	# For now, don't use regime subinfo
	if 'PT2' in vib_ana.regime:

		if (vib_ana.vibana_own_analysis == 'anharm') or (vib_ana.vibana_own_analysis == 'full'):

			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=3),
				target_basis='nm',
				target_units='au')
			)

			# FIXME: Consider implementing extra flag for only semidiagonal force constants needed
			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=4),
				target_basis='nm',
				target_units='au')
			)

			needed_props.append(MolecularProperty(
				{'ops': tuple(['r']), 'freq': (0.0)},
				trivial_name=prop_trivname(ord_rot=1),
				target_basis='nm',
				target_units='au')
			)

			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'r']), 'freq': (0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=2, ord_rot=1),
				target_basis='nm',
				target_units='au')
			)
		
		if vib_ana.vibana_own_analysis == 'anharm':
			needed_props.append({'nc_sqrt_eigvec': None})

	return needed_props


def do_full_vib_analysis(vib_ana: VibAnaSetup, props: list[MolecularProperty],
				   analyzer: Callable[[MolecularSystem, list[MolecularProperty], str, str],
				   tuple[dict, dict, list[VibState], dict]]):
		"""
		Carry a vibrational analysis with the set-up regime: Determine and keep the (harmonic) fundamental
		vibrational energy levels (stored in self.nc_sqrt_eigval), the associated eigenvectors (stored in
		self.nc_eigvec) and the (regime-specific) vibrational states

		props: list of MolecularProperty instances: Molecular properties containing those needed in the analysis
		analyzer: Callable: A reference to an analyzer function. See function definition and attribute explanation in
		__init__ for detailed argument specification: Must take as input a system, a set of properties,
		a choice of regime (and subinfo as relevant) and return fundamental harmonic energy levels,
		the associated eigenvectors and the vibrational states as VibState instances
		system: MolecularSystem instance: The system for which analysis is sought. May optionally already be stored
		with self as self.system
		"""
		if vib_ana.nc_sqrt_eigval is not None or vib_ana.states is not None or vib_ana.nc_eigvec is not None:
			raise AssertionError('Full analysis requested but some of the results are already present')

		if vib_ana.regime is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having chosen an analysis regime')

		if vib_ana.system is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having set the system attribute')
		
		context = {'system': vib_ana.system, 'props': props, 
			 	   'regime': vib_ana.regime, 'regime_subinfo': vib_ana.regime_subinfo}
		
		# To return: nc_sqrt_eigval, nc_eigvec, states, diagn
		return analyzer(**context)
	
def do_anharmonic_analysis(vib_ana: VibAnaSetup, props: list[MolecularProperty], anharmonic_analyzer:
						Callable[[MolecularSystem, list[MolecularProperty], str, str, dict, dict],
						tuple[list[VibState], dict]]):
	"""
	Carry out anharmonic vibrational analysis as set up

	props: list of MolecularProperty instances: Molecular properties containing those needed in the analysis
	analyzer: Callable: A reference to an anharmonic analyzer function. See function definition and attribute 
	explanation in __init__ for detailed argument specification: Must take as input a system, a set of properties,
	a choice of regime (and subinfo as relevant), harmonic fundamental energy levels and associated eigenvectors,
	and return the anharmonically corrected vibrational states as VibState instances.
	system: MolecularSystem instance: The system for which analysis is sought. May optionally already be stored
	with self as self.system
	"""

	if vib_ana.regime is None:
		raise AssertionError('Vibrational analysis cannot be carried out without having chosen an analysis regime')
	else:
		if vib_ana.regime not in ['harmonic', "GVPT2", "VPT2"]:
			raise NotImplementedError('Implemented regime choices are: "GVPT2", "VPT2"')
		elif vib_ana.regime == 'harmonic':
			raise ValueError('Anharmonic analysis requested but chosen vibrational regime is harmonic.')

	if vib_ana.system is None:
		raise AssertionError('Vibrational analysis cannot be carried out without having set the system attribute')

	from inspect import isfunction
	if not isfunction(anharmonic_analyzer):
		raise TypeError('anharmonic_analyzer should be a function')
	
	if vib_ana.nc_sqrt_eigval is None:
		raise ValueError('Missing values for nc_sqrt_eigval, cannot proceed with anharmonic analysis')

	for i in props:
		if i.trivial_name in ['cff', 'qff', 'B', 'coriolis']:
			assert i.vals is not None, f'Missing values for {i.trivial_name}, cannot proceed with anharmonic analysis'

	context = {'system': vib_ana.system, 'props': props, 
				'regime': vib_ana.regime, 'regime_subinfo': vib_ana.regime_subinfo, 
				'nc_sqrt_eigval': vib_ana.nc_sqrt_eigval, 
				'exclude_modes': vib_ana.exclude_modes}
	
	logger.debug(repr(context))

	# To return: states, diagn
	return anharmonic_analyzer(**context)


@dataclass
class SpectralAxis:
	"""
	'Plain' spectral axis for rendering response function freq arg spectra;
	with independent lineshape functions.

	freq_vars is {freq label 1 in this axis: coeff, ...}

	Examples:
		axis1 = ws.main.abstractions.spectralAxis({1: 1})        -- w1
		axis2 = ws.main.abstractions.spectralAxis({1: 1, 2: -1}) -- w1-w2

	simple w1 and w2:
		axis1 = ws.main.abstractions.spectralAxis({1: 1})        -- w1
		axis2 = ws.main.abstractions.spectralAxis({2: 1})        -- w2
	"""
	freq_vars: dict
	
	def __post_init__(self):
		if not isinstance(self.freq_vars, dict):
			raise TypeError('SpectralAxis needs freq_vars to be a dictionary like {freq label 1 in this axis: coeff, ...}')


# simply copying old sketch for now
class SpectralAxisAdvanced:
	"""
	Class to represent an "advanced" spectral axis (involving e.g. variation of experiment parameters
	or possibly other attributes). Not yet implemented.
	"""

	def __init__(self):

		pass

@dataclass
class SpectralGrid:
	"""
	Class to represent a collective set of spectral axes.

	Use example:

	axis1 = ws.main.abstractions.spectralAxis({1: 1})
	axis2 = ws.main.abstractions.spectralAxis({1: 1, 2: -1})
	start = {1: 250, 2: 100}
	end = {1: 3850, 2: 7550}
	spacer = {1: 3.8, 2: 3.8}
	spec_grid = ws.main.abstractions.spectralGrid({1: axis1, 2: axis2}, range_style='uniform',
												  start=start, end=end, spacer=spacer)
	
	----
	axes: Dictionary {axis 1 ID: SpectralAxis instance, axis 2 ID: SpectralAxis instance, ...}: One SpectralAxis
	instance per axis. TODO: Also to support instances being SpectralAxisAdvanced
		this parameter is misleading if :
			e.g., axis1 is with {1: 1} and axis2 is with {1: 1, 2: -1} 
			and `spacer`, `start` and `end` dicts are used as they are now in __post_init__ and `make_mesh_numpy`
			the grid itself right now would not correspond to axis1 is with {1: 1} and axis2 is with {1: 1, 2: -1}

	range_style: String: What sort of range? Intended options at least "uniform" or "custom"
	start: Dictionary {axis 1 ID: starting point (float), ...}: Axis starting points
	end: Dictionary {axis 1 ID: end point (float), ...}: Axis end points
	n_pts: Dictionary {axis 1 ID: number of points (int), ...}: Number of points by axis
	spacer: Dictionary {axis 1 ID: spacer (float), ...}: Spacers by axis
	custom_range: Type not specified: Custom range for each axis. Not yet implemented
	collective_grid: Type not specified (but most likely will be ndarray): (Custom) collective grid for all axes.
	"""
	axes: dict
	range_style: str
	start: dict=None
	end: dict=None
	n_pts: dict=None
	spacer: dict=None
	custom_range: dict=None
	# Optional collective (e.g. adaptive) grid
	# Otherwise intended to default to full granularity grid of individual axes
	collective_grid: Any=None

	def __post_init__(self):
		
		for i in self.axes:
			if not isinstance(self.axes[i], SpectralAxis):
				raise TypeError("Values of axes dict should be SpectralAxis instances")

		if (self.range_style == 'uniform'):

			self.ranges = {}
			n_pts = {}
			spacer = {}

			for i in self.axes:

				if (self.n_pts is None) and (self.spacer is None):
					raise AssertionError('For a uniform setup, either a spacer or a n_pts dictionary must be specified')

				if (self.n_pts is not None) and (self.spacer is not None):
					raise AssertionError('Only one of the arguments n_pts and spacer may be specified')

				if self.n_pts is not None:

					spacer[i] = (self.end[i] - self.start[i])/(self.n_pts[i] + 1)

				elif self.spacer is not None:

					# Underflow possible
					n_pts[i] = int((self.end[i] - self.start[i])/self.spacer[i] + 1)
					if not(self.end[i] == self.start[i] + self.spacer[i]*(n_pts[i] - 1)):
						logger.warning(f'NOTE: Axis defined end {self.end[i]} not precisely at spacer increment of start')

				else:

					raise AssertionError('For uniform grid, must specify either spacer or n_pts')

				# fixme: Other datatype? Should be fine for now
				self.ranges[i] = np.arange(self.start[i], self.end[i], self.spacer[i])
			
			if spacer:
				self.spacer = spacer
			if n_pts:
				self.n_pts = n_pts

		if(self.range_style == 'custom'):
			# rm error to enable skipping `spacer, start, end` - they aren't used meaningfully anyway
			logger.warning('Custom range style is not yet supported')
			pass


	def make_mesh_numpy(self) -> dict:
		"""
		Make a meshgrid using the axes information
		"""

		listofmeshaxes = []
		for ax_label in self.axes:
			if self.spacer is not None:
				wn = np.arange(self.start[ax_label], self.end[ax_label], self.spacer[ax_label])
				listofmeshaxes.append(wn)
			elif self.n_pts is not None:
				wn = np.linspace(self.start[ax_label], self.end[ax_label], self.n_pts[ax_label])
				listofmeshaxes.append(wn)
		meshes = np.meshgrid(*listofmeshaxes)

		mesh_dict = {}
		for i, ax_label in enumerate(self.axes):
			mesh_dict[ax_label] = meshes[i]

		return mesh_dict

	def collGridFromAxes(self):
		"""
		Make collective grid from individual axes linspaces. Not yet implemented
		"""

		pass


@dataclass
class EvaluationVariable:
	"""
	Like SpectralAxis, but a range for an independent variable of the response function (frequency variable)

	range_style: 'uniform' or 'custom'
	"""
	range_style: str
	start: float = None
	end: float = None
	n_pts: int = None
	spacer: float = None
	custom_range: list|np.ndarray = None

	def __post_init__(self):
		"""
		dealing with one range at the time seems to be more clean
		"""
		
		if self.range_style == 'custom':
			raise NotImplementedError('Custom range style is not yet supported')
		
		elif self.range_style == 'uniform':
			
			if (self.n_pts is None) and (self.spacer is None):
				raise AssertionError('For a uniform setup, either a spacer or a n_pts dictionary must be specified')

			if (self.n_pts is not None) and (self.spacer is not None):
				raise AssertionError('Only one of the arguments n_pts and spacer may be specified')
			
			if self.n_pts is not None:
				self.spacer = (self.end - self.start)/(self.n_pts + 1)
				self.range = np.linspace(self.start, self.end, self.n_pts)

			elif self.spacer is not None:

				self.n_pts = int((self.end - self.start)/self.spacer + 1)
				if self.end != self.start + self.spacer*(self.n_pts - 1):
					logger.info(f'NOTE: Axis defined end {self.end} not precisely at spacer increment of start')
				self.range = np.arange(self.start, self.end, self.spacer)


@dataclass
class EvaluationInfo:
	"""
	this feels a bit more "official" than a dict
	and it is warranted because that is a critical info that is needed for the evaluation

	freq_variables - is a dict {variable label: variable data} with a range for each
	fixed_variables - a dict of values for the non-varied fixed variables 
		(e.g., when having a 2D slice of a 3D spectrum at fixed 3rd)
	"""
	freq_variables: dict
	Gamma: float
	Gamma_unit: str
	freq_condition: str = None
	fixed_variables: dict = field(default_factory=lambda: dict())
	# 'diag_margin'- this parameter is specific to the condition ow w2>w1
	spec_result: np.ndarray | dict = None
	margins: dict = None

	@property
	def spec_window_bounds(self):
		"""
		creating `bounds` dict for `check_if_in_window()`
		"""
		bounds = {}
		for key in self.freq_variables:
			bounds[key] = {'left': np.min(self.freq_variables[key]) + self.margins.get(key, 0.), 
						'right': np.max(self.freq_variables[key]) + self.margins.get(key, 0.)}

		return bounds

@dataclass
class RenderingInfo:
	"""
	this feels a bit more "official" than a dict
	and it is warranted because that is a critical info that is needed for the rendiring

	projection: '1d', '2d' or '3d'
	reference_max: normalizing to this reference_max value

	"""
	projection: str = '2d'
	reference_max: float = None
	dynamic_range: float = 100
	num_levels: int = 12
	intensity_normalization_type: NormalizationType = NormalizationType.LOG_SCALE
	title: str = 'plot'
	spec_data_operations: str = 'abs()**2'  # 'abs', 'real', 'imag', 'abs()**2'
	metadata: dict = field(default_factory=lambda: dict())
	to_save: bool = False
	filename: str = 'spectrum.svg'
	backend: str = 'matplotlib'
	# style configurations - currently will work/be used for matplotlib renderer
	style_config: PlotConfig = field(default_factory=lambda: PlotConfig())

# An evaluation setup contains various visualization configuration information
# and information about other relevant evaluation-related choices for a wilsonSimulation instance
#
# Examples of relevant information here:
# Evaluation grid
# System to run simulation on
@dataclass
class SpecEvalSetup:
	"""
	Class for setup information related to spectrum evaluation and rendering
	FIXME: Consider making this into a dataclass
	
	----
	grid: SpectralGrid instance: The grid on which the spectrum is to be evaluated
	ev_info: dict: Setup information which is principally evaluation-related (e.g. dynamic range, relaxation
	parameters etc.)
	rnd_info: dict: Setup information which is principally rendering-related (e.g. number of level ticks, other
	plotting-/visualization-related information)
	FIXME: Consider formalizing which setup attributes may be passed in ev_info and rnd_info -> RenderingInfom and EvaluationInfo
	"""
	grid: SpectralGrid=None
	ev_info: EvaluationInfo = None
	rnd_info: RenderingInfo = None

	def __post_init__(self):
		if self.grid is not None:
			if not isinstance(self.grid, SpectralGrid):
				raise TypeError("Values of axes dict should be SpectralAxis instances")



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

	def addTerms(self, terms: list[VibPerturbedTerm], extend: bool=False):
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
			self.residual_vib_info[k] = data_dict.get(k)


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


	def evaluateAsResponseFunction(self,
								   evaluator: Callable[[
								   MolecularSystem, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup, bool], tuple[np.ndarray, dict]],
								   do_diagn: bool=False):
		"""
		Evaluate the spectrum "as a response function" (i.e. do not use/convolute over
		experiment pulse strength information and without regard to further experiment information except terms)

		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""
		# TODO - checks like in VibAnaSetup.doAnharmonicAnalysis 

		context = dict(system=self.system, derived_terms=self.terms, props=self.props,
				 spec_eval_setup=self.spec_eval_setup, vib_ana_setup=self.vib_ana_setup, 
				 do_diagn=do_diagn)
	
		if do_diagn:
			self.spec, diagn = evaluator(**context)
			self.updDiagnostics(upd_dict=diagn)
			
			if not isinstance(self.diagn, dict):
				raise AssertionError('Diagnostics result must be dictionary')
		else:
			self.spec, _ = evaluator(**context)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')


	def evaluateFull(self, evaluator: Callable[[
								   MolecularSystem, VibExperiment, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup], tuple[np.ndarray, dict]],
								   do_diagn: bool=False):
		"""
		Evaluate the spectrum including experiment context (e.g. convolute over pulse strength)

		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""
		# TODO - checks and context dict like in VibAnaSetup.doAnharmonicAnalysis 

		context = dict(system=self.system, experiment=self.exp, derived_terms=self.terms, props=self.props,
				 spec_eval_setup=self.spec_eval_setup, vib_ana_setup=self.vib_ana_setup, 
				 do_diagn=do_diagn)
		
		self.spec = evaluator(**context)

		if do_diagn:
			self.spec, diagn = evaluator(**context)
			self.updDiagnostics(upd_dict=diagn)
			
			if not isinstance(self.diagn, dict):
				raise AssertionError('Diagnostics result must be dictionary')
		else:
			self.spec, _ = evaluator(**context)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')


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

def find_props(terms: list[VibPerturbedTerm], freqs: str='static') -> list[MolecularProperty]:

	props = []
	
	if terms is None:
		raise AssertionError('There must be terms present to determine needed properties')

	# FIXME: Consider checking if terms are VibPerturbedTerm instances
	for i in terms:

		for a in terms[i]:
			for t in terms[i][a]:
				for j in t.props:

					ops = []

					m = j.dord
					for k in range(m):
						ops.append('g')

					n = len(j.ops)

					for k in range(n):
						ops.append('f')

					if freqs == 'static':
						pdict = {'ops': tuple(ops), 'freq': tuple([0.0 * k for k in range(len(ops))])}

					else:
						raise AssertionError('Managing electronic properties for non-static frequencies not yet implemented')

					new_prop = MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=m, ord_el=n),
													target_basis='nm', target_units='au')

					if new_prop.h(1) not in [k.h(1) for k in props]:
						props.append(copy.deepcopy(new_prop))

	return props

def find_max_state_lvl(terms: list[VibPerturbedTerm]) -> int:

	for i in terms:

		for a in terms[i]:
			for t in terms[i][a]:

				max_state_lvl = 0

				for j in t.freqterms:

					if len(j.sl.q) > max_state_lvl:
						max_state_lvl = len(j.sl.q)
					if len(j.sr.q) > max_state_lvl:
						max_state_lvl = len(j.sr.q)

				for j in t.res:

					if len(j.diff.sl.q) > max_state_lvl:
						max_state_lvl = len(j.diff.sl.q)
					if len(j.diff.sr.q) > max_state_lvl:
						max_state_lvl = len(j.diff.sr.q)

	return max_state_lvl

def find_residual_vib_info(vib_ana: VibAnaSetup) -> tuple[list[MolecularProperty], dict]:

	props = []
	residual_vib_info = {}

	for i in tell_needed_props_for_vib_analysis(vib_ana):
		if isinstance(i, MolecularProperty):
			if i.h(1) not in [k.h(1) for k in props]: # 
				props.append(copy.deepcopy(i))
		else:
			residual_vib_info[i] = None

	return props, residual_vib_info


def find_props_and_max_state_lvl(terms: list[VibPerturbedTerm], 
								 vib_ana: VibAnaSetup, freqs: str='static') -> tuple[list[MolecularProperty], dict, int]:

	props = find_props(terms, freqs)
	props_ext, residual_vib_info = find_residual_vib_info(vib_ana)
	props.extend(props_ext)

	return props, residual_vib_info, find_max_state_lvl(terms)

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
