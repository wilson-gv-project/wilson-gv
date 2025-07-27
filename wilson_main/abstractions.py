from dataclasses import dataclass, field, asdict, is_dataclass, InitVar
from typing import Callable, Any
import json
import numpy as np
import copy

from wilson_utils.prop_trivname import prop_trivname
from wilson_derive.abstractions import VibPerturbedTerm
from wilson_experiment.abstractions import VibExperiment
from wilson_utils.abstractions import VibState

import logging
logger = logging.getLogger("wilson."+__name__)

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
	
	def h(self):
		"""
		Hashing function
		"""

		# For now only returning by name
		return hash(self.name)
	
	def asXyz(self):
		"""
		Convert geometry to e.g. xyz file - unwritten but keep shell here for now
		"""

		# For default geo

		# For geo_extra

		pass


# Program, level of theory, basis set, other setup info (environment for QM/MM?)
# Does not need to reference an actual setup and can also be used for "get from no specific calculation"
@dataclass
class ExternalCalcSetup:
	"""
	Class to represent computational setups for properties obtained external to Wilson
	Does not need to pertain to an actual program and could also be used for "get from no specific calculation"/
	"get from file"

	----
	program: String: Program name if relevant (alt. names like 'from_file' are also fine)
	# FIXME: Consider other name "source" instead of "program"
	lvl_theory: String: Level of theory
	basis: String: Basis set

	NOTE: Other setup parameters currently not used/handled
	other_setup: Dictionary {attribute: value, ...}
	other_setup_identifier: Dictionary {attribute: name (not required to be hashable), ...}
	# FIXME: Not sure how I want this to work, return to it if needed
	"""
	# Strings
	program: str = ''
	lvl_theory: str = ''
	basis: str = ''
	
	# Arbitrary data structure (user-managed)
	other_setup: dict = field(default_factory=lambda: dict())
	other_setup_identifier: dict = field(default_factory=lambda: dict())

	def __post_init__(self):
		if self.other_setup is not None:

			if self.other_setup_identifier is None:
				raise AssertionError('Other setup identifier string must accompany other setup for hashing purposes')

		else:

			if self.other_setup_identifier is not None:
				raise AssertionError('No other setup identifier string may be given if no other setup is given')

	def h(self) -> int:
		"""
		Hashing function
		"""

		# FIXME: Only using other setup keys in hash for now, need to complete this for full hash consistency
		return hash((self.program, self.lvl_theory, self.basis, tuple(self.other_setup.keys())))

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
	in_basis: String: In which basis (e.g. "Cartesian" or "normal modes")?
	in_units: String: In which units?
	system: MolecularSystem instance: For which system?
	calc_setup: ExternalCalcSetup instance: For which calculation setup?
	target_basis: String: In which basis should this property be specified (if not matching in_basis, it means that
	it should be transformed)
	target_units: String: In which units should this property be specified (if not matching in_units, it means that
	it should be converted)

	serial_vals: Any = field(init=False) - serializable dict of vals
	see more in test_main_dataclasses.py::test_MolecularProperty
	"""
	prop_spec: dict
	trivial_name: str=None
	vals: InitVar[Any] = field(default=None, repr=False)
	in_basis: str=None
	in_units: str=None
	target_basis: str=None
	target_units:str=None
	serial_vals: Any = field(init=False)

	def __post_init__(self, vals):
		"""
		Turning ndarray to dict when vals were given during init.
		Can be skipped alltogether is vals would be in dict?
		"""
		from wilson_utils.serialization import ndarray_to_dict
		self.serial_vals = ndarray_to_dict(vals, serial=True) if vals is not None else None

	def make_serial_vals(self):
		"""
		An option to make serial vals from self.vals
		see test_main_dataclasses.py::test_MolecularProperty
		"""
		from wilson_utils.serialization import ndarray_to_dict
		self.serial_vals = ndarray_to_dict(self.vals, serial=True) if self.vals is not None else None

	def to_dict(self):
		return {
			"prop_spec": self.prop_spec,
			"trivial_name": self.trivial_name,
			"in_basis": self.in_basis,
			"in_units": self.in_units,
			"target_basis": self.target_basis,
			"target_units": self.target_units,
			"serial_vals": self.serial_vals,
		}

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

		if htype >= 2:

			hlist.append(self.target_basis)
			hlist.append(self.target_units)

		if htype >= 3:

			hlist.append(self.system.h())
			hlist.append(self.calc_setup.h())

		if htype >= 4:

			hlist.append(self.in_basis)
			hlist.append(self.in_units)

		return hash(tuple(hlist))

	def addSystem(self, system: MolecularSystem):
		"""
		Associate a MolecularSystem instance

		system:	MolecularSystem instance: The system to be attached
		"""

		self.system = system

	def addCalcSetup(self, calc_setup):
		"""
		Associate an ExternalCalcSetup instance

		calc_setup: ExternalCalcSetup instance: The setup to be attached
		"""

		self.calc_setup = calc_setup
	
	# Add values (usually scalars or a numPy array)
	def addValues(self, values: Any, in_basis: str=None, in_units: str=None):
		"""
		Associate values to this property

		values: Undetermined form: The values to be added
		in_basis: string: In which basis are these values?
		in_units: string: In which units are these values?
		"""

		self.vals = values

		self.in_basis = in_basis
		self.in_units = in_units
	
	# convertor is a function reference (must take system, basis, units and convertor_info)
	# convertor_info is further information for the convertor
	def convertValues(self, convertor: Callable[[MolecularSystem, dict, Any, str, str, str, str, dict], Any],
					  convertor_info: dict={}):
		"""
		Convert values from the current basis and units to the target basis and units

		convertor: A function reference of the form specified in the declaration: Will be assumed to be able to
		convert to target basis and units and must fail if unable
		convertor_info: dictionary {attribute: value(s)}: Further information for the convertor if needed
		"""

		# Call convertor
		# Will be assumed to be able to convert to target basis and units and must fail if unable
		self.vals = convertor(self.system, self.prop_spec, self.vals, self.in_basis, self.target_basis,
								self.in_units, self.target_units, convertor_info)

		# Update basis and units as changed
		if self.target_basis is not None:
			self.in_basis = self.target_basis
			
		if self.target_units is not None:
			self.in_units = self.target_units

class MolecularPropertyEncoder(json.JSONEncoder):
	"""
	Helper for JSON encoding of MolecularProperty
	"""
	def default(self, o):
		if isinstance(o, MolecularProperty):
			# Convert the MolecularProperty object to a dictionary
			return {
				"prop_spec": o.prop_spec,
				"trivial_name": o.trivial_name,
				"in_basis": o.in_basis,
				"in_units": o.in_units,
				"target_basis": o.target_basis,
				"target_units": o.target_units,
				"serial_vals": o.serial_vals,
			}
		# Let the base class handle other types
		return super().default(o)



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
	allow_skip_eigvec: Boolean: Is it OK to skip the obtainment of normal mode displacements?
	vibana_prop_need: String: Which kinds of properties will I need to actually carry out the vibrational analysis? 
		Choices: 
			"all": I need properties for both harmonic and (if chosen) anharmonic analysis,
			"anharm": I only need properties to carry out an anharmonic analysis [I will or have already gotten the harmonic data], 
			"none": I don't need any properties [I will or have already gotten both harmonic and anharmonic data]
	external_fill_from: ExternalCalcSetup instance: Specifies requested setup (e.g. lvl of theory etc.) for results
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
	allow_skip_eigvec: bool=False

	# 'all': Will need properties for both harmonic and anharmonic analysis
	# 'anharm': Will only need props. for anharmonic analysis (harmonic results will be provided by external program)
	# 'none': All results will be provided by external program
	vibana_prop_need: str='all'

	# externalCalcSetup instance
	# NOTE: Refers only to vibrational properties that will be directly filled from analysis and not to
	# properties that will be used in own doAnalysis invocation (they may have their own specification)
	external_fill_from: ExternalCalcSetup=None

	# TODO: MODE EXCLUSION, REGISTERING OF FERMI RESONANCES (TO BE PASSED TO EVALUATOR)
	exclude_modes: list=None

	def __post_init__(self):
		if self.exclude_modes is None:
			self.exclude_modes = []
		
		import numpy as np
		self.modes_indices = [int(i) for i in np.arange(self.system.Nnmodes) if i not in self.exclude_modes]
	
	@property
	def serial_states(self):
		return [{'s': vibst.serial_s, 
		   'e': vibst.e, 'd': vibst.d} for vibst in getattr(self, 'states')]

	def tellNeededProps(self) -> list[MolecularProperty]:
		"""
		Tell which molecularProperty instances are required for a specific vibrational analysis

		Returns a list of MolecularProperty instances detailing which properties are required
		"""

		needed_props = []

		if self.vibana_prop_need == 'none':
			return needed_props

		# Check which information is already present
		reg_hess = False

		if self.nc_sqrt_eigval is None:

			if (self.vibana_prop_need == 'all'):

				# FIXME: Not sure about target units
				needed_props.append(MolecularProperty(
					{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
					trivial_name=prop_trivname(ord_geo=2),
					target_basis='cart',
					target_units='au')
				)
				reg_hess = True

		if self.nc_eigvec is None and (not(self.allow_skip_eigvec) and not(reg_hess)):

			if (self.vibana_prop_need  == 'all'):

				# FIXME: Not sure about target units
				needed_props.append(MolecularProperty(
					{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
					trivial_name=prop_trivname(ord_geo=2),
					target_basis='cart',
					target_units='au')
				)
				reg_hess = True

		if self.states is None:

			if (self.vibana_prop_need == 'all'):

				if not reg_hess:

					# FIXME: Not sure about target units
					needed_props.append(MolecularProperty(
						{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
						trivial_name=prop_trivname(ord_geo=2),
						target_basis='cart',
						target_units='au')
					)
					reg_hess = True

			# For now, don't use regime subinfo
			if 'PT2' in self.regime:

				if (self.vibana_prop_need == 'anharm') or (self.vibana_prop_need == 'all'):

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

		return needed_props

	def setStates(self, states: list[VibState]):
		"""
		Set vibrational states

		states: List of VibState instances: The states to be set
		"""

		self.states = states

	def doAnalysis(self, props: list[MolecularProperty],
				   analyzer: Callable[[MolecularSystem, list[MolecularProperty], str, str],
				   tuple[list[VibState], dict, dict]]):
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

		if self.regime is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having chosen an analysis regime')

		if self.system is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having set the system attribute')
		self.nc_sqrt_eigval, self.nc_eigvec, self.states = analyzer(self.system, props,
																	self.regime, self.regime_subinfo)


	def doHarmonicAnalysis(self, props: list[MolecularProperty],
						   harmonic_analyzer: Callable[[MolecularSystem, list[MolecularProperty]], tuple[dict, dict]]):
		"""
		Carry out a harmonic vibrational analysis (regardless of chosen regime) and keep only
		the (harmonic) fundamental vibrational energy levels (stored in self.nc_sqrt_eigval) and
		associated eigenvectors (stored in self.nc_eigvec), but not vibrational state VibState instances.

		props: list of MolecularProperty instances: Molecular properties containing those needed in the analysis
		harmonic_analyzer: Callable: A reference to a harmonic analyzer function. See function definition and
		attribute explanation in __init__ for detailed argument specification: Must take as input a system,
		a set of properties, a choice of regime (and subinfo as relevant) and return
		fundamental harmonic energy levels and associated eigenvectors.
		system: MolecularSystem instance: The system for which analysis is sought. May optionally already be stored
		with self as self.system
		"""

		if self.regime is None:
			logger.warning('WARNING: doHarmonicAnalysis was called but no VibAnaSetup regime was specified')

		if self.system is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having set the system attribute')

		self.nc_sqrt_eigval, self.nc_eigvec = harmonic_analyzer(self.system, props)

	def doAnharmonicAnalysis(self, props: list[MolecularProperty], anharmonic_analyzer:
							Callable[[MolecularSystem, list[MolecularProperty], str, str, dict, dict],
							tuple[list[VibState], dict, dict]]):
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

		if self.regime is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having chosen an analysis regime')

		if self.system is None:
			raise AssertionError('Vibrational analysis cannot be carried out without having set the system attribute')

		context = {'system': self.system, 'props': props, 
			 		'regime': self.regime, 'regime_subinfo': self.regime_subinfo, 
					'nc_sqrt_eigval': self.nc_sqrt_eigval, 
					'exclude_modes': self.exclude_modes}
		self.states, self.diagn = anharmonic_analyzer(**context)


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
	# Must be dictionary: {freq label 1 in this axis: coeff, ...}
	freq_vars: dict


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
						logger.warning('NOTE: Axis defined end', self.end[i], 'not precisely at spacer increment of start')

				else:

					raise AssertionError('For uniform grid, must specify either spacer or n_pts')

				# fixme: Other datatype? Should be fine for now
				self.ranges[i] = np.arange(self.start[i], self.end[i], self.spacer[i])
			
			if spacer:
				self.spacer = spacer
			if n_pts:
				self.n_pts = n_pts

		if(self.range_style == 'custom'):
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
	FIXME: Consider formalizing which setup attributes may be passed in ev_info and rnd_info
	"""
	grid: SpectralGrid=None
	ev_info: dict=None
	rnd_info: dict=None


class CalculationBatch:
	"""
	Class to collect (external) calculations with one setup, make input and collect results
	TODO: Add functionality to make input
	TODO: Extend functinality to collect results
	"""

	def __init__(self, system: MolecularSystem, calc_setup: ExternalCalcSetup, properties: list[MolecularProperty]=None):
		"""
		system: MolecularSystem instance: The system for which calclulation is sought/defined
		calc_setup: ExternalCalcSetup: The calculation setup with respect to which calclulation is sought/defined
		properties: List of MolecularProperty instances: The properties for which calculation is sought/defined
		"""

		self.system = system
		self.calc_setup = calc_setup

		if properties is None:
			self.properties = []

		else:
			self.properties = properties

	def addProperty(self, prop):
		"""
		Add a property to the list of requested properties

		prop: MolecularProperty instance: The property to be added
		"""

		self.properties.append(prop)

	def makeInputs(self):
		"""
		Make input(s) for calculation(s)
		TODO: Implement
		"""

		pass

	def getResults(self, props_to_fill: list[MolecularProperty],
				   vib_ana_setup_to_fill: VibAnaSetup=None, source_type: str='',
				   source_types: list[str]=[], source_loc: Any=None):
		"""
		Get results (values for properties) from a specified source or sources and fill them into the
		MolecularProperty instances in props_to_fill for all such instances that match what self can provide. Also
		optionally fill data when applicable into a provided vibrational analysis setup.

		FIXME: This functionality could do with some cleaning up. Maybe it's better to fill the vibrational analysis
		data in a separate method? Also source type/types and source loc are a bit messy: Maybe use just plural and let
		source_loc be a list of lists of strings.

		props_to_fill: list of MolecularProperty instances: Properties for which filling is to be attempted
		vib_ana_setup_to_fill: VibAnaSetup instance: Vibrational analysis setup for which filling is to be
		attempted. Typically involved if the external program can calculate e.g. energy levels directly.

		source_type: string: Type of data source. Could be e.g. "vault" or "file". Intended for use when only one
		kind of location is relevant.

		source_types: list of strings: Types of data source (in decreasing order of priority).

		source_loc: Format not specified: Location(s) of data source(s).
		"""

		# Currently only vault retrieval
		if not source_type == 'vault':
			raise NotImplementedError('Only vault retrieval currently implemented')

		else:
			self.getResultsFromVault(props_to_fill, vib_ana_setup_to_fill, source_loc)

	def getResultsFromOutputs(self):
		"""
		Get results from program output file(s): Not yet implemented.
		"""
		raise NotImplementedError('Results from program output file(s) not yet implemented')

	def getResultsFromVault(self, props_to_fill: list[MolecularProperty], vib_ana_setup_to_fill: VibAnaSetup,
							source_loc: Any):
		"""
		Get results from data vault. See get_results declarations for argument explanations.
		"""

		# FIXME: There might be ways to make this cleaner. Return to this after vault functionality (e.g. trivial names
		# throughout) is settled more fully.

		from wilson.utils import get_package_root
		wilson_root = get_package_root()

		from CQCParse.relay import DataVault
		vault = DataVault(source_loc)

		logger.info(f'system name: {self.system.name}')

		datadict = vault.make_DatainputDict(self.calc_setup.program, (self.system.name, self.calc_setup.lvl_theory, self.calc_setup.basis), wilson_root)

		if self.calc_setup.program == 'gaussian':
			from CQCParse.parsing import GaussianDataParser as progDataParser

		elif self.calc_setup.program == 'cfour':
			from CQCParse.parsing import CFOURdataParser as progDataParser

		parser_obj = progDataParser(datadict)
		parser_obj.getData()

		for i in props_to_fill:
			if i.calc_setup.h() == self.calc_setup.h():
				i.addValues(getattr(parser_obj, i.trivial_name))

		if vib_ana_setup_to_fill is not None:

			# Take harmonic vibrational analysis results
			if vib_ana_setup_to_fill.vibana_prop_need in ['none', 'anharm']:

				vib_ana_setup_to_fill.nc_sqrt_eigval = parser_obj.fundamentals_harmonic_int # todo: tests...

				if not vib_ana_setup_to_fill.allow_skip_eigvec:
					# FIXME: Find out if these are proper coordinates (and precision) for the intended use (transformation)
					if parser_obj.normal_modes is None:
						raise AssertionError('Normal coordinates (eigenvectors) not found')
					vib_ana_setup_to_fill.nc_eigvec = parser_obj.normal_modes

			# Take states
			if vib_ana_setup_to_fill.vibana_prop_need in ['none']:

				if vib_ana_setup_to_fill.regime not in ['harmonic']:
					extracted_states = parser_obj.anharmonic_states

				else:
					extracted_states = parser_obj.harmonic_states

				processed_states = []

				# For now taking only "single harmonic oscillator state" states when getting from output file
				# TODO: Add parsing capability for re-resolved states with possible admixtures
				for i in extracted_states:
					if len(i) <= vib_ana_setup_to_fill.max_state_lvl:

						# TODO: Exclusion based on mode index or freq cutoff
						# FIXME: Change to integer indexing
						processed_states.append(VibState(s={i: 1.0}, e=extracted_states[i]))
				vib_ana_setup_to_fill.states = processed_states

	def to_dict(self):
		"""
		"""
		attributes = ['system', 'calc_setup', 'properties']
		result = {}
		for key in attributes:
			value = getattr(self, key)
			if is_dataclass(value):
				result[key] = asdict(value)
			elif hasattr(value, "to_dict"):  # if a custom object
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

		return result


# simply copying old sketch for now
class CollEvalSetup:
	"""
	Class to hold information about how to process several jobs together. Not yet implemented.

	FIXME: Consider merging with WilsonSimulations

	Examples of relevant information here:
	Render spectra from each job as a tiled image or as an animation?
	Norm all spectra w.r.t. a collective max?
	"""

	def __init__(self):
		pass


class WilsonSimulation:
	"""
	Class to hold up to a full set of information for a Wilson run and carry out operations related to the run
	workflow.
	"""

	def __init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
				 eval_uniform: ExternalCalcSetup=None, eval_by_prop_name: dict[str: ExternalCalcSetup]=None,
				 props: list[MolecularProperty]=None, calc_batches: dict[int: CalculationBatch]=None,
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
		eval_uniform: ExternalCalcSetup instance: If all properties are (to be) evaluated under the same external
		setup, then providing an ExternalCalcSetup as this argument signifies that
		eval_by_prop_name: dictionary {trivial name: ExternalCalcSetup}: If specific properties are (to be) evaluated
		under specific setups, then signify that with this argument. All properties needed but not referred to in this
		way will instead be assumed to be requested under the setup specified in eval_uniform. If no properties
		are missing specification in eval_by_prop_name, then eval_uniform can be excluded.
		props: List of MolecularProperty instances: Properties needed for evaluation
		calc_batches: Dictionary {hash: CalculationBatch}: Batches of external calculation information
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

			self.eval_uniform = eval_uniform
			self.eval_by_prop_name = eval_by_prop_name

			self.props = props
			self.calc_batches = calc_batches
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



	def getTerms(self, deriver: Callable[[VibExperiment], list[VibPerturbedTerm]]):
		"""
		Get terms based on experiment

		deriver: A function taking a VibExperiment instance and returning a list of VibPerturbedTerm instances:
		The function that will carry out the derivation of terms
		"""

		self.terms = deriver(self.exp)

	def addVibAnaSetup(self, vib_ana_setup: VibAnaSetup):
		"""
		Add a vibrational analysis setup

		vib_ana_setup: VibAnaSetup instance: The vibrational analysis to be added
		"""
		self.vib_ana_setup = vib_ana_setup

	def addPropEvalSetup(self, eval_uniform: ExternalCalcSetup=None,
						 eval_by_prop_name: dict[str: ExternalCalcSetup]=None,):
		"""
		Add a property evaluation setup

		See argument explanation of __init__ method of this class for explanation of these arguments
		"""

		self.eval_uniform = eval_uniform
		self.eval_by_prop_name = eval_by_prop_name

	def addSpecEvalSetup(self, spec_eval_setup: SpecEvalSetup):
		"""
		Add a spectral evaluation/rendering setup

		spec_eval_setup: SpecEvalSetup instance: The setup to be added
		"""

		self.spec_eval_setup = spec_eval_setup

	def findPropsAndMaxStateLvl(self, freqs: str='static'):
		"""
		Make property instances needed to fulfill tasks and set maximum state level in vibrational analysis

		freqs: String: For terms involving properties that may be frequency dependent, use
		experiment information ('exp') or use the static ('static') properties?
		"""

		self.props = []

		if self.terms is None:
			raise AssertionError('There must be terms present to determine needed properties')
		if self.vib_ana_setup is None:
			raise AssertionError('There must be a vibrational analysis setup present to')

		# FIXME: Consider checking if terms are VibPerturbedTerm instances
		for i in self.terms:

			for a in self.terms[i]:
				for t in self.terms[i][a]:
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

						if not new_prop.h(1) in [k.h(1) for k in self.props]:
							self.props.append(copy.deepcopy(new_prop))

					# Currently registering these states without regard to whether harmonic or other regime
					# TODO: Find out if this should be changed

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

					self.vib_ana_setup.max_state_lvl = max_state_lvl

		for i in self.vib_ana_setup.tellNeededProps():
			if not i.h(1) in [k.h(1) for k in self.props]:
				self.props.append(copy.deepcopy(i))

	def dressPropsWithSetup(self):
		"""
		Dress my self.properties with computational setups according to how they are specified in
		self.eval_uniform or self.eval_by_prop_name
		"""

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
			if self.eval_uniform is not None:

				i.addCalcSetup(self.eval_uniform)
				dressed = True

			if not dressed:
				raise AssertionError('Unable to determine calculation setup for property')

	def makeCalculationBatches(self):
		"""
		Make calculation batches needed to fulfill tasks, grouping together properties to be obtained under
		common setups
		"""

		calc_batches = {}

		for i in self.props:

			ih = i.calc_setup.h()

			# TODO: Consider adding calc setup data strip method to molecularProperty to avoid calc setup info duplication here

			if ih in calc_batches:
				calc_batches[ih].addProperty(copy.deepcopy(i))

			else:
				calc_batches[ih] = CalculationBatch(self.system, i.calc_setup, [copy.deepcopy(i)])

		self.calc_batches = calc_batches

	def getResultsFromCalculationBatches(self, source_type: str='', source_types: list[str]=[], source_loc: Any=None):
		"""
		Get results from calculation batches and register in self.properties
		FIXME: See comments in CalculationBatch.getResults

		source_type: string: Type of data source. Could be e.g. "vault" or "file". Intended for use when only one
		kind of location is relevant.

		source_types: list of strings: Types of data source (in decreasing order of priority).

		source_loc: Format not specified: Location(s) of data source(s).
		"""

		for i in self.calc_batches:
			if self.vib_ana_setup.external_fill_from is not None:
				if self.calc_batches[i].calc_setup.h() == self.vib_ana_setup.external_fill_from.h():
					self.calc_batches[i].getResults(self.props, vib_ana_setup_to_fill=self.vib_ana_setup,
													source_type=source_type, source_loc=source_loc)

			else:
				self.calc_batches[i].getResults(self.props, source_type=source_type, source_loc=source_loc)

	def evaluateAsResponseFunction(self,
								   evaluator: Callable[[
								   MolecularSystem, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup], np.ndarray]):
		"""
		Evaluate the spectrum "as a response function" (i.e. do not use/convolute over
		experiment pulse strength information and without regard to further experiment information except terms)

		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""

		self.spec = evaluator(self.system, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')

	def evaluateAsResponseFunctionWithDiagnostics(self, evaluator: Callable[[
								   MolecularSystem, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup], tuple[np.ndarray, dict]]):
		"""
		Evaluate the spectrum "as a response function" (i.e. do not use/convolute over
		experiment pulse strength information and without regard to further experiment information except terms)

		evaluator: As in evaluateAsResponseFunction but must additionally return a dictionary of diagnostics information.
		"""

		self.spec, self.diagn = evaluator(self.system, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')

		if not isinstance(self.diagn, dict):
			raise AssertionError('Diagnostics result must be dictionary')

	def evaluateFull(self, evaluator: Callable[[
								   MolecularSystem, VibExperiment, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup], np.ndarray]):
		"""
		Evaluate the spectrum including experiment context (e.g. convolute over pulse strength)

		evaluator: Callable: A function to carry out the evaluation. Uses attributes described in __init__ of this
		class: Must take a system, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return the spectral data as a numpy ndarray
		"""

		self.spec = evaluator(self.system, self.exp, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')


	def evaluateFullWithDiagnostics(self, evaluator: Callable[[
								   MolecularSystem, VibExperiment, list[VibPerturbedTerm], list[MolecularProperty],
								   SpecEvalSetup, VibAnaSetup], tuple[np.ndarray, dict]]):
		"""
		Evaluate the spectrum including experiment context (e.g. convolute over pulse strength)

		evaluator: Callable: As in evaluateFull but must additionally return a dictionary of diagnostics information.
		"""

		self.spec, self.diagn = evaluator(self.system, self.exp, self.terms, self.props,
										  self.spec_eval_setup, self.vib_ana_setup)

		if not isinstance(self.spec, np.ndarray):
			raise AssertionError('Spectroscopic evaluator result must be numpy.ndarray')

		if not isinstance(self.diagn, dict):
			raise AssertionError('Diagnostics result must be dictionary')

	def render(self, renderer: Callable[[np.ndarray, MolecularSystem, VibExperiment, dict, str, SpecEvalSetup], Any]):
		"""
		Render the spectral data.

		renderer: Callable: A function to carry out the rendering. Uses attributes described in __init__ of this
		class: Must take a system, a list of terms, a collection of properties, an evaluation setup and a
		vibrational analysis setup and return
		"""

		# Consider extending arguments to provide even more info to renderer
		self.rendering = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)

	def renderWithDiagnostics(self, renderer: Callable[[np.ndarray, MolecularSystem, VibExperiment,
														dict, str, SpecEvalSetup], tuple[Any, dict]]):
		"""
		Render the spectral data

		renderer: Callable: A function to carry out the rendering. As in render but must additionally return a
		dictionary of diagnostics information.
		"""

		# Consider extending arguments to provide even more info to renderer
		self.rendering, self.diagn = renderer(self.spec, self.system, self.exp, self.diagn,
											  self.name, self.spec_eval_setup)

		if not isinstance(self.diagn, dict):
			raise AssertionError('Diagnostics result must be dictionary')

	def to_dict(self):
		"""
		__init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
				 eval_uniform: ExternalCalcSetup=None, eval_by_prop_name: dict[str: ExternalCalcSetup]=None,
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
		from wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
		result['terms'] = derived_terms_dict_to_dicts(getattr(self, 'terms'))
		return result

	
	@classmethod
	def from_dict(cls, data):
		"""
		__init__(self, exp: VibExperiment=None, terms: list[VibPerturbedTerm]=[], vib_ana_setup: VibAnaSetup=None,
				 spec_eval_setup: SpecEvalSetup=None, system: MolecularSystem=None,
				 eval_uniform: ExternalCalcSetup=None, eval_by_prop_name: dict[str: ExternalCalcSetup]=None,
				 props: list[MolecularProperty]=None, calc_batches: dict[int: CalculationBatch]=None,
				 spec: Any=None, diagn: dict=None, rendering: Any=None, import_from: str=None, name: str=None):
		
		VibExperiment - dataclass
		terms - list[VibPerturbedTerm] - handle this or maybe no need?
		VibAnaSetup - not a dataclass

		SpecEvalSetup - dataclass
		MolecularSystem - dataclass
		ExternalCalcSetup - dataclass

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
