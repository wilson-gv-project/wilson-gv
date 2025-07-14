
import copy
from wilson_utils.prop_trivname import prop_trivname
from typing import Callable, Any
import numpy as np

# A system is here only the system name, molecular geometry and atoms (masses for isotopes?)
class MolecularSystem:
	"""
	Class to represent a molecular system
	"""

	def __init__(self, name: str, natoms: int=None, geo=None, geo_extra=None):
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


		self.name = name
		self.natoms = natoms
		self.Nnmodes = 3*natoms-6 # fixme

		if (geo is not None) and (geo_extra is not None):
			raise AssertionError('Ambigious definition: Both default form geometry geo and extended form geometry geo_extra was defined')
		if (geo is None) and (geo_extra is None):
			print('Note: Molecular system was instantiated without geometry information')



		if geo is not None:

			self.geo = geo
			self.geo_extra = None

		if geo_extra is not None:

			self.geo_extra = geo_extra
			self.geo = None

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

	def __repr__(self):
		return f'THIS IS MolecularSystem {self.name}\n'

# Program, level of theory, basis set, other setup info (environment for QM/MM?)
# Does not need to reference an actual setup and can also be used for "get from no specific calculation"
class ExternalCalcSetup:
	"""
	Class to represent computational setups for properties obtained external to Wilson
	Does not need to pertain to an actual program and could also be used for "get from no specific calculation"/
	"get from file"
	"""

	def __init__(self, program: str='', lvl_theory: str='', basis: str='', other_setup: dict={}, other_setup_identifier: dict={}):
		"""
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
		self.program = program
		self.lvl_theory = lvl_theory
		self.basis = basis

		# Arbitrary data structure (user-managed)
		self.other_setup = other_setup

		if other_setup is not None:

			if other_setup_identifier is None:
				raise AssertionError('Other setup identifier string must accompany other setup for hashing purposes')

		else:

			if other_setup_identifier is not None:
				raise AssertionError('No other setup identifier string may be given if no other setup is given')

		self.other_setup_id = other_setup_identifier

	def h(self):
		"""
		Hashing function
		"""

		return hash((self.program, self.lvl_theory, self.basis, self.other_setup_id))


class MolecularProperty:
	"""
	Class to represent a molecular (energy derivative or similar) property
	Can both be used "head only" (only prop_spec, target_basis, target_units) to specify only the concept of a property
    and "full" (system, calc_setup) for a particular realization (optional with/without values)
	"""

	def __init__(self, prop_spec, trivial_name: str=None, vals: Any=None, in_basis: str=None, in_units: str=None,
				 system: MolecularSystem=None, calc_setup: ExternalCalcSetup=None,
				 target_basis: str=None, target_units:str=None):
		"""
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
		"""

		self.prop_spec = prop_spec
		self.triv_name = trivial_name
		self.vals = vals

		self.in_basis = in_basis
		self.in_units = in_units

		self.system = system
		self.calc_setup = calc_setup

		self.target_basis = target_basis
		self.target_units = target_units

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
		
	def __repr__(self):
		if self.vals is not None:
			s = ' not'
		else:
			s = ''
		return f'MolecularProperty {self.triv_name}: values are{s} None'

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

		print('system name', self.system.name)

		datadict = vault.make_DatainputDict(self.calc_setup.program, (self.system.name, self.calc_setup.lvl_theory, self.calc_setup.basis), wilson_root)

		if self.calc_setup.program == 'gaussian':
			from CQCParse.parsing import GaussianDataParser as progDataParser

		elif self.calc_setup.program == 'cfour':
			from CQCParse.parsing import CFOURdataParser as progDataParser

		parser_obj = progDataParser(datadict)
		parser_obj.getData()

		trivname_translation = {'dipgrad' : 'dipole_first_derivatives',
								'diphess' : 'dipole_second_derivatives',
								'polgrad' : 'polarizability_first_derivatives',
								'polhess' : 'polarizability_second_derivatives',
								'cff' : 'cubic_force_constants',
								'qff' : 'quartic_force_constants',
								'B' : 'rotational_constant',
								'coriolis' : 'coriolis_constant'
								}

		print([i.triv_name for i in props_to_fill])
		print(dir(parser_obj))


		for i in props_to_fill:
			if i.calc_setup.h() == self.calc_setup.h():
				i.addValues(getattr(parser_obj, trivname_translation[i.triv_name]))

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

				if not(vib_ana_setup_to_fill.regime in ['harmonic']):
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
						processed_states.append(VibState({i: 1.0}, extracted_states[i]))
				vib_ana_setup_to_fill.states = processed_states


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
	def __init__(self, freq_vars):

		# Must be dictionary: {freq label 1 in this axis: coeff, ...}
		self.freq_vars = freq_vars


class SpectralAxisAdvanced:
	"""
	Class to represent an "advanced" spectral axis (involving e.g. variation of experiment parameters
	or possibly other attributes). Not yet implemented.
	"""

	def __init__(self):

		pass

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

	"""

	def __init__(self, axes: dict, range_style: str, start: dict=None, end: dict=None,
				 n_pts: dict=None, spacer: dict=None,
				 custom_range: dict=None, collective_grid: Any=None):
		"""
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

		self.axes = axes

		self.start = None
		self.end = None
		self.n_pts = None
		self.ranges = None

		if (range_style == 'uniform'):

			self.ranges = {}

			self.start = start
			self.end = end

			self.n_pts = {}
			self.spacer = {}

			for i in self.axes:

				if (n_pts is None) and (spacer is None):
					raise AssertionError('For a uniform setup, either a spacer or a n_pts dictionary must be specified')

				if (n_pts is not None) and (spacer is not None):
					raise AssertionError('Only one of the arguments n_pts and spacer may be specified')

				if n_pts is not None:

					self.n_pts[i] = n_pts[i]

					self.spacer[i] = (self.end[i] - self.start[i])/(self.n_pts[i] + 1)

				elif spacer is not None:

					self.spacer[i] = spacer[i]

					# Underflow possible
					self.n_pts[i] = int((self.end[i] - self.start[i])/self.spacer[i] + 1)
					if not(self.end[i] == self.start[i] + self.spacer[i]*(self.n_pts[i] - 1)):
						print('NOTE: Axis defined end', self.end[i], 'not precisely at spacer increment of start')

				else:

					raise AssertionError('For uniform grid, must specify either spacer or n_pts')

				# fixme: Other datatype? Should be fine for now
				self.ranges[i] = np.arange(self.start[i], self.end[i], self.spacer[i])

		if(range_style == 'custom'):

			pass

		# Optional collective (e.g. adaptive) grid
		# Otherwise intended to default to full granularity grid of individual axes
		self.coll_grid = collective_grid

	def make_mesh_numpy(self) -> tuple[np.ndarray]:
		"""
		Make a numpy meshgrid using the axes information

		Returns: A numpy meshgrid
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

	def __repr__(self):
		return "THIS IS SpectralGrid with self.axes,self.n_pts"

# State, energy, displacement
class VibState:
	"""
	Class to represent a vibrational state.
	This is for a "concrete" vibrational state and not the same as its symbolic namesake in wilson-derive.
	TODO: Consider moving this class to wilson-utils.
	"""

	def __init__(self, s: dict, e: float, d: Any=None):
		"""
		s: dictionary {(harm. quanta): coeff, (harm. quanta): coeff, ...}: Specify the state in terms of harm. osc. WFs
		e: float: State energy level
		d: type not specified: Should be some form of vector to represent displacement in terms of atomic coordinates.
		"""

		self.s = s
		self.e = e
		self.d = d

	def __repr__(self):
		return f"vibState {self.s}, energy is {self.e} cm-1"


# CONTINUE HERE: Most likely rewrite to vibanaEvalSetup: Have this tell deriv. and rot. props needed (incl. xform matrix?)
# Includes keywords for energy lvl regime

#
# Under which regime to describe the vibrational states
class VibAnaSetup:
	"""
    Class for setup for vibrational analysis and storage of the resulting information
	"""

	def __init__(self, vib_regime: str='harmonic', system: MolecularSystem=None, vib_regime_subinfo: dict=None,
				 max_state_lvl: int=None, states: list[VibState]=None,
				 nc_sqrt_eigval: dict=None, nc_eigvec: dict=None, allow_skip_eigvec: bool=False,
				 vibana_prop_need: str='all', external_fill_from: ExternalCalcSetup=None,
				 exclude_modes: list=None):
		"""
		vib_regime: string: Vibrational analysis regime (e.g. "harmonic", "GVPT2", "VPT2")
		system: MolecularSystem instance: System to which this instance pertains
		vib_regime_subinfo: dictionary: Extra configuration info for vibrational regime (e.g. skip rotational effects)
		max_state_lvl: integer: Maximum number of vibrational quanta per harmonic state involved in states
		states: List of VibState instances: Specification of each vibrational state in scope
		nc_sqrt_eigval: dictionary {mode index: value}: Harmonic vibrational energy levels
		nc_eigvec: dictionary {mode index: [values]}: Normal mode displacements (canonically in Cartesian basis)
		allow_skip_eigvec: Boolean: Is it OK to skip the obtainment of normal mode displacements?
		vibana_prop_need: String: Which kinds of properties will I need to actually carry out the
		vibrational analysis? Choices: "all": I need properties for both harmonic and (if chosen) anharmonic analysis,
		"anharm": I only need properties to carry out an anharmonic analysis [I will or have already gotten the harmonic
		data], "none": I don't need any properties [I will or have already gotten both harmonic and anharmonic data]
		external_fill_from: ExternalCalcSetup instance: Specifies requested setup (e.g. lvl of theory etc.) for results
		exclude_modes: list: Tells which modes (if any) to exclude in this vibrational analysis
		"""

		self.system = system
		self.regime = vib_regime
		self.regime_subinfo = vib_regime_subinfo
		self.max_state_lvl = max_state_lvl
		self.states = states

		# TODO: MODE EXCLUSION, REGISTERING OF FERMI RESONANCES (TO BE PASSED TO EVALUATOR)
		self.exclude_modes = exclude_modes
		if exclude_modes is None:
			self.exclude_modes = []
		import numpy as np
		self.modes_indices = [i for i in np.arange(system.Nnmodes) if i not in self.exclude_modes]

		# Dictionary: {nm index: w}
		self.nc_sqrt_eigval = nc_sqrt_eigval
		# Matrix
		self.nc_eigvec = nc_eigvec

		self.allow_skip_eigvec = allow_skip_eigvec

		# 'all': Will need properties for both harmonic and anharmonic analysis
		# 'anharm': Will only need props. for anharmonic analysis (harmonic results will be provided by external program)
		# 'none': All results will be provided by external program
		self.vibana_prop_need = vibana_prop_need

		# externalCalcSetup instance
		# NOTE: Refers only to vibrational properties that will be directly filled from analysis and not to
		# properties that will be used in own doAnalysis invocation (they may have their own specification)
		self.external_fill_from = external_fill_from


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

	# Use preanalyzer_harmonic only if the (main) analyzer requires the harmonic part to be done first
	# Use the with_conversion flag if the analyzer (alt. will also be requested to carry out basis conversion
	def doAnalysis(self, props, analyzer, system=None, preanalyzer_harmonic=None, with_conversion=False, convert_in_preanalyzer=False):


		if system is None:
			if self.system is None:
				raise AssertionError('No system specified as argument or stored for vibrational analysis')
			else:
				sys_va = self.system
		else:
			if self.system is not None:
				raise AssertionError('Definition conflict: System specified both as argument and stored in vibAnaSetup class instance')
			else:
				sys_va = system

		if preanalyzer_harmonic is not None:

			if with_conversion:

				if convert_in_preanalyzer:

					self.nc_sqrt_eigval, self.nc_eigvec, props = preanalyzer_harmonic(sys_va, props)
					self.states = analyzer(system, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval,
										   self.nc_eigvec)

				else:

					self.nc_sqrt_eigval, self.nc_eigvec = preanalyzer_harmonic(sys_va, props)
					self.states, props = analyzer(system, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval,
										   self.nc_eigvec)

			else:

				print('CAUTION: No property basis conversion requested but harmonic preanalysis requested')
				self.nc_sqrt_eigval, self.nc_eigvec = preanalyzer_harmonic(sys_va, props)
				self.states = analyzer(system, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval, self.nc_eigvec)

		else:

			# One-stop shop
			if with_conversion:
				self.states, self.nc_sqrt_eigval, self.nc_eigvec, props = analyzer(sys_va, props, self.regime, self.regime_subinfo)

			# Otherwise, will assume that harmonic analysis was already done and properties already in correct basis
			else:
				self.states = analyzer(sys_va, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval, self.nc_eigvec)

	def __repr__(self):
		return 'THIS IS vibAnaSetup with self.regime,self.states'

# An evaluation setup contains various visualization configuration information
# and information about other relevant evaluation-related choices for a wilsonSimulation instance
#
# Examples of relevant information here:
# Evaluation grid
# System to run simulation on
class SpecEvalSetup:

	def __init__(self, grid=None, ev_info=None, rnd_info=None):

		# Must be spectralGrid instance
		self.grid = grid

		self.ev_info = ev_info
		self.rnd_info = rnd_info

	def __repr__(self):
		return 'THIS IS specEvalSetup with self.grid,self.ev_info,self.rnd_info'

class WilsonSimulation:

	def __init__(self, exp=None, terms=None, vib_ana_setup=None, spec_eval_setup=None,
				 system=None, eval_uniform=None, eval_by_prop_name=None, props=None, calc_batches=None,
				 spec=None, diagn=None, rendering=None, import_from=None, name=None):

		if import_from is None:

			self.exp = exp
			# Must for now be VibPerturbedTerm instances
			self.terms = terms
			self.vib_ana_setup = vib_ana_setup
			self.spec_eval_setup = spec_eval_setup
			# Must be molecularSystem instance
			self.system = system

			# Must specify one or both of of eval_uniform or eval_by_prop
			# eval_uniform: If not None then must be externalCalcSetup
			# eval_by_prop_name: Dictionary {trivname: externalCalcSetup}
			# Implied: All properties to be evaluated by eval_uniform setup except eval_by_prop
			# If no eval_uniform then must test if all properties specified in eval_by_prop
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


	def addExperiment(self, experiment):

		self.exp = experiment

	def addSystem(self, system):

		self.system = system

	def addTerms(self, terms):

		self.terms = terms

	# Get terms based on experiment
	def getTerms(self, deriver):

		self.terms = deriver(self.exp)

	def addVibAnaSetup(self, vib_ana_setup):

		self.vib_ana_setup = vib_ana_setup

	def addPropEvalSetup(self, eval_uniform=None, eval_by_prop_name=None):

		self.eval_uniform = eval_uniform
		self.eval_by_prop_name = eval_by_prop_name

	def addSpecEvalSetup(self, spec_eval_setup):

		self.spec_eval_setup = spec_eval_setup

	# Make property instances needed to fulfill tasks and set maximum state level in vibrational analysis
	# Uses both terms and evalSetup config (for VPT2 etc.)
	# freqs can also be 'experiment': Get from UV/VIS part of pulses
	# vibana_need should be 'none', 'harm', 'anharm', 'all' depending on a priori knowledge of which vibrational analysis will
	# be performed using data registered here and which is known to be provided directly from other sources
	def findPropsAndMaxStateLvl(self, term_type='default', freqs='static', vibana_need='anharm'):

		self.props = []

		if term_type == 'default':

			if self.terms is None:
				raise AssertionError('There must be terms present to determine needed properties')
			if self.vib_ana_setup is None:
				raise AssertionError('There must be a vibrational analysis setup present to')

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

		else:

			raise NotImplementedError('Non-default term types not implemented')

	def dressPropsWithSetup(self):

		for i in self.props:

			dressed = False

			# See if property specifically mentioned and if so use that
			if self.eval_by_prop_name is not None:

				if i.triv_name is not None:
					if i.triv_name in self.eval_by_prop_name:
						i.addCalcSetup(self.eval_by_prop_name[i.triv_name])
						dressed=True

				print('WARNING: Property without trivial name encountered but eval_by_prop_name was specified')

			# Otherwise, use uniform eval argument
			if self.eval_uniform is not None:

				i.addCalcSetup(self.eval_uniform)
				dressed = True

			if not dressed:
				raise AssertionError('Unable to determine calculation setup for property')

	# Make calculation batches needed to fulfill tasks
	def makeCalculationBatches(self):

		calc_batches = {}

		for i in self.props:

			ih = i.calc_setup.h()

			# TODO: Consider adding calc setup data strip method to molecularProperty to avoid calc setup info duplication here

			if ih in calc_batches:
				calc_batches[ih].addProperty(copy.deepcopy(i))

			else:
				calc_batches[ih] = CalculationBatch(self.system, i.calc_setup, [copy.deepcopy(i)])

		self.calc_batches = calc_batches

	# Get results from calculation batches and register in self.properties
	# I let (at least for now) source location type and composition vary in form
	def getResultsFromCalculationBatches(self, source_type=None, source_types=None, source_loc=None):

		for i in self.calc_batches:
			if self.vib_ana_setup.external_fill_from is not None:
				if self.calc_batches[i].calc_setup.h() == self.vib_ana_setup.external_fill_from.h():
					self.calc_batches[i].getResults(self.props, vib_ana_setup_to_fill=self.vib_ana_setup,
													source_type=source_type, source_loc=source_loc)

			else:
				self.calc_batches[i].getResults(self.props, source_type=source_type, source_loc=source_loc)

	# Have this carry out "only" numerical evaluation and return as ordered structure of spectral info, analytics data etc.
	# Other later functionality can interact with this and generate reports, images or other requested data
	def evaluate(self, evaluator, include_diagnostics=False):

		if include_diagnostics:
			self.spec, self.diagn = evaluator(self.system, self.exp, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

		else:
			self.spec = evaluator(self.system, self.exp, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

	# After evaluation, render the spectral data as requested
	def render(self, renderer):

		# Consider extending arguments to give even more info
		self.rendering = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)


