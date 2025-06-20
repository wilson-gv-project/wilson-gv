
import copy
from wilson_utils.prop_trivname import prop_trivname


# A system is here only the system name, molecular geometry and atoms (masses for isotopes?)
class molecularSystem:

	def __init__(self, name=None, geo=None, geo_extra=None):

		# String
		self.name = name

		if (geo is not None) and (geo_extra is not None):
			raise AssertionError('Ambigious definition: Both default form geometry geo and extended form geometry geo_extra was defined')
		if (geo is None) and (geo_extra is None):
			print('Note: Molecular system was instantiated without geometry information')


		# Default geometry specification
		# atom_name must be element name
		# default isotopes assumed
		# [[atom name, [x, y, z](Ångström)], ...]
		if geo is not None:

			self.geo = geo
			self.geo_extra = None

		# Extended geometry specification
		# atom_name has no implication for element
		# Must specify number of protons and neutrons in nucleus
		# [[atom name, [x, y, z] (Ångström), (# protons, # neutrons)], ...]
		if geo_extra is not None:

			self.geo_extra = geo_extra
			self.geo = None

	# Hash
	def h(self):

		# For now only returning by name

		return hash(self.name)

	# Here: Various routines to give geo, name as e.g. xyz file
	def asXyz(self):

		# For default geo

		# For geo_extra

		pass
		
# Program, level of theory, basis set, other setup info (environment for QM/MM?)
# Does not need to reference an actual setup and can also be used for "get from no specific calculation"
class externalCalcSetup:

	def __init__(self, program=None, lvl_theory=None, basis=None, other_setup=None, other_setup_identifier=None):

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


	# Hash
	def h(self):

		return hash((self.program, self.lvl_theory, self.basis, self.other_setup_id))


# Properties as derivatives
# Can both be used "head only" (only is_derivative, prop_spec, target_basis, target_units) to specify only a property
# and "full" (system, calc_setup) for a particular realization (optional with/without values)
# TODO: Hash a) "head only" information (only hash(prop_spec)),
# TODO  b) hash "head only" information (also tgt basis, tgt units),
# TODO: c) hash "full" (also system, calc_setup),
# TODO: d) hash "full" (also system, calc_setup, in_basis, in_units)
# TODO: Check for adequate property specification and values format when known
# TODO: Consider enforcing specification of units and basis when values are provided
class molecularProperty:

	def __init__(self, prop_spec, trivial_name=None, vals=None, in_basis=None, in_units=None, system=None, calc_setup=None, target_basis=None, target_units=None):

		# Dictionary {'attr name': val, ...}
		# Info like perturbing operators, frequencies etc.
		# All vals must be hashable
		self.prop_spec = prop_spec

		# Trivial name (string): For simplified reference
		self.triv_name = trivial_name

		# Values (several valid forms)
		# Should be arrays for energy derivatives
		# Could be dictionary {state ref: energy} for energy levels
		self.vals = vals

		# Strings
		self.in_basis = in_basis
		self.in_units = in_units

		# molecularSystem instance
		self.system = system

		# externalCalcSetup instance
		self.calc_setup = calc_setup

		# Strings
		self.target_basis = target_basis
		self.target_units = target_units

	# Hash
	def h(self, htype):

		hlist = []

		if htype < 1:

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

		if htype > 4:

			raise AssertionError('Property hash must be requested with type argument (1-4)')
	
		return hash(tuple(hlist))

	# Attach a molecularSystem instance
	def addSystem(self, system):

		self.system = system

	# Attach an externalCalcSetup instance
	def addCalcSetup(self, calc_setup):
	
		self.calc_setup = calc_setup
	
	# Add values (usually scalars or a numPy array)
	def addValues(self, values, in_basis=None, in_units=None):

		self.vals = values

		self.in_basis = in_basis
		self.in_units = in_units
	
	# convertor is a function reference (must take system, basis, units and convertor_info)
	# convertor_info is further information for the convertor
	def convertValues(self, convertor, convertor_info):

		# Call convertor
		# Will be assumed to be able to convert to target basis and units and must fail if unable
		self.vals = convertor(self.system, self.prop_spec, self.vals, self.in_basis, self.target_basis,
								self.in_units, self.target_units, convertor_info)

		# Update basis and units as changed
		if self.target_basis is not None:
			self.in_basis = self.target_basis
			
		if self.target_units is not None:
			self.in_units = self.target_units
		

# Collects necessary calculations with one setup, makes input, collects results
class calculationBatch:

	def __init__(self, system, calc_setup, properties=None):

		self.system = system
		self.calc_setup = calc_setup

		if properties is None:
			self.properties = []

		else:
			self.properties = properties

	def addProperty(self, prop):

		self.properties.append(prop)

	def makeInputs(self):
	
		pass



	def getResults(self, props_to_fill, vib_ana_setup_to_fill=None, source_type=None, source_types=None, source_loc=None):

		# Currently only vault retrieval
		if not source_type == 'vault':
			raise NotImplementedError('Only vault retrieval currently implemented')

		else:
			self.getResultsFromVault(props_to_fill, vib_ana_setup_to_fill, source_loc)

	def getResultsFromOutputs(self):
	
		pass
		
	def getResultsFromVault(self, props_to_fill, vib_ana_setup_to_fill, source_loc):

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

				vib_ana_setup_to_fill.nc_sqrt_eigval = parser_obj.fundamentals_harmonic_int

				if not vib_ana_setup_to_fill.allow_skip_eigvec:
					# FIXME: Find out if these are proper coordinates (and precision) for the intended use (transformation)
					if parser_obj.normal_modes is None:
						raise AssertionError('Normal coordinates (eigenvectors) not found')
					vib_ana_setup_to_fill.nc_eigvec = parser_obj.normal_modes

			# Take states
			if vib_ana_setup_to_fill.vibana_prop_need in ['none']:

				if not(vib_ana_setup_to_fill.vib_regime in ['harmonic']):
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
						processed_states.append(vibState({i: 1.0}, extracted_states[i]))

	def getResultsAsArrayFromFile(self):
	
		pass			


# A collective evaluation setup contains information about how to process several jobs together
#
# Examples of relevant information here:
# Render spectra from each job as a tiled image or as an animation?
# Norm all spectra w.r.t. a collective max?
class collEvalSetup:

	def __init__(self):
		pass

# TODO: Add collective jobs instructions (i.e. instructions for collections of jobs, not for individual jobs)
class wilsonSimulations:

	def __init__(self, jobs=None, coll_instructions=None):

		self.jobs = jobs
		self.coll_instructions = None

	# A job is a wilsonSimulation instance
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

# "Plain" spectral axis for rendering response function freq arg spectra with independent lineshape functions
class spectralAxis:

	def __init__(self, freq_vars):

		# Must be dictionary: {freq label 1 in this axis: coeff, ...}
		self.fv = freq_vars




# TODO: Implement
# For "advanced" axes: Variation of experiment parameters or possibly other attributes
class spectralAxisAdvanced:

	def __init__(self):

		pass

# Spectral collective axes
class spectralGrid:

	def __init__(self, axes, range_style, start=None, end=None, n_pts=None, spacer=None, custom_range=None, collective_grid=None):

		# Axes must be a dictionary {1: spectralAxisRsp/Advanced instance, 2: ...}
		self.axes = axes

		self.start = None
		self.end = None
		self.n_pts = None
		self.ranges = None

		if (range_style == 'uniform'):

			import numpy as np

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

	# Make collective grid from individual axes linspaces
	def collGridFromAxes(self):

		pass


# State, energy, displacement
class vibState:

	def __init__(self, s, e, d=None):

		# s: {(harm. quanta): coeff, (harm. quanta): coeff, ...}
		self.s = s
		# Energy: Units must be ??
		self.e = e
		# Displacements (optional)
		self.d = d



# CONTINUE HERE: Most likely rewrite to vibanaEvalSetup: Have this tell deriv. and rot. props needed (incl. xform matrix?)
# Includes keywords for energy lvl regime
# Tighter definition for property class?
# Also adapt rest of code to this


# Setup for vibrational analysis and storage of the resulting information
# Under which regime to describe the vibrational states
# props is for "derivative-style props"
class vibAnaSetup:

	def __init__(self, vib_regime='harmonic', system=None, vib_regime_subinfo=None, max_state_lvl=None, states=None,
				 nc_sqrt_eigval=None, nc_eigvec=None, allow_skip_eigvec=False, vibana_prop_need='all', external_fill_from=None):

		self.system = system
		self.regime = vib_regime
		self.regime_subinfo = vib_regime_subinfo
		self.max_state_lvl = max_state_lvl
		self.states = states

		# TODO: MODE EXCLUSION, REGISTERING OF FERMI RESONANCES (TO BE PASSED TO EVALUATOR)

		# Dictionary: {nm index: w}
		self.nc_sqrt_eigval = nc_sqrt_eigval
		# Matrix
		self.nc_eigvec = nc_eigvec

		self.allow_skip_eigvec = allow_skip_eigvec

		# 'all': Will need properties for both harmonic and anharmonic analysis
		# 'anharm': Will only need props. for anharmonic analysis (harmonic results will be provided by external program)
		# 'none': All results will be provided by external program
		# FIXME: Maybe do away with 'harm' option, could be implied by other setup choices
		self.vibana_prop_need = vibana_prop_need

		# externalCalcSetup instance
		# NOTE: Refers only to vibrational properties that will be directly filled from analysis and not to
		# properties that will be used in own doAnalysis invocation (they may have their own specification)
		self.external_fill_from = external_fill_from

	# Tell which molecularProperty instances are required for a specific vibrational analysis
	# Allowed to skip eigenvectors (e.g. if all other data already in nm basis)
	def tellNeededProps(self):

		needed_props = []

		if self.vibana_prop_need == 'none':
			return needed_props

		# Check which information is already present
		reg_hess = False

		if self.nc_sqrt_eigval is None:

			if (self.vibana_prop_need == 'all'):

				# FIXME: Not sure about target units
				needed_props.append( molecularProperty(
					{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
					trivial_name=prop_trivname(ord_geo=2),
					target_basis='cart',
					target_units='au')
				)
				reg_hess = True

		if self.nc_eigvec is None and (not(self.allow_skip_eigvec) and not(reg_hess)):

			if (self.vibana_prop_need  == 'all'):

				# FIXME: Not sure about target units
				needed_props.append(molecularProperty(
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
					needed_props.append(molecularProperty(
						{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
						trivial_name=prop_trivname(ord_geo=2),
						target_basis='cart',
						target_units='au')
					)
					reg_hess = True

			# For now, don't use regime subinfo
			if 'PT2' in self.regime:

				if (self.vibana_prop_need == 'anharm') or (self.vibana_prop_need == 'all'):

					needed_props.append(molecularProperty(
						{'ops': tuple(['g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0)},
						trivial_name=prop_trivname(ord_geo=3),
						target_basis='nm',
						target_units='au')
					)

					# FIXME: Consider implementing extra flag for only semidiagonal force constants needed
					needed_props.append(molecularProperty(
						{'ops': tuple(['g', 'g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0, 0.0)},
						trivial_name=prop_trivname(ord_geo=4),
						target_basis='nm',
						target_units='au')
					)

					needed_props.append(molecularProperty(
						{'ops': tuple(['r']), 'freq': (0.0)},
						trivial_name=prop_trivname(ord_rot=1),
						target_basis='nm',
						target_units='au')
					)

					needed_props.append(molecularProperty(
						{'ops': tuple(['g', 'g', 'r']), 'freq': (0.0, 0.0, 0.0)},
						trivial_name=prop_trivname(ord_geo=2, ord_rot=1),
						target_basis='nm',
						target_units='au')
					)

		return needed_props

	def setStates(self, states):

		self.states = states

	def setMaxStateLvl(self, lvl):

		self.max_state_lvl = lvl

	# Use preanalyzer_harmonic only if the (main) analyzer requires the harmonic part to be done first and the
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



# An evaluation setup contains various visualization configuration information
# and information about other relevant evaluation-related choices for a wilsonSimulation instance
#
# Examples of relevant information here:
# Evaluation grid
# System to run simulation on
class specEvalSetup:

	def __init__(self, axes=None, ev_info=None, rnd_info=None):

		# Must be spectralAxes instance
		self.axes = axes

		self.ev_info = ev_info
		self.rnd_info = rnd_info


class wilsonSimulation:

	def __init__(self, exp=None, terms=None, vib_ana_setup=None, spec_eval_setup=None,
				 system=None, eval_uniform=None, eval_by_prop_name=None, props=None, calc_batches=None,
				 spec=None, diagn=None, rendering=None, import_from=None, name=None):

		if import_from is None:

			self.exp = exp
			# Must for now be vibPerturbedTerm instances
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

							new_prop = molecularProperty(pdict, trivial_name=prop_trivname(ord_geo=m, ord_el=n),
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
				calc_batches[ih] = calculationBatch(self.system, i.calc_setup, [copy.deepcopy(i)])

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
			self.spec = evaluator(self.system, self.exp, self.terms, self.props,  self.spec_eval_setup, self.vib_ana_setup)

	# After evaluation, render the spectral data as requested
	def render(self, renderer):

		# Consider extending arguments to give even more info
		self.rendering = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)


