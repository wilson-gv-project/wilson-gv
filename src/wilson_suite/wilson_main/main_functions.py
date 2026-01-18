from .abstractions import VibState
from .abstractions import (VibAnaSetup, MolecularProperty,
						   MolecularSystem)

from ..wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_utils.termdict_from_symb_term import prop_trivname
from typing import Callable
import copy

import logging
logger = logging.getLogger("wilson")

# VibAnaSetup related functions

def tell_needed_props_for_vib_analysis(vib_ana: VibAnaSetup):
	
	"""
	Tell which MolecularProperty instances are required for a specific vibrational analysis

	Returns a list of MolecularProperty instances detailing which properties are required for curent state of instance
	"""

	needed_props = []

	if vib_ana.vibana_own_analysis == 'none':
		if vib_ana.isAllSet:
			return needed_props
		else:
			needed_props.append({'nc_sqrt_eigval': None})
			if vib_ana.regime == 'harmonic':
				needed_props.append({'harmonic_states': None})
			elif 'PT2' in vib_ana.regime:
				needed_props.append({'anharmonic_states': None})
	
	if (vib_ana.vibana_own_analysis == 'full'):
		# should have share the same setting of origin as nc_sqrt_eigval and nc_eigvec
		# in harmonic analysis procedure it will go from "hess" Property to nc_sqrt_eigval and nc_eigvec

		# FIXME: Not sure about target units
		needed_props.append(MolecularProperty(
			{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
			trivial_name=prop_trivname(ord_geo=2))
		)

	# For now, don't use regime subinfo
	if 'PT2' in vib_ana.regime:

		if (vib_ana.vibana_own_analysis == 'anharm') or (vib_ana.vibana_own_analysis == 'full'):

			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=3))
			)

			# FIXME: Consider implementing extra flag for only semidiagonal force constants needed
			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'g', 'g']), 'freq': (0.0, 0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=4))
			)

			needed_props.append(MolecularProperty(
				{'ops': tuple(['r']), 'freq': (0.0)},
				trivial_name=prop_trivname(ord_rot=1))
			)

			needed_props.append(MolecularProperty(
				{'ops': tuple(['g', 'g', 'r']), 'freq': (0.0, 0.0, 0.0)},
				trivial_name=prop_trivname(ord_geo=2, ord_rot=1))
			)
		
		if vib_ana.vibana_own_analysis == 'anharm':
			needed_props.append({'nc_sqrt_eigval': None})
		
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


# WilsonSimulation related functions

def find_props(terms, freqs: str='static') -> list[MolecularProperty]:

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

					new_prop = MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=m, ord_el=n))

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
			residual_vib_info.update(i)

	return props, residual_vib_info


def find_props_and_max_state_lvl(terms: list[VibPerturbedTerm], 
								 vib_ana: VibAnaSetup, freqs: str='static') -> tuple[list[MolecularProperty], dict, int]:

	props = find_props(terms, freqs)
	props_ext, residual_vib_info = find_residual_vib_info(vib_ana)
	
	existing_hashes = {prop.h(1) for prop in props}
	props.extend(prop for prop in props_ext if prop.h(1) not in existing_hashes)

	return props, residual_vib_info, find_max_state_lvl(terms)

def fill_props_results(props, data_dict: dict):
	"""
	loading data into self.props (and optionally to self.vib_ana_setup)
	
	data_dict: dict - {data_name: values}

	"""

	for p in props:
		p.addValues(data_dict.get(p.trivial_name))

def fill_residual_vib_info_results(vib_ana_setup, residual_vib_info, data_dict: dict):
	"""
	loading data into self.props (and optionally to self.vib_ana_setup)
	
	data_dict: dict - {data_name: values}

	"""
	for k in residual_vib_info:
		if k in ['anharmonic_states', 'harmonic_states']:
			states_list = []
			states_dict: dict = data_dict.get(k)

			for state, energy in states_dict.items():
				states_list.append(VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=','.join(state)))

			vib_ana_setup.setStates(states=states_list)
			residual_vib_info[k] = data_dict.get(k)

		else:
			residual_vib_info[k] = data_dict.get(k)
			setattr(vib_ana_setup, k, data_dict.get(k))

def request_props(props: list[MolecularProperty], data_dict: dict) -> dict:
	"""
	data_dict: dict - {data_name: DataOriginInfo}
	"""
	for p in props:
		data_dict[p.trivial_name] = p.calc_setup
	return data_dict

def request_residual_vib_info(residual_vib_info: dict, data_dict: dict) -> dict:
	"""
	data_dict: dict - {data_name: DataOriginInfo}
	"""
	for k, v in residual_vib_info.items():
		data_dict[k] = v
	
	return data_dict