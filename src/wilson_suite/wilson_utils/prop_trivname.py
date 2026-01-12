def prop_trivname(*, ord_geo: int=0, ord_el: int=0, ord_rot: int=0) -> str:
	"""
	Generate an abbreviated trivial name for a molecular property based on orders of perturbation

	ord_geo: Order of geometric differentiation (perturbation)
	ord_el: Order of electric dipole perturbation
	ord_rot: (Ad hoc) order of rotational interaction: Used for rotational constants and Coriolis coupling constants
	"""

	triv_el = {0: '', 1: 'dip', 2: 'pol', 3: 'hyp', 4: 'shyp', 5: 'thyp', 6: '4hyp'}
	triv_geo = {0: '', 1: 'grad', 2: 'hess', 3: 'cff', 4: 'qff', 5: '5ff', 6: '6ff'}

	if (ord_el == 0) and (ord_geo == 0) and (ord_rot == 0):
		return 'E'

	if ord_rot > 0:

		if (ord_rot == 1) and (ord_geo == 0):

			if ord_el == 0:
				return 'B'

			else:
				raise AssertionError('This property trivial name is not established in prop_trivname')

		elif (ord_rot == 1) and (ord_geo == 2):

			if ord_el == 0:
				return 'coriolis'

			else:
				raise AssertionError('This property trivial name is not established in prop_trivname')

		else:
			raise AssertionError('This property trivial name is not established in prop_trivname')

	if (ord_el > 6) or (ord_geo > 6):
			raise AssertionError('This property trivial name is not established in prop_trivname')

	return triv_el[ord_el] + triv_geo[ord_geo]
