


def prop_trivname(ord_geo=0, ord_el=0, ord_rot=0):

	triv_el = {0: '', 1: 'dip', 2: 'pol', 3: 'hyp', 4: 'shyp', 5: 'thyp', 6: '4hyp'}
	triv_geo = {0: '', 1: 'grad', 2: 'hess', 3: 'cff', 4: 'qff', 5: '5ff', 6: '6ff'}

	if (ord_el == 0) and (ord_geo == 0) and (ord_rot == 0):
		return 'E'

	if ord_rot > 0:

		if (ord_rot == 1) and (ord_geo == 0):
			return 'B'

		elif (ord_rot == 1) and (ord_geo == 2):
			return 'coriolis'

		else:
			raise AssertionError('This property trivial name is not established in prop_trivname')

	if (ord_el > 6) or (ord_geo > 6):
			raise AssertionError('This property trivial name is not established in prop_trivname')

	return triv_el[ord_el] + triv_geo[ord_geo]
