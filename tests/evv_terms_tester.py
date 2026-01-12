import wilson_suite as ws

experiment_a = ws.fixtures.evv_experiment()

fully_enhanced_terms = ws.derive.derive.get_fully_enhanced_terms(experiment_a)

print('\n\n ### Printing fully_enhanced_terms...\n')
print('type(fully_enhanced_terms)', type(fully_enhanced_terms))

for i in fully_enhanced_terms:

	print('\n  --> i in fully_enhanced_terms: i is', i)
	print('  i is an int, an element of anharm_orders dict - sum of anharm orders'
		  '| anharm orders {1: [(1, 0), (0, 1)], 0: [(0, 0)]}')
	print('  value for key i is a dict')

	for j in fully_enhanced_terms[i]:

		print('\n  --> j in fully_enhanced_terms[i]: j is', j)
		print('  j is a tuple ')
		print('  value for key j is a dict')

		print('fully_enhanced_terms[i]', type(fully_enhanced_terms[i]))
		print('fully_enhanced_terms[i][j]', type(fully_enhanced_terms[i][j]))

		if len(fully_enhanced_terms[i][j]) == 0: # it's a list!
			print('\nlen(fully_enhanced_terms[i][j]) == 0  ---->  No terms')
			print(' ')

		else:
			# fully_enhanced_terms[i][j] is a list! of wilson_derive.abstractions.VibPerturbedTerm objects
			for k in fully_enhanced_terms[i][j]:
				print('\n  --> k in fully_enhanced_terms[i][j]: k is', k)
				print('  k is an element of', type(fully_enhanced_terms[i][j]))

				print('')
				print('New term')
				k.present()
				print('')

print('===========================================================================')
print('fully_enhanced_terms')
print()

for i in fully_enhanced_terms:
	for j in fully_enhanced_terms[i]:
		print(i, j, fully_enhanced_terms[i][j], '---', len(fully_enhanced_terms[i][j]))

one_term = fully_enhanced_terms[1][(1,0)][0]
print()
print([i for i in dir(one_term) if '_' not in i])
print()

one_term.present()

print()
print(one_term.props)
print()
print(one_term.freqterms)
print()
print(one_term.res)

print('===========================================================================')
print('fully_enhanced_terms\n')
print(fully_enhanced_terms)
