import numpy as np
import copy

# Calculate transposed 'laser polarization term' (the term (A * f) in (A * f * M * g * P) in JCP 141, 204103)
# The argument pol is a list of vectors
def get_pol_laser(pol):

	A = 1.0

	A = get_pol_tensor(A, pol)
	A = np.reshape(A, tuple([len(pol[i]) for i in range(len(pol))]))

	# print ('A is', A)


	f = get_iso_f(len(pol))

	# print ('iso f in pol laser', f)
	#print dfsafd

	pl = np.zeros((len(f)))

	# print ('pl', pl)
	# print ('shape of A', A.shape)
	# print ('length of f', len(f))
    
	for i in range(len(f)):
		#print 'i in len f loop', i
		for j in range(len(f[i][0])):
			#print 'tuple', j, 'in 0 is', tuple(f[i][0][j])
			pl[i] += A[tuple(f[i][0][j])]
		for j in range(len(f[i][1])):
			#print 'tuple', j, 'in 1 is', tuple(f[i][1][j])
			pl[i] -= A[tuple(f[i][1][j])]

		# print ('pl now', pl)


	return np.transpose(pl)



# Create polarization tensor from individual 3D polarization vectors of incident light
# Polarization vector elements are in general complex-valued
# Inital value of pol_tensor is 1.0
def get_pol_tensor(pol_tensor, pol):

	if (len(pol) == 0):

		return pol_tensor

	else:

		return get_pol_tensor(np.kron(pol_tensor, np.array(copy.deepcopy(pol[len(pol) - 1]))), pol[0:len(pol) -1])

def mdk(a, b):

	return [{(a-1):0, (b-1):0}, {(a-1):1, (b-1):1}, {(a-1):2, (b-1):2}]

def mdl(a, b, c):

	return [[{(a-1):0, (b-1):1, (c-1):2}, {(a-1):1, (b-1):2, (c-1):0}, {(a-1):2, (b-1):0, (c-1):1}], [{(a-1):2, (b-1):1, (c-1):0}, {(a-1):1, (b-1):0, (c-1):2}, {(a-1):0, (b-1):2, (c-1):1}]]




def make_iso_f(n, kron, lc):

	#print 'kron', kron
	#print 'lc', lc

	# Make two lists of lists in iso_f: One for addition and another for subtraction
	iso_f = []

	iso_f_first = meso_iso_f(kron, [[0 for i in range(n)]])
	iso_f.append(iso_f_first)
	
	# Are there only Kronecker deltas to take care of? If so, then no subtraction
	if (len(lc) == 0):
		return [iso_f[0], []]

	# If not, proceed to do Levi-Civita handling
	iso_f.append(copy.deepcopy(iso_f[0]))

	#print 'iso f after kronecker', iso_f

	bperm = [[0], [1]]
	a = binary_perm(len(lc), bperm)

	#print 'binary perm', bperm
	#print 'lc', lc

	for i in range(len(bperm)):

		this_lc = []
		for j in range(len(bperm[i])):
			#print 'i j ', i, j
			this_lc.append(lc[j][bperm[i][j]])
		
		iso_f[sum(bperm[i]) % 2] = meso_iso_f(this_lc, iso_f[sum(bperm[i]) % 2])


		#print 'iso f after kron', iso_f

	#for i in range(len(iso_f)):
		#iso_f[0][i] = tuple(iso_f)
		#iso_f[1][i] = tuple(iso_f)

	# print ('returning iso_f', iso_f)

	return iso_f


def binary_perm(n, combs):

	#print 'bp', n
	#print 'combs', combs

	if (n > 1):

		for i in range(len(combs)):

			nc1 = copy.deepcopy(combs[i])
			nc1.append(0)

			nc2 = copy.deepcopy(combs[i])
			nc2.append(1)
		
			a = binary_perm(n-1, nc1)
			a = binary_perm(n-1, nc2)
		
			combs = copy.deepcopy(nc1)
			combs.extend(copy.deepcopy(nc2))

			return 1

		print ('should not happen')
		return 0			
	else:

		return 1


def meso_iso_f(dicts, iso_f):

		#print 'dicts', dicts
		#print 'iso f', iso_f

		if (len(dicts) > 0):

			new_iso_f = []

			for i in range(len(iso_f)):

				this_iso_f = copy.deepcopy(iso_f[i])
				for j in range(len(dicts[0])):
					curr_iso_f = copy.deepcopy(this_iso_f)
					for k in dicts[0][j].keys():
						#print 'i j k', i, j, k
						#print 'curr iso f', curr_iso_f

						curr_iso_f[k] = dicts[0][j][k]

					new_iso_f.append(copy.deepcopy(curr_iso_f))
						
	
			#print new_iso_f
			iso_f = copy.deepcopy(new_iso_f)
			result = meso_iso_f(dicts[1:len(dicts)], iso_f)
			return result

		else:

			#print 'returning from meso', iso_f
			return iso_f
	


# Currently only 3D
# Maybe necessary to rewrite for higher dimensions for e.g. quadrupole effects
def get_iso_f(n):

	if (n == 2):
		
		return [[[(0,0), (1,1), (2,2)], []]]
		
	if (n == 3):
		
		return [[[(0,1,2), (1,2,0), (2,0,1), (2,1,0), (1,0,2), (0,2,1)], []]]

	if (n == 4):

		return [make_iso_f(4, [mdk(1, 2), mdk(3, 4)], []), \
			make_iso_f(4, [mdk(1, 3), mdk(2, 4)], []),
			make_iso_f(4, [mdk(1, 4), mdk(2, 3)], [])]

	if (n == 5):


		return [make_iso_f(5, [mdk(4, 5)], [mdl(1, 2, 3)]), \
			make_iso_f(5, [mdk(3, 5)], [mdl(1, 2, 4)]), \
			make_iso_f(5, [mdk(3, 4)], [mdl(1, 2, 5)]), \
			make_iso_f(5, [mdk(2, 5)], [mdl(1, 3, 4)]), \
			make_iso_f(5, [mdk(2, 4)], [mdl(1, 3, 5)]), \
			make_iso_f(5, [mdk(2, 3)], [mdl(1, 4, 5)])]

	if (n == 6):

		return [make_iso_f(6, [mdk(1, 2), mdk(3, 4), mdk(5, 6)], []), \
			make_iso_f(6, [mdk(1, 2), mdk(3, 5), mdk(4, 6)], []), \
			make_iso_f(6, [mdk(1, 2), mdk(3, 6), mdk(4, 5)], []), \
			make_iso_f(6, [mdk(1, 3), mdk(2, 4), mdk(5, 6)], []), \
			make_iso_f(6, [mdk(1, 3), mdk(2, 5), mdk(4, 6)], []), \
			make_iso_f(6, [mdk(1, 3), mdk(2, 6), mdk(4, 5)], []), \
			make_iso_f(6, [mdk(1, 4), mdk(2, 3), mdk(5, 6)], []), \
			make_iso_f(6, [mdk(1, 4), mdk(2, 5), mdk(3, 6)], []), \
			make_iso_f(6, [mdk(1, 4), mdk(2, 6), mdk(3, 5)], []), \
			make_iso_f(6, [mdk(1, 5), mdk(2, 3), mdk(4, 6)], []), \
			make_iso_f(6, [mdk(1, 5), mdk(2, 4), mdk(3, 6)], []), \
			make_iso_f(6, [mdk(1, 5), mdk(2, 6), mdk(3, 4)], []), \
			make_iso_f(6, [mdk(1, 6), mdk(2, 3), mdk(4, 5)], []), \
			make_iso_f(6, [mdk(1, 6), mdk(2, 4), mdk(3, 5)], []), \
			make_iso_f(6, [mdk(1, 6), mdk(2, 5), mdk(3, 4)], [])]


def get_iso_mat(n):

	if (n == 2):

		return 1.0

	if (n == 3):

		return 1.0

	if (n == 4):

		return np.array([[4, -1, -1],[-1, 4, -1], [-1, -1, 4]])/30.0

	if (n == 5):

		return np.array([[3,-1,-1,1,1,0],[-1, 3, -1, -1, 0, 1],[-1, -1, 3, 0, -1, -1],[1, -1, 0, 3, -1, 1],[1, 0, -1, -1, 3, -1],[0, 1, -1, 1, -1, 3]])/30.0

	if (n == 6):

		return np.array([[16, -5, -5, -5, 2, 2, -5, 2, 2, 2, 2, -5, 2, 2, -5],[-5, 16, -5, 2, -5, 2, 2, 2, -5, -5, 2, 2, 2, -5, 2], [-5,-5,16,2, 2, -5, 2, -5, 2, 2, -5, 2, -5, 2, 2], [-5, 2, 2, 16, -5, -5, -5, 2, 2, 2, -5, 2, 2, -5, 2], [2, -5, 2, -5, 16, -5, 2, -5, 2, -5, 2, 2, 2, 2, -5], [2, 2, -5, -5, -5, 16, 2, 2, -5, 2, 2, -5, -5, 2, 2,], [-5, 2, 2, -5, 2, 2, 16, -5, -5, -5, 2, 2, -5, 2, 2],[2, 2,-5, 2, -5, 2, -5, 16, -5, 2, -5, 2, 2, 2, -5],[2, -5, 2, 2, 2, -5, -5, -5, 16, 2, 2, -5, 2, -5, 2],[2, -5, 2, 2, -5, 2, -5, 2, 2, 16, -5, -5, -5, 2, 2],[2, 2, -5, -5, 2, 2, 2, -5, 2, -5, 16, -5, 2, -5, 2],[-5, 2, 2, 2, 2, -5, 2, 2, -5, -5, -5, 16, 2, 2, -5],[2, 2, -5, 2, 2, -5, -5, 2, 2, -5, 2, 2,16, -5, -5],[2, -5, 2, -5, 2, 2, 2, 2, -5, 2, -5, 2, -5, 16, -5],[-5, 2, 2, 2, -5, 2, 2, -5, 2, 2, 2, -5, -5, -5, 16]])/210.0




