class mtRspfuncs:

# Class containing information and routine for fetching value of derivatives used in microscopic terms

    def __init__(self, operators, modes):

# The lists 'operators' and 'modes_rsp' contain lists specifying the operators and normal
# modes involved in the differentiation
# Let's say the operators are mu_alpha, mu_beta, and mu_delta diff. w.r.t. modes a (twice)
# and b (once). This is d**3beta / (da**2 * db). This would give
# operators = [0, 1, 3] and modes = [1, 0]
# UPDATE 2014: Identify by name, not number (example above would be ['a', 'b', 'd'] and ['b', 'a']
# Also: allow for several quantities, so each of operators and modes are lists of lists
# FURTHER UPDATE: Go back to identify by number (do mapping from identifiers used in sympy)

        self.operators = operators
        self.modes = modes
        self.ind_cache = {}
        self.value_cache = {}
        self.cachesize = 0

    def val(self, d, tensors, mode_indices, pl, iso_mat, iso_f):


        for i in range(self.cachesize):
            #print 'cache', self.ind_cache[i]
            #print 'mode ind', mode_indices
            if (self.ind_cache[i] == mode_indices):
                #print 'cache match', i
                return self.value_cache[i]

        #print 'mode indices', mode_indices
        #print 'self operators', self.operators
        #print 'self modes', self.modes
        #print 'iso f', iso_f, 'with len', len(iso_f)

        # Not sure about proper dimensions, original assignment below commented out, use len(iso_f) instead for now
        #P = np.zeros(sum([len(self.operators[i]) for i in range(len(self.operators))]))
        P = np.zeros(len(pl))

        for i in range(len(iso_f)):

            #print 'i is', i

            # First add
            for j in range(len(iso_f[i][0])):

                #print 'adding for j', j, 'with len', len(iso_f[i][0])

                new_val = 1.0

                for k in range(len(self.operators)):
                    #print 'self operators', self.operators

                    if not(self.operators[k][0] == 'z'):

                        this_ind = tuple([iso_f[i][0][j][m] for m in [alphanum[p] for p in self.operators[k]]])

                    else:
                        #print 'force constant case'
                        this_ind = (0,)

                    #print 'this ind is', this_ind

                    new_val = new_val * tensors.tensor_value(d, self.operators[k], this_ind, [mode_indices[m] for m in self.modes[k]])
                    #print 'new val is now', new_val

                P[i] += new_val
                #print 'nv'

            # Then subtract
            for j in range(len(iso_f[i][1])):

                #print 'subtracting for j', j, 'with len', len(iso_f[i][1])

                new_val = 1.0

                for k in range(len(self.operators)):

                    if not(self.operators[k][0] == 'z'):

                        this_ind = tuple([iso_f[i][1][j][m] for m in [alphanum[p] for p in self.operators[k]]])

                    else:
                        #print 'force constant case'
                        this_ind = (0,)

                    new_val = new_val * tensors.tensor_value(d, self.operators[k], this_ind, [mode_indices[m] for m in self.modes[k]])

                P[i] -= new_val


        #print 'pl', pl
        #print 'iso mat', iso_mat
        #print 'P', P
        #print 'iso mat * P', np.dot(iso_mat, P)
        #print 'end print'

        #print dfsfdasf

        ans = np.dot(pl, np.dot(iso_mat, P))

        self.ind_cache[self.cachesize] = copy.deepcopy(mode_indices)
        self.value_cache[self.cachesize] = ans
        self.cachesize += 1

        #print 'cache size now', self.cachesize

        return ans
