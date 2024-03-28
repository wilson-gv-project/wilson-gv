import copy

import numpy as np

# import time

# note: unit is E/a0


class NumDiff:
    """Utils for numerical differentiation."""

    @classmethod
    def get_gradient(cls, routine, molecule, order, h=0.01):
        """Compute numerical gradient."""
        coord = molecule.coordinates.ravel()
        dim = len(coord)
        test = routine(molecule)
        if isinstance(test, float):
            m = 1
        else:
            if len(test.ravel()) == dim:
                m = 0

        acc, mapping = NumDiff.find_coeff(1, order + 1, type='central')
        gradient = np.zeros(dim)
        # print('mapping', mapping)

        if m == 1:
            for i in range(dim):
                for k1, v1 in mapping.items():
                    mol = copy.deepcopy(molecule)
                    coord = mol.coordinates.ravel()
                    if len(coord) != dim:
                        raise Exception('dimenson error')
                    coord[i] += float(k1 * h)
                    mol.coordinates = coord.reshape(-1, 3)

                    term = float(routine(mol)) * float(v1) / float(h)

                    gradient[i] += term
        elif m == 0:
            gradient = routine(molecule)

        return gradient

    @classmethod
    def get_hessian(cls, routine, molecule, order, h=0.01):
        """Compute numerical hessian."""
        coord = molecule.coordinates.ravel()
        dim = len(coord)
        test = routine(molecule)
        if isinstance(test, float):
            m = 2
        else:
            if len(test.ravel()) == dim:
                m = 1
            else:
                m = 0

        if m > order:
            raise Exception('given accuracy unreasonlable')

        acc, mapping = NumDiff.find_coeff(1, order + 1, type='central')
        # acc2, mapping2 = NumDiff.find_coeff(2,order+1, type='central')
        # if acc!= order:
        #     print('WARNING: available accuracy fr this order', acc, ' consider modifying order')
        if m == 1:
            total = len(mapping)**m * dim
        elif m == 2:
            total = len(mapping)**m * dim * (dim-1)
        elif m == 0:
            total = 1
        print(f'Number of calculations per point: {len(mapping)**m} (in total : {total})')

        # print('expansion', mapping)
        hessian = np.zeros((dim, dim))

        if m == 2:

            for i in range(dim):
                # for k1, v1 in mapping2.items():
                #     mol = copy.deepcopy(molecule)
                #     coord = mol.coordinates.ravel()
                #     coord[i] += k1*h
                #     mol.coordinates = coord.reshape(-1,3)
                #     term = routine(mol)*v1/(h)**2
                #     hessian[i,i]+=term
                for j in range(i, dim):
                    for k1, v1 in mapping.items():
                        for k2, v2 in mapping.items():
                            mol = copy.deepcopy(molecule)
                            coord = mol.coordinates.ravel()
                            coord[i] += k1 * h
                            coord[j] += k2 * h
                            mol.coordinates = coord.reshape(-1, 3)
                            term = routine(mol) * v1 * v2 / (h)**2
                            hessian[i, j] += term
                    hessian[j, i] = hessian[i, j]

        elif m == 1:
            for i in range(dim):
                for k1, v1 in mapping.items():
                    mol = copy.deepcopy(molecule)
                    coord = mol.coordinates.ravel()
                    coord[i] += k1 * h
                    mol.coordinates = coord.reshape(-1, 3)
                    term = routine(mol).ravel() * v1 / (h)
                    hessian[i, :] += term

        elif m == 0:
            hessian = routine(molecule)

        return hessian

    # print('hessian', print(np.array_str(hessian, precision=2, suppress_small=True)))
    # u,s,vh = np.linalg.svd(hessian)
    # print('EIGENVAlUES', s)

    @classmethod
    def find_coeff(cls, order, ncalc, type='central'):
        """Find expansion coefficients for given order of derivatives and number of calcs with highes acuray order."""
        if order >= ncalc:
            raise Exception(f'numerical derivatives of order {order} requires at least {order+1} calculations')

        # points 1 : +1h, -2 : -2h ,...
        points = [0]
        if type == 'central':
            # if ncalc%2 != 0 :
            # print('central derivaitves only for even number of calculations')
            for i in range(1, ncalc//2 + 1):
                points.append(i)
                points.append(-i)
        elif type == 'forwards':
            for i in range(1, ncalc + 1):
                points.append(i)
        # backwards : identical with forwards, with the mapping coefficients
        # multplied by -1 for odd order of derivatives

        coeff = cls.gen_coeff(order, points)
        for m, n, acc, mapping in coeff:
            if m == order:
                return acc, mapping

    @classmethod
    def gen_coeff(cls, mbig, points, x0=0):
        """Generate coefficients for expansion of derivative."""
        # Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids",
        # Mathematics of Computation, 51 (184): 699–706, doi:10.1090/S0025-5718-1988-0935077-0
        nbig = len(points)
        d = np.zeros((mbig + 1, nbig, nbig))
        c1 = 1.0
        d[0][0][0] = 1.0
        for n in range(1, nbig):
            c2 = 1.0
            val1 = float(points[n] - x0)
            val2 = float(points[n - 1] - x0)
            for nu in range(n):
                c3 = float(points[n] - points[nu])
                c2 = c2 * c3
                d[0][n][nu] = val1 * d[0][n - 1][nu] / c3

                for m in range(1, min(n + 1, mbig + 1)):
                    d[m][n][nu] = (val1 * d[m][n - 1][nu] - float(m) * d[m - 1][n - 1][nu]) / c3
            d[0, n, n] = -c1 / c2 * val2 * d[0][n - 1][n - 1]
            for m in range(1, min(n + 1, mbig + 1)):
                d[m][n][n] = c1 / c2 * (m * d[m - 1][n - 1][n - 1] - val2 * d[m][n - 1][n - 1])
            c1 = c2

        coeff: list = []
        for m in range(1, mbig + 1):
            # print('derivative order', m)
            for n in range(nbig - 1, m - 1, -1):
                # print('order of accuracy', n-m+1)
                # reverted order for picking highest accuracy per number of calcs
                temp = []
                for nu in range(nbig):
                    if d[m][n][nu] == 0.0:
                        continue
                    temp.append((points[nu], d[m][n][nu]))
                # derivative order, number of calc, accuray order, mapping
                coeff.append([m, len(temp), n - m + 1, dict(temp)])
        return coeff


#     # devide everything by h^m (m is order of derivative)

# if __name__ == '__main__':
#
#     import hylleraas as hsp
#     # import numpy as np
#     # from hylleraas import Molecule
#     from qcelemental import PhysicalConstantsContext
#     constants = PhysicalConstantsContext('CODATA2018')
#
#     water = hsp.Molecule({
#         'atoms': ['O', 'H', 'H'],
#         'coordinates': [
#             [0.0, 0.0, 0.0],
#             [0.0, 0.759337, 0.596043],
#             [0.0, -0.759337, 0.596043],
#         ],
#     })
#
#     HF_DZ = hsp.Method({'qcmethod': 'HF', 'basis': 'cc-pVDZ', 'main_input': ['ExtremeSCF', 'Scfconv10']},
#                        program='orca')
#
#     grad_ref = HF_DZ.get_gradient(water).ravel()
#     print('gradient=', grad_ref)
#
#     hvals = np.log10(np.array(list(np.logspace(-10, -2, 10))))
#     diffs = []
#     orders = list(range(1, 11))
#     import itertools
#     xy = list(itertools.product(hvals, orders))
#     xvals = []
#     yvals = []
#     for p in xy:
#         xvals.append(p[0])
#         yvals.append(p[1])
#
#     # for hval in hvals:
#     #     for order in orders:
#     #         grad = NumDiff.get_gradient(HF_DZ.get_energy, water, order, h=hval).ravel()
#     #         grad *= constants.bohr2angstroms
#     #         diff = np.linalg.norm(grad-grad_ref)
#     #         diffs.append(diff)
#     # output:
#     diffs = [
#         0.003253678203232184, 0.003253678203232184, 0.00465050186455665, 0.00465050186455665,
#         0.0054599266930165655, 0.0054599266930165655, 0.00613275114125443, 0.00613275114125443,
#         0.006576877288671117, 0.006576877288671117, 0.00033061000966947925, 0.00033061000966947925,
#         0.00043613269755056857, 0.00043613269755056857, 0.00049382135485336, 0.00049382135485336,
#         0.0005244503135850069, 0.0005244503135850069, 0.0005435589320494325, 0.0005435589320494325,
#         5.864076467465095e-05, 5.864076467465095e-05, 7.026678757619524e-05, 7.026678757619524e-05,
#         7.580899476442038e-05, 7.580899476442038e-05, 8.106570647392463e-05, 8.106570647392463e-05,
#         8.362754658867974e-05, 8.362754658867974e-05, 5.791501936626278e-06, 5.791501936626278e-06,
#         8.578645971497431e-06, 8.578645971497431e-06, 9.963473332655702e-06, 9.963473332655702e-06,
#         1.1199538321346823e-05, 1.1199538321346823e-05, 1.18883241644122e-05, 1.18883241644122e-05,
#         6.827282550519479e-07, 6.827282550519479e-07, 7.874447143728655e-07, 7.874447143728655e-07,
#         8.187471112862283e-07, 8.187471112862283e-07, 8.510235658378788e-07, 8.510235658378788e-07,
#         8.302821159545875e-07, 8.302821159545875e-07, 1.4953448461165067e-07, 1.4953448461165067e-07,
#         1.9971809953223032e-07, 1.9971809953223032e-07, 2.2686826998272838e-07, 2.2686826998272838e-07,
#         2.4197863786819054e-07, 2.4197863786819054e-07, 2.5419802630936806e-07, 2.5419802630936806e-07,
#         5.498007664869499e-08, 5.498007664869499e-08, 5.750693603420465e-08, 5.750693603420465e-08,
#         5.871521111183391e-08, 5.871521111183391e-08, 5.99947807339565e-08, 5.99947807339565e-08,
#         6.072708262040263e-08, 6.072708262040263e-08, 5.624970619416293e-08, 5.624970619416293e-08,
#         4.2546771360866833e-08, 4.2546771360866833e-08, 4.2935607105918345e-08, 4.2935607105918345e-08,
#         4.319406989447165e-08, 4.319406989447165e-08, 4.328590075271881e-08, 4.328590075271881e-08,
#         1.2590513561766969e-06, 1.2590513561766969e-06, 3.99352506712154e-08, 3.99352506712154e-08,
#         3.9876619636953154e-08, 3.9876619636953154e-08, 3.985088166690681e-08, 3.985088166690681e-08,
#         3.983695148651188e-08, 3.983695148651188e-08, 7.401951090515446e-05, 7.401951090515446e-05,
#         8.840796939215706e-08, 8.840796939215706e-08, 4.008287981465355e-08, 4.008287981465355e-08,
#         4.006434689105303e-08, 4.006434689105303e-08, 4.007032219620234e-08, 4.007032219620234e-08
#     ]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     zvals = np.log10(np.array(diffs).reshape(len(orders), len(hvals)))
#     cp = ax.contourf(hvals, orders, zvals)
#     fig.colorbar(cp)
#     ax.set_xlabel('log10(h)')
#     ax.set_ylabel('order')
#     ax.set_title('log10(||g-g_ref||_2)')
#     plt.show()
#
#     print(diffs)


def mol_difference(molecule, variable, hpercent, sign):
    import copy
    newmol = copy.deepcopy(molecule)
    newmol[variable[1], variable[0]] += sign * hpercent

    return newmol

# def numericalDerGeneral(molecule, charges, h, variables):
#     # import analytical_derivatives as nnr
#
#     molUp = mol_difference(molecule, variables[-1], h, sign=1)
#     molDown = mol_difference(molecule, variables[-1], h, sign=-1)
#
#     if len(variables) == 1:
#         return (nuc_repulsionZeroOrder(molUp, charges) - nuc_repulsionZeroOrder(molDown, charges)) / (2 * h)
#
#     else:
#         start = nnr.startingExp(molecule)
#         norderder = nnr.general_derivative_Expression(start, variables[:-1])
#
#         return (nnr.general_derivative_Evaluation(norderder, molUp, charges) - nnr.general_derivative_Evaluation(norderder, molDown, charges)) / (2 * h)

# Equilibrium Cartesian Coordinates
#        0.000000000000000      -1.431855292696819       0.999548494616719
#        0.000000000000000      -0.000000000000000      -0.125961284555014
#        0.000000000000000       1.431855292696819       0.999548494616719
#   Summary of displaced coordinates used in calculation
#
#   Displaced point   1, normal coordinate   7, positive displacement: -1
#       -0.000000000000000      -1.426298373302268       1.006851283857186
#        0.000000000000000      -0.000000000000000      -0.126881568781860
#       -0.000000000000000       1.426298373302267       1.006851283857186
#   Displaced point   2, normal coordinate   7, negative displacement: -1
#        0.000000000000000      -1.437412212091371       0.992245705376252
#       -0.000000000000000      -0.000000000000000      -0.125041000328167
#        0.000000000000000       1.437412212091370       0.992245705376252
#   Displaced point   3, normal coordinate   8, positive displacement: -1
#        0.000000000000000      -1.426796907297962       0.996130181495748
#       -0.000000000000000      -0.000000000000000      -0.125530514948489
#        0.000000000000000       1.426796907297961       0.996130181495748
#   Displaced point   4, normal coordinate   8, negative displacement: -1
#       -0.000000000000000      -1.436913678095677       1.002966807737690
#        0.000000000000000      -0.000000000000000      -0.126392054161539
#       -0.000000000000000       1.436913678095676       1.002966807737690
#   Displaced point   5, normal coordinate   9, positive displacement:  2
#        0.000000000000000      -1.427223486566177       0.995907663646090
#       -0.000000000000000      -0.000583691790011      -0.125961284555014
#        0.000000000000000       1.436487098827460       1.003189325587349

np.set_printoptions(suppress=True, linewidth=150)

with open('cartnormdisplacements') as referencedisplacements:
    all_matching_lines = []

    for line in referencedisplacements:
        line = line.strip()
        if line != '':
            line = [i.strip() for i in line.split('   ') if i!='']
            if len(line) > 6:
                line = np.array([float(i) for i in line])
                all_matching_lines.append(line)

all_matching_lines = np.array(all_matching_lines)
firstgeo = all_matching_lines[0].T
nmodes = all_matching_lines[-3:]
# all_matching_lines = np.array(all_matching_lines).T
# all_matching_lines = np.array(all_matching_lines)[1:]

print(firstgeo)
# print(all_matching_lines)
print('\n')

diffs = all_matching_lines - firstgeo
# print("\differences\n", diffs)
all_matching_lines = np.vstack((all_matching_lines, diffs))

import pandas as pd
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
desired_width = 270
pd.set_option('display.width', desired_width)
# for i, o in enumerate(all_matching_lines):
#     print(i, o-firstgeo)
#
# for i, o in enumerate(all_matching_lines[-2:]):
#     print(i, o)

# print(pd.DataFrame(all_matching_lines))
print(pd.DataFrame(all_matching_lines.T))
# print(pd.DataFrame(diffs))
print(firstgeo)
print(pd.DataFrame(nmodes.T))

# print(all_matching_lines[6]/all_matching_lines[7])
# print(all_matching_lines[6]/diffs[1])
# print(all_matching_lines[6]/diffs[2])
print(diffs[1]/all_matching_lines[6])
print(diffs[2]/all_matching_lines[6])
#
# print(all_matching_lines[7]/diffs[3])
# print(all_matching_lines[7]/diffs[4])

# print((all_matching_lines[7]/diffs[3])/(all_matching_lines[7]/diffs[4]))

# print(all_matching_lines[0]/diffs[4])

# print(all_matching_lines[1]/all_matching_lines[6])
# print(all_matching_lines[2]/all_matching_lines[6])

# print(0.007303/0.005557, 0.999548/-1.431855)
