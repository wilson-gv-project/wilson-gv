#!/usr/bin/env python
import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=10)

import pickle
outfile = '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/polarData.pkl'
with open(outfile, 'rb') as file:
    things = pickle.load(file)

rawfile = '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/polarData_raw.pkl'
with open(rawfile, 'rb') as file:
    rare = pickle.load(file)

dimless = '../../scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/dimensionless.pkl'
with open(dimless, 'rb') as file:
    dless = pickle.load(file)

print(dless[0])
for d in dless[2]:
    print(d)
    print(dless[2][d])

# from scriptsHPC.utils import parseCFOUR
# things['equil'] = parseCFOUR.pTensor(polar_dir+'/../anharm/POLAR')


# print(type(things))
# print(things)

# for i in sorted(things):
#     print(i)
#     print(things[i])
refined = {}

for k in sorted(rare):
    print(k)
    m, R = rare[k]
    print(rare[k][0])
    print('---------')
    print(rare[k][1])
    print('---------')
    temp = np.einsum('ij,jk->ik', R.T, m)
    alpha_prime = np.einsum('ij,jk->ik', temp, R)
    print(alpha_prime)
    refined[k] = alpha_prime
    print('================\n')

def compute_PolarDerivatives(polar_data: dict):
    """

    :param polar_data:

    :return:
    """
    # base_dir = 'equil/displacements'

    # print(sorted(list(polar_data.keys())))
    # quit()

    directories = list(polar_data.keys())
    nums = [h.strip('np').split('_') for h in directories if h!='equil']
    flattened_list = [item for sublist in nums for item in sublist]
    flattened_list = set([int(g) for g in flattened_list])

    # dictionary of first order derivatives
    firstder = {}

    for f in flattened_list:
        firstder[f] = (polar_data[f'{f}p'] - polar_data[f'{f}n']) / 0.02

    secondders = {}
    import copy

    # (∂²α_ij/∂Q_k∂Q_l) ≈ [α_ij(Q_k + ΔQ_k, Q_l + ΔQ_l) - α_ij(Q_k + ΔQ_k, Q_l - ΔQ_l)
    #   - α_ij(Q_k - ΔQ_k, Q_l + ΔQ_l) + α_ij(Q_k - ΔQ_k, Q_l - ΔQ_l)] / (4 * ΔQ_k * ΔQ_l)
    for k in flattened_list:
        for m in flattened_list:
            if k < m:
                val = (polar_data[f'{k}_{m}pp'] - polar_data[f'{k}_{m}pn'] - polar_data[f'{k}_{m}np'] + polar_data[f'{k}_{m}nn']) / (4 * 0.01 * 0.01)
                secondders[(k, m)] = val
                secondders[(m, k)] = val

    # (∂²α_ij/∂Q_k²) ≈ [α_ij(Q_k + ΔQ_k) - 2α_ij(Q_k) + α_ij(Q_k - ΔQ_k)] / (ΔQ_k)²
    for b in flattened_list:
        secondders[(b, b)] = (polar_data[f'{b}p'] - 2 * polar_data['equil'] + polar_data[f'{b}n']) / 0.01 ** 2

    polder = []
    for p in firstder:
        polder.append(firstder[p])
    first = np.array(polder)

    second = np.zeros((6, 6, 3, 3))
    indices_to_insert = list(secondders.keys())
    # print(indices_to_insert)
    # Insert the matrices at the specified indices
    for index, mat in zip(indices_to_insert, list(secondders.values())):
        i, j = index
        second[i - 7, j - 7] = mat

    return first, second

first, second = compute_PolarDerivatives(refined)

print(second)