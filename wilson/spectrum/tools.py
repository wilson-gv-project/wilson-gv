import copy

import numpy as np
from scipy import constants
from dataclasses import dataclass, field
from typing import Type
from CQCParse.parsing import Parser


def convNu2Ene(reciprocal_cm: float | np.ndarray, reverse: bool = False) -> float | np.ndarray:
    """Convert wavenumber (cm-1) to energy (Hartree)"""
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    if not reverse:
        return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)
    else:
        return reciprocal_cm / (100 * constants.h * constants.c / hartree2J)


import itertools
def combinations_with_permutations(iterable, k):
    return (comb for comb in itertools.product(iterable, repeat=k))



@dataclass
class Conditions:
    Gamma_rc: float
    diag_margin_rc: float
    dynamic_range_n: int|float
    omega1: np.ndarray
    omega2: np.ndarray
    program: str
    data_parser: Type[Parser] #CFOURdataParser|GaussianDataParser
    molecule: str
    method: str
    basis: str
    new_idx_dict : dict
    el_terms_selected: list
    mech_terms_selected: list
    list2exclude: list = None
    only_modes: list = None
    vpt2settings: dict = field(default_factory=lambda: {'anharmonic_type': 'GVPT2'})
    vib_levels_harmonic: bool = False
    preview: bool = False


def avrg_abc_tensor(formula: tuple,
                    data: dict[str:np.ndarray], gammaCompsAll: np.array) -> np.ndarray:
    """
    Calculate the averaging tensor for a given formula.
    Indices of the tensor are normal coordinates (NC) indices,
    and the shape of the tensor depends on the nature of the term that is being calculated.
    Shape of the averaging tensor for electrical anharmonicity terms is (n_NC, n_NC)
    Shape of the averaging tensor for mechanical anharmonicity terms is (n_NC, n_NC, n_NC)

    :param formula: example (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)))
    :param data:
    :param gammaCompsAll:
    :return:
    """
    nmodes = data['mu_Q'].shape[0]

    if type(formula[-2]) == str:
        # True for mechanical anharmonicity terms
        formula = formula[:-2]      # removes the Fabc string, to deal only with the averaging part

    # specific case of the gamma_1,0 first term
    if [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
        avrg_tensor = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_Q'][b, alpha, delta] * data['mu_QQ'][a, b, gamma]
                avrg_tensor[a, b] = total/15.
        return avrg_tensor

    # specific case of the gamma_1,0 second term
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
        avrg_tensor = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_QQ'][a, b, alpha, delta] * data['mu_Q'][b, gamma]
                avrg_tensor[a, b] = total/15.
        return avrg_tensor

    # all terms of gamma_0,1 have this structure of averaging part
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_Q']:
        avrg_tensor = np.zeros((nmodes, nmodes, nmodes))
        # this part is changing for different terms
        modes_letters = [i[1] for i in formula]
        for a in range(nmodes):
            for b in range(nmodes):
                for c in range(nmodes):
                    abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                    i1, i2, i3 = [abc[j[0]] for j in modes_letters]
                    total = 0.
                    for comps in gammaCompsAll:
                        alpha, beta, gamma, delta = comps
                        total += data['mu_Q'][i1, beta] * data['alpha_Q'][i2, alpha, delta] * data['mu_Q'][i3, gamma]
                    avrg_tensor[a, b, c] = total/15.
        return avrg_tensor


def get_properties_avrg(formula: tuple, data: dict[str:np.ndarray],
                        gammaCompsAll: np.array,
                        a,b,c=None):

    if type(formula[-2]) == str:
        # True for mechanical anharmonicity terms
        F_formula = formula[-2]      # removes the Fabc string, to deal only with the averaging part
        formula = formula[:-2]      # removes the Fabc string, to deal only with the averaging part

    # specific case of the gamma_1,0 first term
    if [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
        components = {'mu_Q': [],
                      'alpha_Q': [],
                      'mu_QQ': []}
        for comps in gammaCompsAll:
            alpha, beta, gamma, delta = comps
            components['mu_Q'].append(data['mu_Q'][a, beta])
            components['alpha_Q'].append(data['alpha_Q'][b, alpha, delta])
            components['mu_QQ'].append(data['mu_QQ'][a, b, gamma])
        return components

    # specific case of the gamma_1,0 second term
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
        components = {'mu_Q1': [],
                      'alpha_QQ': [],
                      'mu_Q2': []}
        for comps in gammaCompsAll:
            alpha, beta, gamma, delta = comps
            components['mu_Q1'].append(data['mu_Q'][a, beta])
            components['alpha_QQ'].append(data['alpha_QQ'][a, b, alpha, delta])
            components['mu_Q2'].append(data['mu_Q'][b, gamma])
        return components

    # all terms of gamma_0,1 have this structure of averaging part
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_Q']:

        abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
        ijk_indx = tuple([abc[j] for j in F_formula])
        F = data['F_abc'][ijk_indx]

        # this part is changing for different terms
        modes_letters = [i[1] for i in formula]
        abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
        i1, i2, i3 = [abc[j[0]] for j in modes_letters]
        components = {'mu_Q1': [],
                      'alpha_Q': [],
                      'mu_Q2': []}
        for comps in gammaCompsAll:
            alpha, beta, gamma, delta = comps
            components['mu_Q1'].append(data['mu_Q'][i1, beta])
            components['alpha_Q'].append(data['alpha_Q'][i2, alpha, delta])
            components['mu_Q2'].append(data['mu_Q'][i3, gamma])
        components['F_'] = [F]*len(components['mu_Q1'])
        return components


def min_abs_preserve_sign(array):
    abs_array = np.abs(array)
    min_index = np.unravel_index(np.argmin(abs_array), abs_array.shape)
    return array[min_index]


def find_nearest_index(array, value):
    idx = np.abs(array - value).argmin()
    return idx


def fill_subgrid(grid, seed, radius, grid_size):
    top = max(0, seed[0] - radius)
    bottom = min(grid_size[0], seed[0] + radius + 1)
    left = max(0, seed[1] - radius)
    right = min(grid_size[1], seed[1] + radius + 1)
    grid[top:bottom, left:right] += 1


def match_modes(spectrumObj_g16, spectrumObj_c4):
    """
    Finds matching modes between g16 and c4 modes
    """

    g16dict, c4dict = spectrumObj_g16.normal_modes, spectrumObj_c4.normal_modes

    dot_products = {}
    dot_products_all = {}
    keys1 = []
    keys2 = []

    for key1, array1 in g16dict.items():
        for key2, array2 in c4dict.items():
            dot_product = abs(np.dot(array1, array2))
            dot_products_all[(key1, key2)] = dot_product

            if abs(dot_product) > 0.6:
                if key1 in keys1:
                    maximum = max([v for k, v in dot_products.items() if key1 == k[0]]+[dot_product])
                    oldthing = [k for k, v in dot_products.items() if key1 == k[0]]
                    del dot_products[oldthing[0]]
                    dot_products[(key1, key2)] = maximum

                elif key2 in keys2:
                    maximum = max([v for k, v in dot_products.items() if key2 == k[1]]+[dot_product])
                    oldthing = [k for k, v in dot_products.items() if key2 == k[1]]
                    del dot_products[oldthing[0]]
                    dot_products[(key1, key2)] = maximum

                else:
                    keys1.append(key1)
                    keys2.append(key2)
                    dot_products[(key1, key2)] = dot_product

    repetitions_g16 = {k: {} for k in g16dict.keys()}
    for (key1, key2), value in dot_products.items():
        repetitions_g16[key1][(key1, key2)] = abs(value)

    # print({k:v for k,v in repetitions_g16.items() if len(v)>1})

    repetitions_c4 = {k: {} for k in c4dict.keys()}
    for (key1, key2), value in dot_products.items():
        repetitions_c4[key2][(key1, key2)] = abs(value)

    # print({k:v for k,v in repetitions_c4.items() if len(v)>1})

    g16_list = list(g16dict.keys())  # key1
    c4_list = list(c4dict.keys())  # key2
    g16_is_c4 = {}
    number_of_modes = len(copy.deepcopy(g16_list))

    for (key1, key2), value in dot_products.items():
        if key1 in repetitions_g16:
            k = max(repetitions_g16[key1], key=repetitions_g16[key1].get)[1]
            g16_is_c4[key1] = k
            g16_list.remove(key1)
            c4_list.remove(k)

        elif key2 in repetitions_c4:
            k = max(repetitions_c4[key2], key=repetitions_c4[key2].get)[1]
            g16_is_c4[key1] = k
            g16_list.remove(key1)
            c4_list.remove(k)

        else:
            g16_is_c4[key1] = key2
            g16_list.remove(key1)
            c4_list.remove(key2)


    if len(g16_list) == 1:
        g16_is_c4[g16_list[0]] = c4_list[0]
        g16_list.pop(0)
        c4_list.pop(0)
    else:
        print('Result:', g16_is_c4)
        print(dot_products)
        print('Oh no, there are some ambiguities')
        print(g16_list, c4_list)

    if g16_list:
        deltas = {}
        for i in g16_list:
            for j in c4_list:
                deltaw = abs(spectrumObj_g16.all_states[(str(i),)] - spectrumObj_c4.all_states[(str(j),)])
                if deltaw < 100.:
                    deltas[(i,j)] = deltaw
                    g16_is_c4[i] = j
        print(deltas)

    print('\n', dot_products_all)
        # for t in deltas:
        #     g16_list.remove(t[0])
        #     c4_list.remove(t[1])
    #
    # if len(g16_is_c4) == number_of_modes:
    #     return g16_is_c4
    # else:
    #     print(g16_is_c4)
    #     print('Oh no, there still some ambiguities')
    #     print(g16_list, c4_list)


def change_idx_modes(parserObj, new_idx_dict, list2exclude=None, only_modes = None):
    """
    new_idx_dict = {oldkey:newkey}

    To change:
        all_states - after vpt2, so no need to upd cubic_cm_1, quartic_cm_1
        fundamentals_harmonic - because used in calculation of prefactors
        all_states_harmonic - used if harmonic energy levels used

        deriv_data, list2exclude

            ddata = [parserObj.dipole_first_derivatives,
             parserObj.dipole_second_derivatives,
             parserObj.polarizability_first_derivatives,
             parserObj.polarizability_second_derivatives,
             parserObj.cubic_force_constants]
    self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))
    """

    new_dict1 = {}
    new_dict2 = {}
    new_dict3 = {}
    new_dict4 = {}

    # upd self.all_states
    for oldkey, val in parserObj.anharmonic_states.items():
        # newkey = tuple(sorted([str(new_idx_dict[int(i)]) for i in oldkey]))
        newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey])])
        new_dict1[newkey] = val

    sorted_keys = sorted(new_dict1.keys(), key=lambda x: tuple(map(int, x)))
    new_dict1_sort = {key: new_dict1[key] for key in sorted_keys}
    all_states = new_dict1_sort

    # upd self.all_states_harmonic
    for oldkey, val in parserObj.harmonic_states.items():
        # newkey = tuple(sorted([str(new_idx_dict[int(i)]) for i in oldkey]))
        newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey])])
        new_dict2[newkey] = val

    sorted_keys = sorted(new_dict2.keys(), key=lambda x: tuple(map(int, x)))
    new_dict2_sort = {key: new_dict2[key] for key in sorted_keys}

    all_states_harmonic = new_dict2_sort

    # upd self.fundamentals_harmonic
    for oldkey, val in parserObj.fundamentals_harmonic_str.items():
        newkey = str(new_idx_dict[int(oldkey)])
        new_dict3[newkey] = val
    sortKeys = list(new_dict3.keys())
    sortKeys.sort()
    new_dict3_sort = {i: new_dict3[i] for i in sortKeys}

    fundamentals_harmonic = new_dict3_sort

    # upd self.fundamentals
    for oldkey, val in parserObj.fundamentals_anharmonic_str.items():
        newkey = str(new_idx_dict[int(oldkey)])
        new_dict4[newkey] = val
    sortKeys = list(new_dict3.keys())
    sortKeys.sort()
    new_dict4_sort = {i: new_dict4[i] for i in sortKeys}

    fundamentals = new_dict4_sort

    # for old in self.list2exclude:
    #     new_list.append(new_idx_dict[old])
    # self.list2exclude = new_list
    nmodes = parserObj.nmodes

    # input list2exclude and only_modes would be already with new indices
    if list2exclude is not None:
        mode_indices = [i for i in np.arange(nmodes) if i not in list2exclude]
        nmodes -= len(list2exclude)
    else:
        if only_modes is not None:
            mode_indices = only_modes
        else:
            mode_indices = [i for i in np.arange(parserObj.nmodes)]

    newmu1 = np.zeros_like(parserObj.dipole_first_derivatives)
    newmu2 = np.zeros_like(parserObj.dipole_second_derivatives)
    newalpha1 = np.zeros_like(parserObj.polarizability_first_derivatives)
    newalpha2 = np.zeros_like(parserObj.polarizability_second_derivatives)
    newF = np.zeros_like(parserObj.cubic_force_constants)

    # cff_cm_1_new = np.zeros_like(parserObj.cubic_cm_1)
    # qff_cm_1_new = np.zeros_like(parserObj.quartic_cm_1)
    # cor_c_new = np.zeros_like(parserObj.coriolis_constant)

    for oldkey, newkey in new_idx_dict.items():
        newmu1[newkey, :] = parserObj.dipole_first_derivatives[oldkey, :]
        newalpha1[newkey, :, :] = parserObj.polarizability_first_derivatives[oldkey, :, :]

    new_idx_dict_2d = {}
    for old_i, new_i in new_idx_dict.items():
        for old_j, new_j in new_idx_dict.items():
            new_idx_dict_2d[(old_i, old_j)] = (new_i, new_j)

    for (old_i, old_j), (new_i, new_j) in new_idx_dict_2d.items():
        newmu2[new_i, new_j, :] = parserObj.dipole_second_derivatives[old_i, old_j, :]
        newalpha2[new_i, new_j, :, :] = parserObj.polarizability_second_derivatives[old_i, old_j, :, :]

    new_idx_dict_3d = {}
    for old_i, new_i in new_idx_dict.items():
        for old_j, new_j in new_idx_dict.items():
            for old_k, new_k in new_idx_dict.items():
                new_idx_dict_3d[(old_i, old_j, old_k)] = (new_i, new_j, new_k)

    for (old_i, old_j, old_k), (new_i, new_j, new_k) in new_idx_dict_3d.items():
        newF[new_i, new_j, new_k] = parserObj.cubic_force_constants[old_i, old_j, old_k]


    ddata = [newmu1, newmu2, newalpha1, newalpha2, newF]
    deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))

    return fundamentals, fundamentals_harmonic, all_states, all_states_harmonic, deriv_data, mode_indices