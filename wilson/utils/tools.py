"""
Utility functions and classes. Related to different parts of calculations and setup.


"""
import numpy as np
from scipy import constants
from typing import Iterable, Generator
import itertools


def convNu2Ene(reciprocal_cm: float | np.ndarray, reverse: bool = False) -> float | np.ndarray:
    """
    Convert a wavenumber (cm-1) to energy (Hartree)
    """
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    if not reverse:
        return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)
    else:
        return reciprocal_cm / (100 * constants.h * constants.c / hartree2J)


def combinations_with_permutations(iterable: Iterable, k: int) -> Generator:
    """
    Making a generator of combinations of k elements of iterable
    """
    return (comb for comb in itertools.product(iterable, repeat=k))


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

    if type(formula[-2]) is str:
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


def avrg_abc_tensor_general(formula: tuple,
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

    if type(formula[-2]) is str:
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
                        a: int, b: int, c: int = None) -> dict:
    """
    Trivial names aren't updated
    """
    if type(formula[-2]) is str:
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


def min_abs_preserve_sign(array: np.ndarray) -> float|complex|int:
    """
    Find abs min value with original sign
    """
    abs_array = np.abs(array)
    min_index = np.unravel_index(np.argmin(abs_array), abs_array.shape)
    return array[min_index]


def find_nearest_index(array: np.ndarray, value:float|complex|int) -> int:
    """
    Get index in np.ndarray of the closest (difference) value to given value
    """
    idx = np.abs(array - value).argmin()
    return idx


def fill_subgrid(grid: np.ndarray, seed: tuple[float, float], radius: float|int, grid_size: tuple[float, float]) -> np.ndarray:
    """
    To part of grid within a radius around the seed +1; returns updated grid.
    """
    top = max(0, seed[0] - radius)
    bottom = min(grid_size[0], seed[0] + radius + 1)
    left = max(0, seed[1] - radius)
    right = min(grid_size[1], seed[1] + radius + 1)
    grid[top:bottom, left:right] += 1

    return grid


def match_modes(spectrumObj_g16, spectrumObj_c4) -> None:
    """
    Finds matching modes between g16 and c4 modes. 
    
    Does not return, prints on the go
    from wilson.spectrum import Spectrum2D

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

    repetitions_c4 = {k: {} for k in c4dict.keys()}
    for (key1, key2), value in dot_products.items():
        repetitions_c4[key2][(key1, key2)] = abs(value)

    g16_list = list(g16dict.keys())  # key1
    c4_list = list(c4dict.keys())  # key2
    g16_is_c4 = {}

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
