"""
Utility functions and classes. Related to different parts of calculations and setup.


"""
import numpy as np
from scipy import constants
from typing import Iterable, Generator
import itertools


def combinations_with_permutations(iterable: Iterable, k: int) -> Generator:
    """
    Making a generator of combinations of k elements of iterable
    """
    return (comb for comb in itertools.product(iterable, repeat=k))



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
