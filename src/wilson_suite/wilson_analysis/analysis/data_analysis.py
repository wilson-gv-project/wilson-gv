"""
TODO: to be cleaned up
"""
import pandas as pd
import numpy as np

# todo: needs to be updated or removed
def get_resonances_DF():
    """
    Returns dataframes with columns: res (expression), a, b, [c], w_a, w_b, [w_c],
                                     w_2-w_1, w_1, w_2, [FR1, FR2, F_abc], avrg_g

    # Printing pandas options:
    import sys
    pd.set_option('display.max_rows', sys.maxsize)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.width', 5000)
    """



def analyse_mechanical_resonances(dataframe_mech_resonances, computedSpectrum, rec_cm=True) -> pd.DataFrame:
    pass

def analyse_electrical_resonances(dataframe_electric_resonances, computedSpectrum, rec_cm=True) -> pd.DataFrame:
    pass

def allResDF(computedSpectrum):

    # settings
    vib_levels_harmonic = False
    rec_cm = True
    # datain = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    #
    # # spectrum object
    # computedSpectrum = Spectrum2D([], [])
    # computedSpectrum.load_data(GaussianDataParser(datain))
    # computedSpectrum.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=10., vib_levels_harmonic=False)
    # computedSpectrum.add_terms(*terms_selection)
    # if molecule=="ACDM":
    #     computedSpectrum.exclude_modes([34, 35, 36, 37, 38, 39, 40, 41])
    # computedSpectrum.precalculate4fullspectrum()

    # dataframes with resonances_args for used terms in spectrum object
    dfs4terms_el, dfs4terms_mech = get_resonances_DF(computedSpectrum, rec_cm=rec_cm,
                                                              vib_levels_harmonic=vib_levels_harmonic)
    formatted_df2 = analyse_electrical_resonances(dfs4terms_el, computedSpectrum)

    result_later = (
        formatted_df2
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later['term'] = result_later['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later['final'] = abs(result_later['gamma_mn']) ** 2
    result_later['w_1'] = result_later['w_1'].map('{:.3f}'.format)
    result_later['w_2'] = result_later['w_2'].map('{:.3f}'.format)
    result_later['ab_tuples'] = result_later.apply(lambda row: str(list(zip(row['a'], row['b']))), axis=1)
    result_later = result_later.drop(columns=['a', 'b'])
    # max_int = max(result_later['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df1 = result_later.copy()

    formatted_df1['Terms'] = formatted_df1['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    formatted_df4 = analyse_mechanical_resonances(dfs4terms_mech, computedSpectrum)

    # smaller dataframe
    result_later1 = (
        formatted_df4
        .groupby(['w_1', 'w_2'])
        .agg({
            'gamma_mn': 'sum',
            'a': lambda x: list(x),
            'b': lambda x: list(x),
            'c': lambda x: list(x),
            'ii': lambda x: list(set(x))
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_later1['term'] = result_later1['ii'].apply(lambda x: ', '.join(map(str, x)))
    result_later1['final'] = abs(result_later1['gamma_mn']) ** 2
    result_later1['w_1'] = result_later1['w_1'].map('{:.3f}'.format)
    result_later1['w_2'] = result_later1['w_2'].map('{:.3f}'.format)

    result_later1['ab_tuples'] = result_later1.apply(lambda row: str(list(zip(row['a'], row['b'], row['c']))), axis=1)
    result_later1 = result_later1.drop(columns=['a', 'b', 'c'])
    # max_int = max(result_later1['final'])
    # power = float(np.log10(max_int)).__floor__()

    formatted_df3 = result_later1.copy()
    # power = float(np.log10(formatted_df3['final'].max())).__floor__()

    formatted_df3['Terms'] = formatted_df3['term'].map({
        '0': 'Term1',
        '1': 'Term2',
        '2': 'Term3',
        '3': 'Term4',
        '4': 'Term5',
        '5': 'Term6'
    })

    combined_df = pd.concat([formatted_df1, formatted_df3])
    combined_df['gamma_mn_list'] = combined_df['gamma_mn']
    combined_df['gamma_mn'] = np.real(np.copy(combined_df['gamma_mn']))

    result_dfF = (
        combined_df.groupby(['w_1', 'w_2'])
        .agg({
            'Terms': lambda x: list(x),
            'ab_tuples': lambda x: list(x),
            'gamma_mn': 'sum',
            'gamma_mn_list': lambda x: list(x)
        })
        .reset_index()
    )
    result_later['gamma_mn'] = np.real(np.copy(result_later['gamma_mn']))
    result_dfF['ab_tuples'] = result_dfF['ab_tuples'].apply(tuple)
    result_dfF['Terms'] = result_dfF['Terms'].apply(tuple)
    def format_tuple_to_scientific(tpl):
        return tuple(f"{np.real(x):.3e}" for x in tpl)

    result_dfF['gamma_mn_list'] = result_dfF['gamma_mn_list'].apply(format_tuple_to_scientific)

    result_dfF['final_both'] = abs(result_dfF['gamma_mn']) ** 2

    return result_dfF, computedSpectrum

# todo: needs to be updated or removed
def allAddedres(molecule, method, basis, data_vault,
                terms_selection, Gamma_rc, plotHigherThan=1e5):
    pass

def check_unhashable_columns(df):
    unhashable_columns = []
    for column in df.columns:
        try:
            # Attempt to convert the column to a set
            _ = set(df[column])
        except TypeError:
            # If there is a TypeError, the column contains unhashable types
            unhashable_columns.append(column)
    return unhashable_columns


def find_nearest_index(array, value):
    # if value<np.max(array) and value>np.min(array):
    idx = np.abs(array - value).argmin()
    return idx

def fill_subgrid(grid, seed, radius, grid_size):
    top = max(0, seed[0] - radius)
    bottom = min(grid_size[0], seed[0] + radius + 1)
    left = max(0, seed[1] - radius)
    right = min(grid_size[1], seed[1] + radius + 1)
    grid[top:bottom, left:right] += 1


def get_where_matrix(computedSpectrum, radius):

    df_edata, spec1 = allResDF(computedSpectrum)
    # print(df_edata)
    df_edata['list_points_grid'] = list(zip(round(df_edata['w_1'].astype(float)),
                                            round(df_edata['w_2'].astype(float))))
    df_edata['list_points'] = list(zip(df_edata['w_1'].astype(float), df_edata['w_2'].astype(float)))
    seeds_freq = [list(i) for i in df_edata['list_points_grid']]

    seeds_indices = [(find_nearest_index(spec1.w1, seed[0]),
                      find_nearest_index(spec1.w2, seed[1])) for seed in
                     seeds_freq]

    grid_size = (len(spec1.w1), len(spec1.w2))
    grid = np.zeros(grid_size, dtype=int)

    for seed in seeds_indices:
        fill_subgrid(grid, seed, radius, grid_size)

    return grid


def get_where_matrix1seed(computedSpectrum, radius, seed):

    seeds_indices = tuple([(find_nearest_index(computedSpectrum.w1, seed[0]),
                            find_nearest_index(computedSpectrum.w2, seed[1]))])

    grid_size = (len(computedSpectrum.w1), len(computedSpectrum.w2))
    grid = np.zeros(grid_size, dtype=int)

    for seed in seeds_indices:
        fill_subgrid(grid, seed, radius, grid_size)

    return grid

def get_abctuples_res(computedSpectrum):
    df_edata, spec1 = allResDF(computedSpectrum)
    print(df_edata)
    alltuples = []
    for i in df_edata['ab_tuples']:
        for j in i:
            count = j.count(')')
            if count==1:
                it = tuple([int(k) for k in j.strip('[()]').split(',')])
                if it not in alltuples:
                    alltuples.append(it)
            else:
                it = tuple([k.strip('()').strip(' (') for k in j.strip('[]').split('),') ]) # for y in k
                ti = tuple([int(g) for g in f.split(',')] for f in it)
                for r in ti:
                    if r not in alltuples:
                        alltuples.append(tuple(r))
    print(len(alltuples))
    print(alltuples)


def find_peaks(array, dynrange=500):
    """
    Finding peaks in 2D array
    https://codemia.io/knowledge-hub/path/peak_detection_in_a_2d_array
    """
    # rows, cols = len(array), len(array[0])
    rows, cols = array.shape
    peaks = []
    maxarr = np.max(array)
    print(f'Max in find_peaks arr: {maxarr:.3e}')
    print(f'Min for find_peaks arr: {maxarr/dynrange:.3e}')
    for i in range(rows):
        for j in range(cols):
            current = array[i][j]
            # Check neighbors
            neigh8 = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if array[nx][ny] >= current:
                            neigh8.append(False)
                        else:
                            neigh8.append(True)
            if all(neigh8):
                if current>(maxarr/dynrange):
                    peaks.append((i, j))
    return peaks


def assemble_point_amplitude(w1l, w2l, terms, deriv_data, all_states, harm_modes_dict, mode_indices, Gamma_rc, margin=0., condition=None):
    dict_contents = {}
    pairs_ab = {}
    for t in terms:
        # print('term', t)
        # dict_contents[(t.term_label, t.term_id)] = t.get_intensity(w1l, w2l, deriv_data,
        # original_vpt2, mode_indices,
        # Gamma_rc, margin, condition, collect_all=True)
        pairs_ab[t] = []
        countall = 0
        countocoll = 0
        for a in mode_indices:
            for b in mode_indices:
                countall+=1
                w1ab, w2ab = t.get_resonance_location(all_states, a, b)
                if w2ab-margin>w1ab:
                    dict_contents[(t, (a, b))] = t.get_intensity_ab(a, b, w1l, w2l, deriv_data,
                                                                    all_states, harm_modes_dict, mode_indices, Gamma_rc, margin,
                                                                    condition=condition)[0]
                    pairs_ab[t].append((a, b))
                    countocoll += 1
        # print(t, 'all', countall, 'collected', countocoll)

    total = np.sum(np.array(list(dict_contents.values())))
    # print(np.array(list(dict_contents.values())))
    # print(pairs_ab)

    return dict_contents, total

def top_n_abs_values(d, n=5):
    filtered = sorted(d.items(), key=lambda x: np.abs(x[1]), reverse=True)[:n]
    return filtered