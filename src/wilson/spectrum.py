"""
Mock should only:
    input:
        - fundamental frequencies (comb and overt made from them) - harmonic mock OR
                        take also comb and overt
        - skip (avrg=1.) or take mock derivatives
        - spectral window
"""
import time
import numpy as np
np.set_printoptions(linewidth=100000)

from calculations.parseCFOUR_forWilson import CFOURdataParser
from calculations.parseGaussian_forWilson import GaussianDataParser

def rec_cm2rec_s(reciprocal_cm):
    from scipy import constants
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)

class SpectrumEVV:
    """
    SpectrumEVV class
    Attributes:
        w1, w2 - np.arrays of frequencies
        w1_mesh, w2_mesh - grid of frequencies w1 and w2
        shape2d - shape of the grid
        fermirm
    """
    def __init__(self, w1: np.array, w2: np.array, input_data_info: dict, vib_levels_harmonic: bool = True):

        # Define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        # axes as arrays
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape

        self.load_data(input_data_info)
        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        # margin for higher diagonal
        self.diagonal_margin = 10.

        self.vib_levels_harmonic = vib_levels_harmonic
        print(f'\nUsed vibrational energy levels:\n harmonic? - {self.vib_levels_harmonic}')

        self.saved_mech = {}
        self.saved_el = {}

    def load_data(self, input_data_info: dict):
        self.dataInfo = input_data_info # dictionary with input_data_info source and type - inputs

        if input_data_info['source'] == 'cfour':
            dataBank = CFOURdataParser(input_data_info)
        elif input_data_info['source'] == 'gaussian':
            dataBank = GaussianDataParser(input_data_info)
        else:
            dataBank = MockDataParser()

        dataBank.getData()

        self.fundamentals = dataBank.fundamentals_anharmonic_str
        self.fundamentals_harmonic = dataBank.fundamentals_harmonic_str
        self.all_states = dataBank.anharmonic_states
        self.all_states_harmonic = dataBank.harmonic_states

        ddata = [dataBank.dipole_first_derivatives,
                 dataBank.dipole_second_derivatives,
                 dataBank.polarizability_first_derivatives,
                 dataBank.polarizability_second_derivatives,
                 dataBank.cubic_force_constants]
        self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))

    def addTerms(self, electrical_terms_selection, mechanical_terms_selection):
        """Creating functions for computing the expressions for mechanical and electrical anharmonicities"""
        # Terms in expressions
        electrical_terms_str = [('a+b,a', 'zero,a'), ('b,a', 'zero,a')]

        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        electric_avrg_str = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                           [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
                           ]

        mechanical_terms_str = [(('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                              (('c,a', 'zero,a'), ('a+b,c', 'b+c,a')),
                              (('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                              (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                              (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                              (('b,a', 'zero,a'), ('b,a+b', 'a,zero'))]

        # derivatives:
        # mu_Q, alpha_Q - for all 6 terms
        mechanical_avrg_str = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]

        self.ee, self.mm = electrical_terms_selection, mechanical_terms_selection
        factors = [1., 1., 0.5, 0.5, -0.5, -0.5]
        self.mech_factors = [factors[i] for i in mechanical_terms_selection]
        # [pool[i] for i in list_of_indices]
        self.electrical_terms, self.mechanical_terms = [electrical_terms_str[i] for i in self.ee], [mechanical_terms_str[i] for i in self.mm]
        self.electric_avrg, self.mechanical_avrg = [electric_avrg_str[i] for i in self.ee], [mechanical_avrg_str[i] for i in self.mm]
        # here the functions of 2 frequencies
        self.electr_funs = [generate_resonances_functions(i, margin=self.diagonal_margin) for i in self.electrical_terms]
        self.mech_funs = [generate_resonances_functions(*i) for i in self.mechanical_terms]

        nmodes = len(self.fundamentals)
        self.combofuns = [dict(zip(self.electr_funs, self.electric_avrg)),
                          dict(zip(self.mech_funs, self.mechanical_avrg))]

        # setting up the combinations of states for the terms
        self.coords_ab = get_abc_indices(2, len(self.fundamentals)) if self.electrical_terms is not None else []
        self.coords_abc = get_abc_indices(3, len(self.fundamentals)) if self.mechanical_terms is not None else []

        if self.electrical_terms is not None:
            self.el_avrg_tensors = [avrg_abc_tensor(ea, self.deriv_data, self.gammaCompsAll) for ea in self.electric_avrg]
        else:
            self.el_avrg_tensors = []

        if self.mechanical_terms is not None:
            self.mech_avrg_tensors = [avrg_abc_tensor(ma, self.deriv_data, self.gammaCompsAll) for ma in self.mechanical_avrg]
        else:
            self.mech_avrg_tensors = []

        self.combofuns_tensors = [dict(zip(self.electr_funs, self.el_avrg_tensors)),
                                  dict(zip(self.mech_funs, self.mech_avrg_tensors))]

        w_ab = np.zeros((nmodes, nmodes))
        w_abc = np.zeros((nmodes, nmodes, nmodes))
        for state in self.all_states:
            if len(state) == 2:
                w_ab[int(state[0]), int(state[1])] = self.all_states[state]
                w_ab[int(state[1]), int(state[0])] = self.all_states[state]
            elif len(state) == 3:
                w_abc[int(state[0]), int(state[1]), int(state[2])] = self.all_states[state]
                w_abc[int(state[0]), int(state[2]), int(state[1])] = self.all_states[state]
                w_abc[int(state[1]), int(state[0]), int(state[2])] = self.all_states[state]
                w_abc[int(state[1]), int(state[2]), int(state[0])] = self.all_states[state]
                w_abc[int(state[2]), int(state[0]), int(state[1])] = self.all_states[state]
                w_abc[int(state[2]), int(state[1]), int(state[0])] = self.all_states[state]

        self.w_abc = rec_cm2rec_s(w_abc)
        self.w_ab = rec_cm2rec_s(w_ab) # for omega_{a+b} frequencies
        w = rec_cm2rec_s(np.array([v for k,v in self.fundamentals.items()]))
        vib_ene_levels_harmonic = rec_cm2rec_s(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.matrix_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.tensor_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])

        # for i, te in enumerate(self.el_avrg_tensors):
        #     print(f'\nel_avrg_tensors {self.electric_avrg[i]}\n', te)
        #
        # for k, tm in enumerate(self.mech_avrg_tensors):
        #     print(f'mech_avrg_tensors {self.mechanical_avrg[k]}\n', tm)

    def gamma_mn(self, Gamma, a, b, c=False):
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic
        else:
            vib_ene_levels = self.all_states

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:
            total_sum_el = 0
            prefac_el = self.matrix_2d.T[a, b]
            for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
                resonance = el_func(vib_ene_levels, self.w1_mesh, self.w2_mesh,
                                    Gamma, (a, b))
                total_sum_el += elavrg[a, b] * resonance / prefac_el
            return total_sum_el / 24.

        else:
            total_sum_mech = 0
            prefac_mech = self.tensor_3d.T[a, b, c]
            for index, (mech_func, mechavrg) in enumerate(self.combofuns_tensors[1].items()):
                mechavrgF = list(self.combofuns[1].items())[index][1]
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                ijk_indx = tuple([abc[j] for j in mechavrgF[-1]])
                F = self.deriv_data['F_abc'][ijk_indx]
                resonance2 = mech_func(vib_ene_levels, self.w1_mesh, self.w2_mesh, Gamma, (a, b, c))

                total_sum_mech += self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
            return -total_sum_mech / 48.

    def get_total_gamma_sum_el(self, Gamma, a, b):
        """
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic
        else:
            vib_ene_levels = self.all_states

        total_sum_el = 0
        prefac_el = self.matrix_2d.T[a, b]
        for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
            resonance = el_func(vib_ene_levels, self.w1_mesh, self.w2_mesh,
                                Gamma, (a, b))
            total_sum_el += elavrg[a, b] * resonance / prefac_el
        return total_sum_el / 24.

    def get_total_gamma_sum_mech(self, Gamma, a, b, c):
        """
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic
        else:
            vib_ene_levels = self.all_states

        total_sum_mech = 0
        prefac_mech = self.tensor_3d.T[a, b, c]
        # prefac_mech = self.tensor_3d[a, b, c]
        for index, (mech_func, mechavrg) in enumerate(self.combofuns_tensors[1].items()):
            if index not in self.saved_mech:
                self.saved_mech[index] = {}
            mechavrgF = list(self.combofuns[1].items())[index][1]
            abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
            ijk_indx = tuple([abc[j] for j in mechavrgF[-1]])
            F = self.deriv_data['F_abc'][ijk_indx]
            resonance2 = mech_func(vib_ene_levels, self.w1_mesh, self.w2_mesh, Gamma, (a, b, c))

            addition = self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
            is_equal = np.allclose(np.abs(addition),
                                   np.abs(np.full(addition.shape, -0. + 0.j, dtype=complex)))
            if not is_equal:
                self.saved_mech[index][tuple([a, b, c])] = (addition*(-1./48.), mechavrg[a, b, c], F, resonance2)
                total_sum_mech += addition
        return -total_sum_mech / 48.

    def intensity(self, Gamma, savedict, el=True, mech=True):
        Qab, Qabc = self.coords_ab, self.coords_abc
        Z = 0
        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        if el:
            start_time = time.time()
            elall = np.zeros(self.shape2d, dtype='complex128')
            for i in Qab:
                contrib_ab = self.gamma_mn(Gamma, i[0], i[1])
                Qab_contrib_dict[tuple(i)] = contrib_ab
                elall += contrib_ab
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time -| electrical: {execution_time} seconds")
            print('Electrical anharmonicities are calculated')
            Z += elall

        if mech:
            start_time = time.time()
            mechall = np.zeros(self.shape2d, dtype='complex128')
            for i in Qabc:
                contrib_abc = self.gamma_mn(Gamma, i[0], i[1], i[2])
                Qabc_contrib_dict[tuple(i)] = contrib_abc
                mechall += contrib_abc
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\nExecution time - mechanical: {execution_time} seconds")
            print('Mechanical anharmonicities are calculated')
            Z += mechall

        key = self.id+f'_gamma{Gamma}'
        if key not in savedict:
            savedict[key] = {}

        if mech: savedict[key]['mechanical'] = mechall
        if el: savedict[key]['electrical'] = elall
        savedict[key]['Qab_contrib_dict'] = Qab_contrib_dict
        savedict[key]['Qabc_contrib_dict'] = Qabc_contrib_dict

        return Z, savedict

    def intensity_electrical(self, Gamma):
        start_time = time.time()

        Qab_contrib_dict = {}

        elall = np.zeros(self.shape2d, dtype='complex128')
        for i in self.coords_ab:
            contrib_ab = self.get_total_gamma_sum_el(Gamma, i[0], i[1])
            Qab_contrib_dict[tuple(i)] = contrib_ab
            elall += contrib_ab

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time -| electrical: {execution_time} seconds")
        print('Electrical anharmonicities are calculated')

        return elall

    def intensity_mechanical(self, Gamma):
        start_time = time.time()

        Qabc_contrib_dict = {}

        mechall = np.zeros(self.shape2d, dtype='complex128')
        for i in self.coords_abc:
            contrib_abc = self.get_total_gamma_sum_mech(Gamma, i[0], i[1], i[2])
            Qabc_contrib_dict[tuple(i)] = contrib_abc
            mechall += contrib_abc

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time - mechanical: {execution_time} seconds")
        print('Mechanical anharmonicities are calculated')

        return mechall, Qabc_contrib_dict


class SpectrumFigure:

    def __init__(self, sec_hypol_data, w1_mesh, w2_mesh, settings):

        # figure XYZ data
        self.gamma_data = sec_hypol_data
        self.intensities = abs(sec_hypol_data) ** 2
        self.X = w1_mesh
        self.Y = w2_mesh

        self.settings = {'omega1_minus_omega2': False, 'log10': True,
                         'font_dict': {'size': 18}, 'dpi': 200,
                         'figsize': (12, 12)}
        self.settings.update(settings)

        if self.settings['omega1_minus_omega2']:
            self.Y = -(self.X - self.Y)

        # figure settings
        self.figsize = self.settings['figsize']
        self.dpi = self.settings['dpi']
        self.font_dict = self.settings['font_dict'] # font = {'size': 18}
        self.settings['norm_min'] = 1e3
        # self.settings['norm_max'] = 1e8

        el, mech = self.settings['electrical'], self.settings['mechanical']

        # dynamic range max - for setting up the norm and colorbar ticks
        if 'dmax_dict' in self.settings:
            self.d_max = self.settings['dmax_dict'][(el, mech)]
        else:
            print('\nself.intensities.max()==np.max(self.intensities.flatten(), axis=0):',
                  self.intensities.max()==np.max(self.intensities.flatten(), axis=0), '{:.4e}'.format(self.intensities.max()))
            self.d_max = self.intensities.max()
        self.settings['d_max'] = self.d_max
        # dmax_dict = {(True, False): 48778401.3, (False, True): 29519537.48, (True, True): 48218929.9}
        # d_max = dmax_dict[(el_bool, mech_bool)] # m, e, t 29519537.48  48778401.3  48218929.9

    def update_settings(self, settings: dict):

        self.settings.update(settings)

    def plot2Dmatplotlib(self, nametuple: tuple, text_under_the_figure: str = '', to_save=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        if to_save:
            matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30
        matplotlib.rc('font', **self.font_dict)

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1)

        import matplotlib.colors as colors
        colorbar_norm = colors.LogNorm(vmin=self.settings['norm_min'], vmax=self.settings['norm_max'])

        dynamic_range = 300 # stop plotting when lower than this (number times 10) dmax
        num_count = 30
        dynrange_log = np.log10(dynamic_range)
        d_min = (1.0 / float(dynamic_range)) * self.intensities.max()
        dmax_log10 = float(int(np.log10(self.d_max)))

        num_level_ticks = 6
        levels_ticks = [10**(dmax_log10-i) for i in range(num_level_ticks)]
        levels = []
        for i in range(num_count):
            levels.append(self.d_max * 10.0 ** (-1.0 * dynrange_log * (float(num_count - 1 - i) / (num_count - 1))))

        cont = plt.contourf(self.X, self.Y, self.intensities,
                            levels=levels, cmap='hot_r',
                            norm=colorbar_norm)

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")

        # formatting of colorbar tick labels
        import matplotlib.ticker as ticker
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        # https://stackoverflow.com/questions/25983218/scientific-notation-colorbar
        colorbar = plt.colorbar(cont, ticks=levels_ticks, format=ticker.FuncFormatter(fmt))

        # plt.xlabel(r'$\omega_1$')
        # plt.ylabel(r'$\omega_2$')
        xs = self.X[0], self.X[-1]
        ys = self.Y[0], self.Y[-1]

        title_type_dict = {(True, False): r'electrical anharmonicity $|\gamma^{[1,0]}|^2$ only',
                           (False, True): r'mechanical anharmonicity $|\gamma^{[0,1]}|^2$ only',
                           (True, True): r'both $|\gamma^{[1,0]}+\gamma^{[0,1]}|^2$'}

        nicetitle = f'{nametuple[2]}'
        plt.title(nicetitle)
        bbox_args = dict(boxstyle="round,pad=0.8", edgecolor='black', facecolor='lightgray')
        ax.annotate(text_under_the_figure, xy=(0.05, -0.11), xycoords='axes fraction',
                    ha="left", va="top", bbox=bbox_args, fontsize=12)
        plt.tight_layout()
        if to_save:
            plt.savefig(nametuple[0], dpi=self.dpi, format='svg')

        # import shutil
        # shutil.copy2(nametuple[0], '/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/svgs/'+nametuple[0])
        return fig

def read_csv_DB(filepath):
    """

    :param filepath:
    :return:
    """
    # Column names are:
    # code, method, basis_set, c4_ZMAT, c4_outfile_orig_hess,
    # c4_QUADRATURE, pkl_dimensionless, pkl_dipole,
    # c4_dipolexyz, pkl_polar, pkl_polar_raw, pkl_polar_data, c4_cubic,
    # c4_out, pkl_vibdata, g16_3quanta_full
    import pandas as pd
    database = pd.read_csv(filepath)
    # columnsDB = list(database.columns)
    # print('\ncolumnsDB\n', columnsDB)

    return database

def getting_files_DB(sourceProgram: str, printing: bool = False):
    """

    :param printing:
    :param sourceProgram:
    :return:
    """
    DB = read_csv_DB('/mnt/c/Users/vle014/Downloads/files_fram/files_database.csv')

    if sourceProgram == 'gaussian':
        filtered_df = DB.query('g16_3quanta_full.notna() and g16_3quanta_full != ""')
        selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'g16_3quanta_full']]
        return selected_columns_df

    else:
        # c4_ZMAT, c4_outfile_orig_hess,
        # c4_QUADRATURE, pkl_dimensionless, pkl_dipole,
        # c4_dipolexyz, pkl_polar, pkl_polar_raw, pkl_polar_data, c4_cubic,
        # c4_out, pkl_vibdata
        columns_to_check = ['c4_dipolexyz', 'pkl_polar', 'c4_cubic', 'c4_out']
        conditions = " and ".join([f"{col}.notna() and {col} != ''" for col in columns_to_check])
        filtered_df = DB.query(conditions)

        # filtered_df = DB.query('c4_dipolexyz.notna() and c4_dipolexyz != "" and pkl_polar.notna() and pkl_polar != "" ')
        if printing:
            selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'c4_out']]
        else:
            selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'c4_out', 'c4_cubic',
                                               'c4_dipolexyz', 'pkl_polar']]

        return selected_columns_df

def make_DatainputDict(sourceProgram: str, mol_tuple: tuple):
    """

    :param mol_tuple:
    :param sourceProgram:
    :return:
    """
    dataframe = getting_files_DB(sourceProgram)
    mol_code, method, basis = mol_tuple
    files_dict = {'mol_code': mol_code, 'method': method, 'basis': basis}

    narrow_df = dataframe.loc[(dataframe['code'] == mol_code)
                              & (dataframe['method'] == method)
                              & (dataframe['basis_set'] == basis)]
    if len(narrow_df) > 1:
        print('Something is wrong, more than one file found. First one is taken here.')

    if sourceProgram == 'gaussian':
        result = {'source': 'gaussian', 'type': 'log'}

        files_dict.update({'3quanta': narrow_df.iloc[0]['g16_3quanta_full'],
                           'log': narrow_df.iloc[0]['g16_3quanta_full']})
        result['files'] = files_dict
        return result

    elif sourceProgram == 'cfour':
        result = {'source': 'cfour', 'type': 'out'}
        files_dict.update({'out': narrow_df.iloc[0]['c4_out'], 'cubic': narrow_df.iloc[0]['c4_cubic'],
                           'dipolexyz': narrow_df.iloc[0]['c4_dipolexyz'][:-1], 'polar': narrow_df.iloc[0]['pkl_polar'],
                           'out_anharm_final': narrow_df.iloc[0]['c4_out'], 'polar_pkl': narrow_df.iloc[0]['pkl_polar']})
        result['files'] = files_dict
        return result


def get_abc_indices(number_ofIndices: int, number_ofFundamentals: int):
    """
    modes a, b, (c) - combinations of them in pairs (electric anharmonicity) or triplets (mechanical anharmonicity)
    :param number_ofIndices:
    :param number_ofFundamentals:
    :return:
    """
    return np.indices([number_ofFundamentals]*number_ofIndices).reshape(number_ofIndices, -1).T

def get_AlphaBetaGammaDelta_indices(num_f: int):
    """
    pol_g = orientationalaveraging.get_iso_f(num_f)
    pol_g is a list of lists of 2 lists where the second one is empty
          but first one contains the lists of interest

    :param num_f:
    :return: array_of_4greekIndices - an array of arrays of 4 greek indices for second hyperpolarizability :
             [alpha, beta, gamma, delta]
    """
    from wilson import orientationalaveraging
    pol_g = orientationalaveraging.get_iso_f(num_f)
    array_of_4greekIndices = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)
    return array_of_4greekIndices

def avrg_abc_tensor(formula: list[tuple[str, tuple[str]]], data: dict[str:np.ndarray], gammaCompsAll: np.array):
    """
    Calculate the averaging tensor for a given formula.
    Indices of the tensor are normal coordinates (NC) indices,
    and the shape of the tensor depends on the nature of the term that is being calculated.
    Shape of the averaging tensor for electrical anharmonicity terms is (n_NC, n_NC)
    Shape of the averaging tensor for mechanical anharmonicity terms is (n_NC, n_NC, n_NC)

    :param formula:
    :param data:
    :param gammaCompsAll:
    :return:
    """
    nmodes = data['mu_Q'].shape[0]

    if type(formula[-1]) == str:
        # True for mechanical anharmonicity terms
        formula = formula[:-1]

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

# function generator
def generate_resonances_functions(subscripts, fermi=None, margin=10.):
    m1n1m2n2 = [i.split(',') for i in subscripts]
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def function(w_all: dict, w1, w2, Gamma: float, abctuple: tuple[int, int, int], m1n1m2n2=m1n1m2n2, fermi=fermi):
        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all[('zero',)] = 0.

        wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')], key=int))
        wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')], key=int))# if len(m1n1m2n2[0][1].split('+')) > 1 else tuple([m1n1m2n2[0][1]])
        # print([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')])
        # print(m1n1m2n2[1][0])

        wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')], key=int)) if 'zero' not in m1n1m2n2[1][0].split('+') else tuple([m1n1m2n2[1][0]])
        wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')], key=int))

        if fermi is None:
            return np.where(w2-margin > w1, 1 / (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1])
                                                 + rec_cm2rec_s(w1) - rec_cm2rec_s(w2)
                                                 - 1j * Gamma) / (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2])
                                                                  + rec_cm2rec_s(w1) - 1j * Gamma), 0.)

        else:
            w_fr11 = tuple(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')], key=int))
            w_fr21 = tuple(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')], key=int)) if 'zero' not in fermi[0][1].split('+') else tuple([fermi[0][1]])

            w_fr12 = tuple(sorted([str(dictabc[i]) for i in fermi[1][0].split('+')], key=int))
            w_fr22 = tuple(sorted([str(dictabc[i]) for i in fermi[1][1].split('+')], key=int)) if 'zero' not in fermi[1][1].split('+') else tuple([fermi[1][1]])

            t1 = rec_cm2rec_s(w_all[wm1]-w_all[wn1]+w1-w2) - 1j * Gamma
            t2 = rec_cm2rec_s(w_all[wm2]-w_all[wn2]+w1) - 1j * Gamma
            t3 = rec_cm2rec_s(w_all[w_fr11]-w_all[w_fr21])
            t4 = rec_cm2rec_s(w_all[w_fr12]-w_all[w_fr22])

            sumfrac = (1 / t3 + 1 / t4)
            # with open('./fermi', 'a') as file1:
            #     file1.write('\n==============================\n')
            #     file1.write(f'{abctuple}\n{w_fr11} {w_fr21} {w_fr12} {w_fr22}\n{fermi}\n')
            #     file1.writelines(str(t3)+'\n')
            #     file1.writelines(str(t4)+'\n')
            #     file1.writelines(str(sumfrac) + '\n')

            # with open('./fermi_other', 'a') as file1:
            #     file1.write('\n==============================\n')
            #     file1.write(f'{abctuple}\n{m1n1m2n2}\n')
            #     file1.writelines(str(rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]))+'\n')
            #     file1.writelines(str( rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) )+'\n')
            #     file1.writelines(str((1 / t1 / t2)) + '\n')

            return (1 / t1 / t2) * sumfrac

    return function

def get_resonances(electrical_terms_dict, mechanical_terms_dict, w_all, margin=10.):
    """
    subscripts /and fermi
    :param subscripts:
    :param fermi:
    :param margin:
    :return:
    """
    nfunds = len([i for i in w_all if len(i)==1])

    for elTerm in electrical_terms_dict:
        subscripts = electrical_terms_dict[elTerm]
        m1n1m2n2 = [i.split(',') for i in subscripts]


    for mechTerm in mechanical_terms_dict:
        subscripts, fermi = mechanical_terms_dict[mechTerm]
        fermi = [i.split(',') for i in fermi]

    def function(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi):

        resonances_tensor = np.zeros((len(abctuple), len(abctuple)))

        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all[('zero',)] = 0.

        wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')], key=int))
        wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')], key=int))
        # if len(m1n1m2n2[0][1].split('+')) > 1 else tuple([m1n1m2n2[0][1]])

        wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')], key=int)) \
            if 'zero' not in m1n1m2n2[1][0].split('+') else tuple([m1n1m2n2[1][0]])
        wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')], key=int))

        if fermi is None:
            return np.where(w2-margin > w1, 1 / (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1])
                                                 + rec_cm2rec_s(w1) - rec_cm2rec_s(w2)
                                                 - 1j * Gamma) / (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2])
                                                                  + rec_cm2rec_s(w1) - 1j * Gamma), 0.)

        else:
            w_fr11 = tuple(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')], key=int))
            w_fr21 = tuple(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')], key=int)) \
                if 'zero' not in fermi[0][1].split('+') else tuple([fermi[0][1]])

            w_fr12 = tuple(sorted([str(dictabc[i]) for i in fermi[1][0].split('+')], key=int))
            w_fr22 = tuple(sorted([str(dictabc[i]) for i in fermi[1][1].split('+')], key=int)) \
                if 'zero' not in fermi[1][1].split('+') else tuple([fermi[1][1]])

            t1 = rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]) + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma
            t2 = rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma
            t3 = rec_cm2rec_s(w_all[w_fr11]) - rec_cm2rec_s(w_all[w_fr21])
            t4 = rec_cm2rec_s(w_all[w_fr12]) - rec_cm2rec_s(w_all[w_fr22])

            sumfrac = (1 / t3 + 1 / t4)

            return (1 / t1 / t2) * sumfrac

    return function

def printT(tensor):
    import pandas as pd
    pd.set_option('display.float_format', '{:.10f}'.format)

    ndims = len(tensor.shape)

    # mu_Q
    if ndims == 2:
        column_names = ['x', 'y', 'z']
        row_names    = [f'{i}' for i in range(tensor.shape[0])]
        df = pd.DataFrame(tensor, columns=column_names)#, index=row_names)
        df.insert(0, "I", row_names, allow_duplicates=True)
        df.insert(1, "", ['|']*len(row_names), allow_duplicates=True)

        # print(df)
        print(df.to_string(index=False))

    elif ndims == 3:
        # F_abc
        if tensor.shape[0] == tensor.shape[1] == tensor.shape[2]:
            row_names = [f'K {i}' for i in range(tensor.shape[0])]
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)#, index=row_names)
            df.insert(0, "I", ['0']*len(row_names), allow_duplicates=True)
            df.insert(1, "J", indx, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(row_names), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)#, index=row_names)
                dfi.insert(0, "I", [f'{ii+1}']*len(row_names), allow_duplicates=True)
                dfi.insert(1, "J", indx, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(row_names), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

        # mu_QQ
        elif tensor.shape[0] == tensor.shape[1] != tensor.shape[2]:
            row_names = ['x', 'y', 'z']
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)  # , index=row_names)
            df.insert(0, "I", ['0'] * len(indx), allow_duplicates=True)
            df.insert(1, "J", indx, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)  # , index=row_names)
                dfi.insert(0, "I", [f'{ii + 1}'] * len(indx), allow_duplicates=True)
                dfi.insert(1, "J", indx, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

        # alpha_Q
        elif tensor.shape[0] != tensor.shape[1] == tensor.shape[2]:
            row_names = ['x', 'y', 'z']
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)  # , index=row_names)
            df.insert(0, "I", ['0'] * len(indx), allow_duplicates=True)
            df.insert(1, "", row_names, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)  # , index=row_names)
                dfi.insert(0, "I", [f'{ii + 1}'] * len(indx), allow_duplicates=True)
                dfi.insert(1, "", row_names, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

    # alpha_QQ
    elif ndims == 4:
        listdf = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[0]):
                row_names = ['x', 'y', 'z']
                df = pd.DataFrame(tensor[i, j], columns=row_names)  # , index=row_names)
                df.insert(0, "I", [f'{i}'] * 3, allow_duplicates=True)
                df.insert(1, "J", [f'{j}'] * 3, allow_duplicates=True)
                df.insert(2, "", row_names, allow_duplicates=True)
                df.insert(3, "", ['|'] * 3, allow_duplicates=True)

                listdf.append(df)

        dfs = pd.concat(listdf, ignore_index=True)
        n = tensor.shape[2]  # chunk row size
        list_df = [dfs[i:i + n] for i in range(0, dfs.shape[0], n)]

        for dframe in list_df:
            print(dframe.to_string(index=False))

    else:
        print(f"Dimension of the property in not 2, 3 or 4, it's {ndims}")


def printed2DIRtensors(setup: SpectrumEVV):

    ders = setup.deriv_data
    print('\nFundamental frequencies (anharmonic):', list(setup.fundamentals.values()))
    print('Fundamental frequencies (harmonic)  :', list(setup.fundamentals_harmonic.values()), '\n')

    print('All frequencies (anharmonic)  :', setup.all_states, '\n')

    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('==================================\n')


class MockDataParser:

    def __init__(self):
        pass
