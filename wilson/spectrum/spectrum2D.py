import copy
import time
from datetime import timedelta
from typing import Callable

import numpy as np

from .averaging import get_AlphaBetaGammaDelta_indices
from .tools import convNu2Ene, avrg_abc_tensor

import itertools
def combinations_with_permutations(iterable, k):
    return (comb for comb in itertools.product(iterable, repeat=k))

numcombperm = lambda n, k: n**k


class Spectrum2D:
    """
    Contains settings for current EVV derivations.
    Will hold all the necessary data and the terms for the evaluation of the intensities;
    from this general data more specifics can be extracted (data_analysis module, ...)

    input_data_info is a dictionary:

    result = {'source': 'cfour', 'type': 'out', 'files': c4_files_dict, 'source': 'cfour', 'type': 'out'}
    c4_files_dict = {'mol_code': mol_code, 'method': method, 'basis': basis,
                                   'out': c4_out(anharmonic equil.),
                                   'cubic': c4_cubic',
                                   'dipolexyz': c4_dipolexyz (location/name w/o xyz part),
                                   'polar': pkl_polar,
                                   'out_anharm_final': c4_out,
                                   'polar_pkl': pkl_polar}

    result = {'source': 'cfour', 'type': 'out', 'files': c4_files_dict, 'source': 'gaussian', 'type': 'log'}
        a helper can be used - DataVault.make_DatainputDict with specific choice of molecule calculation
    """

    def __init__(self, w1=None, w2=None, print_level=0):
        """
        TODO: remove w1 and w2 from init here; clean up init
        """
        self.print_level = print_level

        if w2 is None:
            w2 = []
        if w1 is None:
            w1 = []
        if type(w1)==list or type(w2)==list:
            self.w1, self.w2 = np.array(w1), np.array(w2)
        else:
            self.w1, self.w2 = w1, w2

        # define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        self.shape2d = self.w1_mesh.shape

        # initialized final spectrum pixels
        self.intensities_grid = np.zeros(self.shape2d, dtype='complex64')

        self.resonances_bank = {}

        self.Gamma = None
        self.Gamma_rc = None
        self.diagonal_margin_rc = None

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        self.deriv_data = None
        self.corrected_levels = None

        self.mechab = False

        self.nmodes = None
        self.nmodes_original = None


    def get_derived_terms_evv(self):
        """
        Currently available for selection EVV terms
        """
        # Terms in the expressions
        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        # mu_Q, alpha_Q - for all 6 terms
        self.allterms_str = {0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)))),
                             1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',)))),
                             2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.)),
                             3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.)),
                             4: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5)),
                             5: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5)),
                             6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5)),
                             7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5))}


    def load_data(self, parserObj,
                  list2exclude = None,
                  vpt2=False, vpt2settings=None):
        """
        Loading the data from a parser object/DataVault
            with the sources given to it

        anharmonic_type options:
            'VPT2'   - don't do_res, don't do_var
            'DVPT2'  - do_res, don't do_var
            'GVPT2'  - do_res, do_var

        default is vpt2settings = {'anharmonic_type': 'VPT2'}
        """
        # TODO - make it more flexible, give an option to supply files
        # parserObj = parser(self.input_data_info)
        self.vpt2 = vpt2

        if vpt2settings is None:
            vpt2settings = {'anharmonic_type': 'VPT2'}

        if list2exclude is None:
            list2exclude = []

        parserObj.getData()
        self.parserObj = parserObj

        self.fundamentals = parserObj.fundamentals_anharmonic_str
        self.fundamentals_harmonic = parserObj.fundamentals_harmonic_str
        self.all_states = parserObj.anharmonic_states
        self.all_states_harmonic = parserObj.harmonic_states

        self.nmodes = len(self.fundamentals)
        self.nmodes_original = len(self.fundamentals)
        self.list2exclude = list2exclude
        if list2exclude:
            print('self.fundamentals', self.fundamentals)
            # exclude indices
            self.mode_indices = [i for i in np.arange(self.nmodes) if i not in list2exclude]
            self.nmodes -= len(list2exclude)
        else:
            self.mode_indices = [i for i in np.arange(self.nmodes)]

        ddata = [parserObj.dipole_first_derivatives,
                 parserObj.dipole_second_derivatives,
                 parserObj.polarizability_first_derivatives,
                 parserObj.polarizability_second_derivatives,
                 parserObj.cubic_force_constants]
        self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))
        # 'mu_Q',  'mu_QQ',  'alpha_Q', 'alpha_QQ', 'F_abc'
        # (6, 3)  (6, 6, 3)  (6, 3, 3) (6, 6, 3, 3) (6, 6, 6) if nmodes = 6

        self.normal_modes = parserObj.normal_modes

        if self.vpt2:
            if parserObj.DD11 or parserObj.DD13 or parserObj.DD22:
                print("Warning: found Darling-Dennison resonances_args in data:")
                print(f"DD 1-1: {parserObj.DD11}")
                print(f"DD 2-2: {parserObj.DD22}")
                print(f"DD 1-3: {parserObj.DD13}")

            one = {i: self.all_states[i] for i in self.all_states if len(i) == 1}
            two = {i: self.all_states[i] for i in self.all_states if len(i) == 2}
            if self.print_level == 1:
                print('\nOriginal anharm corrected:')
                print(dict(sorted(one.items())))
                print(dict(sorted(two.items())), '\n')

            cff_cm_1 = parserObj.cubic_cm_1
            qff_cm_1 = parserObj.quartic_cm_1
            rot_c, cor_c = parserObj.rotational_constant, parserObj.coriolis_constant

            from .vpt2 import anharm_corr_energiesVPT2
            # corrected_levels : funds, over2q, combo2q, over3q, combo3q
            self.corrected_levels = anharm_corr_energiesVPT2(list(self.fundamentals_harmonic.values()),
                                                             cff_cm_1, qff_cm_1, rot_c, cor_c,
                                                             vpt2settings['anharmonic_type'])
            self.all_states_corr = {}
            for i in range(len(self.fundamentals)):
                self.all_states_corr[(str(i),)] = self.corrected_levels[0][i]

                for j in range(i+1):
                    if i==j:
                        self.all_states_corr[tuple([str(i), str(i)])] = self.corrected_levels[1][i]
                    else:
                        self.all_states_corr[tuple([str(el) for el in sorted([i, j])])] = self.corrected_levels[2][i, j]

                    for k in range(len(self.fundamentals)):
                        if i==j==k:
                            self.all_states_corr[tuple([str(i), str(i), str(i)])] = self.corrected_levels[3][i]
                        else:
                            key = tuple([str(el) for el in sorted([i, j, k])])
                            if key not in self.all_states_corr:
                                if self.corrected_levels[4][i, j, k]!=0.:
                                    self.all_states_corr[tuple([str(el) for el in sorted([i, j, k])])] = self.corrected_levels[4][i, j, k]

            self.all_states = copy.deepcopy(self.all_states_corr)
            one = {i: self.all_states[i] for i in self.all_states if len(i) == 1}
            two = {i: self.all_states[i] for i in self.all_states if len(i) == 2}
            if self.print_level == 1:
                print('\nGVPT2 anharm corrected:')
                print(dict(sorted(one.items())))
                print(dict(sorted(two.items())), '\n')


    def change_idx_modes(self, new_idx_dict):
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
        new_list = []

        # upd self.all_states
        for oldkey, val in self.all_states.items():
            # newkey = tuple(sorted([str(new_idx_dict[int(i)]) for i in oldkey]))
            newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey])])
            new_dict1[newkey] = val

        sorted_keys = sorted(new_dict1.keys(), key=lambda x: tuple(map(int, x)))
        new_dict1_sort = {key: new_dict1[key] for key in sorted_keys}
        self.all_states = new_dict1_sort

        # upd self.all_states_harmonic
        for oldkey, val in self.all_states_harmonic.items():
            # newkey = tuple(sorted([str(new_idx_dict[int(i)]) for i in oldkey]))
            newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey])])
            new_dict2[newkey] = val

        sorted_keys = sorted(new_dict2.keys(), key=lambda x: tuple(map(int, x)))
        new_dict2_sort = {key: new_dict2[key] for key in sorted_keys}

        self.all_states_harmonic = new_dict2_sort

        # upd self.fundamentals_harmonic
        for oldkey, val in self.fundamentals_harmonic.items():
            newkey = str(new_idx_dict[int(oldkey)])
            new_dict3[newkey] = val
        sortKeys = list(new_dict3.keys())
        sortKeys.sort()
        new_dict3_sort = {i: new_dict3[i] for i in sortKeys}

        self.fundamentals_harmonic = new_dict3_sort

        # for old in self.list2exclude:
        #     new_list.append(new_idx_dict[old])
        # self.list2exclude = new_list
        if self.list2exclude:
            self.mode_indices = [i for i in np.arange(self.nmodes) if i not in self.list2exclude]
            self.nmodes -= len(self.list2exclude)
        else:
            self.mode_indices = [i for i in np.arange(self.nmodes)]

        newmu1 = np.zeros_like(self.deriv_data['mu_Q'])
        newmu2 = np.zeros_like(self.deriv_data['mu_QQ'])
        newalpha1 = np.zeros_like(self.deriv_data['alpha_Q'])
        newalpha2 = np.zeros_like(self.deriv_data['alpha_QQ'])
        newF = np.zeros_like(self.deriv_data['F_abc'])

        for oldkey, newkey in new_idx_dict.items():
            newmu1[newkey, :] = self.deriv_data['mu_Q'][oldkey, :]
            newalpha1[newkey, :, :] = self.deriv_data['alpha_Q'][oldkey, :, :]

        new_idx_dict_2d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                new_idx_dict_2d[(old_i, old_j)] = (new_i, new_j)

        for (old_i, old_j), (new_i, new_j) in new_idx_dict_2d.items():
            newmu2[new_i, new_j, :] = self.deriv_data['mu_QQ'][old_i, old_j, :]
            newalpha2[new_i, new_j, :, :] = self.deriv_data['alpha_QQ'][old_i, old_j, :, :]

        new_idx_dict_3d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                for old_k, new_k in new_idx_dict.items():
                    new_idx_dict_3d[(old_i, old_j, old_k)] = (new_i, new_j, new_k)

        for (old_i, old_j, old_k), (new_i, new_j, new_k) in new_idx_dict_3d.items():
            newF[new_i, new_j, new_k] = self.deriv_data['F_abc'][old_i, old_j, old_k]


        ddata = [newmu1, newmu2, newalpha1, newalpha2, newF]
        self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))



    def set_spectrum_settings(self, Gamma_rc: float, diag_margin_rc: float = 10., vib_levels_harmonic: bool =True):
        """Settings to be set before computing the intensities.
        rc - reciprocal centimeter.

        vib_levels_harmonic - weather to use harmonic levels for resonance terms
                (useful for the investigations of Fermi resonances_args? or other)
        """
        self.Gamma_rc = Gamma_rc
        self.Gamma = convNu2Ene(Gamma_rc)
        # margin for higher diagonal, to not show/compute data to close to the diagonal
        self.diagonal_margin_rc = diag_margin_rc
        self.convert_units()
        self.vib_levels_harmonic = vib_levels_harmonic
        print(f'\nUsed vibrational energy levels are harmonic? - {self.vib_levels_harmonic}')

    def convert_units(self):
        """
        Eh - Hartree unit
        convNu2Ene converts from wavenumber to Hartree
        """
        self.all_states_harmonic_Eh = {k: convNu2Ene(v) for k, v in self.all_states_harmonic.items()}
        self.all_states_harmonic_Eh[('zero',)] = 0.
        self.all_states_Eh = {k: convNu2Ene(v) for k, v in self.all_states.items()}
        self.all_states_Eh[('zero',)] = 0.

        self.w1_mesh_Eh, self.w2_mesh_Eh = convNu2Ene(self.w1_mesh), convNu2Ene(self.w2_mesh)
        self.diagonal_margin_Eh = convNu2Ene(self.diagonal_margin_rc)


    def add_terms(self, electrical_terms_selection: list, mechanical_terms_selection: list):
        """Creating functions for computing the expressions for mechanical and electrical anharmonicities.
            Different functions because of the difference in terms.

        The terms available for selection are set with self.get_derived_terms_evv() and are currently for EVV experiment
        """

        # setting up terms available for selection (all EVV terms now)
        self.get_derived_terms_evv()

        # now used in the analysis
        self.e_selected, self.m_selected = electrical_terms_selection, mechanical_terms_selection
        self.selection = electrical_terms_selection + mechanical_terms_selection


        self.avrg_tensors_dict = {i: avrg_abc_tensor(self.allterms_str[i][1], self.deriv_data, self.gammaCompsAll)
                                      for i in self.selection}
        self.allfunc_dict = {i: self.generate_resonances_functions(self.allterms_str[i][0][0], self.allterms_str[i][0][1]) for i in self.selection}
        self.nmodes = len(self.fundamentals)


    def precalculate(self):

        # self.res_dict
        self.precalc_locations()

        x, y = zip(*([i[0] for i in self.res_dict[('a+b,a', 'zero,a')]]+[i[0] for i in self.res_dict[('b,a', 'zero,a')]]))
        x = np.array(x)
        y = np.array(y)

        data = {'omega1': np.array([i[0][0] for i in self.res_dict[('a+b,a', 'zero,a')]]+[i[0][0] for i in self.res_dict[('b,a', 'zero,a')]]),
                'omega2': np.array([i[0 ][1] for i in self.res_dict[('a+b,a', 'zero,a')]]+[i[0][1] for i in self.res_dict[('b,a', 'zero,a')]]),
                'a': [int(i[1][0]) for i in self.res_dict[('a+b,a', 'zero,a')]]+[int(i[1][0]) for i in self.res_dict[('b,a', 'zero,a')]],
                'b': [int(i[1][1]) for i in self.res_dict[('a+b,a', 'zero,a')]]+[int(i[1][1]) for i in self.res_dict[('b,a', 'zero,a')]],
                'type': ['a+b,a; zero,a' for i in self.res_dict[('a+b,a', 'zero,a')]] + ['b,a; zero,a' for i in self.res_dict[('b,a', 'zero,a')]]
                }

        # self.resonances_args
        self.precalc_intensities(convNu2Ene(x), convNu2Ene(y))
        x_shape = x.shape

        # Y axis limit for w1mw2
        list_of_sets = [v for k, v in self.res_dict.items()]
        union_result = set()
        for s in list_of_sets:
            union_result = union_result.union(s)

        self.maxX = np.max(np.array([i[0][0] for i in union_result]))
        self.maxY = np.max(np.array([i[0][1] for i in union_result]))
        self.maxYX = np.max(np.array([i[0][1] - i[0][0] for i in union_result]))

        return data, x_shape


    def preview_spectrum(self, w=1100, h=700):

        # dictionary with data
        data, x_shape = self.precalculate()
        # is a 1d array of intensities
        # Z = self.intensity_both(selectionCond=None, shape2d=x.shape, resonances_args=self.resonances_args, mechel_contrib=True)
        if self.vpt2:
            prefix = 'vpt2_'+self.parserObj.program+'_'+self.parserObj.molecule+'_'+str(self.selection)
        else:
            prefix = self.parserObj.program+'_'+self.parserObj.molecule+'_'+str(self.selection)

        typedict = {'a+b,a; zero,a': [0, 2, 4, 5], 'b,a; zero,a': [1, 3, 6, 7]}

        for t in self.e_selected:
            data[f'abs {t}'] = np.zeros(x_shape)
        for t in self.m_selected:
            data[f'abs {t}'] = np.zeros(x_shape)

        data['gamma'] = np.array([])
        # intensity_clean = np.zeros(x_shape)
        gammaabs_clean = np.zeros(x_shape, dtype='complex64')
        for i in range(len(data['omega1'])):
            # single contribution
            int_ab = self.intensity_both(selectionCond=None, shape2d=x_shape,
                                resonances_args=self.resonances_args,
                                selected_ab=[(data['a'][i], data['b'][i])],
                                mechel_contrib=True)

            termtype = data['type'][i]

            for t in self.selection:
                if t in [0,1]:
                    if t in typedict[termtype]:
                        gammaabs_clean[i] += self.el_ab[t][(data['a'][i], data['b'][i])][i]
                elif t in [2,3,4,5,6,7]:
                    if t in typedict[termtype]:
                        gammaabs_clean[i] += self.mech_ab[t][(data['a'][i], data['b'][i])][i]

            data['gamma'] = np.append(data['gamma'], int_ab[i])

            if termtype == 'a+b,a; zero,a':
                for t in self.selection:
                    if t in [0,]:
                        data[f'abs {t}'][i] = abs(self.el_ab[t][(data['a'][i], data['b'][i])][i])
                    elif t in [ 2, 4, 5]:
                        data[f'abs {t}'][i] = abs(self.mech_ab[t][(data['a'][i], data['b'][i])][i])

            elif termtype == 'b,a; zero,a':
                for t in self.selection:
                    if t in [1, ]:
                        data[f'abs {t}'][i] = abs(self.el[t][i])

                    elif t in [3, 6, 7]:
                        data[f'abs {t}'][i] = abs(self.mech[t][i])


        data['Intensity'] = abs(data['gamma'])**2
        data['Intensity_clean'] = abs(gammaabs_clean)**2
        data['abs gamma_clean'] = abs(gammaabs_clean)

        data['log10(Intensity)'] = np.where(data['gamma']!=0., np.log10(abs(data['gamma'])**2), 0.)
        # print(abs(data['gamma']))
        # print(data['log10(Intensity)'])
        data['abs el'] = abs(sum([v for k, v in self.el.items()]))
        data['abs mech'] = abs(sum([v for k, v in self.mech.items()]))


        for t in self.el_ab:

            data[f'factor {t}'] = [self.avrg_tensors_dict[t][data['a'][i], data['b'][i]] / self.prefac_2d[data['a'][i], data['b'][i]] / 24.
                                   if data[f'abs {t}'][i]!=0. else 0. for i in range(len(data['omega1']))]

        for t in self.mech_ab:

            data[f'factor {t}'] = [self.comb_fac_dict[self.allterms_str[t]][data['a'][i], data['b'][i]]
                                   if data[f'abs {t}'][i]!=0. else 0. for i in range(len(data['omega1']))]

        import pandas as pd
        spectrumDF = pd.DataFrame(data)
        spectrumDF['w2mw1'] = spectrumDF['omega2'] - spectrumDF['omega1']

        spectrumDF['factor 0/2'] = spectrumDF['factor 0']/spectrumDF['factor 2']
        spectrumDF['factor 1/3'] = spectrumDF['factor 1']/spectrumDF['factor 3']

        spectrumDF['factor 0/2 sign'] = np.log(abs(spectrumDF['factor 0'])/abs(spectrumDF['factor 2']))
        spectrumDF['factor 1/3 sign'] = np.log(abs(spectrumDF['factor 1'])/abs(spectrumDF['factor 3']))
        spectrumDF["factors sign"] = spectrumDF["factor 0/2 sign"].fillna(0) + spectrumDF["factor 1/3 sign"].fillna(0)

        import altair as alt
        import math
        threshold_slider = alt.binding_range(min=math.floor(min(data['log10(Intensity)']))-1,
                                             max=math.ceil(max(data['log10(Intensity)']))+1,
                                             step=0.1, name='Threshold:')
        threshold_select = alt.selection_point(fields=['log10(Intensity)'], bind=threshold_slider)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        title = alt.TitleParams(f'Terms: {tuple(self.selection)}', anchor='middle')
        chart = alt.Chart(spectrumDF, title=title).mark_circle().encode(
            x='omega1',
            y='w2mw1',
            color=alt.condition(
                alt.datum['log10(Intensity)'] > alt.expr.if_(threshold_select, threshold_select['log10(Intensity)'], 0),
                alt.value('steelblue'),  # Color for points above the threshold
                alt.value('lightgray')  # Color for points below the threshold
            ),
            tooltip=[alt.Tooltip('omega1', format='.2f'),
                     alt.Tooltip('omega2', format='.2f'),
                     alt.Tooltip('w2mw1', format='.2f'),
                     alt.Tooltip('log10(Intensity)', format='.4f'),
                     alt.Tooltip('Intensity', format='.4e'),
                     # alt.Tooltip('relative el/mech', format='.5f'),
                     'a', 'b', 'type'
                     ],
            opacity=alt.condition(
                alt.datum['log10(Intensity)'] > alt.expr.if_(threshold_select, threshold_select['log10(Intensity)'], 0),
                alt.value(1),  # Full opacity for points above the threshold
                alt.value(0.35)  # No opacity for points below the threshold
            )
        ).add_selection(
    threshold_select
).properties(
                    width=w,
                    height=h
                    ).interactive()

        # alt.renderers.enable("browser")
        alt.renderers.enable("jupyterlab")
        # chart.save(prefix+'_resints.html', inline=True, scale_factor=2)

        return chart, spectrumDF


    def precalc_locations(self):

        if self.vib_levels_harmonic:
            # vib_ene_levels = self.all_states_harmonic_Eh
            vib_ene_levels_rc = self.all_states_harmonic
        else:
            # vib_ene_levels = self.all_states_Eh
            vib_ene_levels_rc = self.all_states

        self.res_dict = {}

        w_apbbma_rc = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma_rc = np.zeros((self.nmodes_original, self.nmodes_original))
        # print(vib_ene_levels_rc)

        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if a in self.mode_indices and b in self.mode_indices:
                w_apbbma_rc[a, b] = vib_ene_levels_rc[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels_rc[
                    tuple([str(a)])]
                w_bma_rc[a, b] = vib_ene_levels_rc[tuple([str(b)])] - vib_ene_levels_rc[tuple([str(a)])]

        za_rc = np.array([-vib_ene_levels_rc[tuple([str(k)])] for k in range(self.nmodes_original)])
        za_rc = np.tile(za_rc, self.nmodes_original).reshape(self.nmodes_original, -1).T

        self.res_dict[('a+b,a', 'zero,a')] = []
        self.res_dict[('b,a', 'zero,a')] = []

        self.inwindow = {}
        self.inwindow[('a+b,a', 'zero,a')] = {}
        self.inwindow[('b,a', 'zero,a')] = {}

        mw1, Mw1 = self.w1_mesh.min(), self.w1_mesh.max()
        mw2, Mw2 = self.w2_mesh.min(), self.w2_mesh.max()

        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if a in self.mode_indices and b in self.mode_indices:
                self.res_dict[('a+b,a', 'zero,a')].append(((-za_rc[a, b],
                                                           w_apbbma_rc[a, b] - za_rc[a, b]), (a, b)))
                self.res_dict[('b,a', 'zero,a')].append(((-za_rc[a, b],
                                                         w_bma_rc[a, b] - za_rc[a, b]), (a, b)))

                # will collect those outside the window (with margin)
                margin = 0.
                if not (mw1+margin < -za_rc[a, b] < Mw1-margin) and not (mw2+margin < w_apbbma_rc[a, b] - za_rc[a, b] < Mw2-margin):
                    self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_apbbma_rc[a, b] - za_rc[a, b])
                if not (mw1+margin < -za_rc[a, b] < Mw1-margin) and not (mw2+margin < w_bma_rc[a, b] - za_rc[a, b] < Mw2-margin):
                    self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_bma_rc[a, b] - za_rc[a, b])

        # omega2>omega1 condition
        self.res_dict[('a+b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])
        self.res_dict[('b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])



    def precalc_intensities(self, w1_mesh_Eh=None, w2_mesh_Eh=None):

        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
            # vib_ene_levels_rc = self.all_states_harmonic
        else:
            vib_ene_levels = self.all_states_Eh
            # vib_ene_levels_rc = self.all_states

        self.prefac_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.prefac_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])

        self.resonancesTypes = [(-1, 2), (-1,)]

        if w1_mesh_Eh is None and w2_mesh_Eh is None:
            self.axes = {1: self.w1_mesh_Eh, 2: self.w2_mesh_Eh}
        else:
            self.axes = {1: w1_mesh_Eh, 2: w2_mesh_Eh}

        self.w1w2Condition = self.axes[2] - self.diagonal_margin_Eh > self.axes[1]
        # self.w1w2Condition = np.ones(self.w1_mesh_Eh.shape, dtype=bool)

        # [-1, 2] and [-1] types of terms: w1-w2 or w1, without w_{m,n}
        # these 2d arrays will be added to combinations of wm and wn when looped over combinations of a, b, (c)
        self.resonances_args = {}
        for typelist in self.resonancesTypes:
            self.resonances_args[typelist] = (-1) * sum([np.sign(ix) * np.where(self.w1w2Condition,
                                                                                self.axes[abs(ix)], 0) for ix in typelist]) - 1j * self.Gamma

        w_apbbma = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma = np.zeros((self.nmodes_original, self.nmodes_original))
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if a in self.mode_indices and b in self.mode_indices:
                w_apbbma[a, b] = vib_ene_levels[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels[tuple([str(a)])]
                w_bma[a, b] = vib_ene_levels[tuple([str(b)])] - vib_ene_levels[tuple([str(a)])]

        za = np.array([-vib_ene_levels[tuple([str(k)])] for k in range(self.nmodes_original)])
        self.w_mn_dict = {'a+b,a': w_apbbma, 'b,a': w_bma, 'c,a': w_bma,
                          'zero,a': np.tile(za, self.nmodes_original).reshape(self.nmodes_original, -1).T}

        if self.m_selected:
            st = time.time()
            # setting up a dict for combined mech factors - for each selected mech term
            self.comb_fac_dict = {}
            for key in self.m_selected:
                self.comb_fac_dict[self.allterms_str[key]] = np.zeros((self.nmodes_original, self.nmodes_original))

            # computing combined mech factors - summed over c for each a,b
            for ab in combinations_with_permutations(self.mode_indices, 2):
                a, b = ab
                if a in self.mode_indices and b in self.mode_indices:
                    for key in self.m_selected:
                        self.comb_fac_dict[self.allterms_str[key]][a,b] = self.compute_mech_factors(a, b)[key]
            # print(self.comb_fac_dict)
            elapsed_time = time.time() - st
            print('self.comb_fac_dict collected:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def precalculate_parts(self, *,
                           preview=False,
                           screenmodeswindow=True):
        """
        Precalculate some parts:
            factors (1/wa/wb/wc);
            resonance terms (wmn[-1,2], wmn[-1]);
            diff terms (wmn)
        """
        st0 = time.time()

        # self.res_dict
        self.precalc_locations()
        # self.resonances_args
        self.precalc_intensities()

        # used in get_gamma_el
        self.screenmodeswindow = screenmodeswindow

        # collect resonances below diagonal (without margin)
        fromdiagonal = {('a+b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] >= i[0][1]]),
                        ('b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] >= i[0][1]])}
        # omega2>omega1 condition
        self.res_dict[('a+b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])
        self.res_dict[('b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])


        if preview:

            if self.vpt2:
                prefix = 'vpt2_'+self.parserObj.program+'_'+self.parserObj.molecule
            else:
                prefix = self.parserObj.program+'_'+self.parserObj.molecule

            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 16))
            x, y = zip(*(set([i[0] for i in self.res_dict[('a+b,a', 'zero,a')]])|set([i[0] for i in self.res_dict[('b,a', 'zero,a')]])))
            x1, y2 = zip(*(fromdiagonal[('a+b,a', 'zero,a')]|fromdiagonal[('b,a', 'zero,a')]))

            plt.scatter(x, y)
            plt.scatter(x1, y2, color='r')
            plt.plot(x,x, label=f'X=Y, diag margin {self.diagonal_margin_rc} cm-1', color='g')

            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            x_ticks = np.arange(x_min - (x_min % 100), x_max + 100, 100)
            y_ticks = np.arange(y_min - (y_min % 100), y_max + 100, 100)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            plt.legend()
            plt.savefig(prefix+'_resloc.svg', format='svg')
            plt.cla()

            x = np.array(x)
            y = np.array(y)
            plt.scatter(x, tuple(np.array(y)-np.array(x)))
            plt.savefig(prefix+'_resloc_w1mw2.svg', format='svg')
            plt.cla()

            X, Y = np.meshgrid(x, y)
            Z = self.intensity_both(selectionCond=None, shape2d=X.shape)
            plt.scatter(X.flatten(), Y.flatten(), c=Z.flatten(), cmap='viridis', marker='o')
            plt.savefig(prefix+'_resints.svg', format='svg')

            exit()

        elapsed_time = time.time() - st0
        print('Precalculate full:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    def locate_on_big_grid(self, seed, radius):
        """
        Find corner points of the small square
        """
        stepX = self.w1[1]-self.w1[0]
        stepY = self.w2[1]-self.w2[0]

        # indices of grid point closest to resonance point
        closestXind = round((seed[0][0]-np.min(self.w1))/stepX)
        closestYind = round((seed[0][1]-np.min(self.w2))/stepY)

        # number of steps to reach radius distance (rounded)
        radiusIndX = round(radius/stepX)
        radiusIndY = round(radius/stepY)

        # start-end indices for smaller grid fo X
        endIndX = closestXind+radiusIndX
        strIndX = closestXind-radiusIndX

        # start-end indices for smaller grid fo Y
        endIndY = closestYind + radiusIndY
        strIndY = closestYind - radiusIndY

        # corrections for grid boundaries
        if endIndX>self.w1.shape[0]:
            endIndX = self.w1.shape[0]-1
        if endIndY>self.w2.shape[0]:
            endIndY = self.w2.shape[0]-1
        if strIndX<0:
            strIndX = 0
        if strIndY<0:
            strIndY = 0

        return strIndX, endIndX + 1, strIndY, endIndY + 1

    def find_all_grids(self, radius_rc):
        """
        Find all small squares around each resonance point
        """
        st = time.time()

        allRes = list(self.res_dict[('a+b,a', 'zero,a')] | self.res_dict[('b,a', 'zero,a')])
        resGridsDict = {}

        w1grid_Ha = self.w1_mesh_Eh
        w2grid_Ha = self.w2_mesh_Eh
        w1grid = self.w1_mesh
        w2grid = self.w2_mesh

        for seed in allRes:
            x1, x2, y1, y2 = self.locate_on_big_grid(seed, radius_rc)
            cutout_w1 = w1grid[x1:x2+1, y1:y2+1]
            cutout_w2 = w2grid[x1:x2+1, y1:y2+1]

            cutout_w1_Ha = w1grid_Ha[x1:x2+1, y1:y2+1]
            cutout_w2_Ha = w2grid_Ha[x1:x2+1, y1:y2+1]

            axes = {1: cutout_w1_Ha, 2: cutout_w2_Ha}
            resonances = {}
            resonancesTypes = [(-1, 2), (-1,)]
            for typelist in resonancesTypes:
                resonances[typelist] = ((-1) * sum([np.sign(ix) * axes[abs(ix)] for ix in typelist])
                                                - 1j * self.Gamma)

            resGridsDict[tuple([seed, (x1, x2+1, y1, y2+1)])] = (cutout_w1, cutout_w2,
                                                                 cutout_w1_Ha, cutout_w2_Ha,
                                                                 resonances)
        elapsed_time = time.time() - st
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta)
        print('find_all_grids in:', formatted_time)
        return resGridsDict


    def compute_mech_factors(self, a: int, b: int):
        """
        Precalculate prefactor of mechanical terms - summation over c for each a,b
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        factors = {}
        for m_idx in self.m_selected:

            fac = 0.
            mechterm, termavrg = self.allterms_str[m_idx]
            for c in self.mode_indices:
                prefac_mech = self.prefac_3d[a, b, c]
                mechavrg = self.avrg_tensors_dict[m_idx]
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                ijk_indx = tuple([abc[j] for j in termavrg[-2]])
                F = self.deriv_data['F_abc'][ijk_indx]

                freqDiff = [i.split(',') for i in mechterm[1]]
                letters = ['a', 'b', 'c', 'zero']
                dictabc = dict(zip(letters, (a, b, c) + tuple(['zero'])))

                w_fr11 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][0].split('+')], key=int))
                if 'zero' not in freqDiff[0][1]:
                    w_fr21 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][1].split('+')], key=int))
                else:
                    w_fr21 = tuple([freqDiff[0][1]])
                w_fr12 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][0].split('+')], key=int))
                if 'zero' not in freqDiff[1][1]:
                    w_fr22 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][1].split('+')], key=int))
                else:
                    w_fr22 = tuple([freqDiff[1][1]])

                t3 = vib_ene_levels[w_fr11] - vib_ene_levels[w_fr21]
                t4 = vib_ene_levels[w_fr12] - vib_ene_levels[w_fr22]
                sumfrac = (1 / t3 + 1 / t4)

                fac += termavrg[-1] * sumfrac / prefac_mech * mechavrg[a, b, c] * F / (-48.)
            factors[m_idx] = fac

        return factors


    def intensity_both(self, selectionCond: np.ndarray = None,
                       shape2d = None, resonances_args = None,
                       selected_ab = None,
                       mechel_contrib: bool = False) -> np.ndarray:
        """
        Collects all the contributions to intensity.
        Loop over (a,b) modes combinations.
        """

        if resonances_args is None:
            resonances_args = self.resonances_args

        if shape2d is None:
            intensities_grid = np.zeros(self.shape2d, dtype='complex64')
            shapegrid = self.shape2d
            w1w2Condition = self.w1w2Condition
        else:
            intensities_grid = np.zeros(shape2d, dtype='complex64')
            shapegrid = shape2d
            w1w2Condition = np.ones(shapegrid, dtype=bool)

        import time
        st = time.time()

        if selectionCond is None:
            selectionCond = np.ones(shapegrid, dtype=bool)
        condition = (w1w2Condition & selectionCond)
        np.savetxt('condition.out', condition, delimiter=',', fmt='%i')  # X is an array


        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        count = 0
        numberofcombs = numcombperm(len(self.mode_indices), 2)
        if selected_ab is None:
            combinations_ab = combinations_with_permutations(self.mode_indices, 2)
        else:
            combinations_ab = selected_ab

        if mechel_contrib:
            self.mech = {k: np.zeros(shapegrid, dtype='complex64') for k in self.m_selected}
            self.el = {k: np.zeros(shapegrid, dtype='complex64') for k in self.e_selected}

            self.mech_ab = {k: {} for k in self.m_selected}
            self.el_ab = {k: {} for k in self.e_selected}

        count0 = 0
        for ab in combinations_ab:
            import time
            st_ab = time.time()
            a,b = ab
            count+=1

            self.resonances_bank = {}

            for termID in self.selection:
                res_formula, avrg_formula = self.allterms_str[termID]
                # print(res_formula)
                if ab not in [i[1] for i in self.res_dict[res_formula[0]]]:
                    if mechel_contrib:
                        if termID in [0, 1]:
                            self.el_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                        else:
                            self.mech_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                    continue

                if res_formula[-1] is None:
                    factor = self.avrg_tensors_dict[termID][a, b] / self.prefac_2d[a, b] / 24.
                else:
                    factor = self.comb_fac_dict[self.allterms_str[termID]][a, b]

                if abs(factor)<1e-20:
                    count0 +=1
                    if mechel_contrib:
                        if termID in [0, 1]:
                            self.el_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                        else:
                            self.mech_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                    continue

                if res_formula[0] not in self.resonances_bank:
                    self.resonances_bank[res_formula[0]] = self.allfunc_dict[termID](allLevels_Eh=vib_ene_levels,
                                                                                     w_res_dict=resonances_args,
                                                                                     abctuple=(a, b),
                                                                                     w1w2Condition=condition)
                addition = np.where(condition, factor * self.resonances_bank[res_formula[0]], 0.)
                intensities_grid += addition

                if mechel_contrib:
                    if termID in [0,1]:
                        self.el[termID] += addition
                        self.el_ab[termID][ab] = addition
                    else:
                        self.mech[termID] += addition
                        self.mech_ab[termID][ab] = addition

            if not mechel_contrib and selected_ab is None:
                if count % 10 == 0:
                    print(f'{count}/{numberofcombs} modes combinations -- {count*100/numberofcombs}%; '
                          f'time passed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st_ab))}',
                          f'time passed since start: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')

        if not mechel_contrib and selected_ab is None:
            elapsed_time = time.time() - st
            print('Compute time of looping over abc combinations in intensity_mechanical:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        return intensities_grid


    def generate_resonances_functions(self, subscripts, freqDiff=None) -> Callable:
        """
        Generates a python function for a term given by a formula (subscripts and freqDiff);
                varied argument of that function is abctuple (used in the loop over combinations of modes).
        subscripts - a tuple of strings from the formula; subscripts of omega energy levels in the resonance part;
                        e.g., ('a+b,a', 'zero,a')
        freqDiff - a tuple of strings from the formula; subscripts of omega energy levels in the freq. difference part;
                        e.g., ('a+b+c,0', 'c,a+b'); not None for mech. anharm.
        """
        if freqDiff is not None:
            freqDiff = [i.split(',') for i in freqDiff]

        def compute_res_condition(allLevels_Eh: dict, w_res_dict: dict[str:np.ndarray],
                     abctuple: tuple[int, int] | tuple[int, int, int],
                     w1w2Condition: np.ndarray[bool],
                     freqDiff: list = freqDiff) -> np.ndarray:
            """
            allLevels_Eh_c collects all vibrational energy levels in Hartree; e.g., [('1', '2')] - combination mode
            w_res_dict contains [-1, 2] and [-1] 2d arrays (in s-1)
            abctuple is a tuple of normal mode indices for which current iteration is evaluating resonance term
            """
            # todo: lorentzian shape cutoff

            letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
            dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
            # allLevels_Eh_c = copy.deepcopy(allLevels_Eh)

            if 'c' not in subscripts[0]:
                index_wmn = (abctuple[0], abctuple[1])
            else:
                index_wmn = (abctuple[0], abctuple[2])

            t1 = self.w_mn_dict[subscripts[0]][index_wmn] + w_res_dict[(-1, 2)]  # - 1j * Gamma_hartree

            t2 = self.w_mn_dict[subscripts[1]][abctuple[0], abctuple[1]] + w_res_dict[(-1,)] #- 1j * Gamma_hartree

            if freqDiff is None:
                sumfrac = 1.

            else:
                if self.mechab:
                    w_fr11 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][0].split('+')], key=int))
                    if 'zero' not in freqDiff[0][1].split('+'):
                        w_fr21 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][1].split('+')], key=int))
                    else:
                        w_fr21 = tuple([freqDiff[0][1]])

                    w_fr12 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][0].split('+')], key=int))
                    if 'zero' not in freqDiff[1][1].split('+'):
                        w_fr22 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][1].split('+')], key=int))
                    else:
                        w_fr22 = tuple([freqDiff[1][1]])

                    t3 = allLevels_Eh[w_fr11] - allLevels_Eh[w_fr21]
                    t4 = allLevels_Eh[w_fr12] - allLevels_Eh[w_fr22]

                    sumfrac = (1 / t3 + 1 / t4)
                    # self.mechab = False

                else:
                    sumfrac = 1.

            return  np.where(w1w2Condition, sumfrac / (t1 * t2), 0.)

        return compute_res_condition

