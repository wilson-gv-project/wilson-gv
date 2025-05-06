import copy
import time
from datetime import timedelta
from typing import Callable

import numpy as np

from .averaging import get_AlphaBetaGammaDelta_indices
from .tools import convNu2Ene, avrg_abc_tensor, get_properties_avrg

from CQCParse.parsing import ParsedData

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
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='xy')
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


    def launch_sequence(self, parserObj, spectrum_settings,
                        vpt2settings=None, preview=False, print_level=0):
        """
        Execute the main steps before the intensities calculation.
        Will update the state of the object
        """
        # , list2exclude = None, only_modes = None

        # - load data from data object
        self.load_data(parserObj)

        # - changing indices of list2exclude
        if spectrum_settings.new_idx_dict is not None and spectrum_settings.list2exclude is not None:
            temp_list2exclude = []
            for new in spectrum_settings.list2exclude:
                old_idx = list(spectrum_settings.new_idx_dict.keys())[list(spectrum_settings.new_idx_dict.values()).index(new)]
                temp_list2exclude.append(old_idx)
        else:
            temp_list2exclude = None

        # - get anharmonic vpt2 energies
        if vpt2settings is not None:
            from .vpt2 import get_vpt2_corrected_levels
            all_states = get_vpt2_corrected_levels(self, parserObj, vpt2settings,
                                                   temp_list2exclude, print_level=print_level)
            self.all_states = all_states
            self.fundamentals = {k[0]: v for k, v in all_states.items() if len(k)==1}
        # print('after vpt2 self.fundamentals', self.fundamentals)
        # print('after vpt2 self.fundamentals_harmonic', self.fundamentals_harmonic)

        # - changing indices in all data
        if spectrum_settings.new_idx_dict is not None:
            from .tools import change_idx_modes

            # datain =
            (fundamentals, fundamentals_harmonic, all_states, all_states_harmonic,
             deriv_data, mode_indices) = change_idx_modes(parserObj, spectrum_settings.new_idx_dict,
                                                          spectrum_settings.list2exclude,
                                                          spectrum_settings.only_modes)
            self.fundamentals = fundamentals
            self.fundamentals_harmonic = fundamentals_harmonic
            self.all_states = all_states
            self.all_states_harmonic = all_states_harmonic
            self.deriv_data = deriv_data
            self.mode_indices = mode_indices
            # print(self.fundamentals)
            # print(self.fundamentals_harmonic)

        # - ensure there is list2exclude
        if spectrum_settings.list2exclude is None:
            spectrum_settings.list2exclude = []

        # print('self.fundamentals', self.fundamentals)
        # print('self.fundamentals_harmonic', self.fundamentals_harmonic)

        # - exclude indices
        if spectrum_settings.list2exclude:
            self.mode_indices = [i for i in np.arange(self.nmodes) if i not in spectrum_settings.list2exclude]
            self.nmodes -= len(spectrum_settings.list2exclude)
        else:
            if spectrum_settings.only_modes is not None:
                self.mode_indices = spectrum_settings.only_modes
            else:
                self.mode_indices = [i for i in np.arange(self.nmodes)]


        # adding attributes and convesion of units
        self.set_spectrum_settings(Gamma_rc=spectrum_settings.Gamma_rc,
                                   diag_margin_rc=spectrum_settings.diag_margin_rc,
                                   vib_levels_harmonic=spectrum_settings.vib_levels_harmonic)

        self.add_terms(spectrum_settings.el_terms_selected, spectrum_settings.mech_terms_selected)

        if preview:
            # chart, spectrumDF
            chart, spectrumDF = self.preview_spectrum()
            medium = spectrumDF.drop(['omega2', 'log10(Intensity)', 'gamma', 'abs el',
                                            'abs mech', 'Intensity', 'abs gamma_clean'],
                                            axis=1)
            # if [0 , 1 , 2 , 3] in self.selection:
            if {0, 1, 2, 3}.issubset(self.selection):
                cols = ['omega1', 'w2mw1', 'a', 'b', 'type', 'abs 0', 'abs 2', 'abs 1', 'abs 3',
                        'factor 0', 'factor 2', 'factor 1', 'factor 3',
                        'factor 0/2', 'factor 1/3', 'factor 0/2 sign', 'factor 1/3 sign', 'factors sign', 'Intensity_clean']
            else:
                cols = ['omega1', 'w2mw1', 'a', 'b', 'type',
                        'Intensity_clean']
            medium = medium[cols]
        else:
            chart, spectrumDF, medium = None, None, None

        self.precalculate4fullspectrum()

        return {'spectrum': self,
                'chart': chart,
                'resonancesDF': spectrumDF,
                'mediumDF': medium}


    def launch_sequence1(self, parsed_data: ParsedData, spectrum_settings,
                        print_level=0):
        """
        Execute the main steps before the intensities calculation.
        Will update the state of the object

        1. getting parsed_data initial, before this
        2. change indices - to indices of cfour
        3. exclude modes
        4. vpt2
        """
        vpt2settings = spectrum_settings.vpt2settings
        preview = spectrum_settings.preview
        if spectrum_settings.vib_levels_harmonic:
            vpt2settings = None

        if spectrum_settings.list2exclude is None:
            spectrum_settings.list2exclude = []
        # print('before chng', parsed_data.vib_states.fundamentals_anharmonic_str)

        # - 2. changing indices in list2exclude ------------------------------------------------
        if spectrum_settings.new_idx_dict is not None:
            newKey_oldVal = dict(zip(list(spectrum_settings.new_idx_dict.values()),
                                     list(spectrum_settings.new_idx_dict.keys())))
            list2exclude_vpt2 = [newKey_oldVal[i] for i in spectrum_settings.list2exclude]
        else:
            list2exclude_vpt2 = spectrum_settings.list2exclude

        # print('parsed_data.vib_states.anharmonic_states', parsed_data.vib_states.anharmonic_states)
        # print('after chng', parsed_data.vib_states.fundamentals_anharmonic_str)


        from CQCParse.utils import make_modes_idx

        self.mode_indices = make_modes_idx(len(parsed_data.normal_modes.normal_modes),
                                           modes=spectrum_settings.list2exclude,
                                           include=False)
        self.nmodes = parsed_data.nmodes
        self.nmodes_original = parsed_data.nmodes

        # print('parsed_data.vib_states.anharmonic_states\n', parsed_data.vib_states.anharmonic_states)
        # print(vpt2settings, parsed_data.anharm_treatment)
        # - 4. get anharmonic vpt2 energies with exclusion of list2exclude ------------------------
        if vpt2settings is not None:
            parsed_data.get_vpt2(vpt2settings=vpt2settings,
                                 list2exclude=list2exclude_vpt2,
                                 print_level=print_level)

            # upd dicts
            self.all_states = parsed_data.vib_states.anharmonic_states
            self.fundamentals = {k[0]: v for k, v in self.all_states.items() if len(k)==1}
            # print('self.all_states',self.all_states)
            # print('self.fundamentals', self.fundamentals)
            self.fermi_resonances = parsed_data.anharm_correction_data.fermi_resonance

        else:
            self.fundamentals = parsed_data.vib_states.fundamentals_anharmonic_str
            self.all_states = parsed_data.vib_states.anharmonic_states
            self.fermi_resonances = parsed_data.anharm_correction_data.fermi_resonance

        # print('after vpt2', parsed_data.vib_states.fundamentals_anharmonic_str)

        # - 2. changing indices in all data -------------------------------------------------------
        if spectrum_settings.new_idx_dict is not None:
            parsed_data.vib_states.upd_indices(spectrum_settings.new_idx_dict)
            parsed_data.derivatives.upd_indices(spectrum_settings.new_idx_dict)
            parsed_data.anharm_correction_data.upd_indices(spectrum_settings.new_idx_dict)
            self.fundamentals = {k[0]:v for k,v in parsed_data.vib_states.fundamentals_anharmonic_str.items()}
            self.all_states = parsed_data.vib_states.anharmonic_states
            self.fermi_resonances = parsed_data.anharm_correction_data.fermi_resonance

        parsed_data.list2exclude = spectrum_settings.list2exclude

        print(f'\nFermi resonances: {self.fermi_resonances}\n')

        # load or load and upd states
        self.fundamentals_harmonic = parsed_data.vib_states.fundamentals_harmonic_str
        self.all_states_harmonic = parsed_data.vib_states.harmonic_states

        # add derivatives
        ddata = [parsed_data.derivatives.dipole_first_derivatives,
                 parsed_data.derivatives.dipole_second_derivatives,
                 parsed_data.derivatives.polarizability_first_derivatives,
                 parsed_data.derivatives.polarizability_second_derivatives,
                 parsed_data.derivatives.cubic_force_constants]
        deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))
        self.deriv_data = deriv_data

        # print(parsed_data.anharm_treatment)
        # --- spectrum prep
        self.set_spectrum_settings(Gamma_rc=spectrum_settings.Gamma_rc,
                                   diag_margin_rc=spectrum_settings.diag_margin_rc,
                                   vib_levels_harmonic=spectrum_settings.vib_levels_harmonic)

        self.add_terms(spectrum_settings.el_terms_selected, spectrum_settings.mech_terms_selected)

        if preview:
            chart, spectrumDF = self.preview_spectrum()
            medium = spectrumDF.drop(['omega2', 'log10(Intensity)', 'gamma', 'abs el',
                                            'abs mech', 'Intensity', 'abs gamma_clean'],
                                            axis=1)
            if {0, 1, 2, 3}.issubset(self.selection):
                cols = ['omega1', 'w2mw1', 'a', 'b', 'type', 'abs 0', 'abs 2', 'abs 1', 'abs 3',
                        'factor 0', 'factor 2', 'factor 1', 'factor 3',
                        'factor 0/2', 'factor 1/3', 'factor 0/2 sign', 'factor 1/3 sign',
                        'factors sign', 'Intensity_clean']
            else:
                cols = ['omega1', 'w2mw1', 'a', 'b', 'type',
                        'Intensity_clean']
            medium = medium[cols]
        else:
            chart, spectrumDF, medium = None, None, None

        self.precalculate4fullspectrum()

        return {'spectrum': self,
                'parsed_data': parsed_data,
                'chart': chart,
                'resonancesDF': spectrumDF,
                'mediumDF': medium,
                'settings': spectrum_settings}


    def get_derived_terms_evv(self):
        """
        Currently available for selection EVV terms
        """
        # Terms in the expressions
        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        # mu_Q, alpha_Q - for all 6 terms
        self.allterms_str = {0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',)))),
                             1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',)))),
                             2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',)), 'abc', 1.)),
                             3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acb', 1.)),
                             4: ((('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',)), 'bcc', 0.5)),
                             5: ((('a+b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acc', 0.5)),
                             6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'bcc', -0.5)),
                             7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acc', -0.5))}


    def load_data(self, parserObj):
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

        parserObj.getData()
        self.parserObj = parserObj

        self.fundamentals = parserObj.fundamentals_anharmonic_str
        self.fundamentals_harmonic = parserObj.fundamentals_harmonic_str
        self.all_states = parserObj.anharmonic_states
        self.all_states_harmonic = parserObj.harmonic_states

        self.nmodes = len(self.fundamentals)
        self.nmodes_original = len(self.fundamentals)

        ddata = [parserObj.dipole_first_derivatives,
                 parserObj.dipole_second_derivatives,
                 parserObj.polarizability_first_derivatives,
                 parserObj.polarizability_second_derivatives,
                 parserObj.cubic_force_constants]
        self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))
        # 'mu_Q',  'mu_QQ',  'alpha_Q', 'alpha_QQ', 'F_abc'
        # (6, 3)  (6, 6, 3)  (6, 3, 3) (6, 6, 3, 3) (6, 6, 6) if nmodes = 6

        self.normal_modes = parserObj.normal_modes


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


    def precalculate4resonances(self):
        """
        Precalculationf of intensities parts are done for resonance points
        """
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
        data, x_shape = self.precalculate4resonances()
        # is a 1d array of intensities
        # Z = self.intensity_both(selectionCond=None, shape2d=x.shape, resonances_args=self.resonances_args, mechel_contrib=True)
        # if self.vpt2:
        #     prefix = 'vpt2_'+self.parserObj.program+'_'+self.parserObj.molecule+'_'+str(self.selection)
        # else:
        #     prefix = self.parserObj.program+'_'+self.parserObj.molecule+'_'+str(self.selection)

        typedict = {'a+b,a; zero,a': [0, 2, 4, 5], 'b,a; zero,a': [1, 3, 6, 7]}
        if self.e_selected:
            data['gamma el'] = np.zeros(x_shape)

        if self.m_selected:
            data['gamma mech'] = np.zeros(x_shape)


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
                    addition = self.el_ab[t][(data['a'][i], data['b'][i])][i]
                    data['gamma el'][i] += addition
                    if t in typedict[termtype]:
                        gammaabs_clean[i] += addition
                elif t in [2,3,4,5,6,7]:
                    addition = self.mech_ab[t][(data['a'][i], data['b'][i])][i]
                    data['gamma mech'][i] += addition
                    if t in typedict[termtype]:
                        gammaabs_clean[i] += addition

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
            data[f'avrg {t}'] = [self.avrg_tensors_dict[t][data['a'][i], data['b'][i]] for i in range(len(data['omega1']))]
        for t in self.mech_ab:

            data[f'factor {t}'] = [self.comb_fac_dict[self.allterms_str[t]][data['a'][i], data['b'][i]] / self.prefac_2d[data['a'][i], data['b'][i]] / (-48.)
                                   if data[f'abs {t}'][i]!=0. else 0. for i in range(len(data['omega1']))]
            data[f'avrg {t}'] = [self.avrg_tensors_dict[t][data['a'][i], data['b'][i]] for i in range(len(data['omega1']))]

        import pandas as pd
        spectrumDF = pd.DataFrame(data)

        spectrumDF['w2mw1'] = spectrumDF['omega2'] - spectrumDF['omega1']
        if 0 in self.selection and 2 in self.selection:
            spectrumDF['factor 0/2'] = spectrumDF['factor 0']/spectrumDF['factor 2']
            # spectrumDF['factor 0/2'] = spectrumDF['factor 0/2'].apply('{:.4e}'.format)
            spectrumDF['factor 0/2 sign'] = np.log(abs(spectrumDF['factor 0'])/abs(spectrumDF['factor 2']))
        if 1 in self.selection and 3 in self.selection:
            spectrumDF['factor 1/3'] = spectrumDF['factor 1']/spectrumDF['factor 3']
            # spectrumDF['factor 1/3'] = spectrumDF['factor 1/3'].apply('{:.4e}'.format)
            spectrumDF['factor 1/3 sign'] = np.log(abs(spectrumDF['factor 1'])/abs(spectrumDF['factor 3']))
        if "factor 0/2 sign" in spectrumDF and "factor 1/3 sign" in spectrumDF:
            spectrumDF["factors sign"] = spectrumDF["factor 0/2 sign"].fillna(0) + spectrumDF["factor 1/3 sign"].fillna(0)
        column_to_move = spectrumDF.pop("w2mw1")
        spectrumDF.insert(2, "w2mw1", column_to_move)
        for tt in self.selection:
            spectrumDF[f'factor {tt}'] = spectrumDF[f'factor {tt}'].apply('{:.4e}'.format)

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

    def preview_matplotlib(self, vpt2):

        if vpt2:
            prefix = 'vpt2_'+self.parserObj.program+'_'+self.parserObj.molecule
        else:
            prefix = self.parserObj.program+'_'+self.parserObj.molecule
        # collect resonances below diagonal (without margin)
        fromdiagonal = {('a+b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] >= i[0][1]]),
                        ('b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] >= i[0][1]])}

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



    def precalc_intensities_modes(self):
        """
        Normal modes dependent calculations:

        self.prefac_2d - Eh
        self.prefac_3d - Eh
        self.w_mn_dict - Eh
        self.comb_fac_dict - mechanical factors
        """

        # general
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

        w_apbbma = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma = np.zeros((self.nmodes_original, self.nmodes_original))
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if a in self.mode_indices and b in self.mode_indices:
                w_apbbma[a, b] = vib_ene_levels[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels[
                    tuple([str(a)])]
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


    def precalc_intensities_freqs(self, w1_mesh_Eh=None, w2_mesh_Eh=None):
        """
        self.resonances_args
        self.w1w2Condition
        """
        # grid specific
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
                                                                                self.axes[abs(ix)], 0) for ix in
                                                         typelist]) - 1j * self.Gamma


    def precalc_intensities(self, w1_mesh_Eh=None, w2_mesh_Eh=None):

        # general
        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k, v in self.fundamentals_harmonic.items()]))

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

        w_apbbma = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma = np.zeros((self.nmodes_original, self.nmodes_original))
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if a in self.mode_indices and b in self.mode_indices:
                w_apbbma[a, b] = vib_ene_levels[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels[
                    tuple([str(a)])]
                w_bma[a, b] = vib_ene_levels[tuple([str(b)])] - vib_ene_levels[tuple([str(a)])]

        za = np.array([-vib_ene_levels[tuple([str(k)])] for k in range(self.nmodes_original)])
        self.w_mn_dict = {'a+b,a': w_apbbma, 'b,a': w_bma, 'c,a': w_bma,
                          'zero,a': np.tile(za, self.nmodes_original).reshape(self.nmodes_original, -1).T}

        # grid specific
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
                                                                                self.axes[abs(ix)], 0) for ix in
                                                         typelist]) - 1j * self.Gamma

        # precalc mech factors
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
                        self.comb_fac_dict[self.allterms_str[key]][a, b] = self.compute_mech_factors(a, b)[key]
            # print(self.comb_fac_dict)
            elapsed_time = time.time() - st
            print('self.comb_fac_dict collected:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            # print('FOUND! I THINK\n', self.comb_fac_dict)

    def precalc_combfac_cmech(self):
        # precalc mech factors
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
                        self.comb_fac_dict[self.allterms_str[key]][a, b] = self.compute_mech_factors(a, b)[key]
            # print(self.comb_fac_dict)
            elapsed_time = time.time() - st
            print('self.comb_fac_dict collected:',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    def precalculate4fullspectrum(self):
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

        # omega2>omega1 condition
        # self.res_dict[('a+b,a', 'zero,a')] = set(
        #     [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])
        # self.res_dict[('b,a', 'zero,a')] = set(
        #     [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])

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
            # print('vib_ene_levels = self.all_states_harmonic_Eh')
        else:
            vib_ene_levels = self.all_states_Eh
            # print('vib_ene_levels = self.all_states_Eh')

        # print('vib_ene_levels', vib_ene_levels)
        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k,v in self.fundamentals_harmonic.items()]))
        factors = {}
        # print('self.mode_indices', self.mode_indices)
        for m_idx in self.m_selected:

            fac = 0.
            mechterm, termavrg = self.allterms_str[m_idx]
            for c in self.mode_indices:
                # prefac_mech = self.prefac_3d[a, b, c]
                prefac_mech_c = vib_ene_levels_harmonic[c]
                # print(prefac_mech==prefac_mech_c*self.prefac_2d[a, b])
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
                fac += termavrg[-1] * sumfrac / prefac_mech_c * mechavrg[a, b, c] * F #/ (-48.)
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

        print("('a+b,a', 'zero,a')", len(self.res_dict[('a+b,a', 'zero,a')]))
        # print([i[1] for i in self.res_dict[('a+b,a', 'zero,a')]])
        print("('b,a', 'zero,a')", len(self.res_dict[('b,a', 'zero,a')]))
        # print([i[1] for i in self.res_dict[('b,a', 'zero,a')]])
        count0 = 0
        for ab in combinations_ab:
            import time
            st_ab = time.time()
            a,b = ab
            count+=1

            self.resonances_bank = {}

            for termID in self.selection:
                res_formula, avrg_formula = self.allterms_str[termID]
                # print(f"term {termID}") if termID==0 else None
                # print(res_formula)
                if ab not in [i[1] for i in self.res_dict[res_formula[0]]]:
                    if mechel_contrib:
                        if termID in [0, 1]:
                            self.el_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                        else:
                            self.mech_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                    # print(f'skipping in "for termID in self.selection:": {ab, termID}')
                    continue

                if res_formula[-1] is None:
                    factor = self.avrg_tensors_dict[termID][a, b] / self.prefac_2d[a, b] / 24.
                    # print(f'1/self.prefac_2d[a, b] {1/self.prefac_2d[a, b]:.3e}') if termID==0 else None
                    # print(f'1/self.prefac_2d[b, a] {1/self.prefac_2d[b, a]:.3e}') if termID==0 else None
                    # print(f'self.avrg_tensors_dict[termID][a, b] {self.avrg_tensors_dict[termID][a, b]:.3e}') if termID==0 else None
                else:
                    factor = self.comb_fac_dict[self.allterms_str[termID]][a, b] / self.prefac_2d[a, b] / (-48.)

                # if abs(factor)<1e-20:
                #     count0 +=1
                #     if mechel_contrib:
                #         if termID in [0, 1]:
                #             self.el_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                #         else:
                #             self.mech_ab[termID][ab] = np.zeros(shapegrid, dtype='complex64')
                #     continue

                if res_formula[0] not in self.resonances_bank:
                    self.resonances_bank[res_formula[0]] = self.allfunc_dict[termID](allLevels_Eh=vib_ene_levels,
                                                                                     w_res_dict=resonances_args,
                                                                                     abctuple=(a, b),
                                                                                     w1w2Condition=condition)
                    # print(self.resonances_bank[res_formula[0]])
                # print(f"term {termID}, a,b: {(a, b)}, factor: {factor:.2e}") if termID==0 else None

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


class Term2D:
    """
    Calculations using the expression.
        prefactor_num
        prefactor_ene
        property_1
        property_2
        property_3
        avrg_terms
        CFF (optional)

    {
    0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)))),
    1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',)))),
    2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.)),
    3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.)),
    4: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5)),
    5: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5)),
    6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5)),
    7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5))
    }

    """

    def __init__(self, term_id, expression):
        """
        expression[0] - resonance part
            expression[0][0] - actual resonances
            expression[0][1] - vib levels differences
        expression[1] - orientational average part
            expression[1][0] - property_tuple1: (property, mode indices)
            expression[1][1] - property_tuple2: (property, mode indices)
            expression[1][2] - property_tuple3: (property, mode indices)
            mech: expression[1][3] - CFF indices
            mech: expression[1][4] - term prefactor
        >expression[2] - overall el/mech prefactor (1/24. or -1/48.)

        """
        self.term_id = term_id
        self.expression = expression

        self.resonances_expr = expression[0][0]
        self.viblevelsdiff_expr = expression[0][1]
        if self.viblevelsdiff_expr is not None:
            self.term_label = 'MECH'
            self.F_expr = expression[1][3]
            self.prefactor_term = expression[1][4]

            self.F_vals = {}
        else:
            self.term_label = 'EL'
            self.prefactor_term = 1.

        self.property_tuples = expression[1][:3]
        self.property_tuple1 = expression[1][0]
        self.property_tuple2 = expression[1][1]
        self.property_tuple3 = expression[1][2]

        # self.part_prefactor = expression[2]

    def __repr__(self):
        return f'{self.term_label} - {self.term_id}'

    def get_resonance_location(self, modes_dict, a, b):
        """
        A resonance for this term for ab combination of modes
        """
        a, b = str(a), str(b)
        modes_dict[('zero',)] = 0.
        dict_id = {'a': a, 'b': b, 'zero': 'zero'}
        type12, type1 = self.resonances_expr
        m1_str, n1_str = type1.split(',')

        m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
        n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])
        w1 = modes_dict[n1_tuple] - modes_dict[m1_tuple]

        m12_str, n12_str = type12.split(',')
        m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
        n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
        w2 = modes_dict[m12_tuple] - modes_dict[n12_tuple] + w1

        return w1, w2


    def get_res_factor(self, modes_dict, w1_rc, w2_rc, a, b, Gamma_rc, condition=None):
        """
        A resonance factor for this term for ab combination of modes
        """
        if condition is None:
            condition = np.ones_like(w1_rc, dtype=bool)

        a, b = str(a), str(b)
        w1, w2 = convNu2Ene(w1_rc), convNu2Ene(w2_rc)
        modes_dict_Eh = {k: convNu2Ene(v) for k, v in modes_dict.items()}
        modes_dict_Eh[('zero',)] = 0.
        dict_id = {'a': a, 'b': b, 'zero': 'zero'}
        type12, type1 = self.resonances_expr
        m1_str, n1_str = type1.split(',')

        m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
        n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])

        m12_str, n12_str = type12.split(',')
        m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
        n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
        Gamma_Eh = convNu2Ene(Gamma_rc)
        # print(condition)
        r = np.where(condition,
                     1/(modes_dict_Eh[m12_tuple] - modes_dict_Eh[n12_tuple] + w1 - w2 -1j*Gamma_Eh)/(modes_dict_Eh[m1_tuple] - modes_dict_Eh[n1_tuple] + w1 -1j*Gamma_Eh), 0.)
        return r


    def get_properties(self, properties_data, ABGD, a, b, c=None):
        dict_id = {'a': a, 'b': b, 'c': c}
        # beta, alpha, delta, gamma
        dict_ax_id = {'A': ABGD[0], 'B': ABGD[1], 'G': ABGD[2], 'D': ABGD[3]}
        propdict = {}
        for nn, p in enumerate(self.property_tuples):
            # p is ('mu_QQ', ('a', 'b',), ('G',))
            # p is ('alpha_Q', ('c',), ('A', 'D'))
            indices = [dict_id[i] for i in p[1]] + [dict_ax_id[i] for i in p[2]]
            propdict[f'{nn}_'+p[0]] = properties_data[p[0]][*indices]

        if self.term_label == 'MECH':
            if (a,b,c) not in self.F_vals:
                idx = [dict_id[i] for i in self.F_expr]
                self.F_vals[(a,b,c)] = properties_data['F_abc'][*idx]
            propdict['F_abc'] = self.F_vals[(a,b,c)]

        return propdict


    def get_avrg_properties(self, gammaCompsAll, properties_data, a, b, c=None):
        components = {}
        total = 0.
        for ABGD in gammaCompsAll:
            props_dict = self.get_properties(properties_data, ABGD, a, b, c)
            addition = np.prod(np.array([v for k,v in props_dict.items() if 'mu' in k or 'alpha' in k]))
            total += addition
            components[tuple(ABGD)] = (addition, props_dict)
        return total/15, components


    def get_factor_summed(self, gammaCompsAll, properties_data, modes_dict, harm_modes_dict, mode_indices, a, b):
        """
        Sum of full factor over c index for given a,b
        """
        components = {}
        total = 0.
        for c in mode_indices:
            addition_2 = self.get_full_factor(gammaCompsAll, properties_data, modes_dict, harm_modes_dict, a, b, c)
            total += addition_2[0]
            components[c] = addition_2
        return total, components


    def get_full_factor(self, gammaCompsAll, properties_data, modes_dict, harm_modes_dict, a, b, c=None):
        """
        product of: ene_factor, avrg_properties, (F_abc, viblevelsdiff)
        """
        components = {}
        ene_factor = self.get_ene_factor(harm_modes_dict, a, b, c)
        avrg_properties_2 = self.get_avrg_properties(gammaCompsAll, properties_data, a, b, c)
        product_all = ene_factor*avrg_properties_2[0]

        components['ene_factor'] = ene_factor
        components['avrg_properties'] = avrg_properties_2

        # print(f'ene_factor {(a,b,c)}, {ene_factor:.3e}')
        # print(f'avrg_properties_2 {(a,b,c)}, {avrg_properties_2[0]:.3e}')

        if self.term_label=='MECH':
            product_all *= self.F_vals[(a,b,c)] * self.get_viblevelsdiff(modes_dict, a, b, c)[0]
            components['F_abc'] = self.F_vals[(a,b,c)]
            components['viblevelsdiff'] = self.get_viblevelsdiff(modes_dict, a, b, c)[0]

        return product_all, components


    def get_full_factor_tensor(self, gammaCompsAll, properties_data,
                               modes_dict, harm_modes_dict, mode_indices):

        t2 = np.zeros((max(mode_indices)+1, max(mode_indices)+1))
        t3 = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1))
        for a in mode_indices:
            for b in mode_indices:
                if self.term_label == 'EL':
                    t2[(a, b)] = self.get_full_factor(gammaCompsAll, properties_data,
                                                      modes_dict, harm_modes_dict, a,b)[0]
                else:
                    for c in mode_indices:
                        t3[(a, b, c)] = self.get_full_factor(gammaCompsAll, properties_data,
                                                             modes_dict, harm_modes_dict, a,b,c)[0]

        all_zeros_t2 = not np.any(t2)
        if all_zeros_t2:
            return t3
        else:
            return t2

    @staticmethod
    def get_ene_factor(harm_modes_dict, a, b, c=None):
        """
        1/omega_a/omega_b/omega_c
        """
        modes_dict_Eh = {k: convNu2Ene(v) for k, v in harm_modes_dict.items() if len(k)==1}
        modes_dict_Eh[('zero',)] = 0.
        modes = [a, b] if c is None else [a, b, c]
        return 1./np.prod(np.array([modes_dict_Eh[(str(m),)] for m in modes]))


    def get_viblevelsdiff(self, modes_dict, a, b, c=None):
        """
        1/omega_m,n + 1/omega_k,l
        """
        a, b = str(a), str(b)
        dict_id = {'a': a, 'b': b, 'zero': 'zero'}
        if c is not None:
            c = str(c)
            dict_id['c'] = c

        modes_dict_Eh = {k: convNu2Ene(v) for k, v in modes_dict.items()}
        modes_dict_Eh[('zero',)] = 0.

        total = []
        for e in self.viblevelsdiff_expr:
            m_str, n_str = e.split(',')

            l_m = [str(dict_id[i]) for i in m_str.split('+')]
            l_n = [str(dict_id[i]) for i in n_str.split('+')]

            if 'zero' not in l_m:
                m_tuple = tuple(sorted(l_m, key=int))
            else:
                m_tuple = tuple(l_m)
            if 'zero' not in l_n:
                n_tuple = tuple(sorted(l_n, key=int))
            else:
                n_tuple = tuple(l_n)

            total.append(modes_dict_Eh[m_tuple] - modes_dict_Eh[n_tuple])
        total0 = 1./np.array(total)
        return np.sum(total0), np.array(total)


    def get_avrgprops_tensor(self, gammaCompsAll, properties_data, mode_indices, threshold=1e-18):
        t2 = np.zeros((max(mode_indices)+1, max(mode_indices)+1))
        t3 = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1))
        for a in mode_indices:
            for b in mode_indices:
                if self.term_label == 'EL':
                    t2[(a, b)] = self.get_avrg_properties(gammaCompsAll, properties_data, a, b)[0]
                else:
                    for c in mode_indices:
                        t3[(a, b, c)] = self.get_avrg_properties(gammaCompsAll, properties_data, a, b, c)[0]

        all_zeros_t2 = not np.any(t2)
        if all_zeros_t2:
            return np.where(np.abs(t3)>threshold, t3, 0.)
        else:
            return np.where(np.abs(t3)>threshold, t2, 0.)


    def get_enefactor_tensor(self, harm_modes_dict, mode_indices):

        t2 = np.zeros((max(mode_indices)+1, max(mode_indices)+1))
        t3 = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1))
        for a in mode_indices:
            for b in mode_indices:
                if self.term_label == 'EL':
                    t2[(a, b)] = self.get_ene_factor(harm_modes_dict, a, b)
                else:
                    for c in mode_indices:
                        t3[(a, b, c)] = self.get_ene_factor(harm_modes_dict, a, b, c)

        all_zeros_t2 = not np.any(t2)
        if all_zeros_t2:
            return t3
        else:
            return t2

    def get_term_tree(self, gammaCompsAll, properties_data, modes_dict, harm_modes_dict, mode_indices):

        components = {}
        for a in mode_indices:
            for b in mode_indices:
                if self.term_label == 'EL':
                    # components[((a,b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_full_factor(gammaCompsAll, properties_data,
                    #                                          modes_dict, a,b)
                    components[(a, b)] = self.get_full_factor(gammaCompsAll,
                                                              properties_data,
                                                              modes_dict, harm_modes_dict,
                                                              a, b)
                elif self.term_label == 'MECH':
                    # components[((a, b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_factor_summed(gammaCompsAll, properties_data,
                    #                                             modes_dict, mode_indices, a, b)
                    components[(a, b)] = self.get_factor_summed(gammaCompsAll,
                                                              properties_data,
                                                              modes_dict, harm_modes_dict,
                                                              mode_indices,
                                                              a, b)
        return components


    def get_all_resonances(self, modes_dict, mode_indices, w2mw1=False):
        res = {}
        for a in mode_indices:
            for b in mode_indices:
                w1, w2 = self.get_resonance_location(modes_dict, a, b)

                if w2mw1:
                    res[(a,b)] = (w1, w2-w1)
                else:
                    res[(a,b)] = (w1, w2)
        return res


    def get_intensity(self, w1, w2, properties_data,
                      modes_dict, mode_indices, harm_modes_dict, Gamma_rc, margin,
                      condition=None, collect_all=False):
        """
        gamma = prefnum * prefene * avrg * resonance
        """
        result = 0.
        # from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
        # gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
        skipped = 0

        self.layers = {}
        for a in mode_indices:
            for b in mode_indices:
                w1ab, w2ab = self.get_resonance_location(modes_dict, a, b)
                # check if resonance is in window w1,w2
                # if it's a single pair of w1,w2 -
                if w2ab>w1ab:
                    if ((np.min(w1) + margin <= w1ab <= np.max(w1) - margin)
                            and (np.min(w2) + margin <= w2ab <= np.max(w2) - margin)
                            and (w2ab-margin)>w1ab) or collect_all:
                        result += self.get_intensity_ab(a, b, w1, w2, properties_data,
                              modes_dict, harm_modes_dict, mode_indices, Gamma_rc, margin,
                              condition=condition)[0]
                    # w1ab, w2ab = self.get_resonance_location(modes_dict, a, b)
                    # # if (w1ab<np.max(w1)-50. and w1ab>np.min(w1)+50.) and (w2ab<np.max(w2)-50. and w2ab>np.min(w2)+50.):
                    # if ((np.min(w1) + margin <= w1ab <= np.max(w1) - margin)
                    #         and (np.min(w2) + margin <= w2ab <= np.max(w2) - margin)
                    #         and (w2ab-margin)>w1ab):
                    #     # print('res loc', w1ab, w2ab)
                    #     if self.term_label=='EL':
                    #         # print(self.get_full_factor(gammaCompsAll, properties_data, modes_dict, a, b)[0])
                    #         # full_prefactor * resonance
                    #         product_all, components = self.get_full_factor(gammaCompsAll, properties_data, modes_dict, a, b)
                    #         # result += (self.get_full_factor(gammaCompsAll, properties_data, modes_dict, a, b)[0]*
                    #         #            self.get_res_factor(modes_dict, w1, w2, a, b, Gamma_rc, condition))/24.
                    #         addition = (product_all*self.get_res_factor(modes_dict, w1, w2, a, b, Gamma_rc, condition))/24.
                    #     else:
                    #         # print(self.get_factor_summed(gammaCompsAll, properties_data,
                    #         #                                  modes_dict, mode_indices, a, b)[0])
                    #         # full_prefactor * resonance
                    #         # product_all, components = self.get_factor_summed(gammaCompsAll, properties_data,
                    #         #                                  modes_dict, mode_indices, a, b)
                    #
                    #         addition = (self.get_factor_summed(gammaCompsAll, properties_data,
                    #                                          modes_dict, mode_indices, a, b)[0]*
                    #                    self.get_res_factor(modes_dict, w1, w2, a, b, Gamma_rc, condition))/(-48.)
                    #     self.layers[(a,b, self.term_id)] = addition
                    #     result += addition
                    #
                    else:
                        skipped += 1
                        continue
        return result

    def get_intensity_ab(self, a, b, w1, w2, properties_data,
                      modes_dict, harm_mode_indices, mode_indices, Gamma_rc, margin,
                      condition=None):
        """
        gamma = prefnum * prefene * avrg * resonance
        """
        # from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
        gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        # w1ab, w2ab = self.get_resonance_location(modes_dict, a, b)
        # if ((np.min(w1) + margin <= w1ab <= np.max(w1) - margin)
        #         and (np.min(w2) + margin <= w2ab <= np.max(w2) - margin)
        #         and (w2ab-margin)>w1ab):
        if self.term_label=='EL':
            # full_prefactor * resonance
            product_all, components = self.get_full_factor(gammaCompsAll, properties_data,
                                                           modes_dict, harm_mode_indices, a, b)
            product_all /= 24.
            # print(f"a,b: {(a, b)}, factor: {product_all:.2e}")
            result = (product_all*self.get_res_factor(modes_dict, w1, w2, a, b, Gamma_rc, condition))
            return result, components

        else:
            product_all, components = self.get_factor_summed(gammaCompsAll, properties_data,
                                             modes_dict, harm_mode_indices, mode_indices, a, b)
            product_all /= -48.
            # print(f"a,b: {(a, b)}, factor: {product_all:.2e}")
            shortcomponents = {}
            for k,v in components.items():
                if v[0]!=0.:
                    shortcomponents[k] = {'full_product_abc':v[0], 'ene_factor':v[1]['ene_factor'],
                                          'avrg_properties':v[1]['avrg_properties'][0],
                                          'F_abc':v[1]['F_abc'],
                                          'viblevelsdiff':v[1]['viblevelsdiff']}
            # print(shortcomponents)
            # print(components)
            # quit()
            result = (product_all*self.get_res_factor(modes_dict, w1, w2, a, b, Gamma_rc, condition))

            return result, shortcomponents

    def get_vibdiff_tensor(self, modes_dict, mode_indices):

        viblevelsdiff_tensor = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1))
        viblevelsdiff_tensor2 = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1, 2))
        for a in mode_indices:
            for b in mode_indices:
                for c in mode_indices:
                    viblevelsdiff_tensor[a, b, c] = self.get_viblevelsdiff(modes_dict, a, b, c)[0]
                    viblevelsdiff_tensor2[a, b, c, :] = self.get_viblevelsdiff(modes_dict, a, b, c)[1]

        return viblevelsdiff_tensor, viblevelsdiff_tensor2




    def get_dotspectrum_df(self, properties_data, modes_dict, harm_modes_dict, mode_indices,
                           Gamma_rc, margin, condition=None):
        """

        """
        import pandas as pd

        locations_dict = self.get_all_resonances(modes_dict, mode_indices, w2mw1=False)
        from scipy.spatial import distance
        coords = np.array(list(locations_dict.keys()))
        distances = distance.cdist(coords, coords, 'euclidean')
        # print(distances)

        intensities_dict = {}

        for k in locations_dict:
            w1l, w2l = locations_dict[k]
            if w2l>w1l:
                intensities_dict[k] = self.get_intensity(w1l, w2l, properties_data,
                                                         modes_dict, harm_modes_dict,
                                                         mode_indices,
                                                         Gamma_rc, margin=margin,
                                                         condition=condition)

        data = {
            'ab': [(int(i[0]), int(i[1])) for i in list(intensities_dict.keys())],
            'intensity': [float(i) for i in list(intensities_dict.values())],
            'log10(Intensity)': np.log10(np.array(list(intensities_dict.values()))),
            'w1': [locations_dict[k][0] for k in intensities_dict.keys()],
            'w2': [locations_dict[k][1] for k in intensities_dict.keys()],
            'w2-w1': [locations_dict[k][1]-locations_dict[k][0] for k in intensities_dict.keys()]
        }

        data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        df = pd.DataFrame(data)
        df['term'] = self.term_id

        return df, distances


from dataclasses import dataclass, field

@dataclass
class SpectrumExperimentSetup:
    Gamma_rc: float
    e_selected: list
    m_selected: list
    fundamentals_harmonic: dict
    fundamentals: dict
    all_states: dict
    mode_indices: list | np.ndarray
    vib_levels_harmonic: bool
    maxYX: float


class TermStorage:
    """Stores and manages multiple Term2D objects."""
    def __init__(self):
        self.terms = {}
        self.terms_amplitudes = {}  # Dictionary: term_id -> Term2D object

    def add_term(self, term: Term2D):
        """Adds a Term2D object to storage."""
        if term.term_id in self.terms:
            print(f"Warning: Overwriting existing term {term.term_id}")
        self.terms[term.term_id] = term

    def calculate_all(self, settings):
        """Runs calculations for all stored terms."""
        for k, term in self.terms.items():
            self.terms_amplitudes[k] = term.get_intensity(settings.w1, settings.w2,
                                                          settings.properties_data,
                                                          settings.modes_dict,
                                                          settings.mode_indices,
                                                          settings.Gamma_rc,
                                                          settings.margin,
                                                          condition=settings.condition)

    def get_amplitude(self, term_id):
        """Retrieves the calculated result for a specific term."""
        return self.terms_amplitudes[term_id] if term_id in self.terms else None

    def filter_terms(self, condition_fn):
        """Filters terms based on a given function."""
        return {tid: term for tid, term in self.terms.items() if condition_fn(term)}

    def __repr__(self):
        return f"TermStorage({len(self.terms)} terms)"


def get_resonance_location(resonances_expr, modes_dict, a, b):
    """
    A resonance for this term for ab combination of modes
    """
    a, b = str(a), str(b)
    modes_dict[('zero',)] = 0.
    dict_id = {'a': a, 'b': b, 'zero': 'zero'}
    type12, type1 = resonances_expr
    m1_str, n1_str = type1.split(',')

    m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
    n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])
    w1 = modes_dict[n1_tuple] - modes_dict[m1_tuple]

    m12_str, n12_str = type12.split(',')
    m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
    n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
    w2 = modes_dict[m12_tuple] - modes_dict[n12_tuple] + w1

    return w1, w2


def get_all_resonances(resonances_expr, modes_dict, mode_indices, w2mw1=False):
    res = {}
    for a in mode_indices:
        for b in mode_indices:
            w1, w2 = get_resonance_location(resonances_expr, modes_dict, a, b)

            if w2mw1:
                res[(a,b)] = (w1, w2-w1)
            else:
                res[(a,b)] = (w1, w2)
    return res