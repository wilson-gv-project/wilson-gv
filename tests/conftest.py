import pytest

from CQCParse.parsing import GaussianParser, GaussianOutput
from CQCParse.relay import DataVault

from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices

from .test_config import SimulationConfig


@pytest.fixture
def terms_dict_setup():
    allterms_str = { 0:
                     # {'resonance': (('a+b,a', 'zero,a'), None),
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24},
                     1:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24},
                     2:
                         {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+b+c,zero', 'c,a+b'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'b', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/48.},
                     3:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a+c,b', 'b+c,a'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'b',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 1.,
                          'termA_pref': -1/48.},
                     4:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',))),
                          'non_averaged_props': ('F', ('b', 'c', 'c',)),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1/48.},
                     5:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': 0.5,
                          'termA_pref': -1 / 48.},
                     6:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('a,a+b', 'b,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          'non_averaged_props': (('F', ('b', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1 / 48.},
                     7:
                         {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': ('b,a+b', 'a,zero'),
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                          # 'CFF': ('F', ('a', 'c', 'c',), tuple()), # old
                          'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                          'vibene_denom': ('a','b','c'),
                          'termB_pref': -0.5,
                          'termA_pref': -1 / 48.}
                     }
    return allterms_str


@pytest.fixture
def FORM_setup_parser():
    molecule, method, basis = 'FORM', 'B3LYP', 'cc_pVQZ'

    data_vault = DataVault("/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv")
    dataframe_gaussian = data_vault.getting_files_DB("gaussian")

    aa = dataframe_gaussian[(dataframe_gaussian['code'] == molecule)
                            & (dataframe_gaussian['method'] == method)
                            & (dataframe_gaussian['basis_set'] == basis)]['g16_3quanta_full']
    filename = aa.iloc[0]
    gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)

    parser = GaussianParser(gout)
    parser.load()
    return parser


@pytest.fixture
def avrg_xyz_indices():
    return get_AlphaBetaGammaDelta_indices(num_f=4)


@pytest.fixture
def spectrum_setup(avrg_xyz_indices):
    return SimulationConfig(
        gammaCompsAll=avrg_xyz_indices,
        molecule='FORM',
        method='B3LYP',
        basis='cc_pVQZ',
        Gamma=4.7,
        diag_margin=5.0,
        start1=1000.0,
        end1=3150.0,
        step1=79.8,
        start2=1000.0,
        end2=6150.0,
        step2=79.8,
        old_new_dict={3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5},
        elevels='anharm',
        enelvl=True
    )