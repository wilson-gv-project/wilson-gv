from wilson import spectrum
from wilson.relay import DataVault

import numpy as np
import pandas as pd


def test_wilson_SpectrumEVV_gaussian():
    regions = {1: ((1180., 2050., 10.), (2309., 5350., 10.)),
               2: ((2810., 3210., 10.), (5510., 6050., 10.)),
               3: ((1961.318, 1981.318, 10.), (4931.662, 4951.662, 10.))}
    region = 1
    omega1 = np.arange(*regions[region][0])
    omega2 = np.arange(*regions[region][1])

    g16files = {'log': './test_files_gaussian/g16_inputFull_3q.out',
                '3quanta': './test_files_gaussian/g16_inputFull_3q.out', }

    datain = {'source': 'gaussian',
              'type': 'log',
              'files': g16files}

    spec = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain)
    # setup.addTerms(*terms_selection)
    print('\n----------------------------------')
    print(spec.__dict__)
    print(dir(spec))

def test_read_csv_DB():

    pd.set_option('display.width', 5000)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.max_columns', None)

    data_vault = DataVault()
    DB = data_vault.read_csv_DB()

    # filtered_df = DB[(DB['g16_3quanta_full'].notna()) & (DB['g16_3quanta_full'] != '')]
    filtered_df = DB.query('g16_3quanta_full.notna() and g16_3quanta_full != ""')

    selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'g16_3quanta_full']]
    print('\n----------------------------------')
    print(selected_columns_df)


def test_make_DatainputDict():
    mol_tuple = ('FORM', 'B3LYP', 'cc_pVDZ')
    data_vault = DataVault()

    inpdict = data_vault.make_DatainputDict('gaussian', mol_tuple)

    print('\n', inpdict)

def test_getting_files_DB():
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.max_columns', None)

    data_vault = DataVault()

    selected_columns_df = data_vault.getting_files_DB('gaussian', printing=True)
    print('\n----------------------------------\nFrom Gaussian\n')
    print(selected_columns_df)

    selected_columns_df = data_vault.getting_files_DB('cfour', printing=True)
    print('\n----------------------------------\nFrom CFOUR\n')
    print(selected_columns_df)