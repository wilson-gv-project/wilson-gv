from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.relay import DataVault

import numpy as np
import pandas as pd


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