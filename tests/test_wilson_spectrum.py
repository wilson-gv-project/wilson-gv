from wilson import spectrum
import numpy as np

def test_wilson_SpectrumEVV_gaussian():
    regions = {1: ((1180., 2050., 10.), (2309., 5350., 10.)),
               2: ((2810., 3210., 10.), (5510., 6050., 10.)),
               3: ((1961.318, 1981.318, 10.), (4931.662, 4951.662, 10.))}
    region = 1
    omega1 = np.arange(*regions[region][0])
    omega2 = np.arange(*regions[region][1])

    g16files = {'log': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2b3lypoptanhramanQZ.out',
                '3quanta': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_b3lypanhQZ_3q.out', }

    datain = {'source': 'gaussian',
              'type': 'log',
              'files': g16files}

    spec = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain)
    # setup.addTerms(*terms_selection)
    print('\n----------------------------------')
    print(spec.__dict__)
    print(dir(spec))

def test_read_csv_DB():
    import pandas as pd
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_colwidth', 2000)
    pd.set_option('display.max_columns', None)

    DB = spectrum.read_csv_DB('/mnt/c/Users/vle014/Downloads/files_fram/files_database.csv')

    # filtered_df = DB[(DB['g16_3quanta_full'].notna()) & (DB['g16_3quanta_full'] != '')]
    filtered_df = DB.query('g16_3quanta_full.notna() and g16_3quanta_full != ""')

    selected_columns_df = filtered_df[['code', 'method', 'basis_set', 'g16_3quanta_full']]
    print(selected_columns_df)
