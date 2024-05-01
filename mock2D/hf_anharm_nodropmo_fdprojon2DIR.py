#!/usr/bin/env python

import src.spectrum.c2DIRmain as dd_ir

import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=10)

# type of spectrum - what's on axes
w1mw2 = False
start1, stop1, step = 1300., 1500., 20.
start2, stop2, step = 1300., 1500., 20.

# ranges for 2 frequencies
omega1 = np.arange(start1, stop1, step)
omega2 = np.arange(start2, stop2, step)

y =  omega2 if not w1mw2 else omega2-omega1

# meshgrid for spectrum
x_mesh, y_mesh =  np.meshgrid(omega1, y)

cfourdatafiles = {'vibdata': '../../scriptsHPC/data/hf_anharm_nodropmo_fdprojon_pkl/vibdata.pkl',
                  'cubic': '../../scriptsHPC/data/hf_anharm_nodropmo_fdprojon_pkl/cubic.pkl',
                  'dipole': '../../scriptsHPC/data/hf_anharm_nodropmo_fdprojon_pkl/dipolexyz.pkl',
                  'polar': '../../scriptsHPC/data/hf_anharm_nodropmo_fdprojon_pkl/polders.pkl'
                  }

# cfourdatafiles = {'out': '../scriptsHPC/data/anharm_hf_out',
#                   'CFF': '../scriptsHPC/data/anharm_hf_cubic',
#                   'dipolem': '../scriptsHPC/data/anharm_hf_dipole',
#                   'polar': ''
#                   }

# spectrum is computing intensities on the grid of 2 frequencies
setup = dd_ir.SpectrumEVV(omega1, omega2, data={'source': 'cfour',
                                                'type': 'pkl',
                                                'files':cfourdatafiles})

ders = setup.getDerivs()

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

def printed2DIRtensors():
    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('==================================\n')

quit()

ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]

# add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
setup.addTerms(dd_ir.picks(dd_ir.electrical_terms, ee),
               dd_ir.picks(dd_ir.mechanical_terms, mm),
               dd_ir.picks(dd_ir.electric_avrg, ee),
               dd_ir.picks(dd_ir.mechanical_avrg, mm))

gamma = 1.05

# 2D IR spectrum data
z_mesh = setup.intensity(gamma, {})

# print(z_mesh[0])
#
# for d in z_mesh[1]:
#     print(d)
#     print(z_mesh[1][d])

fig = setup.plot2Dplotly(z_mesh[0],w1mw2=w1mw2, Gamma=gamma, percent=0.85, step=20)

name = './picnew.html'
dd_ir.makehtml(name, fig)

