#!/usr/bin/env python
import time
start_time_global = time.time()
from wilson.spectrum import c2DIRmain
import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=17)
import os
# import sys
#
# print ('argument list', sys.argv)
# dic = {"True": True, "False": False}
# el = dic[sys.argv[1]]
# mech = dic[sys.argv[2]]
# region = int(sys.argv[3]) if len(sys.argv) == 4 else None
# print (f"el = {el}, mech = {mech}")

print(f"""Generated with: 
'getcwd  :      {os.getcwd()}
'__file__:      {__file__}\n\n""")


g16files = {'log': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_coh2b3lypoptanhramanQZ.out',
            '3quanta': '/home/vlew/scriptsHPC/data/dftGaussian/formaldehyde/g16_b3lypanhQZ_3q.out',}

datain = {'source': 'gaussian',
          'type': 'log',
          'files': g16files}

# print(setup.fundamentals)
# print(setup.all_states)
# print('----------\n')
# print(setup.all_states_harm)

log10=True
w1mw2=False
gamma_rc=10.

terms_selection = [0, 1], [0, 1]

regions = {1: ((1180., 2050., 10.), (2309., 5350., 10.)),
           2: ((2810., 3210., 10.), (5510., 6050., 10.)),
           3: ((1961.318, 1981.318, 10.), (4931.662, 4951.662, 10.))}

def one_spectrum_fig(el: bool, mech: bool, datain: dict, region: int = 3, gamma_rc: float = gamma_rc):

    omega1 = np.arange(*regions[region][0])
    omega2 = np.arange(*regions[region][1])

    setup = c2DIRmain.SpectrumEVV(omega1, omega2, data=datain)
    setup.addTerms(*terms_selection)

    step1 = regions[region][0][-1]
    gamma = c2DIRmain.rec_cm2rec_s(gamma_rc)
    gamma_str = f"{gamma_rc:.2f}".replace('.', 'p')
    step_str = f"{step1:.1f}".replace('.', 'p')

    name=f'./specgen_B3LYP_el{str(el)[0]}_mech{str(mech)[0]}_w1mw2{str(w1mw2)[0]}_log10{str(log10)[0]}_gamma{gamma_str}_reg{region}_step{step_str}.svg'

    start_time0 = time.time()
    Z, savedict = setup.intensity(gamma, {}, el=el, mech=mech, printdata=False)
    print('intensity\n', abs(Z)**2)
    end_time0 = time.time()
    execution_time0 = end_time0 - start_time0
    print(f"\nExecution time - setup.intensity: {execution_time0} seconds")

    start_time = time.time()
    setup.plot2Dmatplotlib(Z, w1mw2=w1mw2, nametuple=(name, __file__, "B3LYP/cc-pVQZ"), Gamma=gamma, el=el, mech=mech, dpi=200, log10=log10)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time - setup.plot2Dmatplotlib: {execution_time} seconds")

    end_time_global = time.time()
    execution_time_global = end_time_global - start_time_global

    hours, rem = divmod(execution_time_global, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(f"Execution time - global: {execution_time_global} seconds")
    print('\n===============================================================\n   Next spectrum below\n')

list_figs = [(True, False), (False, True), (True, True)]
# for s in list_figs:
#     one_spectrum_fig(el=s[0], mech=s[1], datain=datain, region=1)

import pandas as pd
import json

# Assuming your data is stored in a JSON file
with open('calcfiles.json', 'r') as file:
    data = json.load(file)
print(data)

flattened_data = []

# First pass to determine all possible keys in the file_path dictionaries
all_keys = set()

for molecule_name, molecule_data in data['molecules'].items():
    for method, method_data in molecule_data.items():
        if method.startswith('_comment'):
            continue  # Skip comments
        if isinstance(method_data, dict):
            for basis_set, basis_set_data in method_data.items():
                if isinstance(basis_set_data, dict):
                    for file_category, files in basis_set_data.items():
                        if isinstance(files, dict):
                            for file_type, file_paths in files.items():
                                if isinstance(file_paths, dict):
                                    all_keys.update(file_paths.keys())

# Second pass to create the flattened data structure
for molecule_name, molecule_data in data['molecules'].items():
    for method, method_data in molecule_data.items():
        if method.startswith('_comment'):
            continue  # Skip comments
        if isinstance(method_data, dict):
            for basis_set, basis_set_data in method_data.items():
                if isinstance(basis_set_data, dict):
                    for file_category, files in basis_set_data.items():
                        if isinstance(files, dict):
                            for file_type, file_paths in files.items():
                                if isinstance(file_paths, dict):
                                    record = {
                                        'molecule_name': molecule_name,
                                        'method': method,
                                        'basis_set': basis_set,
                                        'file_category': file_category,
                                        'file_type': file_type
                                    }
                                    # Add each possible key to the record, with default empty string if not present
                                    for key in all_keys:
                                        record[key] = file_paths.get(key, '')
                                    flattened_data.append(record)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(flattened_data)

print(df)