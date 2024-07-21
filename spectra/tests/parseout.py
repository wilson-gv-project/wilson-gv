#!/usr/bin/env python
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)

import numpy as np
np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=10)
from scipy import constants

mute_vib = True

from src.wilson.retrievedata import CFOURdata, GaussianData
# is it really QZ?????
vibdata_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVQZ/vibdata.pkl'
cubic_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVQZ/cubic.pkl'
dipole_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVQZ/dipolexyz.pkl'
polar_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVQZ/polar.pkl'

files = {'vibdata': vibdata_path, 'cubic': cubic_path, 'dipole': dipole_path, 'polar': polar_path}
data = {'source': 'cfour', 'type': 'pkl', 'files': files}
cfourparser = CFOURdata(data)

allstates_CFOUR = cfourparser.getAllStates()

if not mute_vib:
    print(allstates_CFOUR)
    for i in allstates_CFOUR:
        print(i, allstates_CFOUR[i])

    print('\n-------------------------------------\n')

gaussian_path = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2b3lypoptanhramanQZ.out'
data = {'source': 'gaussian', 'type': 'log', 'files': {'log': gaussian_path}}
gaussianparser = GaussianData(data)

allstates_Gaussian = gaussianparser.getAllStates()

if not mute_vib:
    print(allstates_Gaussian)

    for i in allstates_Gaussian:
        print(i, allstates_Gaussian[i])

    print('\n-------------------------------------\n')


dipole_derivs_CFOUR1, dipole_derivs_CFOUR2 = cfourparser.getDipDers()

# in au
# print(dipole_derivs_CFOUR1)
# print(dipole_derivs_CFOUR2)

# in debye
dipole_derivs_Gaussian1, dipole_derivs_Gaussian2 = gaussianparser.getDipDers()

bohr_radius = constants.physical_constants['Bohr radius'][0]
debye_to_SI = 10**-21/constants.c
au_to_SI = constants.e * bohr_radius
debye_to_au = debye_to_SI / au_to_SI
# print(debye_to_au)

np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True, precision=12)
# print(dipole_derivs_Gaussian1)
# print(dipole_derivs_Gaussian1 * debye_to_au)
# print(dipole_derivs_Gaussian2)

print('\n-------------------------------------')
print('Other Gaussian input_data_info - TZ\n')

gaussian_path = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2hfanh_newopt_raman_new.out'
data = {'source': 'gaussian', 'type': 'log', 'files': {'log': gaussian_path}}
gaussianparser = GaussianData(data)

allstates_Gaussian = gaussianparser.getAllStates()

funds = {k: v for k, v in allstates_Gaussian.items() if len(k) == 1}

# Sorting the dictionary by keys
sorted_data = {k: funds[k] for k in sorted(funds)}
freqs = np.array(list(sorted_data.values()))
# Printing the sorted dictionary
print(repr(freqs))
print(sorted_data)

if not mute_vib:
    print(allstates_Gaussian)

    for i in allstates_Gaussian:
        print(i, allstates_Gaussian[i])

    print('\n-------------------------------------\n')


dipole_derivs_CFOUR1, dipole_derivs_CFOUR2 = cfourparser.getDipDers()

# in au
print(dipole_derivs_CFOUR1)
print(repr(dipole_derivs_CFOUR1))
# print(dipole_derivs_CFOUR2)

# in debye
dipole_derivs_Gaussian1, dipole_derivs_Gaussian2 = gaussianparser.getDipDers()

debye_to_SI = 10**-21/constants.c
au_to_SI = constants.e * bohr_radius
debye_to_au = debye_to_SI / au_to_SI
print(debye_to_au)

np.set_printoptions(linewidth=350, threshold=sys.maxsize, suppress=True)
print(dipole_derivs_Gaussian1)
print(dipole_derivs_Gaussian1 * debye_to_au)
print(repr(dipole_derivs_Gaussian1 * debye_to_au))

# print(dipole_derivs_Gaussian2)

factors_array = np.sqrt(constants.hbar / (4 * np.pi**2 * constants.c**2 * freqs))
factor = np.sqrt(constants.hbar / (4 * np.pi**2 * constants.c**2))

print(factor)
print(factors_array)

quit()

from scriptsHPC.utils import parseGaussian
# file_path = '/home/vlew/scriptsHPC/input_data_info/dftGaussian/formaldehyde/g16_coh2b3lypoptanhramanDZ.out'
file_path = '/home/vlew/scriptsHPC/input_data_info/coh2aldehyde_HFcc-pVQZ/g16_coh2hfoptanhramanQZ.out'

results = parseGaussian.parse_frequencies(file_path)
# get a dictionary from results['Fundamental Bands'] dataframe_gaussian with keys as the first column and values as the third column
f = {tuple([int(k)]): float(v) for k, v in zip(results['Fundamental Bands'][0], results['Fundamental Bands'][2])}
states = {tuple([int(k) for k in t.split()]+[int(k) for k in t.split()]): float(v) for t, v in
          zip(results['Overtones'][0], results['Overtones'][2])}
combinationbands = {tuple([int(k) for k in t1.split()] + [int(l) for l in t2.split()]): float(v) for t1, t2, v in
                    zip(results['Combination Bands'][0], results['Combination Bands'][1], results['Combination Bands'][3])}

allstates = {**f, **states, **combinationbands}
print('\n', allstates)


from scriptsHPC.utils import parseCFOUR
results = parseCFOUR.get_anharmonic_fundamentals(file_path, 'pkl')
print(results)

quit()

# Print the results
# for section, df in results.items():
#     print(f"{section}:\n{df}\n")
# print('\n>>>>> Fundamentals\n---------------------------------------\n')

def parse_cubic_constants(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    start2 = False
    units_lines = []

    for line in lines:
        if "CUBIC FORCE CONSTANTS IN NORMAL MODES" in line:
            start = True
        elif line.strip().startswith("Num. of 3rd derivatives"):
            break
        elif start:
            if line.strip().startswith("I"):
                start2 = True
            elif start2 and line.strip() and not line.isspace():
                parts = line.split()
                results.append(parts)
            elif line.strip().startswith(': FI =') or line.strip().startswith(': k  =') or line.strip().startswith(': K  ='):
                # print(line)
                units_lines.append(line.strip())
    # Convert results to pandas DataFrame
    df = pd.DataFrame(results, columns=["I", "J", "K", "FI(I,J,K)", "k(I,J,K)", "K(I,J,K)"])

    return df, units_lines

# Use the function
df, units_lines = parse_cubic_constants(file_path)

# nm is max value from column I of df
#     I  J  K    FI(I,J,K)   k(I,J,K)  K(I,J,K)
# 0   1  1  1  -1352.37376  -21.04341  -0.71525
# 1   2  1  1     15.32864    0.19078   0.00648
# 2   2  2  1     67.35880    0.67056   0.02279
# 3   2  2  2   -575.59346   -4.58323  -0.15578
nm = int(df['I'].max())

cff_tensor_FI = np.zeros((nm, nm, nm))
cff_tensor_k = np.zeros((nm, nm, nm))
cff_tensor_K = np.zeros((nm, nm, nm))

# there could be 3 tensors from this df , the values are in the last 3 columns
for i, j, k, FI, k_, K in zip(df['I'], df['J'], df['K'], df['FI(I,J,K)'], df['k(I,J,K)'], df['K(I,J,K)']):
    # and all permutations of i, j, k would have the same values but can it be written in compact way
    cff_tensor_FI[int(i)-1, int(j)-1, int(k)-1] = FI
    cff_tensor_FI[int(j)-1, int(i)-1, int(k)-1] = FI
    cff_tensor_FI[int(k)-1, int(i)-1, int(j)-1] = FI
    cff_tensor_FI[int(i)-1, int(k)-1, int(j)-1] = FI
    cff_tensor_FI[int(j)-1, int(k)-1, int(i)-1] = FI
    cff_tensor_FI[int(k)-1, int(j)-1, int(i)-1] = FI

    cff_tensor_k[int(i)-1, int(j)-1, int(k)-1] = k_
    cff_tensor_k[int(j)-1, int(i)-1, int(k)-1] = k_
    cff_tensor_k[int(k)-1, int(i)-1, int(j)-1] = k_
    cff_tensor_k[int(i)-1, int(k)-1, int(j)-1] = k_
    cff_tensor_k[int(j)-1, int(k)-1, int(i)-1] = k_
    cff_tensor_k[int(k)-1, int(j)-1, int(i)-1] = k_

    cff_tensor_K[int(i)-1, int(j)-1, int(k)-1] = K
    cff_tensor_K[int(j)-1, int(i)-1, int(k)-1] = K
    cff_tensor_K[int(k)-1, int(i)-1, int(j)-1] = K
    cff_tensor_K[int(i)-1, int(k)-1, int(j)-1] = K
    cff_tensor_K[int(j)-1, int(k)-1, int(i)-1] = K
    cff_tensor_K[int(k)-1, int(j)-1, int(i)-1] = K

# print(cff_tensor_FI)
print(cff_tensor_k)
# print(cff_tensor_K)

# Print the results
print(df)
# units_lines are not working now
print(units_lines)
print('\n>>>>> Cubic derivatives\n---------------------------------------\n')
quit()


# def parse_dipole_moment(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     results = []
#     start = False
#     units_line = None
#     column_names = ["P", "i", "j", "k", "X", "Y", "Z"]
#     last_ijk = [np.nan, np.nan, np.nan]  # Initialize last seen "i", "j", "k" values
#
#     for line in lines:
#         if line.strip().startswith('Electric Dipole'):
#             start = True
#         elif line.strip().startswith("Polarizability Tensor"):
#             break
#         elif start:
#             if line.strip().startswith("Unit of the property"):
#                 units_line = line.strip()
#             elif line.strip().startswith("P"):
#                 # parts = re.split("[| ]+", line.strip())
#                 parts = line.split('|')
#                 allparts = [parts[0].strip()]
#                 # If "i", "j", "k" values are missing, use last seen values
#                 if parts[1].strip() == '':
#                     allparts.extend(last_ijk)
#                 else:
#                     ijk = parts[1].strip().split()
#                     ijk.extend([np.nan] * (3 - len(ijk)))
#                     allparts.extend(ijk)
#                 allparts.extend([float(s.replace('D', 'e')) for s in parts[2].split()])
#                 # Create a dictionary that maps column names to values
#                 row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
#                 # Fill in missing columns with None
#                 row = [row_dict.get(column_name, np.nan) for column_name in column_names]
#                 results.append(row)
#     # Convert results to pandas DataFrame
#     df = pd.DataFrame(results, columns=column_names)
#
#     return df, units_line


df, units_line = parseGaussian.parse_dipole_moment(file_path)
# pd.set_option('display.float_format', '{:.7f}'.format)

# Print the results
# print(df)
# print(units_line)
# print('\n>>>>> Dipole moment\n---------------------------------------\n')

# from df dataframe_gaussian get P column - P1 values from X, Y, Z columns - into a2d numpy array
# but only where P is P1, also without the P column
a2d = df.loc[df['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
a2d2 = df.loc[df['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy() # this is a 2d array, but i need 3d based on i and j indices
a2d2_3d = np.zeros((6, 6, 3))

for i, j, xyz in zip(df.loc[df['P'] == 'P2', 'i'], df.loc[df['P'] == 'P2', 'j'], a2d2):
    a2d2_3d[int(i)-1, int(j)-1] = xyz
    a2d2_3d[int(j)-1, int(i)-1] = xyz

# print(a2d)
# print(a2d2_3d)



# def parse_polarizability(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     results = []
#     start = False
#     units_line = None
#     column_names = ["P", "i", "j", "k", "comp", "X", "Y", "Z"]
#     last_ijk = [np.nan, np.nan, np.nan]  # Initialize last seen "i", "j", "k" values
#
#     for line in lines:
#         if line.strip().startswith('Polarizability Tensor'):
#             start = True
#         elif line.strip().startswith("============================================") and start:
#             break
#         elif start:
#             if line.strip().startswith("Unit of the property"):
#                 units_line = line.strip()
#             elif line.strip().startswith("P"):
#                 # parts = re.split("[| ]+", line.strip())
#                 parts = line.split('|')
#                 allparts = [parts[0].strip()]
#                 # If "i", "j", "k" values are missing, use last seen values
#                 if parts[1].strip() == '':
#                     allparts.extend(last_ijk)
#                 else:
#                     ijk = [int(i) for i in parts[1].strip().split()]
#                     ijk.extend([np.nan] * (3 - len(ijk)))
#                     allparts.extend(ijk)
#
#                 allparts.extend([parts[2].strip()])
#
#                 if len(parts[3].strip().split()) == 3:
#                     allparts.extend([float(s.replace('D', 'e')) for s in parts[3].split()])
#                 else:
#                     xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
#                     xyz.extend([np.nan] * (3 - len(xyz)))
#                     allparts.extend(xyz)
#                 # print(allparts)
#                 # Create a dictionary that maps column names to values
#                 row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
#                 # Fill in missing columns with None
#                 row = [row_dict.get(column_name, np.nan) for column_name in column_names]
#                 results.append(row)
#
#             elif ('|  X  |' in line or '|  Z  |') and len(line.split('|')) == 4 and not 'i' in line:
#                 parts = line.split('|')
#                 allparts = [np.nan]
#                 allparts.extend([np.nan, np.nan, np.nan])
#                 # allparts.extend([parts[1].strip()])
#                 allparts.extend([parts[2].strip()])
#                 xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
#                 xyz.extend([np.nan] * (3 - len(xyz)))
#                 allparts.extend(xyz)
#                 # Create a dictionary that maps column names to values
#                 row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
#                 # Fill in missing columns with None
#                 row = [row_dict.get(column_name, np.nan) for column_name in column_names]
#                 results.append(row)
#     # Convert results to pandas DataFrame
#     df = pd.DataFrame(results, columns=column_names)
#     array = df.to_numpy()
#
#     # Iterate over the array in steps of 3
#     for i in range(0, len(array), 3):
#         # Get the current 3-row block
#         block = array[i:i + 3]
#         # Find the 'P' value in the second row of the block
#         p_value = block[1, 0]
#         ival = block[1, 1]
#         kval = block[1, 2]
#         jval = block[1, 3]
#         # Replace 'nan' values in the first column of the block with the 'P' value
#         for j in range(3):
#             try:
#                 if np.isnan(block[j, 0]):
#                     block[j, 0] = p_value
#                 if np.isnan(block[j, 1]):
#                     block[j, 1] = ival if np.isnan(ival) else int(ival)
#                 else:
#                     block[j, 1] = int(block[j, 1])
#                 if np.isnan(block[j, 2]):
#                     block[j, 2] = kval if np.isnan(kval) else int(kval)
#                 else:
#                     block[j, 2] = int(block[j, 2])
#                 if np.isnan(block[j, 3]):
#                     block[j, 3] = jval if np.isnan(jval) else int(jval)
#                 else:
#                     block[j, 3] = block[j, 3] if np.isnan(block[j, 3]) else int(block[j, 3])
#             except TypeError:
#                 continue
#         # Replace 'nan' values in the specified positions with the corresponding values
#         if np.isnan(block[0, 6]):
#             block[0, 6] = block[1, 5]
#         if np.isnan(block[0, 7]):
#             block[0, 7] = block[2, 5]
#         if np.isnan(block[1, 7]):
#             block[1, 7] = block[2, 6]
#
#     # Assuming 'array' is your numpy array
#     # pd.set_option('display.float_format', '{:.7f}'.format)
#     df = pd.DataFrame(array)
#     return df, units_line


df, units_line = parseGaussian.parse_polarizability(file_path)

nm = int(df.loc[df[0] == 'P1', 1].max())

p1_3d = np.zeros((nm, 3, 3))

# Iterate over the unique 'i' values for rows where 'P' is 'P1'
for i in df.loc[df[0] == 'P1', 1].unique():
    # Get the rows with the current 'i' value and 'P' is 'P1', and select columns 5, 6, and 7
    xyz = df.loc[(df[0] == 'P1') & (df[1] == i), [5, 6, 7]].values
    # Assign the 2D array 'xyz' to the corresponding slice of the 3D array
    p1_3d[int(i)-1] = xyz

# print(p1_3d)

# Get the maximum 'i' and 'j' values for rows where 'P' is 'P2'
nm_i = int(df.loc[df[0] == 'P2', 1].max())
nm_j = int(df.loc[df[0] == 'P2', 2].max())

# Initialize a 4D numpy array with zeros
p2_4d = np.zeros((nm_i, nm_j, 3, 3))

# Iterate over the unique 'i' and 'j' values for rows where 'P' is 'P2'
# for i in df.loc[df[0] == 'P2', 1].unique():
#     for j in df.loc[df[0] == 'P2', 2].unique():
#         # Get the rows with the current 'i' and 'j' values and 'P' is 'P2', and select columns 5, 6, and 7
#         xyz = df.loc[(df[0] == 'P2') & (df[1] == i) & (df[2] == j), [5, 6, 7]].values
#         # Assign the 2D array 'xyz' to the corresponding slice of the 4D array
#         p2_4d[int(i)-1, int(j)-1] = xyz
# Iterate over the unique 'i' and 'j' values for rows where 'P' is 'P2'
# Iterate over the unique 'i' and 'j' values for rows where 'P' is 'P2'
for i in df.loc[df[0] == 'P2', 1].unique():
    for j in df.loc[df[0] == 'P2', 2].unique():
        # Get the rows with the current 'i' and 'j' values and 'P' is 'P2', and select columns 5, 6, and 7
        xyz = df.loc[(df[0] == 'P2') & (df[1] == i) & (df[2] == j), [5, 6, 7]].values
        # Check if 'xyz' is not empty
        if xyz.shape[0] != 0:
            # Assign the 2D array 'xyz' to the corresponding slice of the 4D array
            p2_4d[int(i)-1, int(j)-1] = xyz
            p2_4d[int(j)-1, int(i)-1] = xyz

print(p2_4d)

# Print the results
print(df)
print(units_line)
print('\n>>>>> Polarizability\n---------------------------------------\n')
