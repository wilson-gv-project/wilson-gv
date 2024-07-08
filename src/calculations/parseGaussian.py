"""
#################################################################################################
##                                                                                             ##
##                             Parsing Gaussian output files                                   ##
##                                                                                             ##
#################################################################################################
#
# Files:
#     - .log  --- the main full output file that contains all the relevant data
#     - .fchk --- formcheck (generated from checkpoint file)
"""

import numpy as np
# np.set_printoptions(linewidth=250, suppress=True, precision=3)
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)

class GaussianDataParser(object):

    def __init__(self, all_files_dict):
        self.all_files_dict = all_files_dict
        # {'log', 'fchk', 'com'}

        self.nModesStart = None

        self.dipole_first_derivatives = None
        self.dipole_second_derivatives = None
        self.polarizability_first_derivatives = None
        self.polarizability_second_derivatives = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None

        self.fundamentals_harmonic_int = None
        self.fundamentals_anharmonic_int = None

        self.harmonic_states = None
        self.anharmonic_states = None
        self.cubic_force_constants = None
        self.quartic_constants = None

        self.equilibrium_geometry = None
        self.Q_normal_coordinates = None
        self.q_normal_coordinates_dimensionless = None

        self.atoms = None
        self.basis = None
        self.lot = None

    def getData(self, linear_molecule: bool = False):
        """Collect the data into the attributes.
        Uses methods:
            parse_output_file,
            pCubicORQuartic, getCubicPost,
            getDipoleDers_anharm,
            'polar_pkl' file <- getPolarDers(getDisplacementsPolarData,
                                getRotationMatrix, pTensor),
                                pklPolder
        """
        # {'log', 'fchk', 'com'}
        self.nModesStart = 6 if linear_molecule else 7

        results_log = parse_frequencies(self.all_files_dict['3quanta'])
        self.fundamentals_anharmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                               results_log['Fundamental Bands'][2])}
        self.fundamentals_harmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                             results_log['Fundamental Bands'][1])}

        self.fundamentals_harmonic_str = {str(k):v for k,v in self.fundamentals_anharmonic_int.items()}
        self.fundamentals_anharmonic_str = {str(k):v for k,v in self.fundamentals_harmonic_int.items()}

        self.anharmonic_states = get_allStates_fromParsedResults(results_log, anharmonic=True)
        self.harmonic_states = get_allStates_fromParsedResults(results_log, anharmonic=False)

        mu = getDipDers_log(self.all_files_dict['log'])
        self.dipole_first_derivatives = mu[0]
        self.dipole_second_derivatives = mu[1]

        alpha = getPolarDers_log(self.all_files_dict['log'])
        self.polarizability_first_derivatives = alpha[0]
        self.polarizability_second_derivatives = alpha[1]

        cubic_df = parse_cubic_constants(self.all_files_dict['log'])[0]
        selected_df = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
        cubic = selected_df.to_numpy()
        self.cubic_force_constants = get_cubic_post(len(self.fundamentals_harmonic_str), cubic)

# used in retrievedata.py
def parse_frequencies(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sections = ["Fundamental Bands", "Overtones", "Combination Bands"]
    results = {section: [] for section in sections}
    results_harm = {section: [] for section in sections}
    current_section = None

    start = False
    units_counter = 0
    for line in lines:
        if "Anharmonic Infrared Spectroscopy" in line:
            start = True
        elif "Units: Transition energies" in line:
            units_counter += 1
            if units_counter == 2:
                start = False
        elif start:
            if any(section in line for section in sections):
                current_section = next(section for section in sections if section in line)
            elif current_section:
                if '------------' not in line:
                    linelist = line.split()
                    # Insert None at the desired index (3rd position, which is index 2)
                    if len(linelist)==5 and current_section=='Combination Bands': linelist.insert(2, None)

                    results[current_section].append(linelist)

    for section, data in results.items():
        if section != 'Overtones':
            results[section] = pd.DataFrame(data[1:-1])
        else:
            results[section] = pd.DataFrame(data[2:-1])

        main_numbers = [i[0] for i in results[section][0]]
        sub_numbers = [i[2] for i in results[section][0]]

        # Insert these columns at specific positions
        results[section].insert(1, 'mode_a', main_numbers)
        results[section].insert(2, 'n_a', sub_numbers)
        results[section].drop(results[section].columns[0], axis=1, inplace=True)

        if section=='Combination Bands':
            main_numbers = [i[0] for i in results[section][1]]
            sub_numbers = [i[2] for i in results[section][1]]
            results[section].insert(3, 'mode_b', main_numbers)
            results[section].insert(4, 'n_b', sub_numbers)
            results[section].drop(results[section].columns[2], axis=1, inplace=True)

            main_numbers = [i[0] if i is not None else i for i in results[section][2]]
            sub_numbers = [i[2] if i is not None else i for i in results[section][2]]

            results[section].insert(5, 'mode_c', main_numbers)
            results[section].insert(6, 'n_c', sub_numbers)
            results[section].drop(results[section].columns[4], axis=1, inplace=True)

    return results

def get_allStates_fromParsedResults(results: pd.DataFrame, anharmonic: bool = False) -> dict:
    """results is a DataFrame from parse_frequencies()"""
    if anharmonic:
        results['Combination Bands']['mode_c'] = results['Combination Bands']['mode_c'].fillna(0)
        results['Combination Bands']['n_c'] = results['Combination Bands']['n_c'].fillna(0)

        funddict = {tuple([int(k) - 1]): float(v) for k, v in
                    zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}
        states = {
            tuple(sorted([int(k) - 1 for k in t.split()] * int(n))): float(v)
            for t, v, n in
            zip(results['Overtones']['mode_a'], results['Overtones'][2], results['Overtones']['n_a'])
        }
        combinationbands = {
            tuple(
                sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [
                    (int(t3) - 1)] * int(n3))
            ): float(v)
            for t1, t2, t3, v, n1, n2, n3 in
            zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                results['Combination Bands']['mode_c'],
                results['Combination Bands'][4], results['Combination Bands']['n_a'],
                results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
        }
        allstates_anharm = {**funddict, **states, **combinationbands}
        allstates_anharm = {key: allstates_anharm[key] for key in sorted(allstates_anharm, key=allstates_anharm.get)}
        return allstates_anharm

    else:
        funddict1 = {tuple([int(k) - 1]): float(v) for k, v in
                     zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
        states1 = {tuple(sorted([int(k) - 1 for k in t.split()] * int(n))): float(v) for t, v, n in
                   zip(results['Overtones']['mode_a'], results['Overtones'][1], results['Overtones']['n_a'])}
        combinationbands1 = {
            tuple(
                sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [
                    (int(t3) - 1)] * int(n3))
            ): float(v)
            for t1, t2, t3, v, n1, n2, n3 in
            zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                results['Combination Bands']['mode_c'],
                results['Combination Bands'][3], results['Combination Bands']['n_a'],
                results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
        }
        allstates_harm = {**funddict1, **states1, **combinationbands1}
        return allstates_harm

def getDipDers_log(logfile: str) -> tuple:
    """
    Dipole derivatives: first order and second order
    Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
    """
    dipl, units = parse_dipole_moment(logfile)
    a2d = dipl.loc[dipl['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
    shpNM = a2d.shape[0]
    a2d2 = dipl.loc[dipl['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy()
    a2d2_3d = np.zeros((shpNM, shpNM, 3))

    for i, j, xyz in zip(dipl.loc[dipl['P'] == 'P2', 'i'], dipl.loc[dipl['P'] == 'P2', 'j'], a2d2):
        a2d2_3d[int(i) - 1, int(j) - 1] = xyz
        a2d2_3d[int(j) - 1, int(i) - 1] = xyz
    return tuple([a2d, a2d2_3d])

def getPolarDers_log(logfile) -> tuple:
    """
    Polarizability derivatives: first order and second order
    Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
    """
    pol = parse_polarizability(logfile)
    shpNM = int(pol.loc[pol[0] == 'P1', 1].max())
    p1_3d = np.zeros((shpNM, 3, 3))
    for i in pol.loc[pol[0] == 'P1', 1].unique():
        # Get the rows with the current 'i' value and 'P' is 'P1', and select columns 5, 6, and 7
        xyz = pol.loc[(pol[0] == 'P1') & (pol[1] == i), [5, 6, 7]].values
        # Assign the 2D array 'xyz' to the corresponding slice of the 3D array
        p1_3d[int(i) - 1] = xyz

    nm_i = int(pol.loc[pol[0] == 'P2', 1].max())
    nm_j = int(pol.loc[pol[0] == 'P2', 2].max())

    p2_4d = np.zeros((nm_i, nm_j, 3, 3))

    for i in pol.loc[pol[0] == 'P2', 1].unique():
        for j in pol.loc[pol[0] == 'P2', 2].unique():
            # Get the rows with the current 'i' and 'j' values and 'P' is 'P2', and select columns 5, 6, and 7
            xyz = pol.loc[(pol[0] == 'P2') & (pol[1] == i) & (pol[2] == j), [5, 6, 7]].values
            # Check if 'xyz' is not empty
            if xyz.shape[0] != 0:
                # Assign the 2D array 'xyz' to the corresponding slice of the 4D array
                p2_4d[int(i) - 1, int(j) - 1] = xyz
                p2_4d[int(j) - 1, int(i) - 1] = xyz

    return tuple([p1_3d, p2_4d])

# used in retrievedata.py
def parse_cubic_constants(file_path: str) -> pd.DataFrame:
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
                units_lines.append(line.strip())
    df = pd.DataFrame(results, columns=["I", "J", "K", "FI(I,J,K)", "k(I,J,K)", "K(I,J,K)"])

    return df, units_lines

# used in retrievedata.py
def get_cubic_post(len_freq: int, cubic: np.ndarray, recipcm: bool = False):
    K3 = np.zeros((len_freq, len_freq, len_freq), dtype=np.float64)

    for fijk in cubic:
        i = int(fijk[0]) - 7
        j = int(fijk[1]) - 7
        k = int(fijk[2]) - 7
        d = np.float64(fijk[3])

        K3[i, j, k] = d
        K3[i, k, j] = d
        K3[k, j, i] = d
        K3[k, i, j] = d
        K3[j, i, k] = d
        K3[j, k, i] = d

    return K3

# used in retrievedata.py
def parse_dipole_moment(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    units_line = None
    column_names = ["P", "i", "j", "k", "X", "Y", "Z"]
    last_ijk = [np.nan, np.nan, np.nan]  # Initialize last seen "i", "j", "k" values

    from scipy import constants
    bohr_radius = constants.physical_constants['Bohr radius'][0]
    debye_to_SI = 10 ** -21 / constants.c
    au_to_SI = constants.e * bohr_radius
    debye_to_au = debye_to_SI / au_to_SI

    for line in lines:
        if line.strip().startswith('Electric Dipole'):
            start = True
        elif line.strip().startswith("Polarizability Tensor"):
            break
        elif start:
            if line.strip().startswith("Unit of the property"):
                units_line = line.strip()
            elif line.strip().startswith("P"):
                # parts = re.split("[| ]+", line.strip())
                parts = line.split('|')
                allparts = [parts[0].strip()]
                # If "i", "j", "k" values are missing, use last seen values
                if parts[1].strip() == '':
                    allparts.extend(last_ijk)
                else:
                    ijk = parts[1].strip().split()
                    ijk.extend([np.nan] * (3 - len(ijk)))
                    allparts.extend(ijk)
                allparts.extend([float(s.replace('D', 'e'))*debye_to_au for s in parts[2].split()])
                # Create a dictionary that maps column names to values
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # Fill in missing columns with None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)

    df = pd.DataFrame(results, columns=column_names)

    return df, units_line

# used in retrievedata.py
def parse_polarizability(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    units_line = None
    column_names = ["P", "i", "j", "k", "comp", "X", "Y", "Z"]
    last_ijk = [np.nan, np.nan, np.nan]  # Initialize last seen "i", "j", "k" values

    for line in lines:
        if line.strip().startswith('Polarizability Tensor'):
            start = True
        elif line.strip().startswith("============================================") and start:
            break
        elif start:
            if line.strip().startswith("Unit of the property"):
                units_line = line.strip()
            elif line.strip().startswith("P"):
                # parts = re.split("[| ]+", line.strip())
                parts = line.split('|')
                allparts = [parts[0].strip()]
                # If "i", "j", "k" values are missing, use last seen values
                if parts[1].strip() == '':
                    allparts.extend(last_ijk)
                else:
                    ijk = [int(i) for i in parts[1].strip().split()]
                    ijk.extend([np.nan] * (3 - len(ijk)))
                    allparts.extend(ijk)

                allparts.extend([parts[2].strip()])

                if len(parts[3].strip().split()) == 3:
                    allparts.extend([float(s.replace('D', 'e')) for s in parts[3].split()])
                else:
                    xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
                    xyz.extend([np.nan] * (3 - len(xyz)))
                    allparts.extend(xyz)
                # Create a dictionary that maps column names to values
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # Fill in missing columns with None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)

            elif ('|  X  |' in line or '|  Z  |') and len(line.split('|')) == 4 and not 'i' in line:
                parts = line.split('|')
                allparts = [np.nan]
                allparts.extend([np.nan, np.nan, np.nan])
                allparts.extend([parts[2].strip()])
                xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
                xyz.extend([np.nan] * (3 - len(xyz)))
                allparts.extend(xyz)
                # Create a dictionary that maps column names to values
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # Fill in missing columns with None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)
    df = pd.DataFrame(results, columns=column_names)
    array = df.to_numpy()

    # Iterate over the array in steps of 3
    for i in range(0, len(array), 3):
        # Get the current 3-row block
        block = array[i:i + 3]
        # Find the 'P' value in the second row of the block
        p_value = block[1, 0]
        ival = block[1, 1]
        kval = block[1, 2]
        jval = block[1, 3]
        # Replace 'nan' values in the first column of the block with the 'P' value
        for j in range(3):
            try:
                if np.isnan(block[j, 0]):
                    block[j, 0] = p_value
                if np.isnan(block[j, 1]):
                    block[j, 1] = ival if np.isnan(ival) else int(ival)
                else:
                    block[j, 1] = int(block[j, 1])
                if np.isnan(block[j, 2]):
                    block[j, 2] = kval if np.isnan(kval) else int(kval)
                else:
                    block[j, 2] = int(block[j, 2])
                if np.isnan(block[j, 3]):
                    block[j, 3] = jval if np.isnan(jval) else int(jval)
                else:
                    block[j, 3] = block[j, 3] if np.isnan(block[j, 3]) else int(block[j, 3])
            except TypeError:
                continue
        # Replace 'nan' values in the specified positions with the corresponding values
        if np.isnan(block[0, 6]):
            block[0, 6] = block[1, 5]
        if np.isnan(block[0, 7]):
            block[0, 7] = block[2, 5]
        if np.isnan(block[1, 7]):
            block[1, 7] = block[2, 6]

    # pd.set_option('display.float_format', '{:.7f}'.format)
    df = pd.DataFrame(array)
    return df#, units_line

# -----------------------------------------------------------------------------

# not used now
def getForceConstants_fchk(fchkfile: str):

    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Cartesian Force Constants")

    end_index = file_content.find("Cartesian 3rd/4th derivatives", start_index)

    hessian_section = file_content[start_index:end_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]

    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

# not used now
def getForceConstants_fchk2(fchkfile: str):

    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Cartesian Force Constants")

    end_index = file_content.find("Nonadiabatic coupling", start_index)
    hessian_section = file_content[start_index:end_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]
    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

# not used now
def getForceConstants_out(outfile: str):

    with open(outfile, 'r') as fileR:
        file_contentR = fileR.read()

    start_indexR = file_contentR.rfind(" Force constants in Cartesian coordinates:")
    end_indexR = file_contentR.find("Leave Link  716 at", start_indexR)
    hessian_sectionR = file_contentR[start_indexR:end_indexR]

    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    hessvecR = np.array([float(i.replace('D', 'e')) for i in hessian_sectionR.strip().split() if '.' in i])

    return hessvecR

# not used now
def getMass_fchk(fchkfile: str):
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Nuclear charges")
    end_index = file_content.find("Current cartesian coordinates", start_index)

    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]
    # supposedly projected out
    massvec = np.array([float(i) for i in coords_section.strip().split()])

    return massvec

# not used now
def getCoords_fchk(fchkfile: str):
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Current cartesian coordinates")
    end_index = file_content.find("Number of symbols in", start_index)
    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]

    # supposedly projected out
    coordsvec = np.array([float(i) for i in coords_section.strip().split()])

    return coordsvec

# used for fchk methods
class FormchkInterface:

    def __init__(self, file_path):
        self.file_path = file_path
        self.natm = NotImplemented
        self.nao = NotImplemented
        self.nmo = NotImplemented
        self.initialization()

    def initialization(self):
        self.natm = int(self.key_to_value("Number of atoms"))
        self.nao = int(self.key_to_value("Number of basis functions"))
        self.nmo = int(self.key_to_value("Number of independent functions"))

    def key_to_value(self, key, file_path=None):
        if file_path is None:
            file_path = self.file_path
        flag_read = False
        expect_size = -1
        vec = []
        with open(file_path, "r") as file:
            for l in file:
                if l[:len(key)] == key:
                    try:
                        expect_size = int(l[len(key):].split()[2])
                        flag_read = True
                        continue
                    except IndexError:
                        try:
                            return float(l[len(key):].split()[1])
                        except IndexError:
                            continue
                if flag_read:
                    try:
                        vec += [float(i) for i in l.split()]
                    except ValueError:
                        break
        if len(vec) != expect_size:
            raise ValueError("Number of expected size is not consistent with read-in size!")
        return np.array(vec)

    def total_energy(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Total Energy", file_path)

    def grad(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Cartesian Gradient", file_path).reshape((self.natm, 3))

    def dipole(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Moment", file_path)

    @staticmethod
    def tril_to_symm(tril: np.ndarray):
        dim = int(np.floor(np.sqrt(tril.size * 2)))
        if dim * (dim + 1) / 2 != tril.size:
            raise ValueError("Size " + str(tril.size) + " is probably not a valid lower-triangle matrix.")
        indices_tuple = np.tril_indices(dim)
        iterator = zip(*indices_tuple)
        symm = np.empty((dim, dim))
        for it, (row, col) in enumerate(iterator):
            symm[row, col] = tril[it]
            symm[col, row] = tril[it]
        return symm

    def hessian(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.tril_to_symm(self.key_to_value("Cartesian Force Constants", file_path))

    def polarizability(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        return self.tril_to_symm(self.key_to_value("Polarizability  ", file_path))

    def polarderiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        # two space after `Polarizability' is to avoid `Polarizability Derivative'
        pre = self.key_to_value("Polarizability Derivatives", file_path).reshape(-1, 6)
        f = []
        for i in pre:
            f.append(self.tril_to_symm(i))
        return np.array(f)

    def dipolederiv(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        return self.key_to_value("Dipole Derivatives", file_path).reshape(-1, 3)
