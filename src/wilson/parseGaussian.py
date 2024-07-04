import numpy as np
# np.set_printoptions(linewidth=250, suppress=True, precision=3)
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)

def getForceConstants_fchk(fchkfile: str):

    # Read the contents of the file fchk
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    hessian_start_index = file_content.find("Cartesian Force Constants")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Cartesian 3rd/4th derivatives", hessian_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    hessian_section = file_content[hessian_start_index:dipole_line_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]

    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

def getForceConstants_fchk2(fchkfile: str):

    # Read the contents of the file fchk
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    hessian_start_index = file_content.find("Cartesian Force Constants")

    # Find the index where the line with "Total dipole moment" appears
    # dipole_line_index = file_content.find("Cartesian 3rd/4th derivatives", hessian_start_index)
    dipole_line_index = file_content.find("Nonadiabatic coupling", hessian_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    hessian_section = file_content[hessian_start_index:dipole_line_index]
    brinx = hessian_section.find("\n")
    hessian_section = hessian_section[brinx:]
    # print(hessian_section)
    # supposedly projected out
    hessvec = np.array([float(i) for i in hessian_section.strip().split()])

    return hessvec

def getForceConstants_out(outfile: str):

    # Read the contents of the file
    with open(outfile, 'r') as fileR:
        file_contentR = fileR.read()

    # Find the index where "Molecular hessian" appears
    hessian_start_indexR = file_contentR.rfind(" Force constants in Cartesian coordinates:")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_indexR = file_contentR.find("Leave Link  716 at", hessian_start_indexR)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    hessian_sectionR = file_contentR[hessian_start_indexR:dipole_line_indexR]


    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    brinxR = hessian_sectionR.find("\n")
    hessian_sectionR = hessian_sectionR[brinxR:]
    # print(hessian_sectionR)
    # print('\n', hessian_sectionR)
    hessvecR = np.array([float(i.replace('D', 'e')) for i in hessian_sectionR.strip().split() if '.' in i])

    return hessvecR

def getMass_fchk(fchkfile: str):
    # Read the contents of the file fchk
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    coords_start_index = file_content.find("Nuclear charges")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Current cartesian coordinates", coords_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[coords_start_index:dipole_line_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]
    # print(coords_section)
    # supposedly projected out
    massvec = np.array([float(i) for i in coords_section.strip().split()])

    return massvec

def getCoords_fchk(fchkfile: str):
    # Read the contents of the file fchk
    with open(fchkfile, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    coords_start_index = file_content.find("Current cartesian coordinates")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Number of symbols in", coords_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[coords_start_index:dipole_line_index]
    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:]
    # print(coords_section)
    # supposedly projected out
    coordsvec = np.array([float(i) for i in coords_section.strip().split()])

    # print('COORDINATES')
    # print(coordsvec)

    return coordsvec

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
                    # print(linelist)
                    # Insert None at the desired index (3rd position, which is index 2)
                    if len(linelist)==5 and current_section=='Combination Bands': linelist.insert(2, None)

                    results[current_section].append(linelist)

    # Convert results to pandas DataFrames
    for section, data in results.items():
        if section != 'Overtones':
            results[section] = pd.DataFrame(data[1:-1])
        # elif section == 'Overtones':
        else:
            results[section] = pd.DataFrame(data[2:-1])
        # elif section == 'Combination Bands':
        #     results[section] = pd.DataFrame(data[1:-1])
        # print(results[section])
        # Extracting the digits before and in the parentheses
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

    # save the header of dataframes of the dictionary results and remove it from dataframe data
    # results = {section: df.iloc[1:] for section, df in results.items()}
    # i also want to change values which contain () such as 2(1) and  1(1) - to 2 and 1, i.e. remove parentheses
    # results = {section: df.replace(r'\(.*\)', '', regex=True) for section, df in results.items()}
    # remove first row of each section dataframe
    # results = {section: df.iloc[1:] for section, df in results.items()}
    # for section, df in results.items():
    #     df.dropna(inplace=True)

    return results

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
                # print(line)
                units_lines.append(line.strip())
    # Convert results to pandas DataFrame
    df = pd.DataFrame(results, columns=["I", "J", "K", "FI(I,J,K)", "k(I,J,K)", "K(I,J,K)"])

    return df, units_lines

def get_cubic_post(freq: dict, cubic: np.ndarray, recipcm: bool = False):
    n = len(freq)
    K3 = np.zeros((n, n, n), dtype=np.float64)

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
    # Convert results to pandas DataFrame
    df = pd.DataFrame(results, columns=column_names)

    return df, units_line

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
                # print(allparts)
                # Create a dictionary that maps column names to values
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # Fill in missing columns with None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)

            elif ('|  X  |' in line or '|  Z  |') and len(line.split('|')) == 4 and not 'i' in line:
                parts = line.split('|')
                allparts = [np.nan]
                allparts.extend([np.nan, np.nan, np.nan])
                # allparts.extend([parts[1].strip()])
                allparts.extend([parts[2].strip()])
                xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
                xyz.extend([np.nan] * (3 - len(xyz)))
                allparts.extend(xyz)
                # Create a dictionary that maps column names to values
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # Fill in missing columns with None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)
    # Convert results to pandas DataFrame
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

    # Assuming 'array' is your numpy array
    # pd.set_option('display.float_format', '{:.7f}'.format)
    df = pd.DataFrame(array)
    return df#, units_line

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

class FchkForceDerivatives:
    """Holder class for force constant derivatives coming out of an fchk file"""
    def __init__(self, derivs):
        self.derivs = derivs
        self._n = None

    def __len__(self):
        return len(self.derivs)

    def _get_n(self):
        if self._n is None:
            l = len(self)
            # had to use Mathematica to get this from the cubic poly
            #  2*(3n-6)*(3n)^2 == 2*l - 2*(3n-6)*(3n)
            l_quad = 81*l**2 + 3120*l - 5292
            l_body = (3*np.sqrt(l_quad) - 27*l - 520)
            if l_body > 0:
                l1 = l_body**(1/3)
            else:
                l1 = -(-l_body)**(1/3)
            n = (1/18)*( 10 + (2**(1/3))*( l1 - 86/l1) )
            self._n = int(np.ceil(n)) # precision issues screw this up in python, but not in Mathematica (I think)
        return self._n

    @property
    def n(self):
        return self._get_n()

    def _get_third_derivs(self):
        # fourth and third derivs are same len
        d = self.derivs
        return d[:int(len(d)/2)]

    def _get_fourth_derivs(self):
        # fourth and third derivs are same len
        d = self.derivs
        return d[int(len(d)/2):]

    @property
    def third_derivs(self):
        return self._get_third_derivs()

    @property
    def fourth_derivs(self):
        return self._get_fourth_derivs()
    @staticmethod
    def _fill_3d_tensor(n, derivs):
        """Makes and fills a 3D tensor for our derivatives
        :param n:
        :type n:
        :param derivs:
        :type derivs:
        :return:
        :rtype: np.ndarray
        """
        dim_1 = (3*n)
        mode_n = 3*n-6

        full_array_1 = np.zeros((mode_n, dim_1, dim_1))
        # set the lower triangle
        inds_1, inds_2 = np.tril_indices(dim_1)
        l_per = len(inds_1)
        main_ind = np.broadcast_to(np.arange(mode_n)[:, np.newaxis], (mode_n, l_per)).flatten()
        sub_ind_1 = np.broadcast_to(inds_1, (mode_n, l_per)).flatten()
        sub_ind_2 = np.broadcast_to(inds_2, (mode_n, l_per)).flatten()
        inds = ( main_ind, sub_ind_1, sub_ind_2 )
        full_array_1[inds] = derivs
        # set the upper triangle
        inds2 = ( main_ind, sub_ind_2, sub_ind_1 ) # basically just taking a transpose
        full_array_1[inds2] = derivs

        return full_array_1
    def _get_third_deriv_array(self):
        """we make the appropriate 3D tensor from a bunch of 2D tensors
        :return:
        :rtype: np.ndarray
        """
        n = self.n
        derivs = self.third_derivs
        return self._fill_3d_tensor(n, derivs)
    @property
    def third_deriv_array(self):
        return self._get_third_deriv_array()

    # def _get_fourth_deriv_array(self):
    #     """We'll make our array of fourth derivs exactly the same as the third
    #     admittedly this should be a 4D tensor, but we only have the diagonal elements so it's just 3D
    #     I should make it a 4D sparse matrix honestly... Apparently we won't need many terms in the 4D tensor so it might
    #     make sense to handle that bloop doop bloop in the schmoop
    #     :return:
    #     :rtype: np.ndarray
    #     """
    #     n = self.n
    #     derivs = self.fourth_derivs
    #     from Sparse import SparseArray
    #     return SparseArray.from_diag(self._fill_3d_tensor(n, derivs))
    @property
    def fourth_deriv_array(self):
        return self._get_fourth_deriv_array()