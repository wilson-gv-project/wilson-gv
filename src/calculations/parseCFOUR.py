"""
########################################################################################################################
##                                                                                                                    ##
##                                             Parsing CFOUR output files                                             ##
##                                                                                                                    ##
########################################################################################################################
#
# Files:
#     - outfile0.out/outfile(n).out  --- the main full output file:
#                            1)
#     - MOLDEN                       --- from starting anharmonic parallel run:
#                            1) equilibrium geometry (a.u.) ; 2) normal coordinates, non-mass-weighted (a.u.)
#     - NORMCO                       --- mass-weighted coordinates: equilibrium and normal coordinates, and frequencies
#     - FCMFINAL                     --- non-mass-weighted Hessian matrix in columns (xyz)
#     - DIPOL                        --- dipole moment (a.u.)
#     - DIPDER                       --- dipole moment first order derivatives (cartesian)
#     - POLAR                        --- static polarizability (a.u.)
#     - out                          --- the final output file in anharmonic parallel procedure:
#                            1) All levels with up to three quanta frequencies; 2) equilibrium geometry;
#                            3) normal coordinates, non-mass-weighted (a.u.); 4) F(IJKK)/a.u ; 5) F(IJKK)/cm-1 ;
#                            6) harmonic and fundamental frequencies and intensities
#     - dipolex(yz)                  --- dipole moment (1st-2nd-3rd) order derivatives (normal coordinates)
#     - cubic                        --- cubic force constants in cm-1 in dimensionless normal modes
#     - quartic                      --- quartic force constants in cm-1 in dimensionless normal modes
"""

import numpy as np
import os
import pickle

class CFOURdataParser(object):
    """A class that contains parsed CFOUR output data"""
    def __init__(self, all_files_dict):
        self.all_files_dict = all_files_dict
        # {'outfile_anharm_start', 'out_anharm_end', 'molden', 'dipolexyz',
        #  'normco', 'quadrature', 'polar', 'dipder', 'dipol', 'cubic', 'fcmfinal'
        #  ''}
        self.nModesStart = None

        self.dipole_first_derivatives = None
        self.dipole_second_derivatives = None
        self.polarizability_first_derivatives = None
        self.polarizability_second_derivatives = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None
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
        # {'outfile_anharm_start', 'out_anharm_final', 'molden', 'dipolexyz',
        #  'normco', 'quadrature', 'polar', 'dipder', 'dipol', 'cubic', 'fcmfinal'
        #  ''}
        self.nModesStart = 6 if linear_molecule else 7

        parsed_data = parse_output_file(self.all_files_dict['out_anharm_final'])
        vib_energy_levels_list, labelsTable, anharmonic_freqs, anharmonic_ints, harmonic_freqs = parsed_data

        anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
        anharm_tuple_dict = {tuple([o - 7 for o in k]): v for k, v in anharm_states_dict.items()}
        harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))
        harm_tuple_dict = {tuple([o - 7 for o in k]): v for k, v in harm_states_dict.items()}

        self.fundamentals_harmonic_str = {str(k[0]):v for k,v in harm_tuple_dict.items() if len(k)==1}
        self.fundamentals_anharmonic_str = {str(k[0]):v for k,v in anharm_tuple_dict.items() if len(k)==1}

        self.anharmonic_states = {tuple(str(i) for i in k): v for k, v in harm_states_dict.items()}
        self.harmonic_states = {tuple(str(i) for i in k): v for k, v in anharm_states_dict.items()}

        cubic = pCubicORQuartic(self.all_files_dict['cubic'])
        self.funds_harm_ints = {int(k): v for k, v in self.fundamentals_harmonic_str.items()}

        self.cubic_force_constants = getCubicPost(self.funds_harm_ints, cubic)
        labelsModes_original = [i+self.nModesStart for i in list(self.funds_harm_ints)]

        mu = getDipoleDers_anharm(self.all_files_dict['dipolexyz'], labelsModes_original, self.nModesStart)
        self.dipole_first_derivatives = mu[0]
        self.dipole_second_derivatives = mu[1]

        with open(self.all_files_dict['polar_pkl'], 'rb') as file:
            alpha = pickle.load(file)
        self.polarizability_first_derivatives = alpha[0]
        self.polarizability_second_derivatives = alpha[1]


# used for things
def parse_output_file(filepath: str):
    """
    Parsing the out file - output of the anharmonic parallel procedure
    :param filepath:
    :return: modes - 1, 2, 3 quanta states - combinations of states
             anharmonic_frequencies
             anharmonic_intensities
             harmonic_transitions
    """
    modes = []
    anharmonic_frequencies = []
    anharmonic_intensities = []
    harmonic_transitions = []

    with open(filepath, 'r') as file:
        file_content = file.read()

    if "All levels with up to three quanta" in file_content:
        start_index = file_content.find("All levels with up to three quanta")

        end_index = file_content.find("Electric dipole moment function in dimensionless normal coordinates",
                                              start_index)

        section = file_content[start_index:end_index]
        section = section.split('\n')
        for line in section:
            if ('-------------' not in line and 'Dipole' not in line
                    and 'quanta' not in line and 'MODE' not in line
                    and 'Intensity' not in line and line.strip()!=''):
                nus = line.strip().split()
                m = tuple([int(f) for f in nus[:-3]])
                modes.append(m)
                anharmonic_frequencies.append(float(nus[-3]))
                anharmonic_intensities.append(float(nus[-2]))
                harmonic_transitions.append(float(nus[-1]))

        modes = np.array(modes)
        anharmonic_frequencies = np.array(anharmonic_frequencies)
        anharmonic_intensities = np.array(anharmonic_intensities)
        harmonic_transitions = np.array(harmonic_transitions)

        labels = np.array('   I    J    K    L    M   NI  NJ  NK  NL  NM '.strip().split())
        modes = [modes_array2tuple(arr) for arr in modes]
        return modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions

    else:
        print('\nNo anharmonic levels information in this file')

# used for polarizability derivatives calculations
def pTensor(filepath: str):
    """
    Parsing DIPOL or POLAR or FCMFINAL files

    :return: np.ndarray tensor of dipole moment or polarizability
    """
    alines = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        start = 1 if len(lines[0].strip().split())==2 else 0
        alines.extend([np.array([float(k) for k in i.strip().split()]) for i in lines[start:]])

    return np.array(alines) #.reshape((-1, 3 * na))

def getRotationMatrix(filepath: str) -> np.array:
    """
    Parsing outfile0.out to get the rotation matrix for xyz axes
    :param filepath:
    :return: OMAT transformation (rotation) matrix from outfile which is printed when PRINT_LEVEL=1
    """
    with open(filepath, 'r') as file:
        file_content = file.read()
    if 'Transformation matrix between QCOM and QCOMP (OMAT)' in file_content:
        start_index = file_content.find("(QCOM = OMAT * QCOMP,")
        end_index = file_content.find("test:  OMAT * QCOMP = ", start_index)
        omat_section = file_content[start_index:end_index]
        newlinebreak = omat_section.find("\n")
        coords_section = omat_section[newlinebreak:][1:-1].split('\n')

        omat_array = []
        for l in coords_section[:-1]:
            ll = np.array([float(i) for i in l.strip().split() if '.' in i])
            omat_array.append(ll)
        omat_array = np.array(omat_array)

        return omat_array
    else:
        print('No rotation matrix found in this outfile')

# used
def pCubicORQuartic(filepath: str):
    """
    Parising cubic and quartic files with CFFs and QFFs
    :param filepath:
    :return: np.ndarray - parsed lines of files
    """
    alllines = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for l in lines:
        elements = l.split()
        alllines.append(np.array([int(k) if i < len(elements) - 1 else float(k) for i, k in enumerate(elements)]))

    return np.array(alllines)

# used
def getCubicPost(freq: dict, cubic: np.ndarray, recipcm: bool = False):
    """ Derives cubic and quartic anharmonic constants.
        It takes reduced values [cm-1] from gaussian output
        and transforms it to :
          * cubic   force constants : [Hartree*amu(-3/2)*Bohr(-3)]
          * quartic force constants : [Hartree*amu(-2  )*Bohr(-4)]
    """
    if recipcm:

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

    else:
        n = len(freq)
        K3 = np.zeros((n, n, n), dtype=np.float64)

        for fijk in cubic:

            i = int(fijk[0]) - 7
            j = int(fijk[1]) - 7
            k = int(fijk[2]) - 7
            d = np.float64(fijk[3])

            d *= np.sqrt(freq[i] * freq[j] * freq[k])

            from scipy import constants
            a = np.sqrt(constants.h / constants.c / constants.physical_constants['unified atomic mass unit'][0] / 100)
            b = 10 ** 10 / 2 / np.pi / constants.physical_constants['Bohr radius'][0] / 10 ** 10
            Fact3R = (constants.physical_constants['hartree-joule relationship'][
                          0] / constants.h / constants.c / 100) * (a * b) ** 3

            d /= Fact3R

            K3[i, j, k] = d
            K3[i, k, j] = d
            K3[k, j, i] = d
            K3[k, i, j] = d
            K3[j, i, k] = d
            K3[j, k, i] = d

        return K3

def getDipoleDers_anharm(filenamebase: str, labels: list, nModesStart: int):
    """
    Getting dipole moment derivatives tensors
    :param nModesStart:
    :param labels:
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return:  dmudqarray - first order derivatives of dipole moment, (3N-6, 3)
              dmudqdarray - second order derivatives of dipole moment, (3N-6, 3N-6, 3)
    """
    dipx = pDipole(filenamebase+'x')
    dipy = pDipole(filenamebase+'y')
    dipz = pDipole(filenamebase+'z')
    # dipx
    # {7: 0.0097191434, (7, 9): -0.0024929592, (7, 10): 0.0068563719,
    # (7, 11): 0.0006583914, (9, 7): -0.0024929355,
    # (11, 11, 7): -0.0004000706, (12, 12, 7): -0.0007561331}

    dq = len(labels)
    dmudq_array = np.zeros((dq, 3))
    dmudqdq_array = np.zeros((dq, dq, 3))

    for l in labels:
        if type(l) ==int:
            if l in dipx:
                dmudq_array[l-nModesStart, 0] = dipx[l]
            if l in dipy:
                dmudq_array[l-nModesStart, 1] = dipy[l]
            if l in dipz:
                dmudq_array[l-nModesStart, 2] = dipz[l]

    for l in dipx:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 0)] = dipx[l]

    for l in dipy:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 1)] = dipy[l]

    for l in dipz:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 2)] = dipz[l]

    return dmudq_array, dmudqdq_array

# used
def getDisplacementsPolarData(polar_dir: str, raw: bool = False):
    """
    Collecting the POLAR files data and also rotates is according to the matrix from the corresponding outfile0.out
    :param raw:
    :param polar_dir: where polar calcs were done

    :return:
    """
    import os
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    allpolardata = {}
    allpolardataRaw = {}

    for directory in directories:
        dir_path = os.path.join(polar_dir, directory)

        polar_file_path = dir_path + '/POLAR'
        poldata = pTensor(polar_file_path)
        R = getRotationMatrix(dir_path + '/outfile0.out')

        allpolardataRaw[directory] = (poldata, R)
        # First, we transform one rank: temp = R α
        temp = np.einsum('ij,jk->ik', R.T, poldata)
        # Then, we transform the other rank: α' = temp R^T
        alpha_prime = np.einsum('ij,jk->ik', temp, R)

        # 'ij, j -> i'
        # 'ij, jk -> ik', 'nk, ik -> in'
        allpolardata[directory] = alpha_prime

    if raw:
        return allpolardataRaw
    else:
        return allpolardata

# used
def getPolarDers(polar_dir: str):
    """
    Computing polarizability derivatives
    :param polar_dir: where optimization was done
    :return:
    """
    # base_dir = 'equil/displacements'

    data = getDisplacementsPolarData(polar_dir)
    data['equil'] = pTensor(polar_dir + '/../anharm/POLAR')
    print(f'Equilibrium polarizability in {polar_dir + "/../anharm/POLAR"}')
    print(sorted(list(data.keys())))

    import re
    # Extract all numbers from directory names
    numbers = set()
    pattern = re.compile(r'\d+')
    for name in list(data.keys()):
        numbers.update(map(int, pattern.findall(name)))

    min_num = min(numbers)
    max_num = max(numbers)
    size = max_num - min_num + 1

    import os
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    nums = [h.strip('np').split('_') for h in directories]
    flattened_list = [item for sublist in nums for item in sublist]
    flattened_list = set([int(g) for g in flattened_list])

    firstder = {}

    for f in flattened_list:
        firstder[f] = (data[f'{f}p'] - data[f'{f}n']) / 0.02

    secondders = {}

    # (∂²α_ij/∂Q_k∂Q_l) ≈ [α_ij(Q_k + ΔQ_k, Q_l + ΔQ_l) - α_ij(Q_k + ΔQ_k, Q_l - ΔQ_l)
    #   - α_ij(Q_k - ΔQ_k, Q_l + ΔQ_l) + α_ij(Q_k - ΔQ_k, Q_l - ΔQ_l)] / (4 * ΔQ_k * ΔQ_l)
    for k in flattened_list:
        for m in flattened_list:
            if k < m:
                val = (data[f'{k}_{m}pp'] - data[f'{k}_{m}pn'] - data[f'{k}_{m}np'] + data[f'{k}_{m}nn']) / (
                            4 * 0.01 * 0.01)
                secondders[(k, m)] = val
                secondders[(m, k)] = val

    # (∂²α_ij/∂Q_k²) ≈ [α_ij(Q_k + ΔQ_k) - 2α_ij(Q_k) + α_ij(Q_k - ΔQ_k)] / (ΔQ_k)²
    for b in flattened_list:
        secondders[(b, b)] = (data[f'{b}p'] - 2 * data['equil'] + data[f'{b}n']) / 0.01 ** 2

    polder = []
    for p in firstder:
        polder.append(firstder[p])
    first = np.array(polder)

    second = np.zeros((size, size, 3, 3))
    indices_to_insert = list(secondders.keys())
    # Insert the matrices at the specified indices
    for index, mat in zip(indices_to_insert, list(secondders.values())):
        i, j = index
        second[i - 7, j - 7] = mat

    return first, second

# not used now
def pklPolder(polar_dir):
    first, second = getPolarDers(polar_dir)
    filenamec = 'polar.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((first, second), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

# not used now
def pklPoldata(polar_dir):
    allpolardata = getDisplacementsPolarData(polar_dir)
    rawdata = getDisplacementsPolarData(polar_dir, raw=True)

    poldataeq = pTensor(polar_dir + '/../anharm/POLAR')
    R = getRotationMatrix(polar_dir + '/../anharm/outfile0.out')
    temp = np.einsum('ij,jk->ik', R.T, poldataeq)
    alpha_prime = np.einsum('ij,jk->ik', temp, R)
    allpolardata['equil'] = alpha_prime

    rawdata['equil'] = (alpha_prime, R)
    filenamec = 'polarData.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(allpolardata, file)

    print(f'\n --> {filenamec} file created\n')

    filenamec2 = 'polarData_raw.pkl'

    with open(filenamec2, 'wb') as file:
        pickle.dump(rawdata, file)

    print(f'\n --> {filenamec2} file created\n')

    return filenamec

# not used now
def pklDimless_normal_modes(quadratureFile):
    equilibrium_geometry, freqs, normal_modes = pQUADRATURE(quadratureFile)
    filenamec = 'dimensionless.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((equilibrium_geometry, freqs, normal_modes), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec


"""
1-parse_output_file,
2-pCubicORQuartic, 3-getCubicPost,
4-getDipoleDers_anharm,
'polar_pkl' file <- 5-getPolarDers(6-getDisplacementsPolarData, 
                    7-getRotationMatrix, 8-pTensor), 
                    9-pklPolder"""
# -----------------------------------------------------------------------------
# not used now
def pklOutFile(outfile):
    import pickle

    data = parse_output_file(outfile)
    filenamedata = 'vibdata.pkl'

    with open(filenamedata, 'wb') as file:
        pickle.dump(data, file)

    print(f'\n --> {filenamedata} file created\n')
    return os.getcwd()+'/'+filenamedata

# not used
def pklDipole(filenamebase, outfile):
    import pickle

    print('\nPickling in pklDipole in', os.getcwd())

    dmudqarray, dmudqdarray = getDipoleDers(filenamebase, outfile)
    filename = 'dipolexyz.pkl'

    with open(filename, 'wb') as file:
        pickle.dump((dmudqarray, dmudqdarray), file)

    print(f'\n --> {filename} file created\n')

    return os.getcwd()+'/'+filename

# used
def pklCubic(cubicfile):

    cubic = pCubicORQuartic(cubicfile)
    filenamec = 'cubic.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(cubic, file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

# -----------------------------------------------------------------------------

# used only in computeRedMass4nm which is not used now
def pMOLDEN(filepath: str, linear: bool = False) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """
    Parsing MOLDEN file (VIB job output, harmonic or anharmonic)
    :param linear:
    :param filepath:
    :return:  geometry_data, atoms, selected_dict
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    atoms = []
    geometry_data = []
    vibrations_data = {}

    in_geometry_section = False
    in_vibration_section = False

    for line in lines:
        if '[FR-COORD]' in line:
            in_geometry_section = True
            in_vibration_section = False
            continue

        elif '[FR-NORM-COORD]' in line:
            in_vibration_section = True
            in_geometry_section = False
            continue

        if in_geometry_section:
            data = line.strip().split()
            atom_label = data[0]
            atoms.append(atom_label)
            geometry_data.append(data[1:])  # Exclude the atom label

        # Capture vibration data
        elif in_vibration_section:
            stripped_line = line.strip()
            line_parts = stripped_line.split()

            if stripped_line.startswith('vibration'):
                current_vibration_index = int(line_parts[-1])
                vibrations_data[current_vibration_index] = []
            else:
                # to check if it was initialized
                if 'current_vibration_index' in locals():
                    vibrations_data[current_vibration_index].append(list(map(float, line_parts)))
                else:
                    print('Error: Vibration number not initialized before data lines.')

    atoms = np.array(atoms)
    geometry_data = np.array(geometry_data, dtype=float)
    vibrations_data = {key: np.array(value) for key, value in vibrations_data.items()}
    # only normal modes, ignore first 6 or 5
    ndiscard = 5 if linear else 6
    normal_modes_dict = {key: value for idx, (key, value) in enumerate(vibrations_data.items()) if idx >= ndiscard}

    return geometry_data, atoms, normal_modes_dict

# not used now
def pNORMCO(filepath: str):
    """
    Getting mass-weighted normal coordinates

    :param filepath:
    :return:  massweightgeo - np.ndarray
    """

    with open(filepath, 'r') as file1:
        linesnormco = file1.readlines()
    #massweighted_normal_coordinates
    massweighted_geometry = []

    # Flags to identify sections
    in_geometry_section = False

    # Loop through each line in the file
    for lin in linesnormco:
        # Check for the start of geometry section
        if '% mass weighted coordinates' in lin:
            in_geometry_section = True
            continue

        # Check for the start of vibration section - can be used to collect normal coordinates
        elif '% frequency' in lin:
            in_geometry_section = False
            break

        # Capture atoms and geometry data
        if in_geometry_section:
            data1 = lin.strip().split()
            massweighted_geometry.append(data1)  # Exclude the atom label

    # Convert lists to numpy arrays
    massweighted_geometry = np.array(massweighted_geometry, dtype=float)
    return massweighted_geometry

# used for polarizability calculations
def pQUADRATURE(filepath) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """Dimensionless normal coordinates are here, in QUADRATURE file"""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    current_frequency = None
    current_matrix = []
    undisplaced_matrix = []
    reading_matrix = False
    dqMat = []
    freqs = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('% frequency'):
            # If there's a current frequency, it means we've reached a new block
            if current_frequency is not None:
                freqs.append(current_frequency)
                dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))
                current_matrix = []
            # Extract the frequency value
            current_frequency = float(lines[i + 1].strip())
        elif line.startswith('% back-transformed dimensionless normal coordinates'):
            reading_matrix = True
        elif line.startswith('% Reference (undisplaced) coordinates are:'):
            # Save the last frequency block before moving to undisplaced coordinates
            if current_frequency is not None:
                freqs.append(current_frequency)
                dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))
                current_frequency = None
                current_matrix = []
            reading_matrix = True
        elif reading_matrix and line:
            # Extract the coordinates
            coords = [float(x) for x in line.split()]
            if len(tuple(coords))==3:
                current_matrix.append(np.array(coords))
        elif not line:
            reading_matrix = False

    # Add the last frequency block if it wasn't added
    if current_frequency is not None:
        freqs.append(current_frequency)
        dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))

    # The last matrix read is the undisplaced matrix
    if current_matrix:
        undisplaced_matrix = current_matrix
    # normal_coordinates = np.vstack(dqMat).T

    normal_coordinates = dict(zip(np.arange(7, len(freqs)+7), dqMat))
    return np.array(undisplaced_matrix), freqs, normal_coordinates

# not used now
def pDIPDER(filepath: str):
    """
    Parsing the DIPDER file

    :param filepath:
    :return: atomsorder, dipolefull - (3*Natoms, 3)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        dipmoment_cartder = []
        for l in lines:
            lr = l.strip().split()
            if len(lr) == 4:
                lr = np.array([float(i) for i in lr])
                dipmoment_cartder.append(lr)
    dipmoment_cartder = np.array(dipmoment_cartder)

    natoms = int(len(dipmoment_cartder) / 3)

    # order of atoms in cartesian coordinates
    atomsorder = dipmoment_cartder[:natoms, 0]

    x = dipmoment_cartder[:natoms, 1:].flatten()
    y = dipmoment_cartder[natoms:2 * natoms, 1:].flatten()
    z = dipmoment_cartder[2 * natoms:3 * natoms, 1:].flatten()

    dipolefull = np.array([x, y, z]).T
    return atomsorder, dipolefull

# used in parse_output_file
def modes_array2tuple(arr: np.array):
    """I J K L M NI NJ NK NL NM information array
    turned into a tuple, where I mode will be mentioned NI times and so on"""
    # Extract the non-zero elements from the array (ignoring the zeros after the first non-zero element)
    non_zero_elements = [x for x in arr[:3] if x != 0]  # We only care about the first two non-zero elements
    tuple_elements = []
    tuple_elements.extend([non_zero_elements[0]] * arr[5])

    if len(non_zero_elements) == 2:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])

    if len(non_zero_elements) == 3:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])
        tuple_elements.extend([non_zero_elements[2]] * arr[7])

    result_tuple = tuple(sorted(tuple_elements))

    return result_tuple

# used
def get_anharmonic_fundamentals(outfile: str, filetype: str = 'out') -> dict:
    """
    Extracts fundamental frequencies with anharmonic corrections from a given file.

    :param outfile: The path to the output file or the preloaded object
    :param filetype: The type of the file to process: 'out' for output files or 'pkl' for pickle files

    :return dict: A dictionary where keys are the mode indices (adjusted by subtracting 7)
            and values are the corresponding anharmonically corrected frequencies
    """
    if filetype == 'out':
        things = parse_output_file(outfile)
    elif filetype == 'pkl':
        with open(outfile, 'rb') as file:
            things = pickle.load(file)
    else:
        raise ValueError('Wrong file type. Choose "out" or "pkl".')

    labels = sorted({t[0] for t in things[0]})
    all_states_dict = dict(zip(things[0], things[2]))
    freqs = {b[0] - 7: all_states_dict[b] for b in (tuple([e]) for e in labels)}

    return freqs

# used
def pDipole(filenamebase: str):
    """
    Parsing dipole(xyz) files dipole(xyz)
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return: dictionary from dipole(xyz) data
    """
    dct = {}
    with open(filenamebase, 'r') as file:
        lines = file.readlines()
    for l in lines:
        nu = np.array([float(k) for k in l.split()])
        indx = tuple([int(i) for i in nu[:-1] if i!=0.0])
        if len(indx) == 1:
            dct[indx[0]] = nu[-1]
        else:
            dct[indx] = nu[-1]

    return dct

# used
def getDipoleDers(filenamebase: str, outfile: str):
    """
    Getting dipole moment derivatives tensors
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :param outfile: out file - output of the anharmonic parallel procedure
    :return:  dmudqarray - first order derivatives of dipole moment, (3N-6, 3)
              dmudqdarray - second order derivatives of dipole moment, (3N-6, 3N-6, 3)
    """
    dipx = pDipole(filenamebase+'x')
    dipy = pDipole(filenamebase+'y')
    dipz = pDipole(filenamebase+'z')

    things = parse_output_file(outfile)
    labels = sorted(list(set([t[0] for t in things[0]])))
    labels = [int(k) for k in labels]

    dmudqdict = {}
    dmudqdqdict = {}
    dq = len(labels)
    dmudqdqdict['x'] = np.zeros((dq, dq))
    dmudqdqdict['y'] = np.zeros((dq, dq))
    dmudqdqdict['z'] = np.zeros((dq, dq))

    for l in labels:
        if type(l) ==int:
            dmudqdict[l] = np.zeros(3)
            if l in dipx:
                dmudqdict[l][0] = dipx[l]
            if l in dipy:
                dmudqdict[l][1] = dipy[l]
            if l in dipz:
                dmudqdict[l][2] = dipz[l]

    for l in dipx:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['x'][(l[0] - 7, l[1] - 7)] = dipx[l]

    for l in dipy:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['y'][(l[0] - 7, l[1] - 7)] = dipy[l]

    for l in dipz:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['z'][(l[0] - 7, l[1] - 7)] = dipz[l]

    dmudqarray = np.array(list(dmudqdict.values()))
    dmudqdarray = np.array(list(dmudqdqdict.values())).T

    return dmudqarray, dmudqdarray

def describe_structure(obj, level=0):
    """
    Recursively describe the structure of a Python object with complex data types
    """
    indent = '  ' * level
    obj_type = type(obj).__name__

    # Base case for non-iterable types
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        return obj_type

    # Handle dictionaries separately
    if isinstance(obj, dict):
        key_descriptions = {describe_structure(k, level + 1): describe_structure(v, level + 1) for k, v in obj.items()}
        dict_description = ',\n'.join([f"{indent}  {k}: {v}" for k, v in key_descriptions.items()])
        return f"dict(\n{dict_description}\n{indent})"

    # Handle other iterables (lists, tuples, sets, etc.)
    if isinstance(obj, (list, tuple, set)):
        # Describe each item in the iterable
        item_descriptions = [describe_structure(item, level + 1) for item in obj]
        # Get the type name ('list', 'tuple', or 'set') for the description
        type_name = obj_type
        iterable_description = ',\n'.join([f"{indent}  {item}" for item in item_descriptions])
        return f"{type_name}(\n{iterable_description}\n{indent})"

    # Fallback for any other types
    return obj_type

# For pickled things
def unpickle(file: str):
    """
    Example here:

    import pickle
    import numpy as np

    vibdata = '../scriptsHPC/cfourscripts/vibdata.pkl'
    hf_cubicarray = '../scriptsHPC/cfourscripts/hf_cubicarray.pkl'
    hf_polarders = '../scriptsHPC/cfourscripts/hf_polarders.pkl'
    hf_vibdata = '../scriptsHPC/cfourscripts/hf_vibdata.pkl'
    # this data is from MOLDEN file
    hf_normalmodes = '../scriptsHPC/cfourscripts/hf_normalmodes.pkl'
    hf_rawdata_polar = '../scriptsHPC/cfourscripts/hf_rawdata_polar.pkl'

    lst = [hf_vibdata, hf_normalmodes, hf_cubicarray, hf_polarders, hf_rawdata_polar]
    for l in lst:
        unpickle(l)

    :param file:
    :return:
    """

    import pickle

    print(f'      Opening a pickle file {file}')

    with open(file, 'rb') as f:
        stuff = pickle.load(f)
    if type(stuff) == tuple and (type(stuff[0]) != str or type(stuff[0]) != float or type(stuff[0]) != int):
        print('  There are several things here')
        baselength = 7
        for i, thing in enumerate(stuff):
            print('\n' + ' ' * 6 + f'{i}:' + ' ' * (baselength - len(str(i))), type(thing))
            if type(thing) == dict:
                dictinfo(thing)
            elif type(thing) == np.ndarray:
                print(' ' * 6 + f'---- Numpy array with shape {thing.shape}')

            print(' ' * 6 + '==' * 20)

    else:
        print(f'  There is just one thing here: {type(stuff)}')
        if type(stuff) == np.ndarray:
            print('\n' + ' ' * 6 + f'---- Numpy array with shape {stuff.shape}')
            print(stuff)
        elif type(stuff) == list:
            print('\n' + ' ' * 6 + f'---- List of length {len(stuff)}')
        elif type(stuff) == dict:
            dictinfo(stuff)

def dictinfo(dct: dict, level: int = 0):
    """
    Getting info from dictionaries

    :param dct:
    :param level:
    :return:
    """
    levels = {0: 6, 1: 12, 2: 18}
    print('\n' + ' ' * levels[level] + f'---- Dictionary with {len(dct)} pairs')
    keystype = type(list(dct.keys())[0])
    valuestype = type(dct[list(dct.keys())[0]])

    print('\n' + ' ' * levels[level] + f'Keys are {keystype} and values are {valuestype}')
    print('\n' + ' ' * levels[level] + f'List of keys:')
    print(' ' * (levels[level] + 6) + str(list(dct.keys())))

    if valuestype == dict:
        print(' ' * levels[level] + '>>' * 30)
        level += 1

        for k1 in dct:
            if k1 == 'metadata':
                width = 25
                print('\n' + ' ' * levels[level] + 'Description' + ' ' * (width - 11), dct['metadata']['description'])
                print('\n' + ' ' * levels[level] + 'Contents' + ' ' * (width - 11))
                for descr in dct['metadata']['contents']:
                    # print(dct['input_data_info'][descr])
                    if type(dct['input_data_info'][descr]) == np.ndarray:
                        extra = 'with shape ' + str(dct['input_data_info'][descr].shape)
                    elif type(dct['input_data_info'][descr]) == list:
                        extra = 'with length ' + str(len(dct['input_data_info'][descr]))
                    else:
                        extra = ''
                    print(' ' * levels[level + 1] + f'{descr}:' + ' ' * (width - len(descr)),
                          dct['metadata']['contents'][descr], extra)

                print('\n')

            else:
                dictinfo(dct[k1], level)

            print(' ' * levels[level] + '>>' * 30)

# this is about getting raw hessian matrix and doing vib analysis with it
def hessianfromout(outfilename: str):
    """
    Getting hessian data from outfile0.out

    :param outfilename:
    :return:
    """
    # Read the contents of the file
    with open(outfilename, 'r') as file:
        file_content = file.read()

    hessian_start_index = file_content.find("Molecular hessian")
    dipole_line_index = file_content.find("Total dipole moment", hessian_start_index)
    hessian_section = file_content[hessian_start_index:dipole_line_index]

    hessian_lines = hessian_section.split('\n\n\n\n')[1]
    h = [k for k in hessian_lines.split('\n') if k!='']

    cleanh = []
    for i in h[1:]:
        if '.' in i:
            cleanh.extend([float(k) for k in i.split() if '.' in k])
    hvec = np.array(cleanh)

    return hvec

def solve_quadratic(a: float, b: float, c: float):
    import cmath  # Import the complex math module

    discriminant = cmath.sqrt(b**2 - 4*a*c)

    # Calculate the two solutions
    root1 = (-b + discriminant) / (2*a)
    root2 = (-b - discriminant) / (2*a)

    return root1, root2

def gethessinmat(hessinvec: np.ndarray):
    """
    Formation of Hessian matrix, for any source of it.
    Example:
        hessmat = gethessinmat(hessvec)
        eigenvalues2, mass_weighted_eigenvectors2 = jn.hessian_diagonalizer(hessmat)
    Or
        newnewhess = jn.project_hessian(mol, hessmat, is_linear= False, internal_coordinates= False)
        eigenvalues, mass_weighted_eigenvectors = jn.hessian_diagonalizer(newnewhess[0])

    Or
        hessmatR = gethessinmat(hessvecR)
        # frequency_analysis(coords, Hessian, elem=None, mass=None, energy=0.0,
        #             temperature=300.0, pressure=1.0, verbose=0, outfnm=None,
        #             note=None, wigner=None, ignore=0, normalized=True)
        gg = g.normal_modes.frequency_analysis(coordsvec, hessmat, mass=massvec)

    :param hessinvec:
    :return:
    """
    # Solve the quadratic equation
    roots = solve_quadratic(a=1, b=1, c=-2*len(hessinvec))

    # Print the solutions
#    print("Root 1:", roots[0])

    ncoords = int(roots[0].real)
    hessmat_mat = np.zeros((ncoords, ncoords))

    ind = 0
    for i in range(ncoords):
        for j in range(i+1):
            hessmat_mat[i, j] = hessinvec[ind]
            hessmat_mat[j, i] = hessinvec[ind]
            ind += 1

    return hessmat_mat

def getrotproj(filename: str):
    """
    Getting Rotationally projected vibrational frequencies from.out file

    E.g, filename = '../data/rawouts/anharm_hf_outfile0.out'
    :return:
    """
    # Read the contents of the file fchk
    with open(filename, 'r') as file:
        file_content = file.read()

    start_index = file_content.find("Rotationally projected vibrational frequencies")
    end_index = file_content.find("Zero-point energy:", start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[start_index:end_index]
    brinx = coords_section.find("\n")
    coords_section = [float(k[1]) for k in [j.strip().split() for j in coords_section[brinx:][1:-1].split('\n')] if k and 'i' not in k[1]]

    print(coords_section[4:])
    print(len(coords_section[4:]))

def extract_anharmvib(filename: str):
    """
    From 'HARMONIC AND FUNDAMENTAL FREQUENCIES.' section of out file

    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        file_content = file.read()
    import re
    import pandas as pd

    # Define start and end patterns for the block
    start_pattern = r'HARMONIC AND FUNDAMENTAL FREQUENCIES.*\n-+\n'
    end_pattern = r'-+\n'

    start_match = re.search(start_pattern, file_content)
    end_match = re.search(end_pattern, file_content[start_match.end():])

    data_block = file_content[start_match.end():start_match.end() + end_match.start()]

    lines = data_block.strip().split('\n')
    head = [word + ' ' for word in lines[0].split()]
    headers = lines[1].split()

    column_names = [str1 + str2 for str1, str2 in zip(head, headers[1:])]
    column_names.insert(0, lines[1].split()[0])

    values = [list(map(float, line.split())) for line in lines[2:]]

    numpy_array = np.array(values)
    df = pd.DataFrame(numpy_array, columns=column_names)

    df['Mode'] = df['Mode'].astype(int)

    return df

# not used now
def computeRedMass4nm(filename: str):
    """
    Source : https://github.com/psi4/psi4/blob/8a781dcad54eac2114f8af0d742ccd8beb036ddb/psi4/driver/qcdb/vib.py#L579
    https://github.com/psi4/psi4/blob/master/psi4/driver/qcdb/vib.py

    :return:
    """
    mass = np.array([15.994914630, 12.000000000, 1.007825035, 1.007825035])
    nat = len(mass)

    # sqrtmmm = np.repeat(np.sqrt(mass), 3)
    # sqrtmmminv = np.divide(1.0, sqrtmmm)

    # filename = '../scriptsHPC/data/rawouts/anharm_hf_MOLDEN'
    with open(filename, 'rb') as f:
        moldendata = pMOLDEN(filename)

    print('Atoms:', moldendata[1])

    wL = np.array([moldendata[2][i] for i in moldendata[2]]).reshape(-1, 3 * nat)

    # this works now
    reduced_mass_u = np.divide(1.0, np.linalg.norm(wL.T, axis=0) ** 2)

    return reduced_mass_u
