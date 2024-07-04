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
#     - DIPDER                       ---
#     - POLAR                        --- static polarizability (a.u.)
#     - out                          --- the final output file in anharmonic parallel procedure:
#                            1) All levels with up to three quanta frequencies; 2) equilibrium geometry;
#                            3) normal coordinates, non-mass-weighted (a.u.); 4) F(IJKK)/a.u ; 5) F(IJKK)/cm-1 ;
#                            6) harmonic and fundamental frequencies and intensities
#     - dipolex(yz)                  ---
#     - cubic                        --- cubic force constants in cm-1 in dimensionless normal modes
#     - quartic                      --- quartic force constants in cm-1 in dimensionless normal modes

import numpy as np
import os
import pickle


def pOutfile(filepath: str):
    """
    Parsing outfile0.out
    :param filepath:
    :return:
    """
    with open(filepath, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    coords_start_index = file_content.find("(QCOM = OMAT * QCOMP,")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("test:  OMAT * QCOMP = ", coords_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[coords_start_index:dipole_line_index]

    brinx = coords_section.find("\n")
    coords_section = coords_section[brinx:][1:-1].split('\n')

    omat_array = []
    for l in coords_section[:-1]:
        ll = np.array([float(i) for i in l.strip().split() if '.' in i])
        omat_array.append(ll)
    omat_array = np.array(omat_array)

    return omat_array

def pMOLDEN(filepath: str) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """
    Parsing MOLDEN file (VIB job, harmonic or anharmonic)

    :param filepath:
    :return:  geometry_data, atoms, selected_dict
    """
    # Read the file content
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Variables to store atoms, geometry, and vibration data
    atoms = []
    geometry_data = []
    vibrations_data = {}

    # Flags to identify sections
    in_geometry_section = False
    in_vibration_section = False

    # Loop through each line in the file
    for line in lines:
        # Check for the start of geometry section
        if '[FR-COORD]' in line:
            in_geometry_section = True
            in_vibration_section = False
            continue

        # Check for the start of vibration section
        elif '[FR-NORM-COORD]' in line:
            in_vibration_section = True
            in_geometry_section = False
            continue

        # Capture atoms and geometry data
        if in_geometry_section:
            data = line.strip().split()
            atom_label = data[0]
            atoms.append(atom_label)
            geometry_data.append(data[1:])  # Exclude the atom label

        # Capture vibration data
        elif in_vibration_section:
            if line.strip().startswith('vibration'):
                vibration_number = int(line.split()[-1])
                vibrations_data[vibration_number] = []
            else:
                vibrations_data[vibration_number].append(list(map(float, line.split())))

    # Convert lists to numpy arrays
    atoms = np.array(atoms)
    geometry_data = np.array(geometry_data, dtype=float)
    vibrations_data = {key: np.array(value) for key, value in vibrations_data.items()}
    # only normal modes
    selected_dict = {key: value for idx, (key, value) in enumerate(vibrations_data.items()) if idx >= 6}

    return geometry_data, atoms, selected_dict

def pNORMCO(filepath: str):
    """
    Getting mass-weighted normal coordinates

    :param filepath:
    :return:  massweightgeo - np.ndarray
    """

    with open(filepath, 'r') as file1:
        linesnormco = file1.readlines()

    # Variables to store atoms, geometry, and vibration data
    massweightgeo = []

    # Flags to identify sections
    in_geometry_section = False

    # Loop through each line in the file
    for lin in linesnormco:
        # Check for the start of geometry section
        if '% mass weighted coordinates' in lin:
            in_geometry_section = True
            continue

        # Check for the start of vibration section
        elif '% frequency' in lin:
            in_geometry_section = False
            break

        # Capture atoms and geometry data
        if in_geometry_section:
            data1 = lin.strip().split()
            massweightgeo.append(data1)  # Exclude the atom label

    # Convert lists to numpy arrays
    massweightgeo = np.array(massweightgeo, dtype=float)
    return massweightgeo

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
#    dqMats = np.vstack(dqMat).T

    dqMats = dict(zip(np.arange(7, len(freqs)+7), dqMat))
    return np.array(undisplaced_matrix), freqs, dqMats


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

def parse_output_file(filepath: str):
    """
    Parsing the out file - output of the anharmonic parallel procedure
    :param filepath:
    :return: modes - 1, 2, 3 quanta states - combinations of states
             anharmonic_frequencies
             anharmonic_intensities
             harmonic_transitions
    """
    # Initialize lists to hold the column data
    modes = []
    anharmonic_frequencies = []
    anharmonic_intensities = []
    harmonic_transitions = []

    # Read the contents of the MOLDEN file
    with open(filepath, 'r') as file:
        file_content = file.read()

    # Find the index where "Molecular hessian" appears
    coords_start_index = file_content.find("All levels with up to three quanta")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Electric dipole moment function in dimensionless normal coordinates",
                                          coords_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[coords_start_index:dipole_line_index]
    coords_section = coords_section.split('\n')
    # print(coords_section)
    # quit()
    for line in coords_section[5:-3]:

        if '-------------' not in line and 'Dipole' not in line:
            nus = line.strip().split()
            # print(nus)
            m = tuple([int(f) for f in nus[:-3]])
            modes.append(m)
            anharmonic_frequencies.append(float(nus[-3]))
            anharmonic_intensities.append(float(nus[-2]))
            harmonic_transitions.append(float(nus[-1]))

    # Convert the lists to NumPy arrays
    modes = np.array(modes)
    anharmonic_frequencies = np.array(anharmonic_frequencies)
    anharmonic_intensities = np.array(anharmonic_intensities)
    harmonic_transitions = np.array(harmonic_transitions)

    # Print the NumPy arrays
    labels = np.array('   I    J    K    L    M   NI  NJ  NK  NL  NM '.strip().split())
    #print(modes)
    #print([modes_array2tuple(arr) for arr in modes])
    modes = [modes_array2tuple(arr) for arr in modes]

    return modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions

def modes_array2tuple(arr: np.array):
    # Extract the non-zero elements from the array (ignoring the zeros after the first non-zero element)
    non_zero_elements = [x for x in arr[:3] if x != 0]  # We only care about the first two non-zero elements

    # Initialize an empty list to hold the elements of the tuple
    tuple_elements = []

    # Add the first non-zero element 'arr[5]' times to the list
    tuple_elements.extend([non_zero_elements[0]] * arr[5])

    # If there is a second non-zero element, add it 'arr[6]' times to the list
    if len(non_zero_elements) == 2:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])

    if len(non_zero_elements) == 3:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])
        # If there is no second non-zero element, add the first element 'arr[6]' more times
        tuple_elements.extend([non_zero_elements[2]] * arr[7])

    # Convert the list to a tuple
    result_tuple = tuple(sorted(tuple_elements))

    return result_tuple

def get_anharmonic_fundamentals(outfile: str, filetype: str = 'out') -> dict:
    """
    Extracts fundamental frequencies with anharmonic corrections from a given file.

    This function reads data from the specified output file or a pre-loaded object,
    and extracts the fundamental frequencies with anharmonic corrections.

    Parameters:
    - outfile (str): The path to the output file or the pre-loaded object.
    - filetype (str): The type of the file to process. Can be 'out' for output files or 'pkl' for pickle files.

    Returns:
    - dict: A dictionary where keys are the mode indices (adjusted by subtracting 7)
            and values are the corresponding anharmonically corrected frequencies.

    Raises:
    - ValueError: If the filetype is not 'out' or 'pkl'.
    """

    if filetype == 'out':
        things = parse_output_file(outfile)
    elif filetype == 'pkl':
        with open(outfile, 'rb') as file:
            things = pickle.load(file)
    else:
        raise ValueError('Wrong file type. Choose "out" or "pkl".')

    labels = sorted({t[0] for t in things[0]})
    tlab = [tuple(element for element in t if element != 0) for t in things[0]]
    dd = dict(zip(tlab, things[2]))
    freqs = {b[0] - 7: dd[b] for b in (tuple([e]) for e in labels)}
    # freqs_harm = {b[0] - 7: dd[b] for b in (tuple([e]) for e in labels)}

    return freqs

def pklOutFile(outfile):
    import pickle

    data = parse_output_file(outfile)
    filenamedata = 'vibdata.pkl'

    with open(filenamedata, 'wb') as file:
        pickle.dump(data, file)

    print(f'\n --> {filenamedata} file created\n')
    return os.getcwd()+'/'+filenamedata

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


def pklDipole(filenamebase, outfile):
    import pickle

    cwd = os.getcwd().split('/')
    print('\nPickling in pklDipole in', os.getcwd())

    ##### get mu derivatives
    # filenamebase = 'dipole'
    # outfile = 'out'

    dmudqarray, dmudqdarray = getDipoleDers(filenamebase, outfile)
    filename = 'dipolexyz.pkl'

    with open(filename, 'wb') as file:
        pickle.dump((dmudqarray, dmudqdarray), file)

    print(f'\n --> {filename} file created\n')

    return os.getcwd()+'/'+filename

def getAllPolarData4Ders(polar_dir: str, raw: bool = False):
    """
    :param raw:
    :param polar_dir: where polar calcs were done

    :return:
    """
    # Base directory where the folders are located
    import os
    # Get a list of all directories in the base directory ("displacements")
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    #print(directories)
    #quit()
    allallpolardata = {}
    allallpolardataRaw = {}

    # Loop through each directory
    for directory in directories:
        # Construct the full path to the directory
        dir_path = os.path.join(polar_dir, directory)

        # Find the POLAR file in the directory
        # Since there's only one, we can use next() to get the first match
        polar_file_path = dir_path + '/POLAR'
        poldata = pTensor(polar_file_path)

        R = pOutfile(dir_path + '/outfile0.out')

        allallpolardataRaw[directory] = (poldata, R)

        # Perform the transformation using einsum
        # First, we transform one rank: temp = R α
        temp = np.einsum('ij,jk->ik', R.T, poldata)
        # Then, we transform the other rank: α' = temp R^T
        alpha_prime = np.einsum('ij,jk->ik', temp, R)

        # 'ij, j -> i'
        # 'ij, jk -> ik', 'nk, ik -> in'
        # Call the function to process the file
        # allallpolardata[directory] = polar(polar_file_path)
        allallpolardata[directory] = alpha_prime

    if raw:
        return allallpolardataRaw
    else:
        return allallpolardata

def pklPoldata(polar_dir):

    allallpolardata = getAllPolarData4Ders(polar_dir)
    rawdata = getAllPolarData4Ders(polar_dir, raw=True)

    poldataeq = pTensor(polar_dir+'/../anharm/POLAR')
    R = pOutfile(polar_dir + '/../anharm/outfile0.out')
    temp = np.einsum('ij,jk->ik', R.T, poldataeq)
    alpha_prime = np.einsum('ij,jk->ik', temp, R)
    allallpolardata['equil'] = alpha_prime

    rawdata['equil'] = (alpha_prime, R)
    filenamec = 'polarData.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(allallpolardata, file)

    print(f'\n --> {filenamec} file created\n')

    filenamec2 = 'polarData_raw.pkl'

    with open(filenamec2, 'wb') as file:
        pickle.dump(rawdata, file)

    print(f'\n --> {filenamec2} file created\n')

    return filenamec

def getPolarDers(polar_dir: str):
    """

    :param polar_dir: where optimization was done

    :return:
    """
    # base_dir = 'equil/displacements'

    data = getAllPolarData4Ders(polar_dir)
    data['equil'] = pTensor(polar_dir+'/../anharm/POLAR')
    print(f'Equilibrium polarizability in {polar_dir+"/../anharm/POLAR"}')
    print(sorted(list(data.keys())))
    
    import re
    # Extract all numbers from directory names
    numbers = set()
    pattern = re.compile(r'\d+')
    for name in list(data.keys()):
        numbers.update(map(int, pattern.findall(name)))

    # Determine the size of the matrix
    min_num = min(numbers)
    max_num = max(numbers)
    size = max_num - min_num + 1

    import os
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    nums = [h.strip('np').split('_') for h in directories]
    flattened_list = [item for sublist in nums for item in sublist]
    flattened_list = set([int(g) for g in flattened_list])

    # dictionary of first order derivatives
    firstder = {}

    for f in flattened_list:
        firstder[f] = (data[f'{f}p'] - data[f'{f}n']) / 0.02

    secondders = {}

    # (∂²α_ij/∂Q_k∂Q_l) ≈ [α_ij(Q_k + ΔQ_k, Q_l + ΔQ_l) - α_ij(Q_k + ΔQ_k, Q_l - ΔQ_l)
    #   - α_ij(Q_k - ΔQ_k, Q_l + ΔQ_l) + α_ij(Q_k - ΔQ_k, Q_l - ΔQ_l)] / (4 * ΔQ_k * ΔQ_l)
    for k in flattened_list:
        for m in flattened_list:
            if k < m:
                val = (data[f'{k}_{m}pp'] - data[f'{k}_{m}pn'] - data[f'{k}_{m}np'] + data[f'{k}_{m}nn']) / (4 * 0.01 * 0.01)
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
    print(size)
    indices_to_insert = list(secondders.keys())
    # print(indices_to_insert)
    # Insert the matrices at the specified indices
    for index, mat in zip(indices_to_insert, list(secondders.values())):
        i, j = index
        #print(i, j)
        second[i - 7, j - 7] = mat

    return first, second

def pklPolder(polar_dir):

    first, second = getPolarDers(polar_dir)
    filenamec = 'polar.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((first, second), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

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

def pklCubic(cubicfile):

    cubic = pCubicORQuartic(cubicfile)
    filenamec = 'cubic.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(cubic, file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

def pklDimless_normal_modes(quadratureFile):
    equilibrium_geometry, freqs, normal_modes = pQUADRATURE(quadratureFile)
    filenamec = 'dimensionless.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((equilibrium_geometry, freqs, normal_modes), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

def getCubicPost(freq: dict, cubic: np.ndarray, recipcm: bool = False):
    """ derives cubic and quartic anharmonic constants.
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
        BohrToAngstrom = 0.5291772086
        HartreeToAttoJoule = 4.3597439
        ToRedCubForceConst = 9.85501E+06

        # freq = get_anharmonic_fundamentals(outfile)
        n = len(freq)
        K3 = np.zeros((n, n, n), dtype=np.float64)
        # Specify the filename from which to load the dictionaries

        # cubic = pCubicORQuartic(cubicfile)

        for fijk in cubic:

            i = int(fijk[0]) - 7
            j = int(fijk[1]) - 7
            k = int(fijk[2]) - 7
            d = np.float64(fijk[3])
            # print('cm-1', d)
            # d *= BohrToAngstrom ** 3
            # d /= HartreeToAttoJoule * ToRedCubForceConst

            d *= np.sqrt(freq[i] * freq[j] * freq[k])

            from scipy import constants
            a = np.sqrt(constants.h / constants.c / constants.physical_constants['unified atomic mass unit'][0] / 100)
            b = 10 ** 10 / 2 / np.pi / constants.physical_constants['Bohr radius'][0] / 10 ** 10
            Fact3R = (constants.physical_constants['hartree-joule relationship'][
                          0] / constants.h / constants.c / 100) * (a * b) ** 3

            d /= Fact3R

            # print('au  ', d, '\n')
            K3[i, j, k] = d
            K3[i, k, j] = d
            K3[k, j, i] = d
            K3[k, i, j] = d
            K3[j, i, k] = d
            K3[j, k, i] = d

        return K3

def describe_structure(obj, level=0):
    """
    Recursively describe the structure of a Python object with complex data types.
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


###
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

    # import sys
    # sys.path.append('../../../')
    # import colorspython as c
    # print(c.CVIOLET+'\n'+'--'*40+c.CEND)
    # print(c.CVIOLET+f'      Opening a pickle file {file}'+c.CEND)
    # print(c.CVIOLET+'--'*40+c.CEND)

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
                    # print(dct['data'][descr])
                    if type(dct['data'][descr]) == np.ndarray:
                        extra = 'with shape ' + str(dct['data'][descr].shape)
                    elif type(dct['data'][descr]) == list:
                        extra = 'with length ' + str(len(dct['data'][descr]))
                    else:
                        extra = ''
                    print(' ' * levels[level + 1] + f'{descr}:' + ' ' * (width - len(descr)),
                          dct['metadata']['contents'][descr], extra)

                print('\n')
                # break

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

    # Find the index where "Molecular hessian" appears
    hessian_start_index = file_content.find("Molecular hessian")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Total dipole moment", hessian_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    hessian_section = file_content[hessian_start_index:dipole_line_index]

    # Split the content into lines
    hessian_lines = hessian_section.split('\n\n\n\n')[1]

    h = [k for k in hessian_lines.split('\n') if k!='']
    # print(h[1:])

    cleanh = []
    for i in h[1:]:
        if '.' in i:
            cleanh.extend([float(k) for k in i.split() if '.' in k])
    hvec = np.array(cleanh)
    # print(len(hvec), hvec)

    return hvec

def solve_quadratic(a: float, b: float, c: float):
    import cmath  # Import the complex math module

    # Calculate the discriminant
    delta = cmath.sqrt(b**2 - 4*a*c)

    # Calculate the two solutions
    root1 = (-b + delta) / (2*a)
    root2 = (-b - delta) / (2*a)

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

    # Find the index where "Molecular hessian" appears
    coords_start_index = file_content.find("Rotationally projected vibrational frequencies")

    # Find the index where the line with "Total dipole moment" appears
    dipole_line_index = file_content.find("Zero-point energy:", coords_start_index)

    # Extract the content starting from "Molecular hessian" up to "Total dipole moment"
    coords_section = file_content[coords_start_index:dipole_line_index]
    brinx = coords_section.find("\n")
    coords_section = [float(k[1]) for k in [j.strip().split() for j in coords_section[brinx:][1:-1].split('\n')] if k and 'i' not in k[1]]

    # coords_section.insert(0, str(len(coords_section))+'\n')
    # coordsstr = '\n'.join(coords_section)
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

    # Find start and end positions of the block
    start_match = re.search(start_pattern, file_content)
    end_match = re.search(end_pattern, file_content[start_match.end():])

    # Extract the block content
    data_block = file_content[start_match.end():start_match.end() + end_match.start()]

    # Splitting the block into lines and extracting relevant information
    lines = data_block.strip().split('\n')
    head = [word + ' ' for word in lines[0].split()]
    headers = lines[1].split()

    # print(head, len(head))
    # print(headers[1:], len(headers[1:]))

    column_names = [str1 + str2 for str1, str2 in zip(head, headers[1:])]
    # column_names = [header.strip() for header in headers]
    column_names.insert(0, lines[1].split()[0])

    values = [list(map(float, line.split())) for line in lines[2:]]

    # Creating a NumPy array
    numpy_array = np.array(values)

    # Creating a Pandas DataFrame
    df = pd.DataFrame(numpy_array, columns=column_names)

    df['Mode'] = df['Mode'].astype(int)

    return df

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
    # print(wL.shape, '\n', wL.T)

    # qL = np.einsum('i,ij->ij', sqrtmmm, wL.T)

    # mwhess = np.einsum('i,ij,j->ij', sqrtmmminv, nmwhess, sqrtmmminv)
    # force_constant_au, qL = np.linalg.eigh(mwhess_proj)
    # vibinfo['q'] = Datum('normal mode', 'a0 u^1/2', qL, comment='normalized mass-weighted')

    # wL = np.einsum('i,ij->ij', sqrtmmminv, qL.T)
    # vibinfo['w'] = Datum('normal mode', 'a0', wL, comment='un-mass-weighted')

    # this works now
    reduced_mass_u = np.divide(1.0, np.linalg.norm(wL.T, axis=0) ** 2)

    return reduced_mass_u
