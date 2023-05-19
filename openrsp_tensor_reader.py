# SpectroscPy x.x.x
# SpectroscPy is a script package developed by and containing contributions from

# Karen Oda Hjorth Minde Dundas
# Magnus Ringholm
# Yann Cornation
# Benedicte Ofstad

# The package is released under a LGPL licence.
# For questions, please contact on karen.o.dundas@uit.no

import numpy as np
import copy
import itertools
from sys import exit


# Class to hold information about a response property
class rspProperty:

    def __init__(self, order, operator, components, frequencies, tensor=None):

        self.order = order
        self.operator = operator
        self.components = components
        self.frequencies = frequencies

        # May optionally add tensor at initialization, marking as default that this is not done
        self.hasTensor = 0

        # Adding tensor if given and if so, specifying that it was given
        if tensor is not None:
            self.hasTensor = 1
            self.tensor = tensor

        # optionally getting Hessian tensor
        self.hasHessian = False

    # May add tensor later if wanted
    def addTensor(self, tensor):

        self.hasTensor = 1
        self.tensor = tensor

    # Print summary of property
    def tellProp(self):

        print('Order is', self.order)
        print('Operators are', self.operator)
        print('Number of components are', self.components)
        print('Frequencies are', self.frequencies)

        # Also print tensor if it has been provided
        if (self.hasTensor):
            pass
        # print('Tensor:')
        # print(self.tensor)

    # find hessian tensor
    def get_hessian(self, fname):
        props, tensors = read_openrsp_tensor_file(fname)

        for i, p in enumerate(props):
            if p.operator == ['GEO', 'GEO']:
                self.hasHessian = True
                p.addTensor(tensors[i])
                return p.tensor

    # transform any tensor's GEO cartesian to normal
    def cart2normal(self, fname_mol, fname_tens):

        shape = self.tensor.shape

        import vib_analysis as va
        coordshere, chargeshere, masseshere = va.read_mol(fname_mol)
        hessian_tensor = self.get_hessian(fname_tens)
        cut_w, cut_T, N_3, min_el, max_el = va.get_vib_harm_freqs_and_eigvecs(coordshere, chargeshere, masseshere,
                                                                              hessian_tensor,
                                                                              outproj=True, print_level=0,
                                                                              harmonic_frequency_limits='Keep all')
        new_tensor = copy.deepcopy(self.tensor)

        for indx, op in enumerate(self.operator):

            if op == 'GEO':
                einstr = str_einsum('ijkl', indx, len(shape))
                new_tensor = np.einsum(einstr, new_tensor, cut_T)

        return new_tensor


# transform any tensor's GEO cartesian to normal
def cart2normal(property, fname_mol, fname_tens):

        shape = property.tensor.shape

        import vib_analysis as va
        coordshere, chargeshere, masseshere = va.read_mol(fname_mol)
        hessian_tensor = property.get_hessian(fname_tens)
        cut_w, cut_T, N_3, min_el, max_el = va.get_vib_harm_freqs_and_eigvecs(coordshere, chargeshere, masseshere,
                                                                              hessian_tensor,
                                                                              outproj=True, print_level=0,
                                                                              harmonic_frequency_limits='Keep all')
        new_tensor = copy.deepcopy(property.tensor)

        for indx, op in enumerate(property.operator):

            if op == 'GEO':
                einstr = str_einsum('ijkl', indx, len(shape))
                new_tensor = np.einsum(einstr, new_tensor, cut_T)

        return new_tensor

# Take the specification prop of a property and an index of the associated tensor
# Generate all equivalent tensor indices based on perturbation equivalence symmetry
# Assumes that perturbations have been sorted according to operator and according
# to frequency within collections of identical operators
# Also assumes that operators with same labels have same number of components
def get_redundant_indices(prop, index):
    # Find number of perturbation_types by perturbation equivalence

    # Initial setup
    # num_perturbation_types = 0
    perturbation_types = [[0, 1]]
    current_perturbation_type = 0

    # Traverse all perturbations
    for i in range(prop.order):

        # If operator is the same, test frequency value
        if (prop.operator[i] == prop.operator[perturbation_types[current_perturbation_type][0]]):

            # If frequency value is the same, set new end point of block to this perturbation
            if (prop.frequencies[i] == prop.frequencies[perturbation_types[current_perturbation_type][0]]):

                perturbation_types[current_perturbation_type][1] = i + 1

            # Otherwise, end this block and start a new one at next perturbation
            else:

                perturbation_types[current_perturbation_type][1] = i
                current_perturbation_type += 1
                perturbation_types.append([i, i + 1])

        # Otherwise, end this block and start a new one at next perturbation
        else:

            perturbation_types[current_perturbation_type][1] = i
            current_perturbation_type += 1
            perturbation_types.append([i, i + 1])

    # For each block:
    # Generate permutations of associated indices but keep only unique
    type_permutations = []
    num_perturbation_types = current_perturbation_type

    for i in range(num_perturbation_types + 1):

        type_permutations.append([list(j) for j in \
                                  itertools.permutations(index[perturbation_types[i][0]:perturbation_types[i][1]])])

        new_type_permutations = []

        for j in range(len(type_permutations[i])):

            if not type_permutations[i][j] in new_type_permutations:
                new_type_permutations.append(copy.deepcopy(type_permutations[i][j]))

        type_permutations[i] = copy.deepcopy(new_type_permutations)

    # Build direct product of block permutations block by block

    redundant_indices = type_permutations[0]

    for i in range(1, num_perturbation_types + 1):

        redundant_indices = [list(j) for j in itertools.product(redundant_indices, \
                                                                type_permutations[i])]

        single_list = []

        # Collapse list of lists into single list
        for j in range(len(redundant_indices)):

            single_list.append([])

            for k in range(len(redundant_indices[j])):
                single_list[j].extend(redundant_indices[j][k])

        redundant_indices = copy.deepcopy(single_list)

    # Turn every permutation thus generated into tuples and decrement by one since counting starts at zero
    for i in range(len(redundant_indices)):
        redundant_indices[i] = tuple([j - 1 for j in redundant_indices[i]])

    return redundant_indices


# Check if keyword inp matches the expected string compare; if not then quit (currently not gracefully)
def keyword_check(inp, compare):
    if (not isinstance(compare, list)):
        compare = [compare]

    keyword_okay = False
    for i in range(len(compare)):
        if (inp == compare[i]):
            keyword_okay = True

    if (keyword_okay):
        return
    else:
        print('\n')
        print('Error in keyword_check')
        print('Input read error: Expected keyword ', compare, 'but read ', inp)
        exit()


# Read a line from a file and skip leading and trailing whitespace and newline
def remove_whitespaces(f):
    return f.readline().rstrip('\n').rstrip().lstrip()


# Read a tensor file, return a list of property class instances and individual data as numpy arrays
def read_openrsp_tensor_file(fname):
    f = open(fname, 'r')

    keyword_check(remove_whitespaces(f), 'VERSION')

    format_version = int(remove_whitespaces(f))

    if not (format_version == 1):
        print('\n')
        print('Eror in read_openrsp_tensor_file')
        print('File format version was given as ', format_version, ' but only version 1 is supported')
        print(' in this routine')
        exit()

    keyword_check(remove_whitespaces(f), 'NUM_PROPERTIES')

    num_properties = int(remove_whitespaces(f))

    rsp_redundant_properties = []  # BETTER NAME
    rsp_tensors = []

    # Loop over number of properties that it was told that this file contains
    for i in range(num_properties):

        keyword_check(remove_whitespaces(f), 'NEW_PROPERTY')

        # Order (number of perturbations) for this property
        keyword_check(remove_whitespaces(f), 'ORDER')
        current_order = int(remove_whitespaces(f))

        # Number of frequency configurations given for this property
        keyword_check(remove_whitespaces(f), 'NUM_FREQ_CFGS')
        current_num_configs = int(remove_whitespaces(f))

        # Specification of perturbing operators associated with this property
        keyword_check(remove_whitespaces(f), 'OPERATORS')

        current_operator = []

        for j in range(current_order):
            current_operator.append(remove_whitespaces(f))

        # Specification of number of components for each operator
        keyword_check(remove_whitespaces(f), ['NUM COMPONENTS', 'NUM_COMPONENTS'])

        current_num_components = []

        for j in range(current_order):
            current_num_components.append(int(remove_whitespaces(f)))

        # Specification of frequencies associated with perturbations
        keyword_check(remove_whitespaces(f), 'FREQUENCIES')

        current_frequencies = []

        # Loop over number of frequency configurations
        for j in range(current_num_configs):

            current_frequencies.append([])

            # Marks start of new frequency configuration
            keyword_check(remove_whitespaces(f), 'CONFIGURATION')

            for k in range(current_order):
                current_frequencies[j].append(float(remove_whitespaces(f)))

        # Specification of values of the associated tensor (or tensors if more than one freq. config.)
        keyword_check(remove_whitespaces(f), 'VALUES')

        # Loop over frequency configurations
        for j in range(current_num_configs):

            # Marks start of new frequency configuration
            keyword_check(remove_whitespaces(f), 'CONFIGURATION')
            current_tensor = np.zeros(tuple(current_num_components))

            # Make new instance of property class for this property with this freq. config.
            new_property = rspProperty(current_order, current_operator, current_num_components, \
                                       current_frequencies[j])

            # Reading values of tensor elements until done
            tensor_done = 0

            while not (tensor_done):

                prev_line = f.tell()

                nline = f.readline()

                # If reached EOF, break read
                if nline == '':
                    break

                nline = nline.rstrip('\n').rstrip().lstrip()

                # If starting new property, undo line read consider this tensor read done
                if (nline == 'NEW_PROPERTY'):
                    tensor_done = 1
                    f.seek(prev_line)

                # If starting new frequency configuration, undo line read and consider this tensor
                # read done
                elif (nline == 'CONFIGURATION'):
                    tensor_done = 1
                    f.seek(prev_line)
                else:

                    # Read index of tensor whose value is to be specified on next line
                    new_index = tuple([int(m) for m in nline.split()])
                    # Read the value of tensor element with this index
                    new_value = float(remove_whitespaces(f))

                    # Get all permutations of indices known to have same value
                    # due to symmetry of perturbations

                    all_redundant_indices = get_redundant_indices(new_property, new_index)

                    # Assign the read tensor element value to all such valid permuted indices
                    for k in all_redundant_indices:
                        current_tensor[k] = new_value

            # Append the property specification and tensor to the arrays to be returned
            rsp_redundant_properties.append(new_property)
            rsp_tensors.append(current_tensor)

    return rsp_redundant_properties, rsp_tensors


def str_einsum(origstr, same_ind, lenshape):
    origstr = origstr[:lenshape]
    neworigstr = origstr[:same_ind] + 'q' + origstr[same_ind + 1:]
    return origstr + f',{origstr[same_ind]}q->' + neworigstr