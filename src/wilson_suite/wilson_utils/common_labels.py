# Common labels to be used throughout wilson-suite

# (Vibrational state labels
state_labels = ['m', 'n', 'o', 'p', 'q', 'r', 's', 't']

# (Vibrational) ground state label
ground_state_label = '0'

# Normal mode indices
nm_inds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

# Operator labels (wilson-derive convention)
op_labels_int = [1, 2, 3, 4, 5, 6, 7, 8]

# "Omega" special operator label
op_omega_label_int = 0

# Operator labels as latinized Greek letters (wilson-intensities convention)
op_labels_greek = ['B', 'G', 'D', 'E', 'Z', 'H', 'T', 'I']

# "Omega" special operator label
op_omega_label_greek = 'A'

# Capital alphabetical labels (primary use: Axis aliases)
cap_alpha_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# numerical index of ordering for Capital alphabetical labels -- alpha label to index mapping
num_cap_alpha_labels = dict(zip(cap_alpha_labels, range(len(cap_alpha_labels))))