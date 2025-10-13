import copy
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def find_pulse_id_tuples_as_axis_vars(id_tuple: tuple, axes: dict):
    """
    Attempt to express a linear combination of pulse IDs in terms of the given axes

    id_tuple: A tuple expressing a  +/- 1 linear combination of pulse IDs
    axes: A dictionary expressing axes as {axis dummy label: independent variable(s) in axis} pairs

    returns (on success): id_tuple_in_axis_vars: {axis label 1: coeff, ...}}: A description
    of which linear combination of the axis variables make up the combination id_tuple

    On fail, throws AssertionError (see end).
    """

    from itertools import product as iter_prod
    from wilson_suite.wilson_utils.common_labels import cap_alpha_labels

    id_tuple_in_axis_vars = {}

    # Crude search: Try all combinations -1, 0, 1 * the axis variables; stop when found
    for i in iter_prod([-1, 0, 1], repeat=len(axes)):

        running_vars = []

        for j in range(len(i)):
            if not i[j] == 0:

                curr_ax = axes[cap_alpha_labels[j]]

                for p in curr_ax:
                    for k in p:
                        if i[j] == -1:

                            if k in running_vars:
                                del running_vars[running_vars.index(k)]
                            else:
                                running_vars.append(-1 * k)

                        elif i[j] == 1:

                            if -1 * k in running_vars:
                                del running_vars[running_vars.index(-1 * k)]
                            else:
                                running_vars.append(k)

        # If success, assemble result and return
        if tuple(sorted(running_vars)) == id_tuple:

            # Assemble ID tuple in axis variables
            for j in range(len(i)):
                id_tuple_in_axis_vars[cap_alpha_labels[j]] = i[j]

            return id_tuple_in_axis_vars


    err_str = 'Error: Pulse ID tuple ' + str(id_tuple) + ' was not expressable in terms of chosen axes ' + str(axes)
    err_str += ('. The occurrence of this error may be due to an ambiguity inherent to the present UV/VIS-range pulse' +
                ' interaction handling arising when more than one partitioning of the same such pulses could result in' +
                ' an IR-range sum frequency.')
    raise AssertionError(err_str)

def translate_one_term_to_axis_variables(term: VibPerturbedTerm, id_tuples_in_axis_vars: dict) -> VibPerturbedTerm:
    """
    Translate a term represented in terms of pulse IDs to be represented in terms of chosen axes

    term: VibPerturbedTerm to be translated

    id_tuples_in_axis_vars: Dictionary {pulse linear combination: {axis label 1: coeff, ...}}: That is, a description
    of which linear combination of the axis variables make up a given oulse linear combination

    returns: return_term: The translated term
    """


    return_term = copy.deepcopy(term)

    # Walk through the resonance conditions and translate according to id_tuples_in_axis_vars
    for i in range(len(return_term.res)):

        idt_dict = id_tuples_in_axis_vars[tuple(sorted(return_term.res[i].pf))]

        new_pf = []

        for j in idt_dict:
            if idt_dict[j] == 1:
                new_pf.append(j)
            elif idt_dict[j] == -1:
                new_pf.append('-' + j)

        return_term.res[i].pf = copy.deepcopy(new_pf)

    return return_term

# FIXME: Currently translating only for resonance conditions: If later using non-static pol props, then may
# need extra handling for UV parts of that? Not sure
def translate_terms_to_axis_variables(terms: list[VibPerturbedTerm], chosen_axes: dict) -> list[VibPerturbedTerm]:
    """
    Translate terms represented in terms of pulse IDs to be represented in terms of chosen axes

    terms: list of VibPerturbedTerm instances: The terms to be translated
    chosen_axes: dictionary of {axis dummy label: list of independent variables in axis} pairs

    Returns: translated_terms: list of VibPerturbedTerm instances: The terms thus translated

    """

    # Walk through all terms and identify all pulse ID tuples used
    pulse_id_tuples = []

    # NOTE: Assumes that resonance conditions have been canonically sorted according to number of perturbing freqs
    for i in terms:
        for j in terms[i]:
            for k in terms[i][j]:
                for m in k.res:
                    if not tuple(sorted(m.pf)) in pulse_id_tuples:
                        pulse_id_tuples.append(tuple(copy.deepcopy(sorted(m.pf))))

    # Take the chosen axes and call fn to express all ind vars in terms of these
    id_tuples_in_axis_vars = {}

    for i in pulse_id_tuples:
        id_tuples_in_axis_vars[i] = find_pulse_id_tuples_as_axis_vars(i, chosen_axes)

    print('idt', id_tuples_in_axis_vars)

    # Go through each term and translate; make structure of same shape as original to return
    translated_terms = {}
    for i in terms:
        translated_terms[i] = {}
        for j in terms[i]:
            translated_terms[i][j] = []
            for k in terms[i][j]:
                translated_terms[i][j].append(translate_one_term_to_axis_variables(k, id_tuples_in_axis_vars))

    return translated_terms