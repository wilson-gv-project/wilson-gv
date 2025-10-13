import copy
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def find_pulse_id_tuples_as_axis_vars(id_tuple, axes):

    from itertools import product as iter_prod
    from wilson_suite.wilson_utils.common_labels import cap_alpha_labels

    id_tuple_in_axis_vars = {}

    # Crude search: Try all combinations -1, 0, 1 * the axis variables; stop when found
    for i in iter_prod([-1, 0, 1], repeat=len(axes)):

        running_vars = []

        for j in range(len(i)):

            if not i[j] == 0:

                print('axes', axes)

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

        if tuple(sorted(running_vars)) == id_tuple:

            # Assemble ID tuple in axis variables
            for j in range(len(i)):
                id_tuple_in_axis_vars[cap_alpha_labels[j]] = i[j]

            return id_tuple_in_axis_vars

    # I think that if the input data was properly assembled, then this condition will never be met; nevertheless
    # included just in case
    raise AssertionError('Error: Pulse ID tuple', id_tuple, 'was not expressable in terms of chosen axes', axes)

def translate_one_term_to_axis_variables(term: VibPerturbedTerm, id_tuples_in_axis_vars):

    return_term = copy.deepcopy(term)

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
def translate_terms_to_axis_variables(terms: list[VibPerturbedTerm], chosen_axes: dict):

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

    # Go through each term and translate; make structure of same shape as original to return

    translated_terms = {}
    for i in terms:
        translated_terms[i] = {}
        for j in terms[i]:
            translated_terms[i][j] = []
            for k in terms[i][j]:
                translated_terms[i][j].append(translate_one_term_to_axis_variables(k, id_tuples_in_axis_vars))

    return translated_terms