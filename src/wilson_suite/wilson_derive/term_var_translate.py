import copy
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def find_pulse_id_tuples_as_axis_vars(id_tuple, axes):

    id_tuple_in_axis_vars = {}

    return id_tuple_in_axis_vars

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
                    #print('pf', m.pf)
                    if not tuple(sorted(m.pf)) in pulse_id_tuples:
                        pulse_id_tuples.append(tuple(copy.deepcopy(sorted(m.pf))))

    print('pulse id tuples', pulse_id_tuples)
    print('chosen axes', chosen_axes)

    id_tuple_in_axis_vars = {}

    for i in pulse_id_tuples:
        id_tuple_in_axis_vars[i] = find_pulse_id_tuples_as_axis_vars(i, chosen_axes)





    # Take the chosen axes and call fn to express all ind vars in terms of these

    # Go through each term and translate


    pass