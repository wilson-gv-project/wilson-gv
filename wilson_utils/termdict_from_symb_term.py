from wilson_derive.abstractions import vibPerturbedTerm
from wilson_utils.prop_trivname import prop_trivname

def state_list_to_str(stl):

    return_str = ''

    for i in stl:
        if not isinstance(i, str):
            raise AssertionError('Normal mode index must be string')
        return_str += i + '+'

    if len(return_str) == 0:
        return 'zero'

    else:
        return return_str[:len(return_str) - 1]

def dict_from_term(term):

    if not isinstance(term, vibPerturbedTerm):
        raise AssertionError('Term being converted must be a vibPerturbedTerm instance')

    result_dict = {}

    # Prefactors

    result_dict['termA_pref'] = term.coeff
    result_dict['termB_pref'] = 1.

    # Properties

    # wilson-derive convention is operators as integers, wilson-intensities convention
    # is operators as (latinized Greek) letters
    numalpha = {0: 'A', 1: 'B', 2: 'G', 3: 'D', 4: 'E', 5: 'Z', 6: 'H', 7: 'T', 8: 'I'}

    averaged_props = []
    non_averaged_props = []

    # FIXME: Handle polarizability vs pure rsp fn sign convention here?
    for i in term.props:

        curr_ops = tuple([numalpha[j.o] for j in i.ops])
        curr_diff_inds = tuple(i.inds)

        if len(curr_ops) > 0:

            averaged_props.append((
                prop_trivname(len(curr_diff_inds), len(curr_ops)),
                curr_diff_inds,
                curr_ops))

        else:

            non_averaged_props.append((
                prop_trivname(len(curr_diff_inds), len(curr_ops)),
                curr_diff_inds))

    result_dict['averaged_props'] = tuple(averaged_props)

    if non_averaged_props == []:
        non_averaged_props = None

    else:
        non_averaged_props = tuple(non_averaged_props)

    result_dict['non_averaged_props'] = non_averaged_props

    # Frequency (difference) terms

    vibenediff = []
    vibene_denom = []

    for i in term.freqterms:

        if i.is_pert_wf_diff:

            vibenediff.append(
                state_list_to_str(i.sl.q) + ',' + state_list_to_str(i.sr.q)
            )

        else:

            if len(i.sr.q) > 0:
                raise AssertionError('Encountered assumed vib energy denominator term with non-zero ket state')

            if not len(i.sl.q) == 1:
                raise AssertionError('Encountered assumed vib energy denominator term with bra state len > 1')

            vibene_denom.append(i.sl.q[0])

        if len(vibene_denom) == 0:
            result_dict['vibene_denom'] = None

        else:
            result_dict['vibene_denom'] = tuple(vibene_denom)

        if len(vibenediff) == 0:
            result_dict['vibenediff'] = None

        else:
            result_dict['vibenediff'] = tuple(vibenediff)

    # Resonance conditions

    resonances = []

    for i in term.res:

        resonances.append((state_list_to_str(i.diff.sl.q) + ',' + state_list_to_str(i.diff.sr.q),
                         tuple(i.pf),))

    result_dict['resonances'] = tuple(resonances)

    return result_dict

