from wilson_derive.abstractions import VibPerturbedTerm
from wilson_utils.prop_trivname import prop_trivname

def state_list_to_str(stl: list):
    """
    Helper function: Generate a string taking a list of vibrational quanta ['a', 'b', ...] and making 'a+b+...'

    stl: List of vibrational quanta
    """

    return_str = ''

    for i in stl:
        if not isinstance(i, str):
            raise AssertionError('Normal mode index must be string')
        return_str += i + '+'

    if len(return_str) == 0:
        return 'zero'

    else:
        return return_str[:len(return_str) - 1]

def dict_from_term(term: VibPerturbedTerm, floats: bool=True):
    """
    Take a VibPerturbedTerm instance and generate a dictionary representation of it for use in wilson-intensities

    term: The VibPerturbedTerm instance to be so represented
    """

    if not isinstance(term, VibPerturbedTerm):
        raise AssertionError('Term being converted must be a VibPerturbedTerm instance')

    result_dict = {}

    # Properties

    # wilson-derive convention is operators as integers, wilson-intensities convention
    # is operators as (latinized Greek) letters
    numalpha = {0: 'A', 1: 'B', 2: 'G', 3: 'D', 4: 'E', 5: 'Z', 6: 'H', 7: 'T', 8: 'I'}

    averaged_props = []
    non_averaged_props = []

    # To keep track of sign convention in electric multipole expansion factor
    # FIXME: Settle if this applies for properties that are not pure electric dipole properties
    # Currently assuming all even nonzero orders involve a factor -1
    rsp_to_mult_exp_conv_fact = 1

    for i in term.props:

        curr_ops = tuple([numalpha[j.o] for j in i.ops])
        curr_diff_inds = tuple(i.inds)

        if len(curr_ops) > 0:

            if (len(curr_ops) % 2) == 0:
                rsp_to_mult_exp_conv_fact *= -1

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

    # Prefactors

    if floats:
        result_dict['termA_pref'] = float(term.coeff * rsp_to_mult_exp_conv_fact)
    else:
        result_dict['termA_pref'] = term.coeff * rsp_to_mult_exp_conv_fact

    result_dict['termB_pref'] = 1.

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


def derived_terms_dict_to_dicts(derived_terms):
    """
    Example:
    derived_terms = {
        1: {(1, 0): [<wilson_derive.abstractions.VibPerturbedTerm object at 0x7ff3b223b260>,
                     <wilson_derive.abstractions.VibPerturbedTerm object at 0x7ff3b223b650>],
            (0, 1): [<wilson_derive.abstractions.VibPerturbedTerm object at 0x7ff3b22582c0>,
                     <wilson_derive.abstractions.VibPerturbedTerm object at 0x7ff3b2258830>,
                     <wilson_derive.abstractions.VibPerturbedTerm object at 0x7ff3b2258110>]},
        0: {(0, 0): []}}

    result_list = []
    """

    result_list = []
    for key_num_anharms in derived_terms:
        for anharms_tuple in derived_terms[key_num_anharms]:
            for term in derived_terms[key_num_anharms][anharms_tuple]:
                result_list.append(dict_from_term(term))

    return result_list


def flip_modes_indices(term_dict, upd_dict):
    """
    take the result of dict_from_term(term) and flip some abc indices

    uniq_res_conds_idx {'c', 'b'}
{'termA_pref': 0.125,
 'termB_pref': 1.0,

 'averaged_props': (('polgrad', ('a',), ('A', 'D')), ('dipgrad', ('b',), ('B',)), ('dipgrad', ('c',), ('G',))),
 'non_averaged_props': (('cff', ('a', 'b', 'c')),),
 'vibene_denom': ('a', 'b', 'c'),
 'vibenediff': ('a+b,c',),
 'resonances': (('zero,b', (-1,)), ('c,b', (-1, 2)))}

 example here: upd_dict = {'b':'a', 'c':'b'}
    """

    upd_term_dict = {}
    locidxs = ['averaged_props', 'non_averaged_props', 'vibene_denom', 'vibenediff', 'resonances']

    for k in term_dict:
        if k not in locidxs:
            upd_term_dict[k] = term_dict[k]
        else:
            upd_term_dict[k] = replace_nested(term_dict[k], upd_dict)

    return upd_term_dict


# chatUiT
def replace_chars(s, replacements):
    # Create a translation table for single-pass replacement
    translation_table = str.maketrans(replacements)
    return s.translate(translation_table)
# Recursive function to apply replacements selectively
def replace_nested(data, replacements):
    if isinstance(data, str):  # If it's a string, apply replacements
        return replace_chars(data, replacements)
    elif isinstance(data, tuple):  # If it's a tuple, process each element
        # Skip replacements for the first element of the top-level tuple
        if len(data) > 0 and isinstance(data[0], str) and data[0] in ('polgrad', 'dipgrad',
                                                                      'polhess', 'diphess', 'cff', 'qff'):
            return (data[0],) + tuple(replace_nested(item, replacements) for item in data[1:])
        else:
            return tuple(replace_nested(item, replacements) for item in data)
    else:  # If it's not a string or tuple, return it as is
        return data