import wilson_suite.wilson_intensities.amplitudes.averaged_props
import wilson_suite.wilson_intensities.amplitudes.term_parts as tparts
import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops

def get_expressions():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    # terms_select = terms_fuller_flat

    return [tparts.PropsCollection(t.props) for t in terms_select]

def test_expr1():
    expression = get_expressions()[0]
    print()
    for prop in expression:
        print(prop)
    nm_indices_symb = sorted(set(expression.get_mode_indices()))
    
    from ...amplitudes.utils import generate_index_choices_general
    idxs = generate_index_choices_general(indlabels_in_motif=nm_indices_symb, labels=['1', '2', '3'])
    print()
    for i in idxs:
        print(i)


def test_identify_unique_avrgmotifs():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    terms_select = terms_fuller_flat

    unique = wilson_suite.wilson_intensities.amplitudes.averaged_props.identify_unique_avrgmotifs(terms_select)
    print('\n\n')
    for i in unique:
        print(i)
    print(len(unique))


def test_make_avrg_props_motif():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    collect_simple = []
    t_inds = [0, -2]
    for tID in t_inds:
        term = terms_fuller_flat[tID]

        # get only avrg props
        # props_with_cart_axes = [prop.to_latex() for prop in term.props if prop.ops]
        props_with_cax_simple = [wilson_suite.wilson_intensities.amplitudes.averaged_props.simple_prop_ID(prop) for prop in term.props if prop.ops]

        collect_simple.append(set(tuple(props_with_cax_simple)))
        pp = wilson_suite.wilson_intensities.amplitudes.averaged_props.make_avrg_props_motif(term.props)
        print(props_with_cax_simple)
        print(pp)