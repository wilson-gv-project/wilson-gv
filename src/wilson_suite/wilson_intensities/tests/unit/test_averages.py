import wilson_suite.wilson_intensities.spectrum.term_parts  as tparts

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
    
    from ...spectrum.pre_eval_treatment import generate_index_choices_general
    idxs = generate_index_choices_general(indlabels_in_motif=nm_indices_symb, labels=['1', '2', '3'])
    print()
    for i in idxs:
        print(i)