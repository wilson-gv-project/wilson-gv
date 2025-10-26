import wilson_suite.wilson_intensities.amplitudes.term_parts  as tparts

    # assert len(unique) == 4

def test_PropsCollection_get_avegaded_props():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    t_inds = [0, 1, -2, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    
    motifs_coll = []

    for t in terms_select:
        t_props = tparts.PropsCollection(props=t.props)
        
        # print('----')
        # motifs_coll.append(t_props.get_avegaded_props())

        # print('all t_props')
        # for tp in t_props:
        #     print(tp)
        # avrg_props = t_props.get_avegaded_props()
        # print(avrg_props)
        # print('avrg_props.get_mode_indices()', avrg_props.get_mode_indices())
        # print('avrg_props.get_cart_axes()', avrg_props.get_cart_axes())
        # print('avrg_props.get_total_difforder()', avrg_props.get_total_difforder())
        # print('avrg_props.identify_avrg_motif()', avrg_props.identify_avrg_motif())

        motifs_coll.append(t_props.identify_avrg_motif())
    for m in motifs_coll:
        print(m)
    # print(motifs_coll)
    print(motifs_coll[2] == motifs_coll[3])
    print(motifs_coll[0] == motifs_coll[2])
    print(motifs_coll[0] == motifs_coll[1])
    print(set(motifs_coll))

"""

PropsCollection(props=[PolProp(ops = [QOperator(o = 0, op_type = None, ax = None), 
                                      QOperator(o = 3, op_type = None, ax = None)], dord = 1, (inds = ['b'])), 
                       PolProp(ops = [QOperator(o = 1, op_type = None, ax = None)], dord = 1, (inds = ['a'])), 
                       PolProp(ops = [QOperator(o = 2, op_type = None, ax = None)], dord = 1, (inds = ['c']))])
"""

def test_ResonanceMotif():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    t_inds = [0, 1, -2, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    for t in terms_select:
        t_resmotif = tparts.ResonanceMotif(t.res)
        print('----')
        print(t_resmotif)
        print(t_resmotif.get_vibdiffs())
        print(t_resmotif.get_freq_axes())
