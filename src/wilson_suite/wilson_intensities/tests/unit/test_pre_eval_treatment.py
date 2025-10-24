import wilson_suite.wilson_intensities.amplitudes.averaged_props
import wilson_suite.wilson_intensities.amplitudes.domains
import wilson_suite.wilson_intensities.amplitudes.resonances
import wilson_suite.wilson_intensities.amplitudes.term_parts  as tparts
import wilson_suite.wilson_intensities.amplitudes.vibene_differences

def generate_only_res_cond_evv_term_selection():

    import wilson_suite.wilson_derive.abstractions as wa
    from fractions import Fraction

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    b_state = wa.HarmOscStateSymbolic(['b'])
    zero_state = wa.HarmOscStateSymbolic([''])

    vd_ab_a = wa.VibDiffTerm(sl = ab_state, sr = a_state)
    vd_0_a = wa.VibDiffTerm(sl=zero_state, sr=a_state)
    vd_b_a = wa.VibDiffTerm(sl=b_state, sr=a_state)

    rc_ab_a_w_A = wa.ResonanceCondition(diff = vd_ab_a, pf = ['A'])
    rc_b_a_w_B = wa.ResonanceCondition(diff=vd_b_a, pf=['B'])
    rc_0_a_w_B = wa.ResonanceCondition(diff=vd_0_a, pf=['B'])
    rc_0_a_w_AmB = wa.ResonanceCondition(diff=vd_0_a, pf=['A', '-B'])

    res_conds_a = [rc_ab_a_w_A, rc_b_a_w_B]
    res_conds_b = [rc_ab_a_w_A]
    res_conds_c = [rc_ab_a_w_A, rc_b_a_w_B]
    res_conds_d = [rc_0_a_w_B, rc_b_a_w_B]
    res_conds_e = [rc_0_a_w_B, rc_0_a_w_AmB]

    term_a = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_a)

    term_b = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_b)

    term_c = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_c)

    term_d = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_d)

    term_e = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_e)

    return [term_a, term_b, term_c, term_d, term_e]

def test_terms_for_motif():
    """
    HMM...
unique_motifs:
(((('a', 'b'), ('a',)), ('A',)),)  ------> ????
(((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
(((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',))) ------> ????
(((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))

    """
    candidate_terms = generate_only_res_cond_evv_term_selection()

    unique_motifs = wilson_suite.wilson_intensities.amplitudes.resonances.identify_unique_resmotifs(candidate_terms)

    terms_for_motif = wilson_suite.wilson_intensities.amplitudes.resonances.terms_for_motif(candidate_terms)

    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))]) == 2
    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)),)]) == 1
    assert not(len(terms_for_motif[(((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))]) == 4)

    assert sorted(unique_motifs) == sorted(list(terms_for_motif.keys()))


def test_find_resonance_locations_wrt_index_choices():
    print()

    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from wilson_suite.wilson_intensities.amplitudes import func_abstractions as f_abst
    allstates = (f_abst.VibState(s={}, state_label='1', e=1234.),
                 f_abst.VibState(s={}, state_label='3', e=3644.),
                 f_abst.VibState(s={}, state_label='4', e=1621.),
                 f_abst.VibState(s={}, state_label='1+1', e=2514.),
                 f_abst.VibState(s={}, state_label='1+4', e=1904.),
                 f_abst.VibState(s={}, state_label='3+4', e=4129.),
                 f_abst.VibState(s={}, state_label='4+4', e=3022.),
                 f_abst.VibState(s={}, state_label='3+3', e=7344.),
                 f_abst.VibState(s={}, state_label='1+3', e=4364.))
    harm_labels = ('1', '3', '4')
    vibdata = f_abst.VibStatesData(allstates=allstates, harmonic_osc_states_labels=harm_labels)
    
    d = wilson_suite.wilson_intensities.amplitudes.resonances.find_resonance_locations_wrt_index_choices(motif=motif1, vibstates_data=vibdata)
    print(d)


def test_motifs_control():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()
    
    r = wilson_suite.wilson_intensities.amplitudes.resonances.motifs_control(candidate_terms)
    print(r)

def test_identify_maximum_axes_in_terms():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()

    wilson_suite.wilson_intensities.amplitudes.resonances.identify_maximum_axes_in_terms(candidate_terms)

def test_is_location_in_window():
    loc1 = {'A': 12., 'B': 33.}
    window1 = {'A': (9., 14.), 'B': (22., 54.)}
    window2 = {'A': (12., 14.), 'B': (30., 54.)}
    window3 = {'A': (9., 14.), 'B': (41., 44.)}
    window4 = {'A': (11., 21.), 'B': (41., 44.)}

    print()
    r1 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window1)
    assert r1

    r2 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window2, margins={'A': (2., 2.)})
    assert r2

    r3 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window3)
    assert not r3

    r4 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window4, margins={'B':(10., 2.)})
    assert r4


def test_find_domain_groups_by_distance():
    print()

    points = [[1., 3.], [5., 11.], [4., 2.], [12., 6.], [8., 2.], [11., 4.]]
    print(points, len(points))

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=10.)
    assert len(groups) == 3
    print(groups)

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=12.)
    assert len(groups) == 2
    print(groups)

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=4.)
    assert len(groups) == 4
    print(groups)

def test_find_domain_distance_threshold():
    print()
    wilson_suite.wilson_intensities.amplitudes.domains.find_distance_threshold(1e6, {'A': 3.8, 'B': 3.8})


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

    # assert len(unique) == 4

def test_identify_unique_vibdiff_motifs():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    res = wilson_suite.wilson_intensities.amplitudes.vibene_differences.identify_unique_vibdiff_motifs(terms_select)
    for i in res:
        print(i)

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
