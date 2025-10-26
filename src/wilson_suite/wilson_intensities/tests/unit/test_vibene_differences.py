import wilson_suite.wilson_intensities.amplitudes.vibene_differences


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