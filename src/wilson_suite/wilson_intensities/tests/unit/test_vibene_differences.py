import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_intensities.amplitudes import func_abstractions as f_abst


def test_identify_unique_vibdiff_motifs():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    res = vediff.identify_unique_vibdiff_motifs(terms_select)
    for i in res:
        print(i)


def test_calculate_vibenedenom_tensor():
    print('\n')

    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = [0, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    vibdata = VibStatesData(allstates=(f_abst.VibState(s={'0':1.}, state_label='0', e=964.),
                                       f_abst.VibState(s={'1':1.}, state_label='1', e=1234.),
                                       f_abst.VibState(s={'2':1.}, state_label='2', e=3644.)),
                                   harmonic_osc_states_labels=(0, 1, 2))

    id_vibenedenom = vediff.identify_vibenedenoms(terms_select)
    print('id_vibenedenom', id_vibenedenom)
    vibenedenom_tensor2d = vediff.calculate_vibenedenom_tensor(vibenedenom_inds=id_vibenedenom[0],
                                                               vibstates_data=vibdata)
    print(vibenedenom_tensor2d)
    vibenedenom_tensor3d = vediff.calculate_vibenedenom_tensor(vibenedenom_inds=id_vibenedenom[1],
                                                               vibstates_data=vibdata)
    print(vibenedenom_tensor3d)
    
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    import numpy as np
    
    assert np.allclose(vibenedenom_tensor3d[0, 0, 0] , 1./ convNu2Ene(964.)**3)
    assert np.allclose(vibenedenom_tensor3d[0, 1, 2] , 1./ (convNu2Ene(964.)*convNu2Ene(1234.)*convNu2Ene(3644.)))
    assert np.allclose(vibenedenom_tensor3d[0, 1, 1] , 1./ (convNu2Ene(964.)*convNu2Ene(1234.)**2))

def test_identify_vibenedenom():
    print('\n')

    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = [0, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    id_vibenedenom = vediff.identify_vibenedenoms(terms_select)
    print(id_vibenedenom)