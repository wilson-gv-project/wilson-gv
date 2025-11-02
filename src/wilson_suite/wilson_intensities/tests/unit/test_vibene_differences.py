import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_main import abstractions as f_abst
import wilson_suite.wilson_derive.abstractions as wa
from wilson_suite.wilson_main.abstractions import VibState


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

    vibdata = VibStatesData(allstates=(f_abst.VibState(harm_quanta_coeffs={'0':1.}, state_label='0', energy=964.),
                                       f_abst.VibState(harm_quanta_coeffs={'1':1.}, state_label='1', energy=1234.),
                                       f_abst.VibState(harm_quanta_coeffs={'2':1.}, state_label='2', energy=3644.)),
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

def test_get_vibdiff_value():
    print()

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    b_state = wa.HarmOscStateSymbolic(['b'])
    zero_state = wa.HarmOscStateSymbolic([])

    vd_ab_a = wa.VibDiffTerm(sl=ab_state, sr=a_state)
    vd_0_a = wa.VibDiffTerm(sl=zero_state, sr=a_state)
    vd_b_a = wa.VibDiffTerm(sl=b_state, sr=a_state)

    print('vd_ab_a key', vediff.make_vibdiff_key(vd_ab_a, {'a': 5, 'b': 7, 'c': 9}))
    print('vd_0_a key', vediff.make_vibdiff_key(vd_0_a, {'a': 5, 'b': 7, 'c': 9}))

    bank = vediff.VibDiffCache()
    bank.register

def test_make_sorted_vibdiff_key():
    print()
    print(vediff.make_sorted_vibdiff_key('5,5,7', '5,5'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5,5', '5,5,7'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5,5,7', '5'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5,5,7', '7'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5', '5,7'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5,7', '5'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('5,5', '5,7'))     # -> ('5', '5,7')
    print(vediff.make_sorted_vibdiff_key('zero', '5'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('5', 'zero'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('5', '7'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('15', '51'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('51', '15'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('zero', 'zero'))    # -> ('zero', '5')
    print(vediff.make_sorted_vibdiff_key('', ''))    # -> ('zero', '5')

def test_compute_vibdiff():
    print()
    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
                                       VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
                                       VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.),
                                       VibState(harm_quanta_coeffs={(1,2,): 1.}, state_label='1,2', energy=4736.)
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))
    t = vediff.compute_vibdiff(('1,2', '1'), vibstates_data=vibdata)
    print(t)
    assert vediff.compute_vibdiff(('2', '1'), vibstates_data=vibdata) == 3644. - 1234.
    assert vediff.compute_vibdiff(('1,2', '1'), vibstates_data=vibdata) == 4736. - 1234.

def test_compute_vibdiff_w_bank():
    print()
    import wilson_suite.wilson_derive.abstractions as wa

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    vd_ab_a = wa.VibDiffTerm(sl=ab_state, sr=a_state)

    vibdiff_bank = vediff.VibDiffCache()
    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
                                       VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
                                       VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.),
                                       VibState(harm_quanta_coeffs={(1,2,): 1.}, state_label='1,2', energy=4736.),
                                       VibState(harm_quanta_coeffs={(0,2,): 1.}, state_label='0,2', energy=4518.)
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))
    res = vediff.compute_vibdiff_w_bank(vibdiff_term=vd_ab_a, 
                                        index_dict={'a': 2, 'b': 0, 'c': 9},
                                        vibdiff_bank=vibdiff_bank,
                                        vibstates_data=vibdata)
    print(res)
    print(vibdiff_bank)

def test_VibDiffCache():
    """Test VibDiffCache functionality"""
    print("\nTesting VibDiffCache...")
    
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.),
            VibState(harm_quanta_coeffs={(1,2): 1.}, state_label='1,2', energy=4736.)
        ),
        harmonic_osc_states_labels=(0, 1, 2)
    )
    
    # Create cache instance
    cache = vediff.VibDiffCache()
    
    # Test 1: Basic caching functionality
    state1 = VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.)
    state2 = VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.)
    diff = vediff.VibDiff(state1, state2)
    
    # Initially should be empty
    assert cache.get(diff) is None
    
    # Add and retrieve
    energy_diff = diff.energy_difference()
    cache.add(diff, energy_diff)
    assert cache.get(diff) == energy_diff
    
    # Test 2: Order invariance
    reverse_diff = vediff.VibDiff(state2, state1)
    assert cache.get(reverse_diff) == -energy_diff
    
    # Test 3: Multiple states
    state3 = VibState(harm_quanta_coeffs={(1,2): 1.}, state_label='1,2', energy=4736.)
    diff2 = vediff.VibDiff(state3, state2)
    energy_diff2 = diff2.energy_difference()
    cache.add(diff2, energy_diff2)
    
    # Both differences should be retrievable
    assert cache.get(diff) == energy_diff
    assert cache.get(diff2) == energy_diff2
    
    # Test 4: Zero state handling
    zero_state = VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.0)
    zero_diff = vediff.VibDiff(zero_state, state1)
    zero_energy_diff = zero_diff.energy_difference()
    cache.add(zero_diff, zero_energy_diff)
    assert cache.get(zero_diff) == zero_energy_diff
    
    # Test 5: Edge cases
    # Same state difference should be zero
    same_state_diff = vediff.VibDiff(state1, state1)
    cache.add(same_state_diff, 0.0)
    assert cache.get(same_state_diff) == 0.0
    
    print("All VibDiffCache tests passed!")

def test_VibDiff_normalized():
    """Test VibDiff normalization"""
    state1 = VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.)
    state2 = VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.)
    
    diff = vediff.VibDiff(state1, state2)
    norm_diff = diff.normalized()
    
    # Should always return the version with smaller state label first
    assert norm_diff.left.state_label <= norm_diff.right.state_label
    
    # Test symmetry
    reverse_diff = vediff.VibDiff(state2, state1)
    reverse_norm = reverse_diff.normalized()
    assert str(norm_diff.left) == str(reverse_norm.left)
    assert str(norm_diff.right) == str(reverse_norm.right)

def test_make_VibDiff_from_symbolic():
    """Test conversion from symbolic to concrete VibDiff"""
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.)
        ),
        harmonic_osc_states_labels=(0, 1, 2)
    )
    
    # Create symbolic term
    symb_term = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['a']),
        sr=wa.HarmOscStateSymbolic(['b'])
    )
    
    # Create index mapping
    index_dict = {'a': 1, 'b': 2}
    
    # Create cache
    cache = vediff.VibDiffCache()
    
    # Convert symbolic to concrete
    vib_diff = vediff.make_VibDiff_from_symbolic(symb_term, index_dict, cache, vibdata)
    print(vib_diff)

    # Verify results
    assert vib_diff.left.state_label == '1'
    assert vib_diff.right.state_label == '2'
    assert vib_diff.energy_difference() == -2410.0  # 1234 - 3644
    assert cache.get(vib_diff) == -2410.0

    print('\n', cache)
    print('\n', type(cache._cache))
    for k in cache._cache:
        print(type(k), k)


def test_VibDiffCache_basic_operations():
    """Simple sanity checks for cache operations"""
    cache = vediff.VibDiffCache()
    
    # Test 1: Cache size tracking
    state1 = VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=100.)
    state2 = VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=200.)
    
    diff = vediff.VibDiff(state1, state2)
    cache.add(diff, -100.0)
    
    # Check cache has exactly 1 entry (or 2 if storing both directions)
    print(f"Cache size after 1 add: {len(cache._cache)}")
    assert len(cache._cache) > 0
    
    # Test 2: Add same diff twice - should overwrite, not duplicate
    initial_size = len(cache._cache)
    cache.add(diff, -100.0)
    assert len(cache._cache) == initial_size
    
    # Test 3: Retrieve non-existent diff returns None
    state3 = VibState(harm_quanta_coeffs={(3,): 1.}, state_label='3', energy=300.)
    missing_diff = vediff.VibDiff(state1, state3)
    assert cache.get(missing_diff) is None
    

def test_VibDiffCache_sign_consistency():
    """Verify energy differences have correct signs"""
    cache = vediff.VibDiffCache()
    
    state_low = VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1_low', energy=500.)
    state_high = VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2_high', energy=1000.)
    
    # Forward: low -> high should be negative
    diff_forward = vediff.VibDiff(state_low, state_high)
    energy_diff = diff_forward.energy_difference()
    
    cache.add(diff_forward, energy_diff)
    
    assert energy_diff < 0, "Low to high should be negative"
    assert cache.get(diff_forward) == energy_diff
    
    # Reverse: high -> low should be positive
    diff_reverse = vediff.VibDiff(state_high, state_low)
    assert cache.get(diff_reverse) == -energy_diff
    assert cache.get(diff_reverse) > 0, "High to low should be positive"

def test_VibDiffCache_multiple_lookups():
    """Test that multiple lookups return consistent values"""
    cache = vediff.VibDiffCache()
    
    state1 = VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=123.)
    state2 = VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=456.)
    
    diff = vediff.VibDiff(state1, state2)
    expected = -333.0
    cache.add(diff, expected)
    
    # Lookup same diff multiple times
    for i in range(5):
        result = cache.get(diff)
        assert result == expected, f"Lookup {i} failed: {result} != {expected}"



def test_process_extra_freqterms_with_bank_basic():
    """Test basic functionality of processing extra frequency terms"""
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.),
            VibState(harm_quanta_coeffs={(3,): 1.}, state_label='3', energy=5000.)
        ),
        harmonic_osc_states_labels=(0, 1, 2, 3)
    )
    
    # Create symbolic terms
    term1 = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['a']),
        sr=wa.HarmOscStateSymbolic(['b'])
    )
    term2 = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['b']),
        sr=wa.HarmOscStateSymbolic(['c'])
    )
    
    extra_freqterms = [term1, term2]
    index_dict = {'a': 1, 'b': 2, 'c': 3}
    cache = vediff.VibDiffCache()
    vibdiffs_bank = {}
    
    # Process terms
    result = vediff.process_extra_freqterms_with_bank(
        extra_freqterms, 
        index_dict, 
        cache, 
        vibdata, 
        vibdiffs_bank
    )
    print('\nresult', result)
    for i in result:
        print('---', i.energy_difference(), i.left.energy, i.right.energy)
        
    # Verify results
    assert len(result) == 2
    assert result[0].left.state_label == '1'
    assert result[0].right.state_label == '2'
    assert result[1].left.state_label == '2'
    assert result[1].right.state_label == '3'
    


def test_process_extra_freqterms_with_bank_retrieval():
    """Test that function retrieves from bank when available"""
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.)
        ),
        harmonic_osc_states_labels=(1, 2)
    )
    
    # Pre-create a VibDiff for the bank
    state1 = vibdata.allstates[0]
    state2 = vibdata.allstates[1]
    precomputed_diff = vediff.VibDiff(state1, state2)
    
    # Create symbolic term
    term = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['a']),
        sr=wa.HarmOscStateSymbolic(['b'])
    )
    
    # Put it in the bank
    vibdiffs_bank = {term: precomputed_diff}
    
    extra_freqterms = [term]
    index_dict = {'a': 1, 'b': 2}
    cache = vediff.VibDiffCache()
    
    # Process - should retrieve from bank, not recompute
    result = vediff.process_extra_freqterms_with_bank(
        extra_freqterms, 
        index_dict, 
        cache, 
        vibdata, 
        vibdiffs_bank
    )
    
    # Verify we got the same object from the bank
    assert len(result) == 1
    assert result[0] is precomputed_diff
    assert result[0].left.state_label == '1'
    assert result[0].right.state_label == '2'
    
    print("Bank retrieval test passed!")


def test_process_extra_freqterms_mixed_sources():
    """Test processing with some terms in bank and some not"""
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964.),
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644.),
            VibState(harm_quanta_coeffs={(3,): 1.}, state_label='3', energy=5000.)
        ),
        harmonic_osc_states_labels=(0, 1, 2, 3)
    )
    
    # Create symbolic terms
    term_in_bank = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['a']),
        sr=wa.HarmOscStateSymbolic(['b'])
    )
    term_not_in_bank = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['c']),
        sr=wa.HarmOscStateSymbolic(['d'])
    )
    
    # Pre-populate bank with only one term
    state1 = vibdata.allstates[1]
    state2 = vibdata.allstates[2]
    precomputed_diff = vediff.VibDiff(state1, state2)
    vibdiffs_bank = {term_in_bank: precomputed_diff}
    
    extra_freqterms = [term_in_bank, term_not_in_bank]
    index_dict = {'a': 1, 'b': 2, 'c': 0, 'd': 3}
    cache = vediff.VibDiffCache()
    
    # Process mixed sources
    result = vediff.process_extra_freqterms_with_bank(
        extra_freqterms, 
        index_dict, 
        cache, 
        vibdata, 
        vibdiffs_bank
    )
    
    # Verify results
    assert len(result) == 2
    
    # First should be from bank
    assert result[0] is precomputed_diff
    assert result[0].left.state_label == '1'
    assert result[0].right.state_label == '2'
    
    # Second should be newly computed
    assert result[1].left.state_label == '0'
    assert result[1].right.state_label == '3'
    
    # Check cache was updated for the new one
    assert cache.get(result[1]) is not None
    
    print("Mixed sources test passed!")


def test_process_extra_freqterms_empty():
    """Test processing empty list of terms"""
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234.),
        ),
        harmonic_osc_states_labels=(1,)
    )
    
    extra_freqterms = []
    index_dict = {}
    cache = vediff.VibDiffCache()
    vibdiffs_bank = {}
    
    result = vediff.process_extra_freqterms_with_bank(
        extra_freqterms, 
        index_dict, 
        cache, 
        vibdata, 
        vibdiffs_bank
    )
    
    assert len(result) == 0
    assert result == []
    
    print("Empty list test passed!")


def test_process_extra_freqterms_cache_population():
    """Test that cache gets populated for newly computed terms"""
    # Setup test data
    vibdata = VibStatesData(
        allstates=(
            VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1000.),
            VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=2000.),
            VibState(harm_quanta_coeffs={(3,): 1.}, state_label='3', energy=3000.)
        ),
        harmonic_osc_states_labels=(1, 2, 3)
    )
    
    # Create symbolic terms (not in bank)
    term1 = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['a']),
        sr=wa.HarmOscStateSymbolic(['b'])
    )
    term2 = vediff.VibDiffTerm(
        sl=wa.HarmOscStateSymbolic(['b']),
        sr=wa.HarmOscStateSymbolic(['c'])
    )
    
    extra_freqterms = [term1, term2]
    index_dict = {'a': 1, 'b': 2, 'c': 3}
    cache = vediff.VibDiffCache()
    vibdiffs_bank = {}  # Empty bank
    
    # Initially cache should be empty
    assert len(cache._cache) == 0
    
    # Process terms
    result = vediff.process_extra_freqterms_with_bank(
        extra_freqterms, 
        index_dict, 
        cache, 
        vibdata, 
        vibdiffs_bank
    )
    
    # Cache should now contain both diffs
    assert len(cache._cache) > 0
    assert cache.get(result[0]) == -1000.0  # 1000 - 2000
    assert cache.get(result[1]) == -1000.0  # 2000 - 3000
    
    print("Cache population test passed!")