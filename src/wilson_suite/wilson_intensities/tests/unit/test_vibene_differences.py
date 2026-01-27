import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_main import abstractions as f_abst
import wilson_suite.wilson_derive.abstractions as wa
from wilson_suite.wilson_main.abstractions import VibState


def test_identify_unique_vibdiff_motifs():
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    res = vediff.identify_unique_vibdiff_motifs(terms_select)
    for i in res:
        print(i)


def test_calculate_vibenedenom_tensor():
    print('\n')

    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    # t_inds = [0, 1, 2, -3, -2, -1]
    t_inds = [0, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]

    vibdata = VibStatesData(allstates=(f_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       f_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       f_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True)),
                                   harmonic_osc_states_labels=(0, 1, 2), number_of_nmodes=3)

    id_vibenedenom = sorted(list(vediff.identify_vibenedenoms(terms_select)))
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

    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

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
    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,): 1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,): 1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,): 1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,2,): 1.}, state_label='1,2', energy=4736., harmonic_WF=False)
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))
    t = vediff.compute_vibdiff(('1,2', '1'), vibstates_data=vibdata)
    print(t)
    assert vediff.compute_vibdiff(('2', '1'), vibstates_data=vibdata) == 3644. - 1234.
    assert vediff.compute_vibdiff(('1,2', '1'), vibstates_data=vibdata) == 4736. - 1234.


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

