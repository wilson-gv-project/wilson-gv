"""
system_data.py — vibrational states, energy differences, molecular data and the data request
(MolSystemData.from_output_file). Built by hand from tiny arrays, no VibPerturbedTerm.
The from_output_file tests read real files.
"""

import re
import warnings
from importlib.resources import files

import numpy as np
import pytest

from wilson_suite.wilson_derive.abstractions import ResonanceCondition
from wilson_suite.wilson_response_evaluate.plan import (
    ResCondKey,
    ResonanceMotif,
)
from wilson_suite.wilson_response_evaluate.tests.helpers import (
    E0,
    E01,
    E1,
    E0_eigval,
    E1_eigval,
    state,
    toy_states,
    vibdiff,
)
from wilson_suite.wilson_system.system_data import (
    FREQUENCY_AND_STATE_KEYS,
    PROP_LAYOUT,
    VPT2_INPUT_KEYS,
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibDiff,
    VibState,
    VibStatesData,
    _vib_states,
    make_state_label,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


@pytest.fixture
def states() -> VibStatesData:
    """
    state('0', E0), state('1', E1), state('0,1', E01), state('0,0', E00)
    """
    return toy_states()


## VibState / VibStatesData -------------------------------------------------

def test_vibstate_serial_roundtrip():
    s = VibState(harm_quanta_coeffs={(0, 1): 0.9, (1,): 0.1}, state_label='0,1')

    assert s.serial_harm_quanta_coeffs == {'0,1': 0.9, '1': 0.1}
    assert s.deserialize_state_dict() == {(0, 1): 0.9, (1,): 0.1}


def test_vibstate_equality_is_label_and_energy():
    assert state('0', 1000.) == state('0', 1000. + 1e-12)
    assert state('0', 1000.) != state('1', 1000.)
    assert state('0', 1000.) != state('0', 1001.)


def test_vibstatesdata_appends_ground_state(states):
    assert 'zero' in states.allstates_map
    assert states.get_state_by_label('zero').energy == 0.
    assert states.get_state_by_label('0,1').energy == E01


def test_vibstatesdata_unknown_label_raises(states):
    with pytest.raises(ValueError):
        states.get_state_by_label('7')


def test_vib_states_joins_quanta_labels():
    states = _vib_states({('0',): E0, ('0', '1'): E01})

    assert [s.state_label for s in states] == ['0', '0,1']
    assert [s.energy for s in states] == [E0, E01]


## VibDiff ------------------------------------------------------------------

def test_from_quanta_maps_symbols_sorts_and_names_ground(states):
    s01, zero = states.get_state_by_label('0,1'), states.get_state_by_label('zero')

    # a=1, b=0 -> '0,1'; no quanta -> 'zero'
    assert VibDiff.from_quanta(('a', 'b'), (), {'a': 1, 'b': 0}, states) == VibDiff(s01, zero)
    assert VibDiff.from_quanta((), ('b', 'a'), {'a': 1, 'b': 0}, states) == VibDiff(zero, s01)


def test_vibdiff_order_sets_equality_and_energy_sign():
    zero, s0, s1 = state('zero', 0.), state('0', E0), state('1', E1)

    assert VibDiff(s0, zero) != VibDiff(zero, s0)
    assert VibDiff(s0, zero).energy_difference() == s0.energy
    assert VibDiff(zero, s0).energy_difference() == -s0.energy

    assert VibDiff(s1, s0) != VibDiff(s0, s1)


def test_vibdiff_energy_difference_cm1_and_au():
    vd = VibDiff(state('0,1', E01), state('0', E0))

    assert vd.energy_difference() == pytest.approx(E01 - E0)
    assert vd.energy_difference(au=True) == pytest.approx(convNu2Ene(E01 - E0))


def test_vibdiff_from_symbolic_and_from_quanta_agree(states):
    index_dict = {'a': 0, 'b': 1}

    symb = VibDiff.from_symbolic(vibdiff(sl='ab', sr='a'), index_dict, states)
    quanta = VibDiff.from_quanta(('a', 'b'), ('a',), index_dict, states)

    assert symb == quanta
    assert symb.energy_difference() == pytest.approx(E01 - E0)

def test_state_labels_agree_for_mode_10():
    assert _vib_states({('10', '2'): 1.})[0].state_label == make_state_label([10, 2]) == '2,10'


def test_resmotif_from_conditions_quanta_resolve_like_the_symbolic_diff(states):
    """A motif keeps only quanta labels; resolving them with VibDiff.from_quanta must land on the
    same states as resolving the original derive-side VibDiffTerm with from_symbolic."""
    rc1 = ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('a',), pert_freqs=('A', 'B'))
    rc2 = ResonanceCondition.make_from_tuples(left_state=('b',), right_state=('a',), pert_freqs=('-A',))
    index_dict = {'a': 1, 'b': 0}

    motif = ResonanceMotif.from_conditions([rc1, rc2])

    assert motif.conditions == (ResCondKey(diff=(('a', 'b'), ('a',)), pf=('A', 'B')),
                                ResCondKey(diff=(('b',), ('a',)), pf=('-A',)))
    for key, rc in zip(motif, (rc1, rc2)):
        assert VibDiff.from_quanta(*key.diff, index_dict, states) == VibDiff.from_symbolic(rc.diff, index_dict, states)

    # a=1, b=0:  E_ab - E_a = E01 - E1 ,  E_b - E_a = E0 - E1
    assert [VibDiff.from_quanta(*k.diff, index_dict, states).energy_difference() for k in motif] \
        == pytest.approx([E01 - E1, E0 - E1])


## Ground state ('zero') -----------------------------------------------------
# The ground state has no quanta. Its label is 'zero' (make_state_label of no modes); VibStatesData
# always holds it with energy 0 and refuses another energy. Mode 0 is something else:
# label '0', one quantum in mode 0.

# -- the label

@pytest.mark.parametrize('modes', [(), [], iter(())])
def test_make_state_label_of_no_modes_is_zero(modes):
    assert make_state_label(modes) == 'zero'


def test_mode_0_is_not_the_ground_state(states):
    assert make_state_label([0]) == '0'
    assert states.get_state_by_label('0').energy == E0
    assert states.get_state_by_label('zero').energy == 0.


# -- VibStatesData

def test_vibstatesdata_raises_for_a_ground_state_away_from_0():
    """Energies are measured from the ground state, so a 'zero' state must be at 0 cm-1."""
    with pytest.raises(ValueError, match=r"the states \(regime 'unknown'\): the ground state "
                                         r"\('zero'\) at 5\.0 cm-1; it must be at 0"):
        VibStatesData(allstates=(state('zero', 5.), state('0', E0)), regime='unknown')


@pytest.mark.parametrize('energy', [-5., 0.], ids=['below the ground state', 'at the ground state'])
def test_vibstatesdata_warns_for_a_state_at_or_below_the_ground_state(energy):
    """a warning, no error: the state set is still built"""
    with pytest.warns(UserWarning, match=rf"the states \(regime 'unknown'\): 1 of 2 state "
                                         rf"energies are at or below 0 cm-1, the ground state; lowest: 1 {energy:.1f}"):
        data = VibStatesData(allstates=(state('0', E0), state('1', energy)), regime='unknown')
    assert data.get_state_by_label('1').energy == energy


# -- the data dict: the ground state key ()

def test_from_data_dict_takes_the_ground_state_key_at_0():
    """The data's () entry becomes the one 'zero' state, at 0; no warning for it or for the other states."""
    data = {**TOY_DATA, 'harmonic_states': {(): 0., **TOY_DATA['harmonic_states']}}

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        molsys = MolSystemData.from_data_dict(data, [])

    harmonic = molsys.states['harmonic']
    assert [s.state_label for s in harmonic.allstates].count('zero') == 1
    assert harmonic.get_state_by_label('zero').energy == 0.


# -- VibDiff

@pytest.mark.parametrize('left, right, expected', [
    ('a', '', E0),        # one quantum -> ground:  E_a - 0
    ('', 'a', -E0),       # ground -> one quantum:  0 - E_a
    ('ab', '', E01),      # combination -> ground:  E_ab - 0
    ('', '', 0.),         # ground -> ground
])
def test_vibdiff_from_symbolic_and_from_quanta_agree_on_the_ground_state(left, right, expected, states):
    """from_symbolic builds its own ground state, from_quanta looks it up in VibStatesData."""
    index_dict = {'a': 0, 'b': 1}

    symb = VibDiff.from_symbolic(vibdiff(sl=left, sr=right), index_dict, states)
    quanta = VibDiff.from_quanta(tuple(left), tuple(right), index_dict, states)

    assert symb == quanta
    assert symb.energy_difference() == pytest.approx(expected)
    assert symb.energy_difference(au=True) == pytest.approx(convNu2Ene(expected))


## MolPropsCollection / MolSystemData ---------------------------------------

@pytest.fixture
def props() -> MolPropsCollection:
    return MolPropsCollection([
        MolecularProperty(trivial_name='cff', extra_data={'ops': ['g', 'g', 'g']}),
        MolecularProperty(trivial_name='polgrad', extra_data={'ops': ['e', 'e', 'g']}),
    ])


def test_molpropscollection_lookup(props):
    assert props.names() == ['cff', 'polgrad']
    assert 'cff' in props and 'hess' not in props
    assert props['polgrad'] is props.get('polgrad')
    assert len(props) == 2
    assert props.get('hess') is None


def test_molpropscollection_contains_a_property_only_as_that_object():
    """== would compare the values: with numpy arrays there it raises ValueError"""
    cff = MolecularProperty(trivial_name='cff', vals=np.ones((2, 2, 2)))
    coll = MolPropsCollection([cff])

    assert cff in coll
    assert MolecularProperty(trivial_name='cff', vals=np.ones((2, 2, 2))) not in coll


def test_molpropscollection_fill_from(props):
    """each property takes its value from the dict, or None; other keys are left out"""
    props.fill_from({'cff': np.zeros(1), 'unrelated': 1})

    assert props['cff'].vals is not None
    assert props['polgrad'].vals is None
    assert props.names() == ['cff', 'polgrad']

# A data dict that fits: 2 modes, their harmonic frequencies, both state lists.
# The data holds every state set it has, by states choice; the evaluation picks one (states_choice).
# Toy data: harmonic fundamentals at the frequencies, 0,1 at their sum; anharmonic 0 -> E0, 1 -> E1, 0,1 -> E01, lower.
FREQS = {'nc_sqrt_eigval': {0: E0_eigval, 1: E1_eigval}}
STATE_LISTS = {'harmonic_states': {('0',): E0_eigval, ('1',): E1_eigval, ('0', '1'): E0_eigval + E1_eigval},
               'anharmonic_states': {('0',): E0, ('1',): E1, ('0', '1'): E01}}
TOY_DATA = {**FREQS, **STATE_LISTS}
HARMONIC_ONLY = {**FREQS, 'harmonic_states': STATE_LISTS['harmonic_states']}


def test_from_data_dict_fills_the_named_properties_and_keeps_the_origin():
    """polgrad: named, but not in the data dict, so no values"""
    origin = DataOriginInfo(source_type='gaussian', base_file_loc='toy.out')
    molsys = MolSystemData.from_data_dict({**TOY_DATA, 'cff': np.ones((2, 2, 2))}, ['cff', 'polgrad'],
                                          data_origin=origin)

    assert molsys.mol_props['cff'].vals.shape == (2, 2, 2)
    assert molsys.mol_props['polgrad'].vals is None
    assert molsys.data_origin is origin


def test_from_data_dict_keeps_every_state_set_by_choice():
    molsys = MolSystemData.from_data_dict(TOY_DATA, [])

    assert set(molsys.states) == {'harmonic', 'anharmonic'}
    assert molsys.states['harmonic'].get_state_by_label('0,1').energy == E0_eigval + E1_eigval
    assert molsys.states['anharmonic'].get_state_by_label('0,1').energy == E01


@pytest.mark.parametrize('kept', ['harmonic', 'anharmonic'])
def test_from_data_dict_builds_only_the_state_sets_the_data_has(kept):
    molsys = MolSystemData.from_data_dict({**FREQS, f'{kept}_states': STATE_LISTS[f'{kept}_states']}, [])

    assert set(molsys.states) == {kept}


@pytest.mark.parametrize('choice', ['GVPT2', 'anharmonic'], ids=['a regime name', 'a set the data lacks'])
def test_state_set_returns_the_set_or_names_the_sets_the_data_has(choice):
    molsys = MolSystemData.from_data_dict(HARMONIC_ONLY, [])

    assert molsys.state_set('harmonic') is molsys.states['harmonic']
    with pytest.raises(ValueError, match=r"the data has the state sets \['harmonic'\]"):
        molsys.state_set(choice)


def test_require_values_names_what_is_missing_and_what_the_collection_has():
    """polgrad: in the collection, vals None; polhess: not in it. Both count as missing, each named once."""
    props = MolPropsCollection([MolecularProperty(trivial_name='cff', vals=np.ones((2, 2, 2))),
                                MolecularProperty(trivial_name='polgrad')])

    props.require_values(['cff'])
    with pytest.raises(ValueError, match=r"no values for \['polgrad', 'polhess'\]: the data has values for \['cff'\]"):
        props.require_values(['cff', 'polgrad', 'polhess', 'polgrad'])


## MolSystemData.from_output_file: reads real files -------------------------------------------

DATA_DIR = files('wilson_suite').joinpath('data_for_tests')
FORMALDEHYDE = DataOriginInfo(source_type='gaussian', base_file_loc=str(DATA_DIR / 'g16_formaldehyde_B3LYPcc_pVQZ.out'))
ACETIC_ACID_DIMER = DataOriginInfo(source_type='gaussian',
                                   base_file_loc=str(DATA_DIR / 'g16_acetic_acid_dimer_B3LYPcc_pVQZ.out'))


def test_from_output_file_fills_properties_frequencies_and_state_sets():
    molsys = MolSystemData.from_output_file(FORMALDEHYDE, ['cff'])

    assert molsys.mol_props['cff'].vals.shape == (6, 6, 6)
    assert sorted(molsys.eigenvals) == [0, 1, 2, 3, 4, 5]
    assert set(molsys.states) == {'harmonic', 'anharmonic'}
    assert molsys.data_origin is FORMALDEHYDE


def test_from_output_file_raises_for_an_unknown_wilson_regime_before_reading():
    """
    'anharmonic' is a states choice, not a regime.
    The file does not exist: a read would raise another error.
    """
    origin = DataOriginInfo(source_type='gaussian', base_file_loc='no_such_file.out')

    with pytest.raises(ValueError, match=r"wilson_regime 'anharmonic': expected None \(the file's states\) "
                                         r"or one of VPT2, DVPT2, GVPT2"):
        MolSystemData.from_output_file(origin, [], wilson_regime='anharmonic')


def test_from_output_file_gvpt2_gives_the_gaussian_files_own_gvpt2_states():
    """
    The Gaussian formaldehyde file holds GVPT2 states. wilson's GVPT2, from the same file's force constants,
    gives the same 83 states, within 0.01 cm-1.
    """
    from_file = MolSystemData.from_output_file(FORMALDEHYDE, [])
    from_wilson = MolSystemData.from_output_file(FORMALDEHYDE, [], wilson_regime='GVPT2')

    file_states, wilson_states = from_file.states['anharmonic'], from_wilson.states['anharmonic']
    assert (file_states.regime, wilson_states.regime) == ('unknown', 'GVPT2')
    labels = sorted(file_states.allstates_map)
    assert len(labels) == 84                                    # 83 states and the ground state
    assert sorted(wilson_states.allstates_map) == labels
    np.testing.assert_allclose([wilson_states.get_state_by_label(label).energy for label in labels],
                               [file_states.get_state_by_label(label).energy for label in labels], rtol=0, atol=0.01)


def test_with_wilson_anharmonic_states_equals_from_output_file_with_a_regime():
    """the two-step way gives the same states as the one call, and the loaded data keeps the file's states"""
    loaded = MolSystemData.from_output_file(FORMALDEHYDE, list(VPT2_INPUT_KEYS))

    in_two_steps = loaded.with_wilson_anharmonic_states('GVPT2')
    in_one_call = MolSystemData.from_output_file(FORMALDEHYDE, [], wilson_regime='GVPT2')

    assert in_two_steps.states['anharmonic'].allstates == in_one_call.states['anharmonic'].allstates
    assert in_two_steps.states['anharmonic'].regime == 'GVPT2'
    assert loaded.states['anharmonic'].regime == 'unknown'


## MolSystemData checks its fields --------------------------------------------------------------
# TOY_DATA fits: 2 modes. Each test breaks one rule.

def test_prop_layout_gives_the_index_order_of_every_property():
    assert len(PROP_LAYOUT) == 49 + 4             # derivatives by modes and field, cff_rc, qff_rc, B, coriolis
    assert PROP_LAYOUT['polhess'] == 'mmxx'
    assert PROP_LAYOUT['cff'] == 'mmm'
    assert PROP_LAYOUT['dipgrad'] == 'mx'
    assert PROP_LAYOUT['coriolis'] == 'xmm'


def test_molsystemdata_takes_data_that_fits():
    """coriolis changes sign when its two modes swap: no symmetry check for it"""
    coriolis = np.zeros((3, 2, 2))
    coriolis[2, 0, 1], coriolis[2, 1, 0] = 0.4, -0.4
    data = {**TOY_DATA, 'polhess': np.ones((2, 2, 3, 3)), 'cff': np.ones((2, 2, 2)), 'dipgrad': np.ones((2, 3)),
            'B': np.array([5.8, 1.3, 1.1]), 'coriolis': coriolis}

    MolSystemData.from_data_dict(data, ['polhess', 'cff', 'dipgrad', 'B', 'coriolis'])


@pytest.mark.parametrize('freqs', [{1: E0_eigval, 2: E1_eigval}, {}, [E0_eigval, E1_eigval],
                                   {0: E0_eigval, '1': E1_eigval}],
                         ids=['modes from 1', 'no modes', 'a list', 'a str key'])
def test_molsystemdata_eigenvals_are_a_dict_with_the_modes_from_0(freqs):
    with pytest.raises(ValueError, match=r"eigenvals \(data key 'nc_sqrt_eigval'\): needs a dict with the modes "
                                         r"0 \.\. n-1"):
        MolSystemData.from_data_dict({**TOY_DATA, 'nc_sqrt_eigval': freqs}, [])


@pytest.mark.parametrize('name, shape', [('dipgrad', (3, 2)), ('polhess', (2, 3, 3)), ('cff', (3, 3, 3)),
                                         ('coriolis', (2, 2, 3))],
                         ids=['x, y, z first', 'a mode index missing', '3 modes instead of 2',
                              'coriolis with x, y, z last'])
def test_molsystemdata_property_shape_follows_its_layout(name, shape):
    with pytest.raises(ValueError, match=re.escape(f'{name}: shape {shape}, expected')):
        MolSystemData.from_data_dict({**TOY_DATA, name: np.ones(shape)}, [name])


def test_molsystemdata_property_is_symmetric_in_its_mode_indices():
    """only cff[0, 1, 1] is filled, not cff[1, 0, 1] and cff[1, 1, 0]: one index order of a full tensor"""
    cff = np.zeros((2, 2, 2))
    cff[0, 1, 1] = 5e-6

    with pytest.raises(ValueError, match=r'cff: not symmetric in its mode indices, cff\[0, 1, 1\] = 5e-06, '
                                         r'cff\[1, 0, 1\] = 0'):
        MolSystemData.from_data_dict({**TOY_DATA, 'cff': cff}, ['cff'])


def test_molsystemdata_symmetry_allows_differences_below_the_zero_tol():
    """1e-20: rounding noise, below the evaluator's zero_tol 1e-18"""
    cff = np.zeros((2, 2, 2))
    cff[0, 1, 1] = 1e-20

    MolSystemData.from_data_dict({**TOY_DATA, 'cff': cff}, ['cff'])


def test_molsystemdata_skips_a_property_without_values_but_refuses_an_unknown_name():
    """a missing value is require_values' job; a name without a layout cannot be checked"""
    MolSystemData.from_data_dict(TOY_DATA, ['polgrad'])
    with pytest.raises(ValueError, match=r'polgard: no layout in PROP_LAYOUT'):
        MolSystemData.from_data_dict(TOY_DATA, ['polgard'])


def test_from_output_file_warning_names_the_source():
    """the acetic acid dimer file has anharmonic states below 0 cm-1, the lowest at -596.7 cm-1"""
    with pytest.warns(UserWarning, match=r"the data from .*acetic_acid_dimer.*: the states \(regime 'unknown'\): "
                                         r"\d+ of \d+ state energies are at or below 0 cm-1"
                      ) as record:
        MolSystemData.from_output_file(ACETIC_ACID_DIMER, [])
    assert len(record) == 1
    assert record[0].filename == __file__           # points at the line that called from_output_file


def test_from_output_file_warns_for_wilsons_states_at_or_below_the_ground_state():
    """wilson's GVPT2 puts some acetic acid dimer states at or below 0 cm-1: the same check as for the file's"""
    with pytest.warns(UserWarning, match=r"the states \(regime 'GVPT2'\): \d+ of \d+ state "
                                         r"energies are at or below 0 cm-1"):
        MolSystemData.from_output_file(ACETIC_ACID_DIMER, [], wilson_regime='GVPT2')


def test_molsystemdata_names_every_problem_at_once():
    cff = np.zeros((2, 2, 2))
    cff[0, 1, 1] = 5e-6

    with pytest.raises(ValueError, match=r'(?s)dipgrad: shape.*cff: not symmetric'):
        MolSystemData.from_data_dict({**TOY_DATA, 'dipgrad': np.ones((3, 2)), 'cff': cff}, ['dipgrad', 'cff'])


@pytest.mark.parametrize('change, message', [
    ({'nc_sqrt_eigval': {0: -E0_eigval, 1: E1_eigval}}, r'eigenvals: \[-1100\.0\] cm-1, at or below 0'),
    ({'anharmonic_states': {('0',): E0, ('2',): E1}}, r"states\['anharmonic'\]: labels with modes that do not "
                                                      r"exist, e\.g\. \['2'\]"),
    ({'anharmonic_states': {('0', '1'): E01, ('1', '0'): E01}}, r"labels that name more than one state, "
                                                                r"e\.g\. \['0,1'\]"),
    ({'harmonic_states': {('0',): E0, ('1',): E1_eigval}}, r"states\['harmonic'\]: fundamentals that differ from "
                                                           r"eigenvals by more than 0\.01 cm-1, e\.g\. mode 0"),
], ids=['an imaginary frequency',
        'a mode that does not exist',
        'two keys with one label',
        'a fundamental off its frequency'])
def test_molsystemdata_states_fit_the_frequencies(change, message):
    with pytest.raises(ValueError, match=message):
        MolSystemData.from_data_dict({**TOY_DATA, **change}, [])


def test_from_output_file_extra_keys_never_collide_with_property_names():
    """
    Guards from_output_file against a name clash between a property and the extra data that every evaluation needs.

    from_output_file asks the reader for everything in one dict, with the names as keys. For example,
    MolSystemData.from_output_file(origin, ['cff']) asks for
        {'cff': origin, 'nc_sqrt_eigval': origin, 'harmonic_states': origin, 'anharmonic_states': origin}
    The last three keys are FREQUENCY_AND_STATE_KEYS: the harmonic frequencies and the two state lists.
    The reader answers with one data dict, with the same keys.

    Suppose a property had the same name as one of those keys, e.g. a property called 'nc_sqrt_eigval'.
    The property and the frequencies would then share one key, so both would get the same value from one read.
    One of the two would hold wrong numbers, and nothing would raise an error.

    So the test checks that no name in PROP_LAYOUT, which holds every property name the data can have, is one of
    FREQUENCY_AND_STATE_KEYS. No name clashes today; the test fails if a new name ever does.
    """
    assert set(FREQUENCY_AND_STATE_KEYS).isdisjoint(PROP_LAYOUT)
