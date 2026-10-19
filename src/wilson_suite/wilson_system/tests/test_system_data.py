"""
system_data.py — vibrational states, energy differences, molecular data and the data request (MolSystemData.load).
Everything here is built by hand from tiny arrays; no VibPerturbedTerm, no data files.
"""

import re
import warnings

import numpy as np
import pytest

from wilson_suite.wilson_derive.abstractions import ResonanceCondition
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ResCondKey,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
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
    EVALUATION_DATA_KEYS,
    PROP_INDEX_COUNTS,
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibDiff,
    VibState,
    VibStatesData,
    _make_hq_states_from_datadict,
    check_data_dict,
    make_state_label,
)
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
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
    assert state('0', 1.) < state('1', 0.)


def test_vibstatesdata_appends_ground_state(states):
    assert 'zero' in states.allstates_map
    assert states.get_state_by_label('zero').energy == 0.
    assert states.get_state_by_label('0,1').energy == E01


def test_vibstatesdata_unknown_label_raises(states):
    with pytest.raises(ValueError):
        states.get_state_by_label('7')


def test_vibstatesdata_harmonic_osc_states_filters_by_label_choice():
    data = VibStatesData(allstates=(state('1', E1), state('0', E0), state('0,1', E01)),
                         harmonic_osc_states_labels=(0,1))

    assert data.get_harmonic_osc_states() == {0: E0, 1: E1}


def test_make_hq_states_from_datadict_joins_quanta_labels():
    harm, anharm = _make_hq_states_from_datadict({'harmonic_states': {('0',): E0, ('0', '1'): E01}})

    assert [s.state_label for s in harm] == ['0', '0,1']
    assert [s.energy for s in harm] == [E0, E01]
    assert anharm == ()


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
    harm, _ = _make_hq_states_from_datadict({'harmonic_states': {('10', '2'): 1.}})
    assert harm[0].state_label == make_state_label([10, 2]) == '2,10'


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
# always holds it with energy 0; VibDiff counts it as energy 0. Mode 0 is something else:
# label '0', the first excited state of mode 0.

# -- the label

@pytest.mark.parametrize('modes', [(), [], iter(())])
def test_make_state_label_of_no_modes_is_zero(modes):
    assert make_state_label(modes) == 'zero'


def test_mode_0_is_not_the_ground_state(states):
    assert make_state_label([0]) == '0'
    assert states.get_state_by_label('0').energy == E0
    assert states.get_state_by_label('zero').energy == 0.


# -- VibStatesData

def test_vibstatesdata_ground_state_energy_is_zero_even_if_the_data_says_otherwise():
    """Energies are measured from the ground state: a 'zero' entry in the data is overridden with energy 0."""
    data = VibStatesData(allstates=(state('zero', 5.), state('0', E0)))

    assert data.get_state_by_label('zero').energy == 0.


# -- VibDiff

def test_vibdiff_counts_a_zero_labelled_state_as_energy_zero():
    """is_zero_state goes by the label, so the energy field of a 'zero' state is ignored."""
    odd_zero = state('zero', 5.)

    assert VibDiff(state('0', E0), odd_zero).energy_difference() == E0
    assert VibDiff(odd_zero, state('0', E0)).energy_difference() == -E0


@pytest.mark.parametrize('left, right, expected', [
    ('a', '', E0),        # excited -> ground:      E_a - 0
    ('', 'a', -E0),       # ground -> excited:      0 - E_a
    ('ab', '', E01),      # combination -> ground:  E_ab - 0
    ('', '', 0.),         # ground -> ground
])
def test_vibdiff_from_symbolic_and_from_quanta_agree_on_the_ground_state(left, right, expected, states):
    """from_symbolic builds its own ground state, from_quanta looks it up in VibStatesData (todo A11)."""
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


def test_molpropscollection_fill_from_and_is_filled(props):
    assert not props.is_filled
    assert props.without_values().names() == ['cff', 'polgrad']

    props.fill_from({'cff': np.zeros(1), 'unrelated': 1})

    assert props['cff'].vals is not None
    assert props.without_values().names() == ['polgrad']
    assert not props.is_filled

# A data dict that fits (check_data_dict): 2 modes, their harmonic frequencies, every fundamental in the states.
# The data holds every state set it has, by states choice; the evaluation picks one (states_choice).
# Toy data: harmonic 0 -> E0, 1 -> E1, 0,1 -> E01; anharmonic: the same states, 10 cm-1 lower.
FREQS = {'nc_sqrt_eigval': {0: E0_eigval, 1: E1_eigval}}
STATE_LISTS = {'harmonic_states': {('0',): E0, ('1',): E1, ('0', '1'): E01},
               'anharmonic_states': {('0',): E0 - 10., ('1',): E1 - 10., ('0', '1'): E01 - 10.}}
TOY_DATA = {**FREQS, **STATE_LISTS}
HARMONIC_ONLY = {**FREQS, 'harmonic_states': STATE_LISTS['harmonic_states']}


def test_from_datadict_does_not_share_mol_props():
    template = MolPropsCollection([MolecularProperty(trivial_name='cff')])
    m1 = MolSystemData.from_datadict(template, {**TOY_DATA, 'cff': np.ones((2, 2, 2))})
    m2 = MolSystemData.from_datadict(template, TOY_DATA)
    assert m1.data_filled and not m2.data_filled
    assert template['cff'].vals is None


def test_from_datadict_keeps_every_state_set_by_choice():
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), TOY_DATA)

    assert set(molsys.states) == {'harmonic', 'anharmonic'}
    assert molsys.states['harmonic'].get_state_by_label('0,1').energy == E01
    assert molsys.states['anharmonic'].get_state_by_label('0,1').energy == E01 - 10.


def test_from_datadict_without_anharmonic_states_has_only_the_harmonic_set():
    assert set(MolSystemData.from_datadict(MolPropsCollection([]), HARMONIC_ONLY).states) == {'harmonic'}


def test_from_datadict_both_sets_get_the_harmonic_fundamentals_as_labels():
    """harmonic_osc_states_labels: the modes with a fundamental among the harmonic states, for both sets"""
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), TOY_DATA)

    assert molsys.states['harmonic'].harmonic_osc_states_labels == (0, 1)
    assert molsys.states['anharmonic'].harmonic_osc_states_labels == (0, 1)


@pytest.mark.parametrize('choice', ['GVPT2', 'anharmonic'], ids=['a regime name', 'a set the data lacks'])
def test_check_states_choice_names_the_sets_the_data_has(choice):
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), HARMONIC_ONLY)

    molsys.check_states_choice('harmonic')
    with pytest.raises(ValueError, match=r"the data has the state sets \['harmonic'\]"):
        molsys.check_states_choice(choice)


def test_check_props_names_what_is_missing_and_what_the_data_has():
    """polgrad: in mol_props, vals None; polhess: not in mol_props. Both count as missing, each named once."""
    molsys = MolSystemData.from_datadict(MolPropsCollection([MolecularProperty(trivial_name='cff'),
                                                             MolecularProperty(trivial_name='polgrad')]),
                                         {**TOY_DATA, 'cff': np.ones((2, 2, 2))})

    molsys.check_props(['cff'])
    with pytest.raises(ValueError, match=r"no values for \['polgrad', 'polhess'\]: the data has values for \['cff'\]"):
        molsys.check_props(['cff', 'polgrad', 'polhess', 'polgrad'])


## MolSystemData.load: one obtainer call, only what the evaluation reads ------------------------

ORIGIN = DataOriginInfo(source_type='gaussian', base_file_loc='toy.out')


def fake_obtainer(data: dict, asked: list):
    """stands in for wilson_data_obtainer: records the request, returns `data` (what the source has)"""
    def obtainer(request: dict) -> dict:
        asked.append(request)
        return data
    return obtainer


def test_load_requests_the_properties_frequencies_and_both_state_lists_only():
    asked = []
    data = {'cff': np.ones((2, 2, 2)), 'dipgrad': np.ones((2, 3)), **TOY_DATA}

    MolSystemData.load(ORIGIN, ['cff', 'dipgrad', 'cff'], obtainer=fake_obtainer(data, asked))

    (request,) = asked
    assert list(request) == ['cff', 'dipgrad', 'nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states']
    assert all(origin is ORIGIN for origin in request.values())


def test_load_fills_properties_frequencies_and_state_sets():
    cff = np.full((2, 2, 2), 3.)
    data = {'cff': cff, **TOY_DATA}

    molsys = MolSystemData.load(ORIGIN, ['cff'], obtainer=fake_obtainer(data, []))

    assert molsys.mol_props['cff'].vals is cff
    assert molsys.eigenvals == {0: E0_eigval, 1: E1_eigval}
    assert set(molsys.states) == {'harmonic', 'anharmonic'}


def test_load_raises_for_a_property_the_source_lacks():
    data = {'cff': np.ones((2, 2, 2)), **TOY_DATA}

    with pytest.raises(ValueError, match=r"the source did not return .*toy\.out.*no values for \['polgrad'\]"):
        MolSystemData.load(ORIGIN, ['cff', 'polgrad'], obtainer=fake_obtainer(data, []))


def test_load_raises_for_data_that_does_not_fit_and_names_the_source():
    data = {'cff': np.ones((2, 2, 2)), **STATE_LISTS}             # no frequencies

    with pytest.raises(ValueError, match=r"(?s)the data from .*toy\.out.* does not fit: .*no 'nc_sqrt_eigval'"):
        MolSystemData.load(ORIGIN, ['cff'], obtainer=fake_obtainer(data, []))


## check_data_dict: the data dict fits what the evaluator reads (wilson_system/todo.txt D5) -----------------
# TOY_DATA fits: 2 modes. Each test breaks one rule; from_datadict runs the same check.

def test_prop_index_counts_every_order_has_its_own_name():
    assert len(PROP_INDEX_COUNTS) == 49
    assert PROP_INDEX_COUNTS['polhess'] == (2, 2)
    assert PROP_INDEX_COUNTS['cff'] == (3, 0)
    assert PROP_INDEX_COUNTS['dipgrad'] == (1, 1)


def test_check_data_dict_passes_data_that_fits():
    data = {**TOY_DATA, 'polhess': np.ones((2, 2, 3, 3)), 'cff': np.ones((2, 2, 2)), 'dipgrad': np.ones((2, 3))}

    check_data_dict(data, ['polhess', 'cff', 'dipgrad'])


@pytest.mark.parametrize('drop, message', [('nc_sqrt_eigval', r"no 'nc_sqrt_eigval'"),
                                           ('harmonic_states', r"no 'harmonic_states'")])
def test_check_data_dict_needs_frequencies_and_harmonic_states(drop, message):
    data = {k: v for k, v in TOY_DATA.items() if k != drop}

    with pytest.raises(ValueError, match=message):
        check_data_dict(data, [])
    with pytest.raises(ValueError, match=message):
        MolSystemData.from_datadict(MolPropsCollection([]), data)


@pytest.mark.parametrize('freqs', [{1: E0_eigval, 2: E1_eigval}, {}, [E0_eigval, E1_eigval]],
                         ids=['modes from 1', 'no modes', 'a list'])
def test_check_data_dict_frequencies_are_a_dict_with_the_modes_from_0(freqs):
    with pytest.raises(ValueError, match=r"'nc_sqrt_eigval': needs a dict with the modes 0 \.\. n-1"):
        check_data_dict({**TOY_DATA, 'nc_sqrt_eigval': freqs}, [])


def test_check_data_dict_needs_every_fundamental_in_the_harmonic_states():
    harmonic = {('0',): E0, ('0', '1'): E01}                                  # no ('1',)

    with pytest.raises(ValueError, match=r"'harmonic_states' lacks the fundamentals of modes \[1\]"):
        check_data_dict({**TOY_DATA, 'harmonic_states': harmonic}, [])


@pytest.mark.parametrize('name, shape', [('dipgrad', (3, 2)), ('polhess', (2, 3, 3)), ('cff', (3, 3, 3))],
                         ids=['x, y, z first', 'a mode index missing', '3 modes instead of 2'])
def test_check_data_dict_property_shape_is_modes_then_xyz(name, shape):
    with pytest.raises(ValueError, match=re.escape(f'{name}: shape {shape}, expected')):
        check_data_dict({**TOY_DATA, name: np.ones(shape)}, [name])


def test_check_data_dict_property_is_symmetric_in_its_mode_indices():
    """only cff[0, 1, 1] is filled, not cff[1, 0, 1] and cff[1, 1, 0]: one index order of a full tensor"""
    cff = np.zeros((2, 2, 2))
    cff[0, 1, 1] = 5e-6

    with pytest.raises(ValueError, match=r'cff: not symmetric in its mode indices, cff\[0, 1, 1\] = 5e-06, '
                                         r'cff\[1, 0, 1\] = 0'):
        check_data_dict({**TOY_DATA, 'cff': cff}, ['cff'])


def test_check_data_dict_symmetry_allows_differences_below_the_zero_tol():
    """1e-20: rounding noise, below the evaluator's zero_tol 1e-18"""
    cff = np.zeros((2, 2, 2))
    cff[0, 1, 1] = 1e-20

    check_data_dict({**TOY_DATA, 'cff': cff}, ['cff'])


def test_check_data_dict_does_not_check_properties_the_data_lacks_or_with_no_known_shape():
    """a missing property is check_props' job; 'coriolis' (rotational) has no entry in PROP_INDEX_COUNTS"""
    check_data_dict({**TOY_DATA, 'coriolis': np.ones(7)}, ['polgrad', 'coriolis'])


@pytest.mark.parametrize('key', ['harmonic_states', 'anharmonic_states'])
@pytest.mark.parametrize('energy', [-5., 0.], ids=['below the ground state', 'at the ground state'])
def test_check_data_dict_warns_for_a_state_at_or_below_the_ground_state(key, energy):
    """a warning, no error: from_datadict still builds the data"""
    data = {**TOY_DATA, key: {**TOY_DATA[key], ('1',): energy}}

    with pytest.warns(UserWarning, match=rf"{key}: 1 of 3 state energies are at or below 0 cm-1, the ground state; "
                                         rf"lowest: \('1',\) {energy:.1f}"):
        molsys = MolSystemData.from_datadict(MolPropsCollection([]), data)
    assert molsys.states[key.removesuffix('_states')].get_state_by_label('1').energy == energy


def test_load_warning_names_the_source():
    data = {**TOY_DATA, 'anharmonic_states': {**TOY_DATA['anharmonic_states'], ('1',): -5.}}

    with pytest.warns(UserWarning, match=r'the data from .*toy\.out.*: anharmonic_states: 1 of 3') as record:
        MolSystemData.load(ORIGIN, [], obtainer=fake_obtainer(data, []))
    assert len(record) == 1
    assert record[0].filename == __file__           # points at the line that called load


def test_check_data_dict_no_warning_for_states_above_the_ground_state():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check_data_dict(TOY_DATA, [])


def test_check_data_dict_names_every_problem_at_once():
    data = {**FREQS, 'harmonic_states': {('0',): E0}, 'dipgrad': np.ones((3, 2))}

    with pytest.raises(ValueError, match=r'(?s)lacks the fundamentals of modes \[1\].*dipgrad: shape'):
        check_data_dict(data, ['dipgrad'])


def test_load_extra_keys_never_collide_with_property_names():
    """
    Guards load against a name clash between a property and the extra data that every evaluation needs.

    load asks the reader for everything in one dict, with the names as keys. For example,
    MolSystemData.load(origin, ['cff']) asks for
        {'cff': origin, 'nc_sqrt_eigval': origin, 'harmonic_states': origin, 'anharmonic_states': origin}
    The last three keys are EVALUATION_DATA_KEYS: the harmonic frequencies and the two state lists.
    The reader answers with one data dict, with the same keys.

    Suppose a property had the same name as one of those keys, e.g. a property called 'nc_sqrt_eigval'.
    The property and the frequencies would then share one key, so both would get the same value from one read.
    One of the two would hold wrong numbers, and nothing would raise an error.

    So the test makes every property name there can be, and checks that none is one of EVALUATION_DATA_KEYS.
    The names: prop_trivname for 0-6 derivatives by normal modes and 0-6 by the electric field (e.g. 'dipgrad',
    'polhess', 'cff'), plus 'B' and 'coriolis'. No name clashes today; the test fails if a new name ever does.
    """
    names = {prop_trivname(ord_geo=g, ord_el=e) for g in range(7) for e in range(7)}
    names |= {prop_trivname(ord_rot=1), prop_trivname(ord_rot=1, ord_geo=2)}   # 'B', 'coriolis'

    assert set(EVALUATION_DATA_KEYS).isdisjoint(names)
