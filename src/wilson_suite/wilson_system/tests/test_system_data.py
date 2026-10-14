"""
system_data.py — vibrational states, energy differences, molecular data and the data request (MolSystemData.load).
Everything here is built by hand from tiny arrays; no VibPerturbedTerm, no data files.
"""

import numpy as np
import pytest

from wilson_suite.wilson_derive.abstractions import ResonanceCondition
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResCondKey,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    E0,
    E01,
    E1,
    state,
    toy_states,
    vibdiff,
)
from wilson_suite.wilson_system.system_data import (
    EVALUATION_DATA_KEYS,
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibDiff,
    VibState,
    VibStatesData,
    _make_hq_states_from_datadict,
    make_state_label,
)
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


@pytest.fixture
def states() -> VibStatesData:
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


def test_vibdiff_normalized_puts_ground_left_then_label_order():
    zero, s0, s1 = state('zero', 0.), state('0', E0), state('1', E1)

    assert VibDiff(s0, zero).normalized() == VibDiff(zero, s0)
    assert VibDiff(s0, zero) != VibDiff(zero, s0)
    assert VibDiff(s0, zero).energy_difference() == s0.energy
    assert VibDiff(zero, s0).energy_difference() == -s0.energy

    assert VibDiff(s1, s0).normalized() == VibDiff(s0, s1)
    assert VibDiff(s1, s0) != VibDiff(s0, s1)
    assert VibDiff(s0, s1).normalized() == VibDiff(s0, s1)


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


def test_from_quanta_works_with_the_zero_entry_of_a_parameter_set(states):
    """get_RHS_motif passes ParameterSet.to_dict(), which carries 'zero': 'zero' next to the mode labels."""
    index_dict = ParameterSet({'a': 0}).to_dict()

    assert VibDiff.from_quanta(('a',), (), index_dict, states).energy_difference() == E0


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

def test_from_datadict_does_not_share_mol_props():
    template = MolPropsCollection([MolecularProperty(trivial_name='cff')])
    m1 = MolSystemData.from_datadict(template, {'cff': np.ones(2)})
    m2 = MolSystemData.from_datadict(template, {})
    assert m1.data_filled and not m2.data_filled
    assert template['cff'].vals is None


# The data holds every state set it has, by states choice; the evaluation picks one (states_choice).
# Toy data: harmonic 0 -> E0, 0,1 -> E01; anharmonic: the same states, 10 cm-1 lower.
STATE_LISTS = {'harmonic_states': {('0',): E0, ('0', '1'): E01},
               'anharmonic_states': {('0',): E0 - 10., ('0', '1'): E01 - 10.}}


def test_from_datadict_keeps_every_state_set_by_choice():
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), STATE_LISTS)

    assert set(molsys.states) == {'harmonic', 'anharmonic'}
    assert molsys.states['harmonic'].get_state_by_label('0,1').energy == E01
    assert molsys.states['anharmonic'].get_state_by_label('0,1').energy == E01 - 10.


def test_from_datadict_has_only_the_state_sets_of_the_data():
    harmonic_only = MolSystemData.from_datadict(MolPropsCollection([]),
                                                {'harmonic_states': STATE_LISTS['harmonic_states']})
    none = MolSystemData.from_datadict(MolPropsCollection([]), {})

    assert set(harmonic_only.states) == {'harmonic'}
    assert none.states == {}


def test_from_datadict_both_sets_get_the_harmonic_fundamentals_as_labels():
    """harmonic_osc_states_labels: the modes with a fundamental among the harmonic states, for both sets"""
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), STATE_LISTS)

    assert molsys.states['harmonic'].harmonic_osc_states_labels == (0,)
    assert molsys.states['anharmonic'].harmonic_osc_states_labels == (0,)


@pytest.mark.parametrize('choice', ['GVPT2', 'anharmonic'], ids=['a regime name', 'a set the data lacks'])
def test_check_states_choice_names_the_sets_the_data_has(choice):
    molsys = MolSystemData.from_datadict(MolPropsCollection([]), {'harmonic_states': STATE_LISTS['harmonic_states']})

    molsys.check_states_choice('harmonic')
    with pytest.raises(ValueError, match=r"the data has the state sets \['harmonic'\]"):
        molsys.check_states_choice(choice)


def test_check_props_names_what_is_missing_and_what_the_data_has():
    """polgrad: in mol_props, vals None; polhess: not in mol_props. Both count as missing, each named once."""
    molsys = MolSystemData.from_datadict(MolPropsCollection([MolecularProperty(trivial_name='cff'),
                                                             MolecularProperty(trivial_name='polgrad')]),
                                         {'cff': np.ones(2)})

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
    data = {'cff': np.ones((2, 2, 2)), 'dipgrad': np.ones((2, 3)), 'nc_sqrt_eigval': {0: 1100., 1: 1580.},
            **STATE_LISTS}

    MolSystemData.load(ORIGIN, ['cff', 'dipgrad', 'cff'], obtainer=fake_obtainer(data, asked))

    (request,) = asked
    assert list(request) == ['cff', 'dipgrad', 'nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states']
    assert all(origin is ORIGIN for origin in request.values())


def test_load_fills_properties_frequencies_and_state_sets():
    cff = np.arange(8.).reshape(2, 2, 2)
    data = {'cff': cff, 'nc_sqrt_eigval': {0: 1100., 1: 1580.}, **STATE_LISTS}

    molsys = MolSystemData.load(ORIGIN, ['cff'], obtainer=fake_obtainer(data, []))

    assert molsys.mol_props['cff'].vals is cff
    assert molsys.eigenvals == {0: 1100., 1: 1580.}
    assert set(molsys.states) == {'harmonic', 'anharmonic'}


def test_load_raises_for_a_property_the_source_lacks():
    data = {'cff': np.ones((2, 2, 2)), 'nc_sqrt_eigval': {0: 1100., 1: 1580.}}

    with pytest.raises(ValueError, match=r"the source did not return .*toy\.out.*no values for \['polgrad'\]"):
        MolSystemData.load(ORIGIN, ['cff', 'polgrad'], obtainer=fake_obtainer(data, []))


def test_load_extra_keys_never_collide_with_property_names():
    """
    load puts the property names and EVALUATION_DATA_KEYS into one request, and from_datadict reads one data dict:
    a shared name would make a property and, e.g., the frequencies one entry.
    """
    names = {prop_trivname(ord_geo=g, ord_el=e) for g in range(7) for e in range(7)}
    names |= {prop_trivname(ord_rot=1), prop_trivname(ord_rot=1, ord_geo=2)}   # 'B', 'coriolis'

    assert set(EVALUATION_DATA_KEYS).isdisjoint(names)
