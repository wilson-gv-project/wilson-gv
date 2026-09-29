"""
evaluate.py — the numeric stage. Everything here is built by hand from tiny arrays;
no VibPerturbedTerm, no data files.
"""

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.evaluate as evaluate_mod
from wilson_suite.wilson_derive.abstractions import (
    HarmOscStateSymbolic,
    PolProp,
    QOperator,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    PropsCollection,
    ResCondKey,
    ResLocGeoObject,
    ResonanceCondition,
    ResonanceMotif,
)
from wilson_suite.wilson_system.system_data import *
from wilson_suite.wilson_system.system_data import (
    _make_hq_states_from_datadict,
    _make_vibdiff_key,
    _sys_info_request,
)
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


def polprop(ops: tuple[int, ...] = (), inds: str = '') -> PolProp:
    p = PolProp(ops=[QOperator(o=i) for i in ops], dord=len(inds))
    p.setInds(list(inds))
    return p


def vibdiff(sl: str = '', sr: str = '', pert: bool = False) -> VibDiffTerm:
    return VibDiffTerm(sl=HarmOscStateSymbolic(list(sl)), sr=HarmOscStateSymbolic(list(sr)), is_pert_wf_diff=pert)


def state(label: str, energy: float) -> VibState:
    return VibState(harm_quanta_coeffs={}, energy=energy, state_label=label)


# Two modes; energies in cm-1 chosen so every difference is distinct.
E0, E1, E01, E00 = 1000., 1500., 2600., 1950.
E0_eigval, E1_eigval = 1100., 1580.


@pytest.fixture
def states() -> VibStatesData:
    return VibStatesData(allstates=(state('0', E0), state('1', E1), state('0,1', E01), state('0,0', E00)),
                         harmonic_osc_states_labels=(0, 1))


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

def test_make_vibdiff_key_maps_symbols_sorts_and_names_ground():
    key = _make_vibdiff_key(vibdiff(sl='ab', sr=''), {'a': 1, 'b': 0})
    assert key == ('0,1', 'zero')

    key = _make_vibdiff_key(vibdiff(sl='', sr='abc'), {'a': 1, 'b': 0, 'c': 2})
    assert key == ('zero', '0,1,2')

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
    assert harm[0].state_label == _make_vibdiff_key(vibdiff(sl='ab'), {'a': 10, 'b': 2})[0] == '2,10'


## MolPropsCollection -------------------------------------------------------

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


## ResonanceMotif -----------------------------------------------------------
# A motif is a set of resonance conditions  E_left - E_right = sum_j s_j * w_j , one per
# ResCondKey: `diff` names the two states by quanta labels, `pf` the frequency axes w_j with
# sign s_j ('-B' -> s = -1). Fixing the mode labels (a, b, ...) to modes turns the motif into
# a linear system  LHS @ w = RHS  whose solution is where on the axes the term resonates.
#
# The four motifs below are the ones used for the pre-refactor amplitudes/resonances.py.
# With the `states` fixture and a=0, b=1:  E_a = E0, E_b = E1, E_ab = E01, E_0 = 0.

MOTIF_AB = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))        # one axis per condition
MOTIF_A = (((('a', 'b'), ('a',)), ('A',)),)                                    # single condition
MOTIF_MIXED = ((((), ('a',)), ('B',)), (((), ('a',)), ('A', '-B')))             # A - B in one condition
MOTIF_B_TWICE = ((((), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))           # two conditions, one axis


def resonance_residuals(motif: ResonanceMotif, location: ResLocGeoObject, index_dict: dict, states) -> list[float]:
    """E_left - E_right - sum_j s_j w_j for every condition, written out independently of evaluate.py."""
    def energy(quanta):
        label = ','.join(str(i) for i in sorted(index_dict[q] for q in quanta))
        return states.get_state_by_label(label).energy if label else 0.

    return [energy(c.left) - energy(c.right)
            - sum((-1. if ax.startswith('-') else 1.) * location[ax.strip('-')] for ax in c.pf)  # type: ignore
            for c in motif]


def test_resmotif_from_conditions_quanta_resolve_like_the_symbolic_diff(states):
    """A motif keeps only quanta labels; resolving them with VibDiff.from_quanta must land on the
    same states as resolving the original derive-side VibDiffTerm with from_symbolic."""
    rc1 = ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('a',), pert_freqs=('A', 'B'))
    rc2 = ResonanceCondition.make_from_tuples(left_state=('b',), right_state=('a',), pert_freqs=('-A',))
    index_dict = {'a': 1, 'b': 0}

    motif = ResonanceMotif.from_conditions([rc1, rc2])
    
    axes = tuple(sorted(motif.get_max_different_freq_axes()))
    print(f'\naxes from get_max_different_freq_axes(): {axes}\n')

    assert motif.conditions == (ResCondKey(diff=(('a', 'b'), ('a',)), pf=('A', 'B')),
                                ResCondKey(diff=(('b',), ('a',)), pf=('-A',)))
    for key, rc in zip(motif, (rc1, rc2)):
        assert VibDiff.from_quanta(*key.diff, index_dict, states) == VibDiff.from_symbolic(rc.diff, index_dict, states)

    # a=1, b=0:  E_ab - E_a = E01 - E1 ,  E_b - E_a = E0 - E1
    assert [VibDiff.from_quanta(*k.diff, index_dict, states).energy_difference() for k in motif] \
        == pytest.approx([E01 - E1, E0 - E1])


## DATA REQUEST

## DATA REQUEST -------------------------------------------------------------
# The request is a {trivial_name: DataOriginInfo} list of what to fetch: one entry per
# averaged and non-averaged property of the term (named by prop_trivname(dord, len(ops))),
# plus the fixed system-info keys that MolSystemData.from_datadict needs.

SYS_INFO_KEYS = {'anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval', 'normal_modes',
                 'atoms', 'equilibrium_geometry'}
ORIGIN = DataOriginInfo(source_type='cfour', lvl_theory='CCSD(T)', basis_set='ANO0')


def bare_term(avrg=(), non_avrg=()) -> CompiledTerm:
    return CompiledTerm(PropsCollection(list(avrg)), PropsCollection(list(non_avrg)),
                        ResonanceMotif(()), FreqTermsCollection([]), 1., idx_summ=(), idx_nonsumm=())


def test_build_data_request_for_term_names_avrg_nonavrg_and_sys_info(term_and_precalc):
    term, _ = term_and_precalc

    request = build_data_request_for_term(term, ORIGIN)

    assert set(request) == {'polgrad', 'cff'} | SYS_INFO_KEYS


def test_build_data_request_for_term_every_entry_uses_the_given_origin(term_and_precalc):
    term, _ = term_and_precalc

    request = build_data_request_for_term(term, ORIGIN)

    assert all(v == ORIGIN for v in request.values())


def test_build_data_request_for_term_without_props_is_sys_info_only():
    assert set(build_data_request_for_term(bare_term(), ORIGIN)) == SYS_INFO_KEYS


@pytest.mark.parametrize('avrg, non_avrg, expected_props', [
    ((),                                    (polprop(inds='abc'), polprop(inds='abc')), {'cff'}),      # repeated factor
    ((),                                    (polprop(inds='abc'), polprop(inds='ab')),  {'cff', 'hess'}),
    ((polprop(ops=(0, 1), inds='a'),),       (polprop(inds='a'),),                        {'polgrad', 'grad'}),
    ((polprop(ops=(0,)), polprop(ops=(0, 1))), (),                                        {'dip', 'pol'}),  # dord=0
])
def test_build_data_request_for_term_one_entry_per_distinct_trivial_name(avrg, non_avrg, expected_props):
    request = build_data_request_for_term(bare_term(avrg, non_avrg), ORIGIN)

    assert set(request) - SYS_INFO_KEYS == expected_props


def test_build_data_request_for_term_names_match_molecular_property(term_and_precalc):
    """The request must ask for exactly the names from_datadict will look up to fill mol_props,
    i.e. what MolecularProperty.from_polprop calls the same PolProps."""
    term, _ = term_and_precalc
    props = [*term.avrg_props.props, *term.non_avrg_props.props]

    request = build_data_request_for_term(term, ORIGIN)

    assert {MolecularProperty.from_polprop(p).trivial_name for p in props} == set(request) - SYS_INFO_KEYS


def test_build_data_request_for_term_is_accepted_by_the_obtainer_type_check(term_and_precalc):
    """wilson_data_obtainer rejects anything that is not dict[str, DataOriginInfo] and, with
    get_geometry=True, reads request['nc_sqrt_eigval']."""
    term, _ = term_and_precalc

    request = build_data_request_for_term(term, ORIGIN)

    assert all(isinstance(k, str) and isinstance(v, DataOriginInfo) for k, v in request.items())
    assert 'nc_sqrt_eigval' in request


def test_build_data_request_for_term_does_not_mutate_the_term(term_and_precalc):
    term, _ = term_and_precalc
    before = (repr(term.avrg_props), repr(term.non_avrg_props))

    build_data_request_for_term(term, ORIGIN)

    assert (repr(term.avrg_props), repr(term.non_avrg_props)) == before


from wilson_suite.wilson_utils.prop_trivname import prop_trivname


def test_sys_info_keys_never_collide_with_property_names():
    """build_data_request_for_term merges property names and system-info keys into one dict,
    with sys info applied last; a shared name would silently merge a property and, e.g., the
    normal modes into one data_dict entry."""
    prop_names = {prop_trivname(ord_geo=g, ord_el=e) for g in range(7) for e in range(7)}
    # dictA |= dictB means dictA is updated with keys,values from dictB
    prop_names |= {prop_trivname(ord_rot=1), prop_trivname(ord_rot=1, ord_geo=2)}   # 'B', 'coriolis'

    assert set(_sys_info_request(DataOriginInfo())).isdisjoint(prop_names)


## Evaluation kernels -------------------------------------------------------

CFF = np.arange(8, dtype=float).reshape(2, 2, 2) + 1.   # cff[a, b, c] = 1 + 4a + 2b + c
POLGRAD_AVRG = np.array([10., 20.])                    # already-averaged <polgrad>[a]


@pytest.fixture
def molsys(states) -> MolSystemData:
    props = MolPropsCollection([MolecularProperty(trivial_name='cff', vals=CFF, extra_data={})])
    return MolSystemData(name='toy', eigenvals={0: E0_eigval, 1: E1_eigval}, eigenvecs=None, mol_props=props, states=states)


