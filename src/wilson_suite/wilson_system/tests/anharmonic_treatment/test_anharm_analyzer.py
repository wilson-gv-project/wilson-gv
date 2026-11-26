import pytest

from ....wilson_system.system_data import DataOriginInfo, MolSystemData
import wilson_suite as ws
from importlib.resources import files
data_dir = files('wilson_suite').joinpath('data_for_tests')

WATER = DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G',
                       base_file_loc=data_dir / 'g16_h2o_HF_STO3G.out')
VPT2_INPUTS = ['cff_rc', 'qff_rc', 'B', 'coriolis']     # force constants in cm-1, rotational, Coriolis constants

'''
compute_vpt2_states() returns list[Vibstate] and diagn dict. Vibstates energies are with anharmonic corrections

compute_vpt2_states() needs:
    - props as MolPropsCollection: cff_rc, qff_rc (cm-1), B, coriolis
    - nc_sqrt_eigval as dict
    - regime info - str
    - exclude_modes list if needed
'''

def test_anharm_analyzer_vibana():
    """
    Water, read with MolSystemData.from_output_file. GVPT2 moves the fundamentals away from the harmonic
    frequencies, to the anharmonic fundamentals the file prints (3 decimals).
    """
    molsys = MolSystemData.from_output_file(WATER, VPT2_INPUTS)
    print(f'eigenvals: {molsys.eigenvals}')

    # ---- do analysis
    states, diagn = ws.system.anharmonic_treatment.compute_vpt2_states(props=molsys.mol_props,
                                                                       nc_sqrt_eigval=molsys.eigenvals,
                                                                       regime='GVPT2',
                                                                       exclude_modes=None)
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    assert molsys.eigenvals != states_1quantum_corrected

    file_fundamentals = {k: molsys.states['anharmonic'].get_state_by_label(str(k)).energy for k in molsys.eigenvals}
    assert states_1quantum_corrected == pytest.approx(file_fundamentals, abs=2e-3)

    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)

    print('\n1 quantum levels')
    for k,v in molsys.eigenvals.items():
        print(k, '--', v, '--', states_1quantum_corrected[k])



def test_anharm_analyzer_vibana_excludemodes():
    """
    Water, read with MolSystemData.from_output_file, GVPT2 with mode 1 left out: no state of mode 1 comes back,
    and the diagnostics name the mode. An unknown mode raises.
    """
    molsys = MolSystemData.from_output_file(WATER, VPT2_INPUTS)
    print(f'eigenvals: {molsys.eigenvals}')

    # ---- do analysis
    states, diagn = ws.system.anharmonic_treatment.compute_vpt2_states(props=molsys.mol_props,
                                                                       nc_sqrt_eigval=molsys.eigenvals,
                                                                       regime='GVPT2',
                                                                       exclude_modes=[1])
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    assert set(states_1quantum_corrected) == {0, 2}
    assert not [key for s in states for key in s.harm_quanta_coeffs if '1' in key]
    assert diagn['exclude_modes'] == (1,)

    for k,v in st_dict.items():
        print(k.ljust(10), v)
    print(diagn)

    print('\n1 quantum levels')
    for k,v in molsys.eigenvals.items():
        print(k, '--', v, '--', states_1quantum_corrected.get(k))

    with pytest.raises(ValueError, match=r'modes \[3\] are out of range'):
        ws.system.anharmonic_treatment.compute_vpt2_states(props=molsys.mol_props,
                                                           nc_sqrt_eigval=molsys.eigenvals,
                                                           regime='GVPT2', exclude_modes=[3])


def test_compute_vpt2_states_names_the_inputs_without_values():
    """Water read without B and coriolis: the error names both, before any VPT2 step."""
    molsys = MolSystemData.from_output_file(WATER, ['cff_rc', 'qff_rc'])

    with pytest.raises(ValueError, match=r"no values for \['B', 'coriolis'\]"):
        ws.system.anharmonic_treatment.compute_vpt2_states(props=molsys.mol_props,
                                                           nc_sqrt_eigval=molsys.eigenvals,
                                                           regime='GVPT2', exclude_modes=[])
