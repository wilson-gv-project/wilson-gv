"""
"""
import wilson_suite as ws
from wilson_suite.wilson_system.system_data import MolecularProperty, MolPropsCollection
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from wilson_suite.wilson_utils.serialization import unpickle_smth_from

pickles_dir = SUITE_ROOT+'/wilson_system/tests/anharmonic_treatment/datafiles'


def vpt2_inputs(props) -> MolPropsCollection:
    """
    The VPT2 inputs from the pickled properties of the old workflow. The pickles keep the force
    constants in cm-1 in extra_data of cff and qff; the collection names them cff_rc and qff_rc.
    """
    by_name = {p.trivial_name: p for p in props}
    return MolPropsCollection([MolecularProperty(trivial_name='cff_rc', vals=by_name['cff'].extra_data),
                               MolecularProperty(trivial_name='qff_rc', vals=by_name['qff'].extra_data),
                               MolecularProperty(trivial_name='B', vals=by_name['B'].vals),
                               MolecularProperty(trivial_name='coriolis', vals=by_name['coriolis'].vals)])

def test_form_vpt2():
    file = 'HF_STO_3G_VPT2.pkl'
    vib_ana, props = unpickle_smth_from(filenamepkl=file, load_from=pickles_dir)
    # save anharmonic corrected from g16 output
    states_g16 = vib_ana.states
    
    print()
    print([state.energy for state in vib_ana.states if ',' not in state.state_label])
    # ---- do analysis - 
    states, diagn = ws.system.anharmonic_treatment.anharm_analyzer_data(props=vpt2_inputs(props),
                                                                       nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                       regime="VPT2", #vib_ana.regime
                                                                       exclude_modes=None)        
    # ---- check results
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    st_dict_g16 = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states_g16}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states_g16 if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    
    print(states_1quantum_corrected)

    for k in st_dict_g16:
        print(k, '--', st_dict_g16[k], '--', round(st_dict[k], 4))
    
    for k in st_dict_g16:
        # not sure how to estimate the agreement
        assert abs(st_dict_g16[k] - round(st_dict[k], 4)) <= 0.003001


def test_form_gvpt2():
    file = 'HF_STO_3G_GVPT2.pkl'
    vib_ana, props = unpickle_smth_from(filenamepkl=file, load_from=pickles_dir)
    # save anharmonic corrected from g16 output
    states_g16 = vib_ana.states
    
    print()
    print([state.energy for state in vib_ana.states if ',' not in state.state_label])
    # ---- do analysis - 
    states, diagn = ws.system.anharmonic_treatment.anharm_analyzer_data(props=vpt2_inputs(props),
                                                                       nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                       regime="GVPT2", #vib_ana.regime
                                                                       exclude_modes=None)        
    # ---- check results
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    st_dict_g16 = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states_g16}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states_g16 if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    
    print(states_1quantum_corrected)

    for k in st_dict_g16:
        print(k, '--', st_dict_g16[k], '--', round(st_dict[k], 4))
    
    for k in st_dict_g16:
        # not sure how to estimate the agreement
        assert abs(st_dict_g16[k] - round(st_dict[k], 4)) <= 0.0014

def test_form_gvpt2_nocoriolis():
    file = 'HF_STO_3G_GVPT2_NoCoriolis_really.pkl'
    vib_ana, props = unpickle_smth_from(filenamepkl=file, load_from=pickles_dir)
    # save anharmonic corrected from g16 output
    states_g16 = vib_ana.states
    
    print()
    print([state.energy for state in vib_ana.states if ',' not in state.state_label])
    # coriolis values are zeros
    for p in props:
        if p.trivial_name == 'coriolis':
            assert (p.vals == 0).all()

    # ---- do analysis - 
    states, diagn = ws.system.anharmonic_treatment.anharm_analyzer_data(props=vpt2_inputs(props),
                                                                       nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                       regime="GVPT2", #vib_ana.regime
                                                                       exclude_modes=None)        
    # ---- check results
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    st_dict_g16 = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states_g16}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states_g16 if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    
    print(states_1quantum_corrected)

    for k in st_dict_g16:
        print(k, '--', st_dict_g16[k], '--', round(st_dict[k], 4))
    
    for k in st_dict_g16:
        # not sure how to estimate the agreement
        assert abs(st_dict_g16[k] - round(st_dict[k], 4)) <= 0.0013


def test_form_dvpt2():
    file = 'HF_STO_3G_DVPT2.pkl'
    vib_ana, props = unpickle_smth_from(filenamepkl=file, load_from=pickles_dir)
    # save anharmonic corrected from g16 output
    states_g16 = vib_ana.states
    
    print()
    print([state.energy for state in vib_ana.states if ',' not in state.state_label])
    # ---- do analysis - 
    states, diagn = ws.system.anharmonic_treatment.anharm_analyzer_data(props=vpt2_inputs(props),
                                                                       nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                       regime="DVPT2", #vib_ana.regime
                                                                       exclude_modes=None)        
    # ---- check results
    st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
    st_dict_g16 = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states_g16}
    states_1quantum_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states_g16 if len(list(s.harm_quanta_coeffs.keys())[0])==1}
    
    print(states_1quantum_corrected)

    for k in st_dict_g16:
        print(k, '--', st_dict_g16[k], '--', round(st_dict[k], 4))
    
    for k in st_dict_g16:
        # not sure how to estimate the agreement
        assert abs(st_dict_g16[k] - round(st_dict[k], 4)) <= 0.0014


