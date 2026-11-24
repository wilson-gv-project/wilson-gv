from wilson_suite.wilson_main.wilson_simulation import WilsonSimulation
from wilson_suite.wilson_main.simulation_core import SimulationCore
from wilson_suite.wilson_experiment.canned_experiments.evv_2scan import evv_2scan
from wilson_suite.wilson_system.system_data import DataOriginInfo, MolSystemData
from wilson_suite.wilson_utils.paths import SUITE_ROOT

def test_wilson_simulation():
    """
    For now organizing as unit test but could at least partly be considered an integration test
    """

    # Canned EVV-2-scan experiment
    tot_anharm_limit = 1
    mech_anharm_limit = 1
    el_anharm_limit = 1

    sim_core = SimulationCore(experiment=evv_2scan(), total_anharm_limit=tot_anharm_limit,
                              mech_anharm_limit=mech_anharm_limit, el_anharm_limit=el_anharm_limit)

    data_origin = DataOriginInfo(source_type='gaussian',
                                 base_file_loc=f'{SUITE_ROOT}/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    molsys_data = MolSystemData.load(data_origin, sim_core.props.names())

    wilson_sim = WilsonSimulation(core=sim_core, system_data=molsys_data, states_choice='anharmonic', dynrange=1000,
                                  integration_resolution = 50.0, integration_style = 'auto', n_dev_w_int = 5.0)

