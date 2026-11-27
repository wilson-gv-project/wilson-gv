from wilson_suite.wilson_main.wilson_simulation import WilsonSimulation
from wilson_suite.wilson_main.simulation_core import SimulationCore
from wilson_suite.wilson_experiment.canned_experiments.evv_2scan import evv_2scan_one_magn
from wilson_suite.wilson_system.system_data import DataOriginInfo, MolSystemData
from wilson_suite.wilson_utils.paths import SUITE_ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Canned EVV-2-scan experiment
tot_anharm_limit = 1
mech_anharm_limit = 1
el_anharm_limit = 1

sim_core = SimulationCore(experiment=evv_2scan_one_magn(), total_anharm_limit=tot_anharm_limit,
                          mech_anharm_limit=mech_anharm_limit, el_anharm_limit=el_anharm_limit)




data_origin = DataOriginInfo(source_type='gaussian',
                             base_file_loc=f'{SUITE_ROOT}/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

molsys_data = MolSystemData.load(data_origin, sim_core.props.names())

wilson_sim = WilsonSimulation(core=sim_core, system_data=molsys_data, states_choice='anharmonic', dynrange=1000,
                              integration_resolution = 10.0, integration_style = 'auto', n_dev_w_int = 4.0)


# To hold instances of FieldAnalysis
acc_res = np.zeros(sim_core.experiment.field_analyses.shape)



for scan_elem in np.ndindex(acc_res.shape):

    acc_res[scan_elem] = wilson_sim.int_results.int_results[scan_elem]


a1_range = 1.0* np.array([i * sim_core.experiment.scans[0].scan_objs[0].coeff + sim_core.experiment.field.pulses[0].cf
                     for i in sim_core.experiment.scans[0].range])


a2_range = np.array([i * sim_core.experiment.scans[1].scan_objs[0].coeff + sim_core.experiment.field.pulses[1].cf for i in sim_core.experiment.scans[1].range
])

z_max = np.amax(acc_res)
spec_fig = plt.figure(figsize=(6, 6))
ax_def = spec_fig.add_subplot(1, 1, 1)
# Logarithmic plot
palette = ax_def.pcolor(a2_range,
                        a1_range,
                        acc_res,
                        norm=colors.LogNorm(vmin=z_max / 1000, vmax=z_max),
                        cmap='hot_r', shading='nearest')
ax_def.set_aspect('equal')
colorbar = spec_fig.colorbar(palette, ax=ax_def, extend='max')
spec_fig.savefig('scan_plot.png', dpi=180)
plt.close(spec_fig)