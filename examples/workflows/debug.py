#!/usr/bin/env python

from wilson_suite.wilson_utils.serialization import unpickle_smth_from
from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow
from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation

eval_wf_file = '/home/vlev/monorepo/examples/workflows/eval_wf.pkl'

eval_wf: EvaluationWorkflow = unpickle_smth_from(eval_wf_file)

print(eval_wf.artifacts.__dict__.keys())
print()

print('eval_wf.artifacts.data_configs.number_of_nmodes', eval_wf.artifacts.data_configs.number_of_nmodes)
print('eval_wf.inputs.number_of_modes', eval_wf.inputs.number_of_modes)
print('eval_wf.inputs.vib_ana_setup', eval_wf.inputs.vib_ana_setup.system)
print('eval_wf.inputs.vib_ana_setup.system.natoms*3-6', eval_wf.inputs.vib_ana_setup.system.natoms*3-6)
print('eval_wf.inputs.vib_ana_setup.nc_sqrt_eigval', eval_wf.inputs.vib_ana_setup.nc_sqrt_eigval)

print()
last = 0
last_key = 0

for k,v in eval_wf.artifacts.__dict__.items():
    if v is not None:
        last = v
        last_key = k

print("\n     Last step was:", last_key)
print()

print()

if isinstance(last, dict):
    for k,v in last.items():
        print('\n ----  ', k)
        print()
        print(type(v), len(v))
        print(v)

# if last_key == 'spec_window':
#     print('Found features:')

#     for feat in eval_wf.artifacts.features:
#         print(feat.location)


wsim_eval_success: WilsonSimulation = unpickle_smth_from('/home/vlev/monorepo/examples/workflows/wsim_after_eval.pkl')

features = wsim_eval_success._workflow.artifacts.features
for f in features:
    if all([i[1]>0 for i in f.location.coordinates]):
        print(f.location.coordinates)