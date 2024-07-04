#!/usr/bin/env python
import os
from scriptsHPC.utils import calcsCFOUR
from input_parameters import settingsCalc, configHPC, configHPCdispl, config_fja

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--stage", help="Select 1 to submit main anharm parallel job.\nSelect 2 to submit projeccing of fja files.")
args = vars(parser.parse_args())

# Set up parameters
stage = args["stage"]
print(stage)
# where optimization was done
basedir = os.getcwd()
print(f'Base directory is {basedir}')

if stage == '1':

    # - 4 - ZMAT for ANH_ALGORITHM=PARALLEL, VIBRATION=ANALYTIC
    calcsCFOUR.fromZmatNew2Zmat('ZMATnew', settingsCalc)
    os.chdir(settingsCalc['jobtype'])

    # - 5 - submit.sh
    calcsCFOUR.generateSubmitPy(configHPC, "submitpy.sh")

    # - 6 - extend submitpy.sh - prepare for equilibrium anharmonic parallel calculation
    calcsCFOUR.extendSubmitEquilParAnh('./submitpy.sh', configHPCdispl)

    # - 7 - sbatch submit.sh for anharm
    jobid = calcsCFOUR.sumbitSbatch("submitpy.sh")
    # os.chdir(basedir)

    print('\nSubmitted the main anharmonic parallel job')

elif stage == '2':
    # check if finished main anharm parallel
    # st0 = calcsCFOUR.checkStatus('id', jobid)
    # print(st0)

    # import time
    # time.sleep(30)

    # check if finished all other anharm parallel
    # st1 = calcsCFOUR.checkStatus('name', basedir)

    calcsCFOUR.process_fja(config_fja)

    print('The last step was submitted. The final results will be in "save" directory')

else:
    print('Unknown option.\nSelect 1 to submit main anharm parallel job.\nSelect 2 to submit projeccing of fja files.')
