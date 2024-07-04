#!/usr/bin/env python
##############################################################################
##                                                                          ##
##                    Hagakure, The Way of the Samurai                      ##
##                                                                          ##
##############################################################################
#
# 1. Optimize structure --> ZMATnew
# 2. With ZMATnew geometry, run ANH_ALGORITHM=PARALLEL, VIBRATION=ANALYTIC,
#                               FD_PROJECT=ON --> zmat0* files
#
#          xcfour > "$output_filename"
#   2a. Add lines to generated submit.sh
#          mkdir save
#          cp JOBARC ./save/
#          cp JAINDX ./save/
#          ../../../../../scriptsHPC/cfourscripts/vpt2_parallel/mkzmatdirs
# 3. Run all the new zmat0* in their directories
#
#           xcfour > "$output_filename"
#    3a. Add lines to generated submit.sh
#           cp DCT dct0
#           xja2fja >> out1
#           cp FJOBARC ../save/fja.004
# 4. Run post-processing script for fja.0* files
#
#         # Copy fja.x to FJOBARC
#         cp "$file" FJOBARC
#         # Execute xja2fja
#         xja2fja
#         # Execute xcubic and append output to out file
#         xcubic >> out
# 5. Make pickles from output files and save them in the data directory
# 6. Tadaaa
#

import os

from scriptsHPC.utils import calcsCFOUR
from input_parameters import settingsOpt, configHPCopt, molecules

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--molecule", help="Molecule name from input_params.py")
parser.add_argument("-t", "--hours", type=str, help="Hours in sbatch settings of time")
args = vars(parser.parse_args())

# Set up parameters
molecule = args["molecule"]
hours = args["hours"]
configHPCopt['hours'] = hours

# get the working directory where the optimization will be run
basedir = os.getcwd()

# - 1 - ZMAT opt
# pick a molecule from:
# coh2 - formaldehyde; co2h2 - formic acid; conh3 - formamide;
calcsCFOUR.makeOptZmat(settingsOpt, molecules[molecule])

# - 2 - submit.sh
calcsCFOUR.generateSubmitPy(configHPCopt, "submitpy.sh")

# - 3 - sbatch submit.sh for opt
jobid = calcsCFOUR.sumbitSbatch("submitpy.sh")
# quit()
# resultJob = calcsCFOUR.checkJobStatusID(jobid)

#while resultJob == 'RUNNNING':
#    import time
#    time.sleep(20)
#    resultJob = calcsCFOUR.checkJobStatus(jobid)

# print(resultJob)
#print('Optimization job has finished now')
print('Optimization job has been submitted')
