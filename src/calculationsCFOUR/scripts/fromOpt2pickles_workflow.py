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
# 5. Make pickles from output files and save them in the input_data_info directory
# 6. Tadaaa
#

import os

from scriptsHPC.utils import calcsCFOUR

# get the working directory where the optimization will be run
basedir = os.getcwd()

# - 1 - ZMAT opt
settingsOpt = {'level of theory': tuple(['HF', 'cc-pVTZ']),
                'geoconv':12, 'ccconv':12, 'scfconv':12, 'lineqconv':12}
calcsCFOUR.makeOptZmat(settingsOpt)

# - 2 - submit.sh
configHPC = {'machine':'fram', 'minutes':30, 'hours':'00', 'nodes':1, 'dir3':False,
             'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}
calcsCFOUR.generateSubmitPy(configHPC, "submitpy.sh")

# - 2 - submit.sh
# bash_script_path = '../submit_utils/scrmaster_fram'
# configHPC = {'machine':'fram', 'minutes':10, 'hours':'00'}
# generateSubmit(bash_script_path, config=configHPC)

# - 3 - sbatch submit.sh for opt
calcsCFOUR.sumbitSbatch("submitpy.sh")

quit()
# check if finished optimization

# - 4 - ZMAT for ANH_ALGORITHM=PARALLEL, VIBRATION=ANALYTIC
settingsCalc = {'level of theory': tuple(['HF', 'cc-pVTZ']),
                'geoconv':12, 'ccconv':12, 'scfconv':12, 'lineqconv':12,
                'job':'ANH_ALGORITHM=PARALLEL\nVIBRATION=ANALYTIC\nFD_PROJECT=ON\nPRINT=1',
                'jobtype':'anharm'}
zmatnew = 'ZMAT'
calcsCFOUR.fromZmatNew2Zmat(zmatnew, settingsCalc)
os.chdir(settingsCalc['jobtype'])

# - 5 - submit.sh
configHPC = {'machine':'fram', 'minutes':40, 'hours':'00', 'nodes':1, 'dir3':True,
             'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}
calcsCFOUR.generateSubmitPy(configHPC, "submitpy.sh")

# - 5 - submit.sh
# bash_script_path = '../submit_utils/scrmaster_fram'
# configHPC = {'machine':'fram', 'minutes':10, 'hours':'00'}
# generateSubmit(bash_script_path, config=configHPC)

# - 6 - extend submit.sh - prepare for equilibrium anharmonic parallel calculation
# extendSubmitEquilParAnh('./submit.sh')

# - 6 - extend submitpy.sh - prepare for equilibrium anharmonic parallel calculation
calcsCFOUR.extendSubmitEquilParAnh('./submitpy.sh')

# - 7 - sbatch submit.sh for anharm
# sumbitSbatch("submitpy.sh")
os.chdir(basedir)

# check if finished all anharm parallel

# - 6 - go to 'save' dir, make submit.sh and run it -- process fja files
# config = {'machine':'fram', 'minutes':'03', 'hours':'00', 'nodes':1}
# process_fja(config)

# # - 7 - now deal with polarizability
# # - 7_1 - ZMAT for PROPS=SECOND_ORDER
# settingsCalc = {'level of theory': tuple(['HF', 'cc-pVTZ']),
#                 'geoconv':12, 'ccconv':12, 'scfconv':12, 'lineqconv':12,
#                 'job':'PROPS=SECOND_ORDER\nPRINT=1',
#                 'jobtype':'polar'}
# fromZmatNew2Zmat('ZMAT', settingsCalc)
# os.chdir(settingsCalc['jobtype'])
#
# # - 7_2 - submit.sh
# configHPC = {'machine':'fram', 'minutes':10, 'hours':'00', 'nodes':1, 'dir3':True}
# generateSubmitPy(configHPC)
#
# # - 7_3 - sbatch submit.sh for polar equilibrium
# # sumbitSbatch("submitpy.sh")
#
# # - 7_4 -
# makeDisplacements(delta=0.01)
# # - 6 - make pickles and save to input_data_info dir


