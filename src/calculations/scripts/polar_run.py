#!/usr/bin/env python
import os
from calculations import calcsCFOUR
from input_parameters import settingsCalcPolar, configHPCpolar

# - 7 - now deal with polarizability
# - 7_1 - ZMAT for PROPS=SECOND_ORDER

calcsCFOUR.fromZmatNew2Zmat('ZMAT', settingsCalcPolar)
os.chdir(settingsCalcPolar['jobtype'])
#
# # - 7_2 - submit.sh
# calcsCFOUR.generateSubmitPy(configHPCpolar, 'submitpy.sh')
#
# # - 7_3 - sbatch submit.sh for polar equilibrium
# calcsCFOUR.sumbitSbatch("submitpy.sh")

#os.makedirs('polar', exist_ok=True)
#os.chdir('./polar')

# calcsCFOUR.fromZmatNew2Zmat('ZMAT', settingsCalcPolar)

# - 7_4 -
calcsCFOUR.makeDisplacementsNew(delta=0.01, config=configHPCpolar)
#calcsCFOUR.makeDisplacements(delta=0.01, config=configHPCpolar)
