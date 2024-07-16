#!/usr/bin/env python
"""
Expected directory structure:
.
├── GENBAS -> /cluster/projects/nn14654k/vle014/cfour_serial/bin/../basis/GENBAS
├── ZMAT
├── ZMATnew     ----> optimized geometry
├── anharm      ----> directory with input_data_info
│    ├── save           ----> directory with input_data_info
├── outfile0.out
├── polar       ----> directory with input_data_info
│    ├── 10_12pn    ...
│    ├── 8p         ...
├── slurm-5778462.out
└── submitpy.sh ----> sbatch submit script

The output should be 7 pickle files:
                'vibdata.pkl',
                'cubic.pkl',
                'dipolexyz.pkl',
                'polar.pkl',
                'polarData.pkl',
                'polarData_raw.pkl',
                'dimensionless.pkl'
"""

from calculations import parseCFOUR_forWilson
import os

curdir = os.getcwd()
print(curdir)

#outfile = curdir+'/anharm/save/out'
#fname = parseCFOUR_forWilson.pklOutFile(outfile)

#dipolexfile = curdir+'/anharm/save/dipole'
# d1, d2 = parseCFOUR_forWilson.getDipoleDers(dipolexfile, outfile)
#dippkl = parseCFOUR_forWilson.pklDipole(dipolexfile, outfile)

pol_dir = curdir+'/polar'
# first, second = parseCFOUR.getPolarDers(pol_dir)
pfile = parseCFOUR_forWilson.pklPolder(pol_dir)
polraw = parseCFOUR_forWilson.pklPoldata(pol_dir)

#cubicfile = curdir+'/anharm/save/cubic'
# cff = parseCFOUR_forWilson.pCubicORQuartic(cubicfile) -- ??
#cfile = parseCFOUR_forWilson.pklCubic(cubicfile)

quadratureFile = curdir+'/anharm/QUADRATURE'
# equilibrium_geometry, freqs, normal_modes = parseCFOUR_forWilson.pQUADRATURE(quadratureFile) - normal_modes would be a dict
dimlessFile = parseCFOUR_forWilson.pklDimless_normal_modes(quadratureFile)
