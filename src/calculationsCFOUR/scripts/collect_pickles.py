#!/usr/bin/env python
"""
Expected directory structure:
.
├── GENBAS -> /cluster/projects/nn14654k/vle014/cfour_serial/bin/../basis/GENBAS
├── ZMAT
├── ZMATnew     ----> optimized geometry
├── anharm      ----> directory with data
│    ├── save           ----> directory with data
├── outfile0.out
├── polar       ----> directory with data
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

from scriptsHPC.utils import parseCFOUR
import os

curdir = os.getcwd()
print(curdir)

#outfile = curdir+'/anharm/save/out'
#fname = parseCFOUR.pklOutFile(outfile)

#dipolexfile = curdir+'/anharm/save/dipole'
# d1, d2 = parseCFOUR.getDipoleDers(dipolexfile, outfile)
#dippkl = parseCFOUR.pklDipole(dipolexfile, outfile)

pol_dir = curdir+'/polar'
# first, second = parseCFOUR.getPolarDers(pol_dir)
pfile = parseCFOUR.pklPolder(pol_dir)
polraw = parseCFOUR.pklPoldata(pol_dir)

#cubicfile = curdir+'/anharm/save/cubic'
# cff = parseCFOUR.pCubicORQuartic(cubicfile) -- ??
#cfile = parseCFOUR.pklCubic(cubicfile)

quadratureFile = curdir+'/anharm/QUADRATURE'
# equilibrium_geometry, freqs, normal_modes = parseCFOUR.pQUADRATURE(quadratureFile) - normal_modes would be a dict
dimlessFile = parseCFOUR.pklDimless_normal_modes(quadratureFile)
