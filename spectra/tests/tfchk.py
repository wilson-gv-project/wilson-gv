#!/usr/bin/env python
import numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=12)

dimlessFile = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/QUADRATURE'
fchkname = '/home/vlew/scriptsHPC/data/coh2aldehyde_HFcc-pVTZ/coh2hfanh_newopt_raman.fchk'

from mock2D.spectrum.callbacks2DIR import GaussianData, tensors2dimlessNMbasis, getDimensionlessNM
from

data={'source': 'gaussian',
      'type': 'fchk',
      'files': {'fchk': fchkname}, 'dimensionless': dimlessFile}

e = GaussianData(data)

dipdc = e.getDipDersCart()
np.set_printoptions(linewidth=250, suppress=True, precision=12)

print(dipdc.shape)
dipd = tensors2dimlessNMbasis(dipdc, [0], dimlessFile)

print(dipd)

smth = e.getPolarDersCart()
print(smth.shape)

pold = tensors2dimlessNMbasis(smth, [0], dimlessFile)
# transfMatrix = getDimensionlessNM(dimlessFile)
# pold = np.einsum('ijk,iq->qjk', smth, transfMatrix)
# pold = np.einsum('ijk,q->qjk', pold, transfMatrix)
print(pold)

# d34 =
# 'Cartesian 3rd/4th derivatives'


