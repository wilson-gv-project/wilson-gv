#!/usr/bin/env python
"""
./wilson_script.py --new y --sparse 0. --preview n --vpt2 n --w1mw2 n
"""
import argparse
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning,
                        message="Signature .* for <class 'numpy.longdouble'> does not match any known type: falling back to type probe function.")
import time
from datetime import timedelta

import numpy as np
np.set_printoptions(legacy='1.25')

from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import GaussianDataParser, CFOURdataParser
from CQCParse.relay import DataVault
from wilson.utils import get_package_root

st0 = time.time()

wilson_root = get_package_root()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--new", type=str2bool, default=True)
parser.add_argument('-s', "--sparse", type=float, default=0.,
                    help='Radius of small grids; default is 0. which means full window will be calculated')
parser.add_argument('-p', "--preview", type=str2bool, default=False)
parser.add_argument('-a', "--vpt2", type=str2bool, default=False)
parser.add_argument('-w', "--w1mw2", type=str2bool, default=False,
                    help='(w1,w2) or (w1,w2-w1) spectrum format')

args = parser.parse_args()

prefix = None

new = args.new
sparse = args.sparse
if sparse == 'n':
    sparse = 0.

preview = args.preview
vpt2 = args.vpt2
w1mw2 = args.w1mw2
# print(new, sparse, preview, vpt2, w1mw2)

if vpt2:
    prefix = 'vpt2'

# data_vault = DataVault(wilson_root+'/tests/test_database/mini_files_database.csv')
data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')

dataframe_gaussian = data_vault.getting_files_DB("gaussian")
# print(dataframe_gaussian[dataframe_gaussian['code']!='FORM'])

# set up big spectrum window
omega1 = np.arange(1130., 2050., 2.91)
omega2 = np.arange(1300., 5150., 2.91)

# METH - methanol, ACAC - acetic acid, ACDM - acetic acid dimer, FORM - formaldehyde, FOAC - formic acid,
# OXAC1 - oxalic acid 1, OXAC2 - oxalic acid 2
molecule = 'FORM'
method = 'B3LYP' # HF, B3LYP
basis = 'cc_pVQZ'
Gamma_rc = 4.7
list2exclude = []
diag_margin_rc=180.
terms_selection = [0,1], [0,1]  # FIXME!!
screenmodeswindow = True

if molecule=='FORM':
    omega1 = np.arange(1000., 2950., 3.8)
    omega2 = np.arange(1400., 5650., 3.8)

    # omega1 = np.arange(1150., 2050., 3.8)
    # omega2 = np.arange(2400., 5150., 3.8)

if molecule=='FOAC':
    omega1 = np.arange(300., 3850., 3.8)
    omega2 = np.arange(450., 7250., 3.8)

if molecule=='ACDM':
    omega1 = np.arange(350., 3450., 3.8)
    omega2 = np.arange(450., 6250., 3.8)

if molecule=='ACAC':
    omega1 = np.arange(0., 3750., 3.8)
    omega2 = np.arange(250., 7150., 3.8)

if molecule=='METH':
    omega1 = np.arange(100., 3850., 3.8)
    omega2 = np.arange(250, 7350., 3.8)


print('\n     Calculation:', (molecule, method, basis))
print('    E:', terms_selection[0])
print('    M:', terms_selection[1], '\n')


# fixme: try defaultdict ?
datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
gParser = GaussianDataParser(datadict)

dictInputs = {'parserObject': gParser,
              'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

# ------- setting up a Spectrum2D object
spectrumObj = Spectrum2D(omega1, omega2)
spectrumObj.load_data(dictInputs['parserObject'], vpt2=vpt2)


spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=diag_margin_rc, vib_levels_harmonic=False)
spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select']) # currently requires diag_margin_rc attribute to be set

if molecule=='ACDM':
    print('     Number of normal modes:', spectrumObj.nmodes)
    list2exclude = [34, 35, 36, 37, 38, 39, 40, 41]
print('    coords_ab', len(spectrumObj.coords_ab))

spectrumObj.precalculateParts(list2exclude=list2exclude,
                              preview=preview,
                              screenmodeswindow=screenmodeswindow)
print(sorted(list(spectrumObj.all_states.values())))
print(spectrumObj.all_states)
# exit()

mask = None

if sparse!=0.:

    d1 = spectrumObj.findAllGrids(sparse)
    prefix = 'windows'

    print('         Number of grids:', len(d1))
    print('   ---> sparse')
    new_w1_mesh = np.zeros(spectrumObj.w1_mesh.shape, dtype='complex64')
    new_w2_mesh = np.zeros(spectrumObj.w2_mesh.shape, dtype='complex64')
    for r in d1:
        new_w1_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][2]
        new_w2_mesh[r[1][0]: r[1][1], r[1][2]: r[1][3]] = d1[r][3]

    spectrumObj.w1_mesh_Eh = new_w1_mesh
    spectrumObj.w2_mesh_Eh = new_w2_mesh
    allp = spectrumObj.w2_mesh.shape[0]*spectrumObj.w2_mesh.shape[1]
    print(np.count_nonzero(spectrumObj.w1_mesh_Eh), allp, np.count_nonzero(spectrumObj.w1_mesh_Eh)/allp)
    mask = spectrumObj.w1_mesh_Eh != 0.


np.set_printoptions(precision=6)

finalIntGrid = np.zeros(spectrumObj.shape2d, dtype='complex64')
print('     Number of normal modes again:', spectrumObj.nmodes)

# ------- computing anharmonicities
if new:

    st = time.time()

    sec_hypol_dataALL = spectrumObj.intensity_both(selectionCond=mask)

    elapsed_time = time.time() - st
    elapsed_timedelta = timedelta(seconds=elapsed_time)
    formatted_time = str(elapsed_timedelta)
    print('Calculated intensities with opt in:',
          formatted_time)

else:

    st = time.time()

    sec_hypol_dataALL = 0

    sec_hypol_data1 = 0
    if dictInputs['el_terms_select']:
        spectrumObj.intensity_electrical()

    sec_hypol_data2 = 0
    if dictInputs['mech_terms_select']:
        spectrumObj.intensity_mechanical()

    sec_hypol_dataALL = spectrumObj.intensities_grid

    elapsed_time = time.time() - st
    elapsed_timedelta = timedelta(seconds=elapsed_time)
    formatted_time = str(elapsed_timedelta)
    print('Calculated intensities in:',
          formatted_time)



#########################################################################################################
# intensity grid for plotting spectrum
intensity = abs(sec_hypol_dataALL) ** 2

from wilson import rendering
import os

settings_here = {'w1mw2': w1mw2,
                 'font_dict': {'size': 14}, 'figsize': (22, 42),
                 'norm_max': None, 'norm_min': None,
                 'dynamic_range_n': 3000, 'num_color_levels': 10, 'num_level_ticks': 8,
                 'levels_ticks': None, 'levels': None}

artist = rendering.SpectrumFigure(sec_hypol_dataALL, spectrumObj, spectrumObj.w1_mesh, spectrumObj.w2_mesh, settings_here)
title_on_top, text_under_the_figure = rendering.make_texts4fig(datadict, spectrumObj, artist, directory='.')
name = rendering.make_name(datadict, spectrumObj, artist, directory='.', prefix=prefix)

print('name:', name)
print(os.path.join(os.path.dirname('__file__')))

fig = artist.plot2Dmatplotlib(nametuple=(name, os.path.join(os.path.dirname('__file__')), title_on_top),
                              text_under_the_figure=text_under_the_figure, diagonal=False, to_save=True)


elapsed_time = time.time() - st0
elapsed_timedelta = timedelta(seconds=elapsed_time)
formatted_time = str(elapsed_timedelta)
print('All done in:',
      formatted_time)
