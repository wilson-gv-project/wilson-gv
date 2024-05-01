#!/usr/bin/env python
###########################################################
##                                                       ##
##    Tests : 2DIR spectra calculation examples          ##
##                                                       ##
###########################################################

import main2DIR as dd_ir
import numpy as np
import pandas as pd
# pd.set_option('display.float_format', '{:.8f}'.format)
# np.set_printoptions(linewidth=250, suppress=True, precision=10)
import pickle

# Terms in expressions
electrical_terms = [('a+b,a', 'zero,a'), ('b,a', 'zero,a') ]

# derivatives:
# 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
# 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
electric_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                 [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))] ]

mechanical_terms = [[('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')],
                    [('c,a', 'zero,a'), ('a+b,c', 'b+c,a')],
                    [('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')],
                    [('b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')] ]

# derivatives:
# mu_Q, alpha_Q - for all 6 terms
mechanical_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'] ]

def picks(pool, listofinds):
    return [pool[i] for i in listofinds]

acetonitrileBool = False
coh2 =True


#=========================================================================
# Data

# Load the data structure from the pickle file
filename = '../scriptsHPC/cfourscripts/vibdata.pkl'
with open(filename, 'rb') as f:
    loaded_data_with_metadata = pickle.load(f)

# Extract the metadata and data
metadata = loaded_data_with_metadata['metadata']
data = loaded_data_with_metadata['data']

# print(metadata)
# print(data['modes'])
# print(data['anharmonic_frequencies'])

combd = dict(zip([tuple([str((t[:3] + t[5:8])[i]-7) for i in range(3) for _ in range((t[:3] + t[5:8])[i + 3])]) for t in data['modes']], data['anharmonic_frequencies']))
Delta = {tuple(sorted(k)):v for k, v in combd.items() if len(k)>1}
# print(Delta)
# quit()

#==============================================================

filename = '../scriptsHPC/cfourscripts/dipolexyz.pkl'

# Load the dictionaries from the file
with open(filename, 'rb') as file:
    dipx, dipy, dipz = pickle.load(file)

dipall = {'x': dipx, 'y': dipy, 'z': dipz}

labels = sorted(list(set([t[0] for t in data['modes']])))

dmulist = []
d2mulist = []

dmudqdict = {}
dmudqdqdict = {}
dq = len(labels)
dmudqdqdict['x'] = np.zeros((dq, dq))
dmudqdqdict['y'] = np.zeros((dq, dq))
dmudqdqdict['z'] = np.zeros((dq, dq))

for l in labels:
    # print(l, type(l))
    if type(l) == int:
        dmudqdict[l] = np.zeros(3)
        if l in dipx:
            dmudqdict[l][0] = dipx[l]
        if l in dipy:
            dmudqdict[l][1] = dipy[l]
        if l in dipz:
            dmudqdict[l][2] = dipz[l]
        dmulist.append(dmudqdict[l])
for l in dipx:
    if type(l)!=int:
        if len(l) == 2:
            dmudqdqdict['x'][(l[0]-7, l[1]-7)] = dipx[l]

for l in dipy:
    if type(l)!=int:
        if len(l) == 2:
            dmudqdqdict['y'][(l[0]-7, l[1]-7)] = dipy[l]

for l in dipz:
    if type(l)!=int:
        if len(l) == 2:
            dmudqdqdict['z'][(l[0]-7, l[1]-7)] = dipz[l]

dmuarray = np.array(dmulist)
# dictionary of mu derivatives by mode
# print(dmudqdict)

# 2d array of mu derivatives
# print('------')
# print(dmuarray.shape, dmuarray)

# dictionary of mu second derivatives by modes 2d arrays (3N-6)x(3N-6)
# print(dmudqdqdict)

#==============================================================

# Specify the filename from which to load the dictionaries
filename = '../scriptsHPC/cfourscripts/polarders.pkl'

# Load the dictionaries from the file
with open(filename, 'rb') as file:
    firstder, secder = pickle.load(file)

#==============================================================

# Specify the filename from which to load the dictionaries
filename = '../scriptsHPC/cfourscripts/cubicarray.pkl'

# Load the dictionaries from the file
with open(filename, 'rb') as file:
    cubic = pickle.load(file)

cubicmat = np.zeros((6, 6, 6))

for e in cubic:
    # print((int(e[0]), int(e[1]), int(e[2])))
    els = [int(e[0])-7, int(e[1])-7, int(e[2])-7]
    import itertools
    permutations = list(itertools.permutations(els))
    for p in permutations:
        cubicmat[p] = e[3]
# print(cubicmat[0])

def HarmonicFrequenciesC4(picklefilevib):
    """freqs from cfour pickeled data"""
    # Load the data structure from the pickle file
    with open(picklefilevib, 'rb') as f:
        loaded_data_with_metadata = pickle.load(f)

    data = loaded_data_with_metadata['data']
    # normal mode labels
    labels = sorted(list(set([t[0] for t in data['modes']])))
    # tuples of combinations of normal modes
    tlab = [tuple(element for element in t if element != 0) for t in data['modes']]
    # dictionary of tuples and corresponding frequencies
    dd = dict(zip(tlab, data['anharmonic_frequencies']))

    # only fundamental frequencies
    freqs = np.array([dd[b] for b in [tuple([e, 1]) for e in labels]])

    return freqs

def cubicpost(picklefilevib, cubicpickle):
    """ derives cubic and quartic anharmonic constants.
        It takes reduced values [cm-1] from gaussian output
        and transforms it to :
          * cubic   force constants : [Hartree*amu(-3/2)*Bohr(-3)]
          * quartic force constants : [Hartree*amu(-2  )*Bohr(-4)]
    """
    BohrToAngstrom = 0.5291772086
    HartreeToAttoJoule = 4.3597439
    ToRedCubForceConst = 9.85501E+06

    freq = HarmonicFrequenciesC4(picklefilevib)
    n = len(freq)
    K3 = np.zeros((n, n, n), dtype=np.float64)
    # Specify the filename from which to load the dictionaries

    # Load the dictionaries from the file
    with open(cubicpickle, 'rb') as file:
        cubic = pickle.load(file)

    for fijk in cubic:

        i = int(fijk[0]) - 7
        j = int(fijk[1]) - 7
        k = int(fijk[2]) - 7
        d = np.float64(fijk[3])
        # print('cm-1', d)
        # transform to [Hartree*amu(-3/2)*Bohr(-3)]
        d *= np.sqrt(freq[i] * freq[j] * freq[k])
        d *= BohrToAngstrom ** 3
        d /= ToRedCubForceConst * HartreeToAttoJoule
        #
        # print('au  ', d, '\n')
        K3[i, j, k] = d
        K3[i, k, j] = d
        K3[k, j, i] = d
        K3[k, i, j] = d
        K3[j, i, k] = d
        K3[j, k, i] = d

    return K3

cubicpickle = '../scriptsHPC/cfourscripts/cubicarray.pkl'
picklefilevib='../scriptsHPC/cfourscripts/vibdata.pkl'

Fijk = cubicpost(picklefilevib, cubicpickle)

np.set_printoptions(linewidth=250, suppress=True, precision=10)
# print(Fijk[0])
# print(cubicmat==Fijk)
# quit()
#==============================================================

# Function to print the dictionary nicely
def print_nicely(data):
    pairs = {k: v for k, v in data.items() if len(k) == 2}
    triples = {k: v for k, v in data.items() if len(k) == 3}
    singles = {k: v for k, v in data.items() if len(k) == 1}

    # Print singles
    print("Singles:")
    for k, v in singles.items():
        print(f"{k}: {v:.3f}")
    print("\n")

    # Print pairs
    print("Pairs:")
    for k, v in pairs.items():
        print(f"{k}: {v:.3f}")
    print("\n")

    # Print triples
    print("Triples:")
    for k, v in triples.items():
        print(f"{k}: {v:.3f}")

def get_fundamentalsc4(picklefilevib):
    # Load the data structure from the pickle file
    with open(picklefilevib, 'rb') as f:
        loaded_data_with_metadata = pickle.load(f)

    # Extract the metadata and data
    # metadata = loaded_data_with_metadata['metadata']
    data = loaded_data_with_metadata['data']

    # normal mode labels
    labels = sorted(list(set([t[0] for t in data['modes']])))

    # tuples of combinations of normal modes
    tlab = [tuple(element for element in t if element != 0) for t in data['modes']]

    # dictionary of tuples and corresponding frequencies
    dd = dict(zip(tlab, data['anharmonic_frequencies']))
    # print('============\n', dd)
    # only fundamental frequencies
    freqs = np.array([dd[b] for b in [tuple([e, 1]) for e in labels]])
    # new labels
    funds = dict(zip([str(l - 7) for l in labels], freqs))

    # getting the rest of frequencies?
    combd = dict(zip([tuple([str((t[:3] + t[5:8])[i]-7) for i in range(3) for _ in range((t[:3] + t[5:8])[i + 3])]) for t in data['modes']], data['anharmonic_frequencies']))
    # print('============\n', combd)
    Delta = {tuple(sorted(k)):v for k, v in combd.items() if len(k)>1}

    return funds, Delta

get_fundamentalsc4(picklefilevib='../scriptsHPC/cfourscripts/vibdata.pkl')

labels = sorted(list(set([t[0] for t in data['modes']])))
tlab = [tuple(element for element in t if element != 0) for t in data['modes']]
dd = dict(zip(tlab, data['anharmonic_frequencies']))
freqs = np.array([dd[b] for b in [tuple([e, 1]) for e in labels]])

funds = dict(zip([str(l-7) for l in labels], freqs))

# print(funds, freqs)
# quit()
# set up frequencies for x and y axes

# works
# w1, w2 = np.arange(min(freqs)-70., max(freqs)+150., 25), np.arange(2*min(freqs)-70., 3*max(freqs)+150., 35)
# no
step = 5.
# w1, w2 = np.arange(min(freqs)-290., max(freqs)+280., step), np.arange(2*min(freqs)-320., 3*max(freqs)+40., step)
w1, w2 = np.arange(2660., 2960., step), np.arange(5100., 5700., step)

h = dd_ir.SpectrumEVV(w1, w2, funds, Delta=Delta)

# # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
derivData = h.getDerivs(source={'source':'cfour'})

ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]
# add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
           picks(electric_avrg, ee), picks(mechanical_avrg, mm))

# plot (and save plot) 2D spectrum
w1mw2 = False
mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
coh2_name = f'coh2_{mech}_{mm}_{elec}_{ee}_w1mw2{w1mw2}' if h.coords_abc is not None else f'h2o_{elec}_{ee}'

# h.plot2D(figname=coh2_name, source={'source':'cfour'}, w1mw2=w1mw2, style='contour')
# h.plot2Dplotly(figname=coh2_name, source={'source':'cfour'}, w1mw2=w1mw2, style='scatter', threshold=-100)
# h.plot2Dplotly(figname=coh2_name, source={'source':'cfour'}, w1mw2=w1mw2, style='contour', threshold=-100)
# h.plot2Dseabornheatmap(source={'source':'cfour'}, w1mw2=w1mw2, style='contour')
# h.plot2Dmatplotlib(source={'source':'cfour'}, w1mw2=w1mw2, style='contour')

# f1 = h.plot2Dplotly(source={'source':'cfour'}, w1mw2=w1mw2, style='contour', Gamma=Gamma)

# gammalist = [0.001, 0.9, 4.5] #, 35.7, 69.8, 112.2, 150.4]
# gammalist = [0.9, 2.5] #, 35.7, 69.8, 112.2, 150.4]
gammalist = [0.9] #, 35.7, 69.8, 112.2, 150.4]

figures = []
savedict = {}
for i, g in enumerate(gammalist):
    print(f'----------------------\nnumber {i}')
    Z, savedict = h.totInt('contour', {'source':'cfour'}, g, savedict)
    fig1 = h.plot2Dplotly(Z.T, w1mw2=w1mw2, Gamma=g, percent=0.75, step=step)
    figures.append(fig1)
    # fig2 = h.plot2Dplotly(Z, w1mw2=w1mw2, Gamma=g, percent=0.8750, step=step)
    # figures.append(fig2)
    # fig3 = h.plot2Dplotly(Z, w1mw2=w1mw2, Gamma=g, percent=0.875001, step=step)
    # figures.append(fig3)
    # fig3 = h.plot2Dplotly(Z, w1mw2=w1mw2, Gamma=g, percent=0.875004, step=step)
    # figures.append(fig3)
    h.print2file(coh2_name, w1mw2, 'contour', {'source':'cfour'}, g, step)

top1 = 0.75
np.set_printoptions(linewidth=250, suppress=False, precision=10)

print('\ncccccccccccccccccccccccccccccccccccccccccccc\n')
print(savedict.keys())
print(savedict[list(savedict.keys())[0]].keys())
print('\ncccccccccccccccccccccccccccccccccccccccccccc\n')
print('electrical')
print(savedict['w12660.0_2955.0w25100.0_5695.0_gamma0.9']['electrical'])
print('-------------\n')
print('mechanical')
print(savedict['w12660.0_2955.0w25100.0_5695.0_gamma0.9']['mechanical'])
# quit()

# HTML template
# html_template = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Interactive Plot</title>
#     <!-- Plotly.js -->
#     <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
# </head>
# <body>
# <!--       <h1>My Interactive Plot</h1>  -->
#     <div id="my-plot">
#         {plot_div}
#     </div>
# </body>
# </html>
# """

# HTML template with a three-column layout using inline CSS
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiple Plots</title>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Simple grid layout with two columns taking up 45% each */
        .row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start; /* Align columns to the start of the row */
        }}
        .column {{
            flex: 0 0 40%; /* Do not grow or shrink, base size is 45% */
            padding: 5px;
            box-sizing: border-box;
            # margin-right: 7%; /* Right margin of 10% (adjust as needed) */
        }}
        /* Remove right margin for the last column */
        .column:last-child {{
            margin-right: 0;
        }}
    </style>
</head>
<body>
    <div class="row">
        {plot_divs}
    </div>
</body>
</html>
"""


plot_divs = ""
for i, fig in enumerate(figures):
    include_plotlyjs = 'cdn' if i == 0 else False  # Include Plotly.js only in the first plot
    plot_div = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    plot_divs += f'<div class="column">{plot_div}</div>'

# Insert the plot DIVs into the template
html_content = html_template.format(plot_divs=plot_divs)

# Save the HTML content to a file
with open(f'./picsnew/small_w1mw2{w1mw2}_step{step}_t{top1}_n.html', 'w') as f:
    f.write(html_content)


# # Insert the plot_div into the template
# html_content = html_template.format(plot_div=plot_div)
#
# # Save the HTML content to a file
# with open('plot.html', 'w') as f:
#     f.write(html_content)