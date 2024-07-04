#!/usr/bin/env python
import time
import pandas as pd
import numpy as np
def plot2Dmatplotlib(X, Y, Z, w1mw2, name, Gamma, dpi=500, contour_levels=100, log10=True, shift_scale=None):
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import Normalize

    matplotlib.use('Agg')
    plt.rcParams['path.simplify'] = True
    plt.rcParams['agg.path.chunksize'] = 10000
    # X, Y = self.w1_mesh, self.w2_mesh
    # if w1mw2:
    #     y = -(X - Y)
    #     ystr = 'w2-w1'
    #     ystr_mesh = ystr + '_'
    # else:
    #     y = Y
    #     ystr = 'w2'
    #     ystr_mesh = ystr + '_'
    # Z_positive = abs(Z) ** 2
    # if log10:
    #     Z_data = np.log10(Z_positive)
    # else:
    #     Z_data = Z_positive
    #
    # df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': Z_data}
    from matplotlib.colors import ListedColormap
    # Define the original levels from the image
    # original_levels = [-21, -18, -15, -12, -9, -6]

    # New levels to group lower values together
    # new_levels = [-18, -15, -12, -9, -6]
    new_levels = [-18, -9, -6]
    # new_levels = np.linspace(-18, -6, 4)

    # Extract the original colormap
    # cmap = plt.get_cmap('viridis')
    # Define a custom colormap
    colors = ['white', 'red']
    new_cmap = ListedColormap(colors)

    # Extract the top 4 colors from the original colormap (grouping the lowest levels)
    # top_colors = cmap(np.linspace(0.5, 1, len(new_levels) - 1))

    # Create a new colormap with the top 4 colors
    # new_cmap = ListedColormap(top_colors)

    plt.figure(figsize=(12, 11))

    if shift_scale is not None:
        import matplotlib.colors as colors
        class SkewNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, skew_factor=2, clip=False):
                self.skew_factor = skew_factor
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                normalized_value = super().__call__(value, clip)
                return np.minimum(normalized_value ** self.skew_factor, 1)

        norm = SkewNormalize(vmin=np.log10(Z).min(), vmax=np.log10(Z).max(), skew_factor=shift_scale)
    else:
        norm = None

    start_time = time.time()
    # cont = plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', norm=norm)
    cont = plt.contourf(X, Y, Z, levels=new_levels, cmap=new_cmap, norm=norm)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time - plt.contourf: {execution_time} seconds")

    # This is the fix for the white lines between contour levels
    for c in cont.collections:
        c.set_edgecolor("face")
    # plt.colorbar()  # Add a colorbar to show the Z scale
    cbar = plt.colorbar(cont, ticks=new_levels)
    cbar.ax.set_yticklabels(new_levels)

    plt.xlabel(r'$\omega_1$, cm-1')
    plt.ylabel(r'$\omega_2$, cm-1')
    xs = X[0], X[-1]
    ys = Y[0], Y[-1]
    plt.title(
        f'plot2Dmatplotlib().\ndpi={dpi} clevels={contour_levels} x{xs[0][0]}..{xs[-1][-1]} y{ys[0][0]}..{ys[-1][-1]}\n{name}\nGamma={Gamma}')

    start_time = time.time()
    plt.savefig(name, dpi=dpi, format='svg')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time - plt.savefig: {execution_time} seconds")

def plot2Dmatplotlib_rawZ(X, Y, Z, w1mw2, name, Gamma, dpi=500, contour_levels=100, log10=True, shift_scale=None):
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import Normalize

    matplotlib.use('Agg')
    plt.rcParams['path.simplify'] = True
    plt.rcParams['agg.path.chunksize'] = 10000
    # X, Y = self.w1_mesh, self.w2_mesh
    if w1mw2:
        y = -(X - Y)
        ystr = 'w2-w1'
        ystr_mesh = ystr + '_'
    else:
        y = Y
        ystr = 'w2'
        ystr_mesh = ystr + '_'

    print(X)
    print(type(X), type(Y), type(Z))
    # print(Z)
    Z_positive = np.abs(Z) ** 2
    if log10:
        Z_data = np.log10(Z_positive)
    else:
        Z_data = Z_positive

    df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': Z_data}
    from matplotlib.colors import ListedColormap

    # new_levels = [-18, -9, -6]
    # # new_levels = np.linspace(-18, -6, 4)
    # colors = ['white', 'red']
    # new_cmap = ListedColormap(colors)

    # Extract the original colormap
    # cmap = plt.get_cmap('viridis')


    # Extract the top 4 colors from the original colormap (grouping the lowest levels)
    # top_colors = cmap(np.linspace(0.5, 1, len(new_levels) - 1))

    # Create a new colormap with the top 4 colors
    # new_cmap = ListedColormap(top_colors)

    plt.figure(figsize=(12, 11))

    if shift_scale is not None:
        import matplotlib.colors as colors
        class SkewNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, skew_factor=2, clip=False):
                self.skew_factor = skew_factor
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                normalized_value = super().__call__(value, clip)
                return np.minimum(normalized_value ** self.skew_factor, 1)

        norm = SkewNormalize(vmin=np.log10(Z).min(), vmax=np.log10(Z).max(), skew_factor=shift_scale)
    else:
        norm = None

    start_time = time.time()
    cont = plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', norm=norm)
    # cont = plt.contourf(X, Y, Z_data, levels=new_levels, cmap=new_cmap, norm=norm)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time - plt.contourf: {execution_time} seconds")

    # This is the fix for the white lines between contour levels
    for c in cont.collections:
        c.set_edgecolor("face")
    plt.colorbar()  # Add a colorbar to show the Z scale
    # cbar = plt.colorbar(cont, ticks=new_levels)
    # cbar.ax.set_yticklabels(new_levels)

    plt.xlabel(r'$\omega_1$, cm-1')
    plt.ylabel(r'$\omega_2$, cm-1')
    xs = X[0], X[-1]
    ys = Y[0], Y[-1]
    plt.title(
        f'plot2Dmatplotlib().\ndpi={dpi} clevels={contour_levels} x{xs[0][0]}..{xs[-1][-1]} y{ys[0][0]}..{ys[-1][-1]}\n{name}\nGamma={Gamma}')

    start_time = time.time()
    plt.savefig(name, dpi=dpi, format='svg')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time - plt.savefig: {execution_time} seconds")

# Custom function to convert string representation of complex numbers to complex type
def to_complex(val):
    try:
        return complex(val.strip('()'))
    except ValueError:
        return np.nan

dir = '/home/vlew/mock2D/spectra/hcoh_HFccpVQZ_Gaussian/svgs/'
datafile_eFmT_0p70 = 'newgammaopt2_w1mw2F_elF_mechT_log10T_gamma5p00_st10p70.csv'
datafile_eTmF_0p70 = 'newgammaopt2_w1mw2F_elT_mechF_log10T_gamma5p00_st10p70.csv'
datafile_eFmT_0p75 = 'newgammaopt2_w1mw2F_elF_mechT_log10T_gamma5p00_st10p75.csv'
datafile_eTmF_0p75 = 'newgammaopt2_w1mw2F_elT_mechF_log10T_gamma5p00_st10p75.csv'

dir2 = '/home/vlew/mock2D/spectra/hcoh_HFccpVQZ_Gaussian/'
d1 = "w1mw2F_elT_mechF_log10T_gamma3p00_st10p70_x1180t2150_y2310t5350_pureD.csv"
d2 = "w1mw2F_elF_mechT_log10T_gamma3p00_st10p70_x1180t2150_y2310t5350_pureD.csv"
d3 = "w1mw2F_elT_mechT_log10T_gamma3p00_st10p70_x1180t2150_y2310t5350_pureD.csv"

# start1, stop1, step1 = 1080., 2450., 0.7
# start2, stop2, step2 = 2609., 6200., 0.7
# omega1 = np.arange(start1, stop1, step1)
# omega2 = np.arange(start2, stop2, step2)
# x, y = np.meshgrid(omega1, omega2, indexing='ij')
# shp1 = x.shape

start1, stop1, step1 = 1180., 2150., 0.7
start2, stop2, step2 = 2310., 5350., 0.7
omega1 = np.arange(start1, stop1, step1)
omega2 = np.arange(start2, stop2, step2)
x, y = np.meshgrid(omega1, omega2, indexing='ij')
shp1 = x.shape

# fname = datafile_eTmF_0p70      # <-------------------------
# df = pd.read_csv(dir+fname)
# X = df['x'].values.reshape(shp1)
# Y = df['y'].values.reshape(shp1)
# Z = df['z'].values.reshape(shp1)
#
# print(X.shape, Y.shape, Z.shape)
#
# w1mw2=False # fixed in input_data_info
# log10=True # fixed in input_data_info
# contour_levels = 6
# dpi = 200
# pref = 'upd2_'
# plot2Dmatplotlib(X, Y, Z, w1mw2=w1mw2, name=pref+f'clev{contour_levels}_dpi{dpi}'+fname[:-4]+'.svg', Gamma=5.0,
#                  dpi=dpi, contour_levels=contour_levels, log10=log10, shift_scale=None)

fname = d1      # <-------------------------
df = pd.read_csv(dir2+fname)
# Convert the 'w1', 'w2', and 'z' columns to complex numbers
df['w1'] = df['w1'].apply(to_complex)
df['w2'] = df['w2'].apply(to_complex)
df['z'] = df['z'].apply(to_complex)

X = df['w1'].values.reshape(shp1)
Y = df['w2'].values.reshape(shp1)
Z = df['z'].values.reshape(shp1)

print(X.shape, Y.shape, Z.shape)
w1mw2=False
log10=True
contour_levels = 6
dpi = 200
pref1= 'rawZ_'
plot2Dmatplotlib_rawZ(X, Y, Z, w1mw2=w1mw2, name=pref1+f'clev{contour_levels}_dpi{dpi}'+fname[:-4]+'.svg', Gamma=5.0,
                 dpi=dpi, contour_levels=contour_levels, log10=log10, shift_scale=None)

print('\n-----------------------------------')
print('plotted first one\n')

# w1mw2=False # fixed in input_data_info
# log10=True # fixed in input_data_info
# contour_levels = 4
# dpi = 200
# plot2Dmatplotlib(X, Y, Z, w1mw2=w1mw2, name=pref+f'clev{contour_levels}_dpi{dpi}'+fname[:-4]+'.svg', Gamma=5.0,
#                  dpi=dpi, contour_levels=contour_levels, log10=log10, shift_scale=None)
