import numpy as np
from dataclasses import dataclass, field

@dataclass
class FactorTensor:
    """
    for a term, contains all (a,b)
    """
    term_id: int
    properties: np.ndarray
    ene_denominator: np.ndarray = None
    CFF: np.ndarray = None


@dataclass
class ComponentsLayer:
    """
    factors for product of floats
    """
    term_id: int
    ab_comb: tuple[int,int]
    prefactor: float
    resonance: np.ndarray # resonance term -1,-12
    factor: float # need only one value?


def combine_into_layer(comps_layer: ComponentsLayer):
    """
    prefactor * resonance * properties
    prefactor * resonance * summed_c(properties * ene_denominator * CFF)

    prefactor * resonance * factors --- general
    """
    if comps_layer.factor!=0.:
        return np.where(comps_layer.resonance!=0., comps_layer.prefactor*comps_layer.resonance*comps_layer.factor, 0.)
    else:
        return np.zeros_like(comps_layer.resonance)

# ComponentsLayer(term_id, ab_comb, prefactor, resonance, factor)
# ComponentsLayer --> combine_into_layer(ComponentsLayer) --> layer_data
# term_layers = [layer_data] (all (a,b) for term) --> term_cake = np.stack(term_layers) --> term_amplitudes = term_cake.sum(axis=0) - for 1 term
# cake_layers = [term_layers] (all terms) --> cake = np.stack(term_layers) --> cake_amplitudes = cake.sum(axis=0) - for full spectrum
# cake_layers = [layer_data] (all (a,b) for all terms) --> cake = np.stack(term_layers) --> cake_amplitudes = cake.sum(axis=0) - for full spectrum

def combine_into_cake(layers):
    return np.stack(layers)

def sum_cake(cake):
    return cake.sum(axis=0)

def get_slice(row_start, row_end, col_start, col_end, cake):
    return cake[:, row_start:row_end, col_start:col_end]

def get_slice_rc(w1_start, w1_end, w2_start, w2_end, cake, w1, w2):
    step1 = w1[1]-w1[0]
    step2 = w2[1]-w2[0]

    row_start = int((w1_start-w1[0])//step1)
    row_end = int((w1_end-w1[0])//step1)
    col_start = int((w2_start-w2[0])//step2)
    col_end = int((w2_end-w2[0])//step2)

    print(row_start, row_end, col_start, col_end)

    return cake.T[:, row_start:row_end, col_start:col_end]


def get_slice_w1_y(x_min, x_max, y_min, y_max, cake, w1, w2):

    w1_grid, w2_grid = np.meshgrid(w1, w2, indexing='xy')  # or 'ij' depending on data orientation
    delta_w = w2_grid - w1_grid

    mask = (w1_grid >= x_min) & (w1_grid <= x_max) & (delta_w >= y_min) & (delta_w <= y_max)

    indices = np.argwhere(mask)
    # print(indices)

    if len(indices) == 0:
        raise ValueError("No data points found within the specified bounds.")

    i_min, j_min = indices.min(axis=0)
    i_max, j_max = indices.max(axis=0) + 1  # +1 to include the upper bound

    if cake.ndim == 2:
        return cake[i_min:i_max, j_min:j_max]
    elif cake.ndim == 3:
        return cake[:, i_min:i_max, j_min:j_max]
    else:
        raise ValueError("Cake must be 2D or 3D.")

from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices

gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
corners_cake_w2mw1 = lambda cake: np.array([cake[0][0, 0], cake[0][0, -1], cake[1][0, 0], cake[1][-1, 0]])
high_value_slice_indices = lambda smallcake, threshold: np.where(np.any(np.abs(smallcake) > threshold, axis=(1, 2)))[0]

def dicts_layers_ab(term, deriv_data, states_dict, mode_indices,
                    w1_mesh, w2_mesh, margin, Gamma_rc,
                    w2mw1min, w2mw1max, res_thresh=3e7):
    """
    collect data about layers
    """
    # dict of layers; (a, b): 2darr
    t_layers_baked = {}
    t_layers_instnces = {}
    # index dict; index: (a, b)
    order_dict = {}
    resonances_idx_w2mw1 = {}
    components_dict = {}
    xs, xe, ys, ye = corners_cake_w2mw1(np.stack([w1_mesh, w2_mesh]))

    count = 0
    for a in mode_indices:
        for b in mode_indices:
            w1ab, w2ab = term.get_resonance_location(states_dict, a, b)
            if ((xs + margin <= w1ab <= xe - margin)
                    and (ys + margin <= w2ab <= ye - margin)
                    and (w2ab-margin)>w1ab):
                if term.term_label == 'MECH':
                    factor, dictcomp = term.get_factor_summed(gammaCompsAll, deriv_data, states_dict, mode_indices, a, b)
                    components_dict[(a,b)] = {k:v for k,v in dictcomp.items() if v[0]!=0.}

                elif term.term_label == 'EL':
                    factor, dictcomp = term.get_full_factor(gammaCompsAll, deriv_data, states_dict, a, b, c=None)
                    components_dict[(a,b)] = dictcomp
                w1_r, w2_r = term.get_resonance_location(states_dict, a, b)
                print('factor', a,b, w1_r, w2_r-w1_r, factor)
                # get_res_factor(self, modes_dict, w1_rc, w2_rc, a, b, Gamma_rc, condition=None)
                resres = term.get_res_factor(states_dict, w1_mesh, w2_mesh, a, b, Gamma_rc=Gamma_rc)
                resonance = np.where(np.abs(resres)>res_thresh, resres, 0.)
                # components_dict['facotor'] =
                if abs(factor)>1e-20:
                    if w2_r-w1_r>w2mw1min and w2_r-w1_r<w2mw1max:
                        # ComponentsLayer(term_id, ab_comb, prefactor, resonance, factor)
                        cl = ComponentsLayer(term.term_id, (a, b), term.prefactor_term, resonance, factor)
                        t_layers_instnces[(a,b)] = cl

                        layer_baked = combine_into_layer(cl)
                        t_layers_baked[(a,b)] = layer_baked
                        order_dict[count] = (a,b)
                        resonances_idx_w2mw1[(a, b)] = (w1_r, w2_r-w1_r)
                        count+=1

    return t_layers_baked, t_layers_instnces, order_dict, resonances_idx_w2mw1, components_dict


def indetify_main_layers(order_dict, resonances_idx_w2mw1, smallcake, threshold=0.):
    """
    get info about the main (most contributing) layers in small cake piece
    """
    max_vals_per_slice = {}
    vals = {}
    # go in array of slice/layer indices with high values
    for i in high_value_slice_indices(smallcake, threshold):
        max_val = np.max(np.abs(smallcake[i]))
        # w1_r, w2_r = term.get_resonance_location(original_harmonic, *order_dict[i])
        vals[i] = max_val
        max_vals_per_slice[i] = {'ab': order_dict[i], '(w1,w2mw1)': resonances_idx_w2mw1[order_dict[i]]}

    for k in max_vals_per_slice:
        max_vals_per_slice[k]['value'] = vals[k]
        max_vals_per_slice[k]['val %'] = vals[k]*100/max(vals.values())
    return max_vals_per_slice