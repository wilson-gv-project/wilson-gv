"""
DOMAINS of RESONANCE LOCATIONS
"""
import numpy as np


def find_domain_groups_by_distance(res_locations, distance_threshold):
    """
    using scikit-learn to cluster points with distance threshold
    """
    from sklearn import cluster
    ward = cluster.AgglomerativeClustering(linkage="ward",
                                           distance_threshold=distance_threshold,
                                           n_clusters=None)
    labels = ward.fit_predict(res_locations)
    groups: dict[int, list] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(res_locations[i])
    return groups


def find_distance_threshold(dynamic_range, Gamma_axes: dict):
    """
    Gamma is a dictionary {'A': Gamma_A, 'B': Gamma_B, ...}

    at Gamma   - 1/2 of maximum
    at Gamma/2 - 4/5 of maximum
    """
    multiplier = np.sqrt((dynamic_range-1)/dynamic_range)
    gammas = [-1j*G for G in Gamma_axes.values()]
    dist_ax = [G*multiplier for G in Gamma_axes.values()]
    print(dist_ax)
    gamma_prod = np.prod(gammas)
    base_intensity = 1./gamma_prod
    min_to_show = base_intensity/dynamic_range

    raise NotImplementedError('find_distance_threshold not finished yet')


def determine_domains_and_features(features_to_draw):
    """
    features_to_draw is a dict:
        {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
                (500., 1400.): [(1, 4)], ...},
        motif 2: {}}

    features_to_draw[i][(state_tuple), (location_tuple)] for i in res_motifs = coeff as float
    """
    return


def get_domain_grids(domains_with_features):

    pass