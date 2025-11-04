"""
DOMAINS of RESONANCE LOCATIONS
"""
import numpy as np
from ..amplitudes.term_parts import SpectralFeature

def find_clusters_by_distance(res_locations: list[tuple], 
                              distance_threshold: float, linkage="single"):
    """
    using scikit-learn to cluster points with distance threshold

    linkage="single", "complete", "ward"

    Use single if you want to ensure points are connected through a chain of nearby points
    Use ward if you want more compact, equally-sized clusters
    Use complete if you want to ensure ALL points within a cluster are within the threshold distance of each other
    """
    from sklearn import cluster
    clustering = cluster.AgglomerativeClustering(linkage=linkage,
                                                 distance_threshold=distance_threshold,
                                                 n_clusters=None)
    labels = clustering.fit_predict(res_locations)
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


def determine_domains_and_features(features_to_draw: dict[tuple, SpectralFeature]):
    """
    features_to_draw is a dict:
        {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
                (500., 1400.): [(1, 4)], ...},
        motif 2: {}}

    features_to_draw[i][(state_tuple), (location_tuple)] for i in res_motifs = coeff as float

 {
(('A', 964.0), ('B', 0.0)): (-544.5807923279399, SpectralFeature(location=(('A', 964.0), ('B', 0.0)), term_contributions=[TermParametersChoice(term_keys=(3269843836520877394,), 
                                                                                                                    states_parameters=(ParameterSet({'a': 0, 'b': 0}),))])), 
(('A', 1234.0), ('B', 270.0)): (0.0, SpectralFeature(location=(('A', 1234.0), ('B', 270.0)), term_contributions=[TermParametersChoice(term_keys=(3269843836520877394,), 
                                                                                                                    states_parameters=(ParameterSet({'a': 0, 'b': 1}), ParameterSet({'a': 0, 'b': 2})))])), 
(('A', 964.0), ('B', -270.0)): (0.0, SpectralFeature(location=(('A', 964.0), ('B', -270.0)), term_contributions=[TermParametersChoice(term_keys=(3269843836520877394,), 
                                                                                                                    states_parameters=(ParameterSet({'a': 1, 'b': 0}), ParameterSet({'a': 2, 'b': 0})))])), 
 }     
    """
    return


def get_domain_grids(domains_with_features):

    pass