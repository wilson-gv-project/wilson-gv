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

def find_clusters_by_distance(res_locations: list[tuple], 
                            distance_threshold: list[float]|tuple[float], 
                            linkage="single"):
    """
    Using scikit-learn to cluster points with component-wise distance thresholds.
    
    Args:
        res_locations: List of tuples containing resonance locations
        distance_threshold: List/tuple of thresholds, one for each dimension
        linkage: Clustering linkage method ("single", "complete", "ward")
    """
    from sklearn import cluster
    
    # Convert inputs to numpy arrays with correct shapes
    locations = np.array(res_locations)  # Shape: (n_points, n_dimensions)
    thresholds = np.array(distance_threshold)  # Shape: (n_dimensions,)
    
    # Scale each dimension by its corresponding threshold
    scaled_locations = locations / thresholds[np.newaxis, :]
    
    # Now use a unit threshold since data is pre-scaled
    clustering = cluster.AgglomerativeClustering(
        linkage=linkage,
        distance_threshold=1.0,
        n_clusters=None
    )
    
    labels = clustering.fit_predict(scaled_locations)
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
    multiplier = float(np.sqrt((dynamic_range-1)/dynamic_range))
    gammas = [-1j*G for G in Gamma_axes.values()]
    dist_ax = [G*multiplier for G in Gamma_axes.values()]
    print('dist_ax', dist_ax)
    gamma_prod = np.prod(gammas)
    base_intensity = 1./gamma_prod
    min_to_show = base_intensity/dynamic_range

    return dist_ax
    # raise NotImplementedError('find_distance_threshold not finished yet')


def determine_domains_and_features(features_to_draw: dict[tuple, SpectralFeature],
                                   dynamic_range, Gamma_axes: dict):
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
    features_locs = [tuple(ax[1] for ax in loc_tuple) for loc_tuple in features_to_draw]
    
    dist_threshold = find_distance_threshold(dynamic_range, Gamma_axes)
    
    clusters = find_clusters_by_distance(res_locations=features_locs, 
                                         distance_threshold=dist_threshold, 
                                         linkage='single')
    return list(clusters.values())


def unzip_tuples(list_of_tuples: list[tuple]) -> tuple[tuple]:
    """
    Transform list of n-dimensional tuples into n tuples of corresponding elements.
    
    Example:
        Input: [(1,2,3), (4,5,6), (7,8,9)]
        Output: ((1,4,7), (2,5,8), (3,6,9))
    """
    return tuple(zip(*list_of_tuples))

def draw_domain_bounds(locs_in_domain: list[tuple], margins: dict):
    """
    Examples of locs_in_domain:
        [(1864.0, 900.0)], 
        [(2255.0, 1291.0), (2274.0, 1310.0)], 
        [(2255.0, 1021.0), (2274.0, 1040.0)], 
        [(2368.0, 1134.0), (2360.0, 1126.0), (2362.0, 1128.0)], 
        [(964.0, 0.0)], [(1234.0, 270.0)], 
        [(964.0, -270.0)], [(1234.0, 0.0)]
    """
    dimension_groups = unzip_tuples(locs_in_domain)
    axes_lims = [tuple([min(g), max(g)]) for g in dimension_groups]
    
    return axes_lims

def get_domain_grids(domains_with_features):

    pass