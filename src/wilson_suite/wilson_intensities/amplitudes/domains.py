"""
DOMAINS of RESONANCE LOCATIONS
"""
import numpy as np
from ..amplitudes.term_parts import SpectralFeature, GeometricObject, RectangularWindow, SpectralWindow

def find_points_clusters_by_distance(res_locations: list[tuple], 
                            distance_thresholds: dict[str, float], 
                            linkage="single") -> dict[int, list[tuple]]:
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
    thresholds = np.array(list(distance_thresholds.values()))  # Shape: (n_dimensions,)
    print('thresholds', thresholds)
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

def find_feature_clusters_by_distance(features: dict[GeometricObject, 
                                                     tuple[float, SpectralFeature]], 
                                      distance_thresholds: dict[str, float], 
                                      window_type: str = 'rectangular',
                                      linkage: str = "single") -> dict[int, RectangularWindow]:
    """
    take features dict, find clusters of them based on location and distances

    return a dict of int key and SpectralWindow instance value
    """
    features_locs = {loc_geo_obj.values:features[loc_geo_obj][1] for loc_geo_obj in features}

    clusters = find_points_clusters_by_distance(res_locations=list(features_locs.keys()), 
                                         distance_thresholds=distance_thresholds,
                                         linkage=linkage)
    rec_windows_dict = {}
    
    if window_type == 'rectangular':
        window_class = RectangularWindow
    else:
        raise NotImplementedError('Other kinds of SpectralWildow are not implemented')
    
    for g in clusters:
        rec_windows_dict[g] = window_class.from_features([features_locs[i] for i in clusters[g]])

    return rec_windows_dict

def get_distance_threshold(dynamic_range: float|int, Gamma_axes: dict) -> dict:
    """
    Gamma is a dictionary {'A': Gamma_A, 'B': Gamma_B, ...}

    at Gamma   - 1/2 of maximum
    at Gamma/2 - 4/5 of maximum
    """
    multiplier = float(np.sqrt((dynamic_range-1.)/dynamic_range))
    # gammas = [-1j*G for G in Gamma_axes.values()]
    # dist_ax = [G*multiplier for G in Gamma_axes.values()]
    dist_ax = {ax: G*multiplier for ax, G in Gamma_axes.items()}

    # gamma_prod = np.prod(gammas)
    # base_intensity = 1./gamma_prod
    # min_to_show = base_intensity/dynamic_range

    return dist_ax
    # raise NotImplementedError('find_distance_threshold not finished yet')


def get_domain_grids(domains_with_features):

    pass