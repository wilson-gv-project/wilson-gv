"""
DOMAINS of RESONANCE LOCATIONS
"""
import numpy as np
from typing import TYPE_CHECKING, Callable

# from .spectrum_composition import RectangularDomain, ResLocGeoObject, SpectralWindow
if TYPE_CHECKING:
    from .spectrum_composition import SpectralFeature, RectangularDomain, Box

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
    if len(res_locations) == 1:
        return {0: res_locations}
    
    from sklearn import cluster
    
    # Convert inputs to numpy arrays with correct shapes
    locations = np.array(res_locations)  # Shape: (n_points, n_dimensions)
    thresholds = np.array(list(distance_thresholds.values()))  # Shape: (n_dimensions,)

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

# def find_feature_clusters_by_distance(features: dict['GeometricObject', 
#                                                      tuple[float, 'SpectralFeature']], 
#                                       distance_thresholds: dict[str, float], 
#                                       spec_window_full: 'SpectralWindow',
#                                       linkage: str = "single") -> dict[int, 'RectangularDomain']:
#     """
#     take features dict, find clusters of them based on location and distances

#     return a dict of int key and SpectralWindow instance value
#     """
#     features_locs = {loc_geo_obj.values:features[loc_geo_obj] for loc_geo_obj in features}

#     clusters = find_points_clusters_by_distance(res_locations=list(features_locs.keys()), 
#                                          distance_thresholds=distance_thresholds,
#                                          linkage=linkage)
#     rec_windows_dict = {}
    
#     for g in clusters:
#         domain = RectangularDomain.from_features([features_locs[i] for i in clusters[g]])
#         domain.bounds.intersect(spec_window_full)
#         rec_windows_dict[g] = domain

#     return rec_windows_dict

from typing import List, Tuple, Dict, Union, Optional
# def compute_box_adjacency(boxes: List[List[Tuple[float, float]]]) -> np.ndarray:
def compute_box_adjacency(boxes: List['Box']) -> np.ndarray:
    """
    Compute adjacency for N-dimensional rectangular boxes (bounds given as min/max per dimension).
    """
    n = len(boxes)
    adjacency = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            overlap = all(
                max_i >= min_j and max_j >= min_i
                for (min_i, max_i), (min_j, max_j) in zip(boxes[i], boxes[j])
            )
            if overlap:
                adjacency[i, j] = adjacency[j, i] = True

    return adjacency

def compute_box_adjacency(
    boxes: List["Box"],
    *,
    touch_inclusive: bool = True,
    axis_order: Optional[Tuple[str, ...]] = None,
) -> np.ndarray:
    """
    Minimal adjacency for axis-labeled Boxes.
    Assumes all boxes are comparable on the given axis_order.
    No axis compatibility checks are performed here.
    Parameters:
      boxes: list[Box]
      touch_inclusive: True => touching counts as adjacent (>=); False => strict overlap only (>)
      axis_order: optional explicit axis order to use. If None, uses boxes[0].axes.
    Returns:
      n x n boolean adjacency matrix
    """
    n = len(boxes)
    adj = np.zeros((n, n), dtype=bool)
    if n == 0:
        return adj
    
    axes = axis_order if axis_order is not None else boxes[0].axes
    
    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            bj = boxes[j]
            overlap = True
            for ax in axes:
                mn_i, mx_i = bi.bounds[ax]
                mn_j, mx_j = bj.bounds[ax]
                if touch_inclusive:
                    if not (mx_i >= mn_j and mx_j >= mn_i):
                        overlap = False
                        break
                else:
                    if not (mx_i > mn_j and mx_j > mn_i):
                        overlap = False
                        break
            if overlap:
                adj[i, j] = adj[j, i] = True
    return adj

# def points_to_bounds(points: List[Dict[str,float]], 
#                      halfwidths_list: List[Dict[str,float]]) -> List[Tuple[Tuple[float,float]]]:
#     axes = list(points[0].keys())
#     return [
#         tuple([(p[axis]-hw[axis], p[axis]+hw[axis]) for axis in axes])
#         for p, hw in zip(points, halfwidths_list)
#     ]

def points_to_bounds(points: List[Dict[str,float]], 
                     halfwidths_list: List[Dict[str,float]]) -> List[Tuple[Tuple[float,float]]]:
    axes = list(points[0].keys())
    return [
        {axis: (p[axis]-hw[axis], p[axis]+hw[axis]) for axis in p}
        for p, hw in zip(points, halfwidths_list)
    ]

def make_domains_from_feat_clusters(clusters: dict[int, list['SpectralFeature']]):
    """
    1. make bounds for each cluster
    2. adjacency matrix for cluster boxes
    3. domains identification - clustering of clusters
    """

    return


# ----------------------------------------------
def connected_components_from_adjacency(adjacency: np.ndarray, box_objects: list) -> dict[int, list]:
    """
    Generic DFS-based connected component finder.
    adjacency: n x n boolean matrix
    objects: list of objects corresponding to rows of adjacency
    Returns: dict[label] -> list of objects
    """
    n = len(box_objects)
    visited = np.zeros(n, dtype=bool)
    clusters = {}
    label_counter = 0

    for i in range(n):
        if not visited[i]:
            stack = [i]
            members = []
            while stack:
                k = stack.pop()
                if visited[k]:
                    continue
                visited[k] = True
                members.append(box_objects[k])
                neighbors = np.where(adjacency[k])[0]
                stack.extend(neighbors)
            clusters[label_counter] = members
            label_counter += 1
    return clusters

def features_to_clusters(features: list['SpectralFeature']) -> dict[int, list['SpectralFeature']]:
    """
    """
    # points_from_features = [feat.location._coord_dict for feat in features]
    # halfwidths_list_from_features = [feat.lineshape_parameter for feat in features]
    # feature_boxes = points_to_bounds(points_from_features, halfwidths_list_from_features)
    feature_boxes = [f.feat_box for f in features]
    feature_ls = [f.lineshape_parameter for f in features]
    # print('feature_boxes', feature_boxes)
    # print('feature_ls', feature_ls)
    adjacency = compute_box_adjacency(feature_boxes)

    return connected_components_from_adjacency(adjacency, features)


def feat_clusters_to_domains(clusters: list['RectangularDomain']) -> dict[int, list['RectangularDomain']]:
    """
    """
    cluster_boxes = [c.box.bounds for c in clusters]
    print('\ncluster_boxes', cluster_boxes)
    cluster_adjacency = compute_box_adjacency(cluster_boxes)
    print('\ncluster_adjacency', cluster_adjacency)
    return connected_components_from_adjacency(cluster_adjacency, clusters)

# --------------------------------------------

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

def cut_grid_with_indices_dict_nd(grid: dict[str, np.ndarray], 
                                  domains: list['RectangularDomain']) -> dict['RectangularDomain', dict]:
    """
    General N-dimensional version.
    
    Given:
        grid: dict mapping axis names (e.g., 'A', 'B', 'C') to np.ndarray grids of identical shape
        domains: list of objects, each with .box.bounds dict {axis_name: (min, max)}

    Returns:
        subgrids: dict mapping each domain.box -> {
            "grid": {axis_name: subarray},
            "indices": tuple(slice_i, slice_j, ...)
        }
    """
    # get axis names and coordinate arrays
    axes = list(grid.keys())
    shape = next(iter(grid.values())).shape
    
    # Each axis has coordinates varying along its dimension
    axis_coords = {}
    for ax in axes:
        # Take unique coordinate values along that axis
        # Find the dimension where variation occurs
        arr = grid[ax]
        axis_dim = np.argmin([np.allclose(np.take(arr, 0, i), arr) for i in range(arr.ndim)])
        axis_coords[ax] = np.unique(np.moveaxis(arr, axis_dim, 0)[..., 0])

    subgrids = {}

    for domain in domains:
        box = domain.box
        bounds = box.bounds

        slices = []
        for ax in axes:
            coords = axis_coords[ax]
            mn, mx = bounds[ax]

            i_min = np.searchsorted(coords, mn, side="right") - 1
            i_max = np.searchsorted(coords, mx, side="left")
            i_min = max(i_min, 0)
            i_max = min(i_max, len(coords) - 1)

            slices.append(slice(i_min, i_max + 1))

        # slice the subgrids
        subgrid = {ax: grid[ax][tuple(slices)] for ax in axes}
        subgrids[domain] = {"grid": subgrid, "indices": tuple(slices)}

    return subgrids

def insert_results_to_grid_nd(grid: dict[str, np.ndarray],
                              subgrids: dict['RectangularDomain', dict],
                              result_func: Callable,
                              result_key="result") -> None:
    """
    Compute results for each subgrid and place them back into the full grid.
    
    Args:
        grid: main grid dict (each axis -> np.ndarray)
        subgrids: output of cut_grid_with_indices_dict_nd
        result_func: callable(subgrid_dict) -> np.ndarray of same shape as subgrid
        result_key: str, key to store in main grid dict
    
    Returns:
        None (updates grid[result_key] in place)
    """
    first_axis = next(iter(grid.values()))
    if result_key not in grid:
        grid[result_key] = np.zeros_like(first_axis, dtype=float)

    for domain, info in subgrids.items():
        subgrid = info["grid"]
        indices = info["indices"]

        # Compute result for this subgrid
        result_sub = result_func(subgrid)
        grid[result_key][indices] += result_sub
