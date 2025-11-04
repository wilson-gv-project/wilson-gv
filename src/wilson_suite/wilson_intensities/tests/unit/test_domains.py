from wilson_suite.wilson_intensities.amplitudes import domains


def test_find_domain_groups_by_distance():
    print()

    points = [(1., 3.), (5., 11.), (4., 2.), (12., 6.), (8., 2.), (11., 4.)]

    groups1 = domains.find_clusters_by_distance(points, distance_threshold=10., linkage='single')
    groups2 = domains.find_clusters_by_distance(points, distance_threshold=10., linkage='ward')

    assert len(groups1) == 1
    assert len(groups2) == 3

    groups = domains.find_clusters_by_distance(points, distance_threshold=12., linkage='ward')
    assert len(groups) == 2

    groups1 = domains.find_clusters_by_distance(points, distance_threshold=4., linkage='single')
    groups2 = domains.find_clusters_by_distance(points, distance_threshold=4., linkage='ward')
    assert len(groups1) == 3
    assert len(groups2) == 4
    print(groups2)


def test_find_domain_distance_threshold():
    print()
    domains.find_distance_threshold(1e6, {'A': 3.8, 'B': 3.8})

def test_terms():
    print()
