import wilson_suite.wilson_intensities.amplitudes.domains


def test_find_domain_groups_by_distance():
    print()

    points = [[1., 3.], [5., 11.], [4., 2.], [12., 6.], [8., 2.], [11., 4.]]
    print(points, len(points))

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=10.)
    assert len(groups) == 3
    print(groups)

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=12.)
    assert len(groups) == 2
    print(groups)

    groups = wilson_suite.wilson_intensities.amplitudes.domains.find_domain_groups_by_distance(points, distance_threshold=4.)
    assert len(groups) == 4
    print(groups)


def test_find_domain_distance_threshold():
    print()
    wilson_suite.wilson_intensities.amplitudes.domains.find_distance_threshold(1e6, {'A': 3.8, 'B': 3.8})