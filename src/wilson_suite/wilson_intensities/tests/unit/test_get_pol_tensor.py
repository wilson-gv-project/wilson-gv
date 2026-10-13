"""
A15: get_pol_tensor (amplitudes/averaging.py) puts the polarization vectors in reverse order.
Plain tests on get_pol_tensor itself. They fail until the fix (the end of this docstring). No orientation
averaging is needed to see the error.

What get_pol_tensor must give. get_pol_tensor(1.0, pol), reshaped to (3,) * n, is the product of the vectors in
list order:
    A[i0, i1, i2, ...] = pol[0][i0] * pol[1][i1] * pol[2][i2] * ...        index k of A belongs to pol[k]
Three places in the code fix this order:
  1. get_pol_tensor's docstring: the Kronecker product of the vectors. np.kron(a, b)[3 * i + j] = a[i] * b[j]:
     the first vector gets the first index.
  2. The caller. get_pol_laser reshapes A to (3,) * n and reads A at the index tuples of get_iso_f. Position k of
     such a tuple is slot k, e.g. rank 3: f = epsilon, +1 at (0, 1, 2), (1, 2, 0), (2, 0, 1), -1 at the others.
  3. Slot k is pol[k]. VibExperiment.all_polarizations = [detector, pulse 1, pulse 2, ...], and derive numbers the
     slots the same way (op.o: 0 = detector, k = pulse k).

What it gives today: the last vector first, so index k of A belongs to pol[n-1-k]:
    get_pol_tensor(1.0, [a, b, c]) = np.kron(np.kron(c, b), a)          right: np.kron(np.kron(a, b), c)
The cause: the recursion takes pol[len(pol) - 1] first (the last line of get_pol_tensor).
The fix, one line:
    return get_pol_tensor(np.kron(pol_tensor, np.array(pol[0])), pol[1:])

Why the EVV results are right anyway: at rank 2 and 4, get_pol_laser reads A only through deltas (dot products),
which give the same number in either order (test_get_pol_laser_rank_2_is_the_dot_product passes today). At rank 3
it reads A through epsilon (a determinant), whose sign depends on the order: test_get_pol_laser_rank_3_is_the_determinant.
"""

from functools import reduce

import numpy as np
import pytest

from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser, get_pol_tensor

X, Y, Z = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)


def random_vectors(n: int) -> list[np.ndarray]:
    """n polarization vectors with different directions (not unit length; fixed seed)"""
    return list(np.random.default_rng(n).normal(size=(n, 3)))


def test_x_y_z_has_its_one_at_index_0_1_2():
    """
    pol = [x, y, z]. The product is 1 at exactly one index, (x, y, z) = (0, 1, 2), and 0 everywhere else.
    Today the 1 sits at (2, 1, 0): the index of (z, y, x).
    """
    A = get_pol_tensor(1.0, [X, Y, Z]).reshape(3, 3, 3)

    ones = np.argwhere(A != 0).tolist()
    assert ones == [[0, 1, 2]], f'A15: the 1 sits at {ones}, the index of the reversed list z, y, x'


@pytest.mark.parametrize('rank', [2, 3, 4, 5, 6])
def test_is_the_kronecker_product_in_list_order(rank):
    """
    get_pol_tensor(1.0, pol) = np.kron(...np.kron(np.kron(pol[0], pol[1]), pol[2])..., pol[n-1]), as its docstring
    says. Today it is the product of the reversed list, at every rank: rank 2 and 4 included.
    """
    pol = random_vectors(rank)

    A = get_pol_tensor(1.0, pol)

    right, of_reversed_list = reduce(np.kron, pol), reduce(np.kron, pol[::-1])
    assert np.allclose(A, right, rtol=1e-14, atol=0), (
        f'A15: rank {rank}: get_pol_tensor equals the product of the reversed list: '
        f'{np.allclose(A, of_reversed_list, rtol=1e-14, atol=0)}')


def test_get_pol_laser_rank_3_is_the_determinant():
    """
    The consequence. Rank 3: get_pol_laser = sum of epsilon_ijk * A[i, j, k] = a . (b x c) = det(a, b, c),
    the triple product of the three vectors in list order (a = detector, b = pulse 1, c = pulse 2).
    Today det(c, b, a) = -det(a, b, c): the sign flips, e.g. x, y, z gives [-1.0] instead of [+1.0].
    """
    a, b, c = random_vectors(3)

    value = get_pol_laser([a, b, c])

    det = float(np.linalg.det(np.array([a, b, c])))
    assert value == pytest.approx([det], rel=1e-12, abs=0), f'A15: get_pol_laser {value}, det(a, b, c) = {det:+.6f}'


def test_get_pol_laser_rank_2_is_the_dot_product():
    """
    Control, passes today. Rank 2: get_pol_laser = sum of delta_ij * A[i, j] = a . b, and a . b = b . a,
    so the reversed product gives the right number. The same holds at rank 4: the EVV terms are right.
    """
    a, b = random_vectors(2)

    assert get_pol_laser([a, b]) == pytest.approx([a @ b], rel=1e-12, abs=0)
