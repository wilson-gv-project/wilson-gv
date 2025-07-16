import numpy as np

# generated derivatives data with the generateDerivs() which is below
mu_Q = np.array([[0.5864, 0.    , 0.6642],
                   [0.    , 0.    , 0.5725],
                   [0.8658, 0.3108, 0.0144],
                   [0.1663, 0.    , 0.1712]])
mu_QQ = np.array([[[0.1162, 0.    , 0.    ],
                    [0.    , 0.6974, 0.    ],
                    [0.    , 0.6685, 0.    ],
                    [0.4136, 0.    , 0.8808]],

                   [[0.    , 0.6974, 0.    ],
                    [0.    , 0.7685, 0.    ],
                    [0.1628, 0.    , 0.    ],
                    [0.    , 0.9276, 0.018 ]],

                   [[0.    , 0.6685, 0.    ],
                    [0.1628, 0.    , 0.    ],
                    [0.    , 0.    , 0.475 ],
                    [0.0595, 0.    , 0.    ]],

                   [[0.4136, 0.    , 0.8808],
                    [0.    , 0.9276, 0.018 ],
                    [0.0595, 0.    , 0.    ],
                    [0.3646, 0.    , 0.    ]]])
alpha_Q = np.array([[[0.    , 0.    , 0.2897],
                        [0.    , 0.3714, 0.    ],
                        [0.2897, 0.    , 0.4978]],

                       [[0.    , 0.    , 0.3845],
                        [0.    , 0.    , 0.587 ],
                        [0.3845, 0.587 , 0.    ]],

                       [[0.0164, 0.0683, 0.    ],
                        [0.0683, 0.1326, 0.    ],
                        [0.    , 0.    , 0.    ]],

                       [[0.7922, 0.    , 0.    ],
                        [0.    , 0.    , 0.    ],
                        [0.    , 0.    , 0.    ]]])
alpha_QQ = np.array([[[[0.5223, 0.2507, 0.6181],
                         [0.2507, 0.7552, 0.    ],
                         [0.6181, 0.    , 0.    ]],

                        [[0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.2471],
                         [0.    , 0.2471, 0.    ]],

                        [[0.    , 0.    , 0.3841],
                         [0.    , 0.    , 0.    ],
                         [0.3841, 0.    , 0.0497]],

                        [[0.    , 0.2646, 0.0646],
                         [0.2646, 0.    , 0.    ],
                         [0.0646, 0.    , 0.    ]]],


                       [[[0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.2471],
                         [0.    , 0.2471, 0.    ]],

                        [[0.    , 0.    , 0.    ],
                         [0.    , 0.2594, 0.3421],
                         [0.    , 0.3421, 0.    ]],

                        [[0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.3795]],

                        [[0.    , 0.    , 0.4842],
                         [0.    , 0.0128, 0.    ],
                         [0.4842, 0.    , 0.    ]]],


                       [[[0.    , 0.    , 0.3841],
                         [0.    , 0.    , 0.    ],
                         [0.3841, 0.    , 0.0497]],

                        [[0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.    ],
                         [0.    , 0.    , 0.3795]],

                        [[0.    , 0.4965, 0.    ],
                         [0.4965, 0.    , 0.    ],
                         [0.    , 0.    , 0.    ]],

                        [[0.    , 0.0864, 0.154 ],
                         [0.0864, 0.5992, 0.3504],
                         [0.154 , 0.3504, 0.3127]]],


                       [[[0.    , 0.2646, 0.0646],
                         [0.2646, 0.    , 0.    ],
                         [0.0646, 0.    , 0.    ]],

                        [[0.    , 0.    , 0.4842],
                         [0.    , 0.0128, 0.    ],
                         [0.4842, 0.    , 0.    ]],

                        [[0.    , 0.0864, 0.154 ],
                         [0.0864, 0.5992, 0.3504],
                         [0.154 , 0.3504, 0.3127]],

                        [[0.    , 0.    , 0.1261],
                         [0.    , 0.6995, 0.316 ],
                         [0.1261, 0.316 , 0.087 ]]]])


def random_with_zeros(shape: tuple, zero_prob: float=0.5) -> np.ndarray:
    """
    Chatgpt

    """
    # Generate a mask with True at places that should be zero, based on zero_prob
    mask = np.random.rand(*shape) < zero_prob
    # Generate random values in the desired shape
    random_values = np.random.rand(*shape)
    # Apply the mask: set values to zero where mask is True
    random_values[mask] = 0.
    return random_values


def generateDerivs() -> None:
    """
    Generate derivatives data

    mu_Q, mu_QQ, alpha_Q, alpha_QQ
    """
    # (4, 3) array
    mu_Q = random_with_zeros((4, 3), zero_prob=0.55)
    # (4, 4, 3) array with symmetric (4, 3) slices
    mu_QQ = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(i, 4):  # Only fill upper triangle for symmetry
            values = random_with_zeros((3,), zero_prob=0.6)

            mu_QQ[i, j, :] = values
            mu_QQ[j, i, :] = values  # Make it symmetric

    # (4, 3, 3) array with symmetric (3, 3) sub-arrays
    alpha_Q = np.zeros((4, 3, 3))
    for i in range(4):
        matrix = random_with_zeros((3,3), zero_prob=0.65)
        symmetric_matrix = (matrix + matrix.T) / 2  # Make it symmetric
        alpha_Q[i] = symmetric_matrix

    # (4, 4, 3, 3) array with symmetric (3, 3) sub-arrays in both the last two dimensions
    alpha_QQ = np.zeros((4, 4, 3, 3))
    for i in range(4):
        for j in range(i, 4):  # Only fill upper triangle for symmetry
            matrix = random_with_zeros((3,3), zero_prob=0.7)
            symmetric_matrix = (matrix + matrix.T) / 2  # Make it symmetric
            alpha_QQ[i, j] = symmetric_matrix
            alpha_QQ[j, i] = symmetric_matrix  # Make it symmetric in both (4,4) and (3,3)

    print('\n')
    np.set_printoptions(precision=4)
    print(repr(mu_Q))
    print(repr(mu_QQ))
    print(repr(alpha_Q))
    print(repr(alpha_QQ))
