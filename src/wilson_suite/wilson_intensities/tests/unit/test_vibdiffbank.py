# from rich.pretty import pprint
# import numpy as np

# from ...amplitudes.vibene_differences import VibDiffBank
# from ...amplitudes.func_evaluation import get_resonance_loc

# def test_VibDiffBank():
#     # Example dummy energy function
#     def dummy_energy(state):
#         """
#         state is a tuple of integers-indices
#         """
#         return sum(i**2 for i in state)

#     indices = list(range(1, 3*9-5))
#     max_quanta = 3

#     bank = VibDiffBank(indices, max_quanta, dummy_energy)
#     pprint(bank.state_values[(1,)])
#     pprint(bank.state_values[(1,2)])

#     pprint(bank.mode)
#     pprint(bank.get_vibdiff_number('a+b,a', (1, 2, 3, 4, 5)))
#     pprint(dummy_energy((1, 2))-dummy_energy((1,)))

#     pprint(bank.get_vibdiff_number('a+b,a', (2, 4, 3, 5, 1)))
#     pprint(dummy_energy((2, 4))-dummy_energy((2,)))

#     pprint(bank.get_vibdiff_number('a+b,c', (2, 4, 3)))
#     pprint(dummy_energy((2, 4))-dummy_energy((3,)))


# def test_get_resonance_loc():
    
#     def dummy_energy(state):
#         """
#         state is a tuple of integers-indices
#         """
#         return sum(i**2 for i in state)

#     indices = list(range(1, 3*9-5))
#     max_quanta = 3

#     bank = VibDiffBank(indices, max_quanta, dummy_energy)
    
#     result = get_resonance_loc(resonances=(('zero,a', (-1,)), ('a+b,a', (-1, 2))),
#                                ind_tuple=(1, 2, 3), vibdiffbank=bank)
#     assert result == {'w1': np.float64(1.0), 'w2': np.float64(5.0)}

