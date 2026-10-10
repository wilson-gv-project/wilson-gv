from ..builders import show_valid_axis_combs, show_term_latex, make_SpectralAxisSet
import wilson_suite as ws
from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxis, SpectralAxisChoices, SpectralAxisSet, IndependentVariableSet, SignedPulseTuple

evv_exp = ws.fixtures.evv_experiment()
terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
flat_list_orig = ws.utils.termdict_from_symb_term.derived_terms_flat(terms, tolist=True)

axis_choice = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})
translated_terms = ws.derive.term_var_translate.translate_terms_to_axis_variables(flat_list_orig, axis_choice)

flat_dict_orig = ws.utils.termdict_from_symb_term.derived_terms_flat(terms, tolist=False)
flat_list = translated_terms


def test_show_valid_axis_combs():
    print('\n')
    show_valid_axis_combs(evv_exp.valid_axis_combs)

def test_make_SpectralAxisSet():
    """
    1 SignedPulseTuple instance per indep variable - 
        making several SignedPulseTuples per IndependentVariableSet
    """
    # independent vars here are -1 and 2
    axis_choice = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})

    varsA = IndependentVariableSet((SignedPulseTuple((-1,)),))
    axA = SpectralAxis('A', varsA)

    varsB = IndependentVariableSet((SignedPulseTuple((-1,)),SignedPulseTuple((2,))))
    axB = SpectralAxis('B', varsB)

    refaxes = SpectralAxisSet((axA, axB))
    assert axis_choice.axes == refaxes.axes

    # -------------------------------------------------------------
    # independent vars here are -1+2 and 3
    axis_choice = make_SpectralAxisSet({'A': [(-1,2),], 'B': [(-1, 2), 3]})
    print(axis_choice)
    varsA = IndependentVariableSet((SignedPulseTuple((-1,2)),))
    axA = SpectralAxis('A', varsA)

    varsB = IndependentVariableSet((SignedPulseTuple((-1,2)),SignedPulseTuple((3,))))
    axB = SpectralAxis('B', varsB)

    refaxes = SpectralAxisSet((axA, axB))
    assert axis_choice.axes == refaxes.axes


    # independent vars here are -1+2 and 3
    axis_choice = make_SpectralAxisSet({'A': [(-1,2),], 'B': [3,]})
    print(axis_choice)
    varsA = IndependentVariableSet((SignedPulseTuple((-1,2)),))
    axA = SpectralAxis('A', varsA)

    varsB = IndependentVariableSet((SignedPulseTuple((3,)),))
    axB = SpectralAxis('B', varsB)

    refaxes = SpectralAxisSet((axA, axB))
    assert axis_choice.axes == refaxes.axes


def test_show_term_latex():

    for k,v in flat_dict_orig.items():
        print('\n', k)
        print(show_term_latex(v))
