from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_intensities.amplitudes.term_parts import VibStatesData
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject


def get_resonances_report(derived_terms: dict, 
                          axis_set_choice: 'SpectralAxisSet' = None, 
                          vib_states_data: 'VibStatesData' = None):
    """
    take terms, optionally - a choice of axes

    1. printing resonance part of the terms
    2. 
    """
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat

    print('\nderived_terms\n')
    flat_terms: dict[str,'VibPerturbedTerm'] = derived_terms_flat(derived_terms)
    for id, term in flat_terms.items():
        print('&'+term.to_latex(part='res') + r' \\')

    if axis_set_choice is not None:
        from wilson_suite.wilson_derive.term_var_translate import translate_terms_to_axis_variables
        print('\n', axis_set_choice, '\n')

        terms_in_axis_choice = translate_terms_to_axis_variables(derived_terms, axis_set_choice)
        print('\n---------')
        print('\nterms_dict_in_axis_choice\n')
        flat_terms_dict_in_axis_choice: dict[str,'VibPerturbedTerm'] = derived_terms_flat(terms_in_axis_choice)
        for id, term in flat_terms_dict_in_axis_choice.items():
            print('&'+term.to_latex(part='res') + r' \\')
    
    from wilson_suite.wilson_intensities.amplitudes.resonances import find_resonance_locations_wrt_index_choices
    from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif
    
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    cache = VibDiffCache()


    if vib_states_data is not None:
        terms_selection = ['0_(1, 0)', '8_(0, 1)']
        select_dict = {k: flat_terms_dict_in_axis_choice[k]  for k in terms_selection}
        motifs = []

        all_res_locs = {}

        for id, term in select_dict.items():
            print('\n', id, '\n', term.res, '\n')
            motifs.append(ResonanceMotif(term.res))
            # ResonanceMotif key - dict val [ResLocGeoObject key - list val]
            rl_dict = find_resonance_locations_wrt_index_choices(motif=ResonanceMotif(term.res), 
                                                                 vibdiff_cache=cache, vibstates_data=vib_states_data)
            res_locs = {}
            for res_motif, dict1 in rl_dict.items():
                for res_loc_obj, list_params in dict1.items():
                    if res_loc_obj not in res_locs:
                        res_locs[res_loc_obj] = [(res_motif, list_params)]
                    else:
                        print('yo')
                        res_locs[res_loc_obj].append((res_motif, list_params))
            all_res_locs[id] = list(res_locs.keys())
        
        for tid, reslocs in all_res_locs.items():
            plot_reslocs(reslocs, title=tid, axis_set=axis_set_choice)

    else:
        print("\nNo vib states data provided. Cannot compute resonances locations.")

    return

def coord_tuples_to_one(coord_tuples: tuple[tuple,tuple]):
    lst = []
    for i in coord_tuples:
        lst.append(i[1])
    return tuple(lst)

def plot_reslocs(reslocs: list['ResLocGeoObject'], title: str, axis_set):
    """
    
    """
    if len(reslocs[0].values)!=2:
        raise NotImplementedError("Plotting objects with more than 2 dimensions is not implemented")
    
    import matplotlib.pyplot as plt    
    x, y = [], []
    for i in reslocs:
        x.append(i.values[0])
        y.append(i.values[1])
    xlbl = ''
    ylbl = ''
    for ax in axis_set.axes:
        pulse_refs = list(p.pulse_refs for p in ax.var_set.var_set)
        if ax.label=='A':
            xlbl = 'A: '+str(pulse_refs)
        if ax.label=='B':
            ylbl = 'B: '+str(pulse_refs)
    plt.scatter(x,y)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.show()

def get_features_report():
    """
    """
    return