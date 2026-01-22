#!/usr/bin/env python
"""
Single EVV spectrum workflow.

1. Set up an experiment -> get derived EVV terms
    - phasematching condition - should be set as config option?
    - axes choices options
2. Configure calculation:
    - molecular system - label
    - vibrational analysis: regime='GVPT2', vibana_own_analysis='anharm'
    - calc_setup: to retrieve data from the QC program outputs (or submit, get outputs then retrieve)
    - 

EVV_EXPERIMENT.magn_conditions -- w2 > w1


core_paper1_setup = ''

-- fixed
experiment: EVV + phasematching (-1, 2, 3) + magn_condition 'w2>w1'
vib_analysis: anharmonic GVPT2
axes_choice: based on experiment but here either (w1,w2) or (w1,w2-w1)
SpecEvalSetup fixed: 
    PlotConfig
    RenderingInfo[all except reference max]
    EvaluationInfo[all except Gamma, dynamic range, grid_resolution?]

-- variables:
system
calc_setup - DataOriginInfo
SpecEvalSetup variables: 
    RenderingInfo[reference max]
    EvaluationInfo[Gamma and dynamic range]

"""
import argparse

import wilson_suite as ws
from wilson_suite.fixtures import evv_experiment
from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
from wilson_suite.wilson_experiment.indep_vars_and_axes import PhaseMatchingCondition, SignedPulseTuple
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import Box, SpectralWindow
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
from wilson_suite.wilson_intensities.anharmonic_treatment.anharmonic_analyzer import anharm_analyzer_data

# --- Vault stuff
from CQCParse.relay import DataVault

csvfile = '/home/vlev/sprint/calculations/calculations.csv'
vault = DataVault(csvfile)
db = vault.read_csv_DB()
# ['Basis', 'Calc_Type', 'Conformer_Description', 'Conformer_ID',
#    'Full_Name', 'Method', 'Name', 'Status', 'cff', 'dipolex', 'dipoley',
#    'dipolez', 'file_location', 'file_location_pathtype', 'molden', 'out',
#    'polar_pkl', 'qff']


# ---------- PREPARE PARTS FOR WilsonSimulation
# assuming this function will set up and return a correct EVV experiment
EVV_EXPERIMENT = evv_experiment()


# TODO: need to have an API for phasematching condition choice as a part of configs? 
# TODO: also, after the axis choice phasematching condition will be defined as well?
EVV_PHASEMATCH_COND = PhaseMatchingCondition(pulses=SignedPulseTuple(pulse_refs=(-1, 2, 3)), id=0)
assert EVV_EXPERIMENT.relevant_phasematch[0] == EVV_PHASEMATCH_COND

print('EVV_EXPERIMENT.canonical_axes', EVV_EXPERIMENT.canonical_axes)

# BTW: valid_axis_combs[0] is unclear API from the POV of the user
# EVV_EXPERIMENT.valid_axis_combs[0].present_spectral_axis_choices()

# setting up a SpectralAxisSet - axes based on possible combination of independent variables
axes_choice: ws.main.spectrum_abstractions.SpectralAxisSet = make_SpectralAxisSet({'A': [-1], 'B': [2]})
# now would be useful to check wheather constructed SpectralAxisSet makes sense here - is it in valid_axis_combs?


# original derived terms with independent variables - needed for the manuscript
DERIVED_EVV_TERMS = ws.derive.derive.get_fully_enhanced_terms(experiment=EVV_EXPERIMENT)
from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat
flat_terms_dict = derived_terms_flat(DERIVED_EVV_TERMS)
for id, term in flat_terms_dict.items():
    # print()
    print('&'+term.to_latex(part='res') + r' \\')

# next step is to translate terms wrt axes_choice

def system_calculation_setup(calc_choice):
    """
    preparing MolecularSystem and DataOriginInfo for the choice of data [molecule+calc_setup]
    """
    vault_df_row = db.iloc[int(calc_choice)]

    print("\nYour selection:")
    print(f"{vault_df_row['Full_Name']} [{vault_df_row['Name']}] - {vault_df_row['Method']}/{vault_df_row['Basis']}\n")

    base_filepath = vault.make_data_input_by_index(db, int(calc_choice))  

    # would always be a user input - ?
    molecular_system = ws.main.abstractions.MolecularSystem(name=vault_df_row['Name'], natoms=vault_df_row["N_atoms"])

    if isinstance(base_filepath, dict):
        source_type = 'cfour'
    elif isinstance(base_filepath, str):
        source_type = 'gaussian'

    # DataOriginInfo - to get data from QC program outputs
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type=source_type, 
                                                     lvl_theory=vault_df_row['Method'], 
                                                     basis_set=vault_df_row['Basis'], 
                                                     base_file_loc=base_filepath)
    return molecular_system, calc_setup


def evv_SpecEvalSetup_paper1(*, reference_max: float,
                                Gamma_rc: float, 
                                dynamic_range: float,
                                window_bounds_dict: dict[str,tuple[float,float]],
                                grid_resolution: dict[str,int],
                                fig_file: str):
    """
    SpecEvalSetup vars:
        reference_max
        Gamma_rc
        dynamic_range
        window_bounds_dict
        grid_resolution
    
    window_bounds_dict = {'A': (1000., 3100.), 'B': (-100., 2500.)} - example
    grid_resolution = {'A': 70, 'B': 100} - example
    fig_file = '.svg'

    """
    # set up SpectralWindow -- should be flexible/changable in wilsonsim object
    spectral_window = SpectralWindow(box=Box(bounds=window_bounds_dict))

    # set up EvaluationInfo
    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                            'Gamma': Gamma_rc, 'Gamma_unit': 'cm-1',
                                                            'dynamic_range': dynamic_range,
                                                            'grid_resolution': grid_resolution})

    # set up RenderingInfo
    style_config = ws.main.spectrum_abstractions.PlotConfig(tick_step=50.)
    rnd = ws.main.spectrum_abstractions.RenderingInfo(filename=fig_file, 
                                                      reference_max=reference_max,
                                                      style_config=style_config)

    # put configs together in SpecEvalSetup
    return ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi, rnd_info=rnd)



def main():
    '''

    parser = argparse.ArgumentParser(
        description="A script to demonstrate CLI functionality and print initial information."
    )

    parser.add_argument(
        "--example", 
        action="store_true", 
        help="An example flag for demonstration."
    )

    args = parser.parse_args()

    '''
    
    print("\n------    WilsonSimulation evaluate and render script!\n")
    print("There are entries with different Calc_Types:", db["Calc_Type"].unique(), '\n\n')

    filter_db = db[db["Calc_Type"] == 'full']
    result = filter_db[["Full_Name", "Conformer_ID", "Name", "Method", "Basis"]]
    print(result)


    filtered_ids = filter_db.index.tolist()
    calc_choice = None
    while calc_choice is None or int(calc_choice) not in filtered_ids:
        calc_choice = input('\n\nSelect a number from the table: ')
        
        try:
            calc_choice = int(calc_choice)  # Convert input to integer
            if calc_choice not in filtered_ids:
                print(f"Invalid choice. Please select a valid number from: {filtered_ids}")
                calc_choice = None  # Reset calc_choice to stay in the loop
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            calc_choice = None  # Reset calc_choice to stay in the loop

    
    # would always be a user input - ?
    molecular_system, calc_setup = system_calculation_setup(calc_choice=calc_choice)

    # user configs
    vib_ana = ws.main.abstractions.VibAnaSetup(system=molecular_system, regime='GVPT2', vibana_own_analysis='anharm')
    # doesn't have to have a molecular system

    import json
    with open("/home/vlev/monorepo/examples/workflows/config.json", "r") as file:
        params = json.load(file)

    
    if params['reference_max'] == 'None':
        params['reference_max'] = None
    params['window_bounds_dict'] = {k: tuple(v) for k,v in params['window_bounds_dict'].items()}
    print(params)

    eval_setup = evv_SpecEvalSetup_paper1(**params)

    '''
    reference_max: float,
    Gamma_rc: float, 
    dynamic_range: float,
    window_bounds_dict: dict[str,tuple[float,float]],
    grid_resolution: dict[str,int],
    fig_file: str
    '''

    # ---------- WilsonSimulation
    sim = ws.main.workflow_abstractions.WilsonSimulation()

    # -- setting attributes
    sim.addExperiment(experiment=EVV_EXPERIMENT)
    sim.addTerms(terms=DERIVED_EVV_TERMS)
    sim.addSystem(system=molecular_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    sim.addSpecEvalSetup(eval_setup)


    # ---- chng of state
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    # ---- chng of state
    sim.dressPropsWithSetup()


    # ---- chng of state
    sim.setAxisChoiceAndTranslateTerms(axes_choice) # set axes and prepare terms for evaluation 

    # ---- chng of state
    sim.getResults(obtainer=wilson_data_obtainer)

    ws.main.main_functions.do_anharmonic_analysis(vib_ana=sim.vib_ana_setup, 
                                                  props=sim.props, anharmonic_analyzer=anharm_analyzer_data)

    print('sim.vib_ana_setup.isAllSet', sim.vib_ana_setup)
    
    print('\nsim.vib_ana_setup.states\n')
    print(sim.vib_ana_setup.states[:3], '\n\n')

    # ---- chng of state? or just setting attributes?
    sim.evaluate()
    sim.save_to_pkl(filename='/home/vlev/monorepo/examples/workflows/wsim_after_eval.pkl')

    # ---- just setting attributes?
    sim.render(renderer=ws.analysis.render.render_spectrum)


if __name__ == "__main__":
    main()