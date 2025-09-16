from wilson_intensities.wilson.spectrum import func_evaluation
from wilson_derive.abstractions import VibPerturbedTerm
from wilson_main.abstractions import WilsonSimulation, SpecEvalSetup, SpectralAxis
from wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
from wilson_utils.useful_shortcuts import makeSpecSetup2D

import numpy as np
from rich.pretty import pprint
from collections import defaultdict

class DataAnalyzer:
    """
    Pre spectrum evaluation analysis with:

        - extract_bounds_from_reslocs
        - extract_bounds_from_multiple_reslocs
        - points_scatter_plt
        - extract_resonances
        - show_terms (no return)
    
    """    
    @staticmethod
    def extract_bounds_from_reslocs(resloc_data: dict):
        """
        Extract min and max bounds for each key across all dictionaries in the data structure.
        
        Args:
            resloc_data: Dictionary where values are lists of dictionaries containing numeric data
        
        Returns:
            dict: Dictionary with keys as the original dictionary keys, and values as dicts with 'min' and 'max'
        """
        # Dictionary to store min and max values for each key
        bounds = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        # Iterate through all EvalTerm entries
        for eval_term, dict_list in resloc_data.items():
            # Iterate through all dictionaries in the list for this EvalTerm
            for data_dict in dict_list:

                # Iterate through all key-value pairs in each dictionary
                for key, value in data_dict.items():
                    # Convert numpy float64 to regular float for comparison
                    float_value = float(value)
                    
                    # Update min and max for this key
                    if float_value < bounds[key]['min']:
                        bounds[key]['min'] = float_value
                    if float_value > bounds[key]['max']:
                        bounds[key]['max'] = float_value
        
        # Convert defaultdict to regular dict
        return dict(bounds)
    
    
    def extract_bounds_from_multiple_reslocs(self, resloc_data_list: list):
        """
        Extract min and max bounds for each key across multiple resloc_data dictionaries.

        Args:
            resloc_data_list: List of resloc_data dictionaries, where each dictionary 
                                has values that are lists of dictionaries containing numeric data

        Returns:
            dict: Dictionary with keys as the original dictionary keys, and values as dicts with 'min' and 'max'
        """
        # Dictionary to store overall min and max values for each key
        overall_bounds = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})

        # Process each resloc_data dictionary
        for resloc_data in resloc_data_list:
            # Get bounds for current resloc_data using existing method
            current_bounds = self.extract_bounds_from_reslocs(resloc_data=resloc_data)
            
            # Update overall bounds with current bounds
            for key, bounds in current_bounds.items():
                if bounds['min'] < overall_bounds[key]['min']:
                    overall_bounds[key]['min'] = bounds['min']
                if bounds['max'] > overall_bounds[key]['max']:
                    overall_bounds[key]['max'] = bounds['max']

        # Convert defaultdict to regular dict
        return dict(overall_bounds)
    

    @staticmethod
    def points_scatter_plt(resloc_data: dict):
        """
        takes one resloc_data (for one system with all the terms)
        returns scatter_points_dict[key] = {'x': x, 'y': y}
        """
        scatter_points_dict = {}

        for key, points_list in resloc_data.items():
            x = [p['w1'] for p in points_list]
            y = [p['w2'] for p in points_list]
            scatter_points_dict[key] = {'x': x, 'y': y}
        
        return scatter_points_dict
    

    @staticmethod
    def show_terms(wilsonsim: WilsonSimulation = None):
        """
        prints all the terms in the wilsonsim
        """
        terms_list = derived_terms_dict_to_dicts(wilsonsim.terms)
        terms_list = [func_evaluation.EvalTerm(**i) for i in terms_list]

        term_type = [isinstance(i, VibPerturbedTerm) for i in terms_list]
        if all(term_type):
            raise NotImplementedError('Analysis of VibPerturbedTerm is not yet supported')
        
        term_type = [isinstance(i, func_evaluation.EvalTerm) for i in terms_list]
        
        if all(term_type):
            for t in terms_list:
                print(f"Term {t.short_id}:")
                pprint(t.__dict__)
                print('---')
        return
    

    def make_vibdiffbank(self, wilsonsim: WilsonSimulation = None, vibdiffbank_mode: str = 'anharmonic'):
        """
        makes self.vibdiffbank attribute
        """
        get_state = func_evaluation.make_state_value_func(wilsonsim.vib_ana_setup.states)
        
        self.vibdiffbank = func_evaluation.VibDiffBank(indices=wilsonsim.vib_ana_setup.modes_indices, 
                                                       max_quanta=wilsonsim.vib_ana_setup.max_state_lvl,
                                                       state_value_func=get_state, mode=vibdiffbank_mode)
        return self.vibdiffbank

    def extract_resonances(self, wilsonsim: WilsonSimulation, vibdiffbank_mode: str = None) -> dict[tuple, func_evaluation.Resonance]:
        """
        Returns a lis of func_evaluation.Resonance instances based on 
            WilsonSimulation instance terms and mode_indices
        """
        self.make_vibdiffbank(wilsonsim=wilsonsim, vibdiffbank_mode=vibdiffbank_mode)
        
        resonance_registry: dict[tuple, func_evaluation.Resonance] = {}
        
        dict_terms = derived_terms_dict_to_dicts(wilsonsim.terms)
        eval_terms = [func_evaluation.EvalTerm(**dict_terms[i]) for i in range(len(dict_terms))]

        for term in eval_terms:
            
            indices = set([k for i in term.resonances for k in i[0].split(',') if k!='zero'])
            for comb in func_evaluation.combinations_with_permutations(iterable=wilsonsim.vib_ana_setup.modes_indices,
                                                                        k=len(indices)):
                res_loc_dict = func_evaluation.solve_linear_system_resonaces(resonance_tuples=term.resonances,
                                                                    ind_tuple=comb,
                                                                    vibdiffbank=self.vibdiffbank)

                key = tuple([float(e) for e in list(res_loc_dict.values())])
                if key not in resonance_registry:
                    res = func_evaluation.Resonance(location=tuple(res_loc_dict.values()))
                    res.add_producer(term_id=term.short_id, 
                                        term_res_pattern=term.resonances,
                                        assignment=comb)

                    resonance_registry[key] = res
                else:
                    resonance_registry[key].add_producer(term_id=term.short_id, 
                                                            term_res_pattern=term.resonances,
                                                            assignment=comb)

        return resonance_registry
    
    @staticmethod
    def extract_oneTerm_resonances(term: func_evaluation.EvalTerm, vibdiffbank: func_evaluation.VibDiffBank):
        """
        Returns a lis of func_evaluation.Resonance instances based on 
            one EvalTerm instance and mode_indices
        """
        list_res = []
        indices = set([k for i in term.resonances for k in i[0].split(',') if k!='zero'])
        for comb in func_evaluation.combinations_with_permutations(iterable=vibdiffbank.indices,
                                                                    k=len(indices)):
            res_loc_dict = func_evaluation.solve_linear_system_resonaces(resonance_tuples=term.resonances,
                                                                ind_tuple=comb,
                                                                vibdiffbank=vibdiffbank)

            res = func_evaluation.Resonance(location=tuple(res_loc_dict.values()))
            res.add_producer(term_id=term.short_id, 
                                term_res_pattern=term.resonances,
                                assignment=comb)
            list_res.append(res)
        return list_res
    
    @staticmethod
    def get_bounds_from_resonances(res_dict: dict[tuple, func_evaluation.Resonance]):
        right_bounds = []
        left_bounds = []

        # Extract the keys from the dictionary
        keys = list(res_dict.keys())
        # Use zip to separate elements by their positions
        separated_elements = list(zip(*keys))
        # Convert each group to a list (optional)
        separated_lists = [list(group) for group in separated_elements]
        
        for g in separated_lists:
            right_bounds.append(max(g))
            left_bounds.append(min(g))
        return left_bounds, right_bounds
    
    @staticmethod
    def get_common_biggest_grid(list_of_spec_eval_setups: list[SpecEvalSetup]):
        
        spectral_grids = [i.grid for i in list_of_spec_eval_setups]
        
        # Initialize minima for start and maxima for end
        new_start = {'x': float('inf'), 'y': float('inf')}
        new_end = {'x': float('-inf'), 'y': float('-inf')}
        new_spacer = {'x': float('inf'), 'y': float('inf')}
        
        # Iterate through each SpectralGrid instance
        for grid in spectral_grids:
            # Update minima for start
            new_start['x'] = min(new_start['x'], grid['start']['x'])
            new_start['y'] = min(new_start['y'], grid['start']['y'])
            
            new_spacer['x'] = min(new_spacer['x'], grid['spacer']['x'])
            new_spacer['y'] = min(new_spacer['y'], grid['spacer']['y'])

            # Update maxima for end
            new_end['x'] = max(new_end['x'], grid['end']['x'])
            new_end['y'] = max(new_end['y'], grid['end']['y'])

        axis1 = SpectralAxis(freq_vars={'w1': 1})
        axis2 = SpectralAxis(freq_vars={'w1': -1, 'w2': 1})
        axes = {'x': axis1, 'y': axis2}        
        configs = dict(dynamic_range=500)

        spec_eval = makeSpecSetup2D(start=new_start, end=new_end, 
                                    spacer=new_spacer,
                                    axes=axes, configs=configs)
        return spec_eval

