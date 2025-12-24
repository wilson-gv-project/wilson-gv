from wilson_suite.wilson_main.abstractions import DataOriginInfo

def wilson_data_obtainer(requested_data_dict: dict[str,DataOriginInfo]):
    """
    1. group by origin
    """
    dict_with_data = {}

    origin_to_req_data: dict[DataOriginInfo, list] = {}

    requested_data_dict.update({'atoms': requested_data_dict['nc_sqrt_eigval'], 
                                'normal_modes': requested_data_dict['nc_sqrt_eigval']})

    for k, v in requested_data_dict.items():
        
        if v not in origin_to_req_data:
            origin_to_req_data[v] = [k]

        else:
            origin_to_req_data[v].append(k)

    
    for o in origin_to_req_data:

        if o.source_type in ['cfour', 'gaussian']:

            from CQCParse.parsing import parse_from_source
            from dataclasses import asdict

            these_results_dict = parse_from_source(requested_data=origin_to_req_data[o], **asdict(o))
            
            dict_with_data.update(these_results_dict)

        elif o.source_type in ['wilson']:

            raise NotImplementedError('In-house retrieval not yet implemented')
            # FIXME: MR to implement:
            #   - use base_file_loc as a parsing starting point. Support one or
            #     more of:
            #     a) Formatted or unformatted numpy arrays delimited by header info
            #     b) Other systematic (e.g. OpenRSP format) tensors
            #     c) Header info plus locator for values in formats a) or b)

        else:
            raise AssertionError('Unsupported source type for data obtainer')
    
    return dict_with_data


def test_do():
    dict_with_data = {'cff': DataOriginInfo(lvl_theory='B3LYP'), 
                      'qff': DataOriginInfo(lvl_theory='CCSD')}
