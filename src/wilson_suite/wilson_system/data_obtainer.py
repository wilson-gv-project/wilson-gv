from collections import Counter
from dataclasses import asdict

from wilson_suite.wilson_system.system_data import DataOriginInfo


def wilson_data_obtainer(requested_data_dict: dict[str,DataOriginInfo]):
    """
    Read data from its sources: one parse_from_source call per origin.

    requested_data_dict  {name: DataOriginInfo}, the names to read and where each comes from
    returns              {name: value}, the requested names, plus extra names a call returns

    Names with equal origins go to one call together, e.g.
        {'cff': g16, 'qff': g16, 'dipgrad': c4} -> ['cff', 'qff'] from g16, ['dipgrad'] from c4
    A call can return names that were not requested, e.g. cff_rc with cff. Merging the calls:
        a requested name   always from its own origin
        an extra name      kept if no origin was asked for it and only one origin returns it
    Mixing origins assumes mode k is the same vibration in each; nothing checks this.

    raises TypeError if requested_data_dict is not {name: DataOriginInfo},
        ValueError for an unknown source_type, NotImplementedError for 'wilson';
        all before any file is read
    """
    if not isinstance(requested_data_dict, dict):
        raise TypeError(f'requested_data_dict should be dict[str, DataOriginInfo], '
                        f'not {type(requested_data_dict).__name__}')
    wrong = {k: type(v).__name__ for k, v in requested_data_dict.items() if not isinstance(v, DataOriginInfo)}
    if wrong:
        raise TypeError(f'requested_data_dict should be dict[str, DataOriginInfo]; not a DataOriginInfo: {wrong}')

    origin_to_req_data: dict[DataOriginInfo, list] = {}
    for k, v in requested_data_dict.items():
        origin_to_req_data.setdefault(v, []).append(k)

    for o, datakeys in origin_to_req_data.items():
        if o.source_type == 'wilson':
            raise NotImplementedError(f"source_type 'wilson' for {datakeys}: in-house retrieval not yet implemented")
            # FIXME: MR to implement:
            #   - use base_file_loc as a parsing starting point. Support one or
            #     more of:
            #     a) Formatted or unformatted numpy arrays delimited by header info
            #     b) Other systematic (e.g. OpenRSP format) tensors
            #     c) Header info plus locator for values in formats a) or b)
        if o.source_type not in ('cfour', 'gaussian'):
            raise ValueError(f"unknown source_type {o.source_type!r} for {datakeys}; expected 'gaussian' or 'cfour'")

    # imported here, not at the top: wilson imports without CQCParse
    from CQCParse.parsing import parse_from_source

    dict_with_data = {}
    extras = []     # per call, the names it returned without being asked
    for o, datakeys in origin_to_req_data.items():
        these_results_dict = parse_from_source(requested_data=datakeys, **asdict(o))
        dict_with_data.update({k: these_results_dict[k] for k in datakeys if k in these_results_dict})
        extras.append({k: v for k, v in these_results_dict.items() if k not in datakeys})

    # an extra name is kept if no origin was asked for it and only one origin returns it
    counts = Counter(k for e in extras for k in e)
    for e in extras:
        dict_with_data.update({k: v for k, v in e.items() if k not in requested_data_dict and counts[k] == 1})

    return dict_with_data


def save_datadict_json(dict_with_data: dict, filename: str, save_to_dir):
    raise NotImplementedError("Need to be able to handle non-serializable dicts, and that's not implemented")


def save_datadict_pkl(dict_with_data: dict, filename: str, save_to_dir):
    from wilson_suite.wilson_utils.serialization import pickle_this_to
    pickle_this_to(dict_with_data, filename, save_to=save_to_dir)


def save_obtained_data(dict_with_data: dict, format: str, filename: str = 'obtained_data_dict', save_to_dir=''):
    """
    Saving data wrapper function

    :param dict_with_data: return dict from wilson_data_obtainer
    :param format: options are json or pkl
    :param filename: filename with or without extention; if no extention - will be added as `.format` value
    """
    if format not in ['json', 'pkl']:
        raise NotImplementedError("Cannot save this file format")

    if '.' not in filename:
        filename += '.' + format

    if format == 'json':
        save_datadict_json(dict_with_data, filename, save_to_dir)
    elif format == 'pkl':
        save_datadict_pkl(dict_with_data, filename, save_to_dir)