import itertools
import os
from collections import deque
from copy import deepcopy
from typing import List

from cw2 import util
from cw2.cw_config import conf_path
from cw2.cw_config import cw_conf_keys as KEY
from cw2.cw_data import cw_logging


_EXPLICIT_SEED_PATHS = {
    ("seed",),
    ("sampler", "args", "seed"),
}


def _is_explicit_seed_path(parameter_path: tuple) -> bool:
    return tuple(parameter_path) in _EXPLICIT_SEED_PATHS


def _normalize_explicit_seed(value, parameter_path: tuple) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Explicit seed sweep value at {'.'.join(parameter_path)} must "
            f"be an integer, got {value!r}."
        )
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Explicit seed sweep value at {'.'.join(parameter_path)} must "
            f"be an integer, got {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"Explicit seed sweep value at {'.'.join(parameter_path)} must "
            f"be an integer, got {value!r}."
        )
    return seed


def unfold_exps(exp_configs: List[dict], debug: bool, debug_all: bool) -> List[dict]:
    """unfolds a list of experiment configurations into the different
    hyperparameter runs and repetitions

    Args:
        exp_configs (List[dict]): list of experiment configurations

    Returns:
        List[dict]: list of unfolded experiment configurations
    """
    param_expansion = expand_experiments(exp_configs, debug, debug_all)
    unrolled = unroll_exp_reps(param_expansion)
    return unrolled


def expand_experiments(
    _experiment_configs: List[dict], debug: bool, debug_all: bool
) -> List[dict]:
    """Expand the experiment configuration with concrete parameter instantiations

    Arguments:
        experiment_configs {List[dict]} -- List with experiment configs

    Returns:
        List[dict] -- List of experiment configs, with set parameters
    """

    # get all options that are iteratable and build all combinations (grid) or tuples (list)
    experiment_configs = deque(deepcopy(_experiment_configs))
    if debug or debug_all:
        for ec in experiment_configs:
            ec[KEY.REPS] = ec["iterations"] = ec[KEY.REPS_PARALL] = ec[
                KEY.REPS_P_JOB
            ] = 1

    expanded_config_list = []

    while len(experiment_configs) > 0:
        config = experiment_configs.popleft()

        # Set Default Values
        # save path argument from YML for grid modification
        if KEY.i_BASIC_PATH not in config:
            config[KEY.i_BASIC_PATH] = config.get(KEY.PATH)
        # save name argument from YML for grid modification
        if KEY.i_EXP_NAME not in config:
            config[KEY.i_EXP_NAME] = config.get(KEY.NAME)
        # add empty string for parent DIR in case of grid
        if KEY.i_NEST_DIR not in config:
            config[KEY.i_NEST_DIR] = ""
        # set debug flag
        config[KEY.i_DEBUG_FLAG] = debug or debug_all

        expansion = None
        for key in config:
            if key.startswith(KEY.GRID):
                expansion = params_combine(config, key, itertools.product)
                break
            if key.startswith(KEY.LIST):
                expansion = params_combine(config, key, zip)
                break
            if key.startswith(KEY.ABLATIVE):
                expansion = ablative_expand(config, key)
                break

        if expansion is not None:
            if debug and not debug_all:
                expansion = expansion[:1]
            experiment_configs.extend(expansion)
        else:
            expanded_config_list.append(config)

    return conf_path.normalize_expanded_paths(expanded_config_list)


def params_combine(config: dict, key: str, iter_func) -> List[dict]:
    """combines experiment parameter with its list/grid combinations

    Args:
        config (dict): an single experiment configuration
        key (str): the combination key, e.g. 'list' or 'grid'
        iter_func: itertool-like function for creating the combinations

    Returns:
        List[dict]: list of parameter-combined experiments
    """
    if iter_func is None:
        return [config]

    combined_configs = []
    # convert list/grid dictionary into flat dictionary, where the key is a tuple of the keys and the
    # value is the list of values
    tuple_dict = util.flatten_dict_to_tuple_keys(config[key])
    _param_names = [".".join(t) for t in tuple_dict]
    name_parameter_indices = [
        index
        for index, path in enumerate(tuple_dict)
        if not _is_explicit_seed_path(path)
    ]

    explicit_seed_paths = [
        path for path in tuple_dict if _is_explicit_seed_path(path)
    ]
    if len(explicit_seed_paths) > 1:
        formatted_paths = ", ".join(".".join(path) for path in explicit_seed_paths)
        raise ValueError(
            f"Sweep {key!r} defines seed more than once ({formatted_paths}). "
            "Use either seed or sampler.args.seed."
        )

    param_lengths = map(len, tuple_dict.values())
    if key.startswith(KEY.LIST) and len(set(param_lengths)) != 1:
        cw_logging.getLogger().warning(
            f'experiment "{config[KEY.NAME]}" list params [{key}] are not of equal length.'.format()
        )

    # create a new config for each parameter setting
    for values in iter_func(*tuple_dict.values()):
        _config = deepcopy(config)

        # Remove Grid/List Argument
        del _config[key]

        if KEY.PARAMS not in _config:
            _config[KEY.PARAMS] = {}

        # Expand Grid/List Parameters
        explicit_seed = None
        for i, t in enumerate(tuple_dict.keys()):
            value = values[i]
            if _is_explicit_seed_path(t):
                value = _normalize_explicit_seed(value, t)
                explicit_seed = value
            util.insert_deep_dictionary(
                d=_config.get(KEY.PARAMS), t=t, value=value
            )

        if explicit_seed is not None:
            # A seed sweep enumerates concrete runs. Do not multiply every
            # explicit seed by the experiment's repetition count.
            _config["seed"] = explicit_seed
            _config[KEY.REPS] = 1

        if name_parameter_indices:
            _config = extend_config_name(
                _config,
                [_param_names[index] for index in name_parameter_indices],
                [values[index] for index in name_parameter_indices],
            )
        combined_configs.append(_config)
    return combined_configs


def ablative_expand(config: dict, key: str):
    tuple_dict = util.flatten_dict_to_tuple_keys(config[key])
    _param_names = [".".join(t) for t in tuple_dict]
    combined_configs = []
    for i, t in enumerate(tuple_dict.keys()):
        for val in tuple_dict[t]:
            _config = deepcopy(config)

            # Remove Grid/List Argument
            del _config[key]

            if KEY.PARAMS not in _config:
                _config[KEY.PARAMS] = {}
            util.insert_deep_dictionary(d=_config.get(KEY.PARAMS), t=t, value=val)
            # TODO: TEST
            _config = extend_config_name(_config, [_param_names[i]], [val])

            combined_configs.append(_config)
    return combined_configs


def extend_config_name(config: dict, param_names: list, values: list) -> dict:
    """extend an experiment name with a shorthand derived from the parameters and their values

    Args:
        config (dict): experiment config
        param_names (list): list of parameter names
        values (list): list of parameter values

    Returns:
        dict: experiment config with extended name
    """
    # Rename and append
    _converted_name = util.convert_param_names(param_names, values)

    # Use __ only once as a seperator
    sep = "__"
    if KEY.i_EXP_NAME in config and sep in config.get(KEY.i_EXP_NAME):
        sep = "_"

    config[KEY.i_EXP_NAME] = config.get(KEY.i_EXP_NAME) + sep + _converted_name
    config[KEY.i_NEST_DIR] = config.get(KEY.NAME)
    return config


def unroll_exp_reps(exp_configs: List[dict]) -> List[dict]:
    """unrolls experiment repetitions into their own configuration object

    Args:
        exp_configs (List[dict]): List of experiment configurations

    Returns:
        List[dict]: List of unrolled experiment configurations
    """
    unrolled_exps = []

    for config in exp_configs:
        if KEY.i_REP_IDX in config:
            unrolled_exps.append(config)
            continue

        for r in range(config[KEY.REPS]):
            c = deepcopy(config)
            c[KEY.i_REP_IDX] = r
            c[KEY.i_REP_LOG_PATH] = os.path.join(
                c.get(KEY.LOG_PATH), "rep_{:02d}".format(r)
            )
            unrolled_exps.append(c)
    return unrolled_exps
