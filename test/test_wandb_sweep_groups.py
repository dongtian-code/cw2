from cw2.cw_data.cw_wandb_logger import (
    WandBLogger,
    build_sweep_group_name,
    group_parameters,
)


def test_group_is_unchanged_without_sweep_parameters():
    assert build_sweep_group_name("study", "experiment") == "study"


def test_each_sweep_gets_a_distinct_readable_group():
    first = build_sweep_group_name(
        "study",
        "experiment__pol.arg.tra.nl2_pol.arg.tra.nh4",
    )
    second = build_sweep_group_name(
        "study",
        "experiment__pol.arg.tra.nl3_pol.arg.tra.nh4",
    )

    assert first == "study | pol_arg_tra_[nh4,nl2]"
    assert second == "study | pol_arg_tra_[nh4,nl3]"
    assert first != second


def test_repetitions_of_the_same_sweep_share_the_group(tmp_path):
    config = {
        "_experiment_name": "experiment__sam.arg.smpsl8",
        "params": {},
        "wandb": {
            "enabled": False,
            "group": "study",
            "group_by_sweep": True,
            "log_model": False,
            "project": "project",
        },
    }

    first = WandBLogger()
    first.init_fields(config, rep=0, rep_log_path=str(tmp_path / "rep_00"))
    second = WandBLogger()
    second.init_fields(config, rep=7, rep_log_path=str(tmp_path / "rep_07"))

    assert first.group == "study | sam.arg.smpsl8"
    assert second.group == first.group
    assert first.runname != second.runname


def test_run_name_uses_concrete_seed_instead_of_repetition_index(tmp_path):
    config = {
        "_experiment_name": "experiment",
        "params": {},
        "seed": 4,
        "wandb": {
            "enabled": False,
            "log_model": False,
            "project": "project",
        },
    }

    logger = WandBLogger()
    logger.init_fields(config, rep=0, rep_log_path=str(tmp_path / "rep_04"))

    assert logger.rep_idx == 0
    assert logger.rep == 4
    assert logger.runname == "experiment_rep_04"


def test_excluded_environment_does_not_create_a_distinct_group():
    parameters = {
        "sampler": {
            "args": {
                "env_id": "metaworld/coffee-pull-v2",
            }
        }
    }
    first = build_sweep_group_name(
        "study",
        "experiment__sam.arg.eimetaworld/coffee-pull-v2_pol.arg.tra.nl2",
        excluded_parameter_paths=["sampler.args.env_id"],
        parameters=parameters,
    )
    parameters["sampler"]["args"]["env_id"] = "metaworld/coffee-push-v2"
    second = build_sweep_group_name(
        "study",
        "experiment__sam.arg.eimetaworld/coffee-push-v2_pol.arg.tra.nl2",
        excluded_parameter_paths=["sampler.args.env_id"],
        parameters=parameters,
    )

    assert first == "study | pol.arg.tra.nl2"
    assert second == first


def test_excluded_seed_does_not_create_a_distinct_group():
    parameters = {
        "seed": 5,
        "policy": {"args": {"learning_rate": 0.001}},
    }
    first = build_sweep_group_name(
        "study",
        "experiment__s5_pol.arg.lr0.001",
        excluded_parameter_paths=["seed"],
        parameters=parameters,
    )
    parameters["seed"] = 6
    second = build_sweep_group_name(
        "study",
        "experiment__s6_pol.arg.lr0.001",
        excluded_parameter_paths=["seed"],
        parameters=parameters,
    )

    assert first == "study | pol.arg.lr0.001"
    assert second == first


def test_non_excluded_sweep_parameters_still_create_distinct_groups():
    parameters = {
        "sampler": {
            "args": {
                "env_id": "metaworld/coffee-pull-v2",
            }
        }
    }
    first = build_sweep_group_name(
        "study",
        "experiment__sam.arg.eimetaworld/coffee-pull-v2_pol.arg.tra.nl1",
        excluded_parameter_paths=["sampler.args.env_id"],
        parameters=parameters,
    )
    second = build_sweep_group_name(
        "study",
        "experiment__sam.arg.eimetaworld/coffee-pull-v2_pol.arg.tra.nl2",
        excluded_parameter_paths=["sampler.args.env_id"],
        parameters=parameters,
    )

    assert first != second


def test_long_sweep_names_are_shortened_without_collisions():
    common = "pol.arg." + ("verylongparameter" * 10)
    first = build_sweep_group_name(
        "study",
        f"experiment__{common}A",
        max_parameter_length=32,
    )
    second = build_sweep_group_name(
        "study",
        f"experiment__{common}B",
        max_parameter_length=32,
    )

    first_suffix = first.split(" | ", 1)[1]
    second_suffix = second.split(" | ", 1)[1]
    assert len(first_suffix) == 32
    assert len(second_suffix) == 32
    assert "~" in first_suffix
    assert first != second


def test_group_by_sweep_is_disabled_by_default(tmp_path):
    config = {
        "_experiment_name": "experiment__sam.arg.smpsl8",
        "params": {},
        "wandb": {
            "enabled": False,
            "group": "study",
            "log_model": False,
            "project": "project",
        },
    }

    logger = WandBLogger()
    logger.init_fields(config, rep=0, rep_log_path=str(tmp_path / "rep_00"))

    assert logger.group == "study"


def test_logger_applies_sweep_group_exclusions(tmp_path):
    config = {
        "_experiment_name": (
            "experiment__sam.arg.eimetaworld/task-v2_pol.arg.tra.nl2"
        ),
        "params": {
            "sampler": {
                "args": {
                    "env_id": "metaworld/task-v2",
                }
            }
        },
        "wandb": {
            "enabled": False,
            "group": "study",
            "group_by_sweep": True,
            "sweep_group_exclude_parameters": ["sampler.args.env_id"],
            "log_model": False,
            "project": "project",
        },
    }

    logger = WandBLogger()
    logger.init_fields(config, rep=0, rep_log_path=str(tmp_path / "rep_00"))

    assert logger.group == "study | pol.arg.tra.nl2"


def test_logger_does_not_expand_a_pre_resolved_group_twice(tmp_path):
    config = {
        "_experiment_name": "experiment__pol.arg.tra.nl2",
        "params": {},
        "wandb": {
            "enabled": False,
            "group": "study | pol.arg.tra.nl2",
            "group_by_sweep": False,
            "sweep_group_resolved": True,
            "log_model": False,
            "project": "project",
        },
    }

    logger = WandBLogger()
    logger.init_fields(config, rep=0, rep_log_path=str(tmp_path / "rep_00"))

    assert logger.group == "study | pol.arg.tra.nl2"


def test_identical_tokens_do_not_recurse_forever():
    # Two swept parameters can abbreviate to the same short name (cw2's
    # shorten_param maps both *_min_step_length and *_max_step_length to
    # "...msl"), so a sweep row where they hold the same value yields two
    # identical tokens. Grouping those used to recurse until RecursionError.
    assert group_parameters(["a", "a"]) == ("a", 1)
    assert group_parameters(["pol.arg.pumsl16", "pol.arg.pumsl16"]) == (
        "pol_arg_pumsl16",
        1,
    )
    assert group_parameters(
        ["pol.arg.trslTrue", "pol.arg.tmpsl1", "pol.arg.tmpsl1"]
    ) == ("pol_arg_[tmpsl1,trslTrue]", 1)


def test_tokens_differing_only_in_trailing_dots_do_not_recurse_forever():
    # A duplicate in the input is not the only way to reach the degenerate
    # state: any group whose members all peel down to "" gets there, so
    # de-duplicating the input alone would not be enough.
    assert group_parameters(["a", "a."]) == ("a", 1)
    assert group_parameters(["", ""]) == ("", 1)
    # Reached one level down, where the group is ["", "."]: still terminates,
    # at the cost of a trailing separator in a name no real parameter produces.
    assert group_parameters(["a.", "a.."]) == ("a_", 1)


def test_repeated_separators_in_a_job_name_do_not_recurse_forever():
    # The use_group_parameters path feeds job_name.split("_") straight in,
    # without dropping empty tokens.
    assert group_parameters("a___b".split("_")) == (",a,b", 3)


def test_grouping_of_distinct_parameters_is_unchanged():
    # Guards the fix against changing any name it must not change.
    assert group_parameters(
        [
            "local",
            "mod.enc.tidentity",
            "mod.hea.nhl5",
            "mod.hea.ioFalse",
            "mod.enc.hd64",
        ]
    ) == ("local,mod_[enc_[hd64,tidentity],hea_[ioFalse,nhl5]]", 2)
    assert group_parameters(["pol.arg.pumsl1", "pol.arg.pumsl8"]) == (
        "pol_arg_[pumsl1,pumsl8]",
        1,
    )


def test_sweep_row_with_equal_min_and_max_keeps_a_distinct_group():
    # The four rows of the implicit_action_repetition sweep:
    # (min, max) = (1, 8), (1, 16), (16, 16), (1, 32). The third one used to
    # crash config processing outright.
    groups = [
        build_sweep_group_name(
            "study",
            f"experiment__pol.arg.pumsl{minimum}_pol.arg.pumsl{maximum}",
        )
        for minimum, maximum in [(1, 8), (1, 16), (16, 16), (1, 32)]
    ]

    assert groups == [
        "study | pol_arg_[pumsl1,pumsl8]",
        "study | pol_arg_[pumsl1,pumsl16]",
        "study | pol_arg_pumsl16",
        "study | pol_arg_[pumsl1,pumsl32]",
    ]
    assert len(set(groups)) == len(groups)


def test_equal_valued_pair_does_not_share_a_group_with_a_single_parameter():
    # Collapsing the duplicate to one token must not make a two-parameter row
    # indistinguishable from a row that swept only one of them.
    pair = build_sweep_group_name(
        "study", "experiment__pol.arg.pumsl16_pol.arg.pumsl16"
    )
    single = build_sweep_group_name("study", "experiment__pol.arg.pumsl16")

    assert pair != single
