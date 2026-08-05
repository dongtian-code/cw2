from cw2.cw_data.cw_wandb_logger import (
    WandBLogger,
    build_sweep_group_name,
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
