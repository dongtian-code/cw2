from types import SimpleNamespace

import pytest

from cw2 import cw_error
from cw2.cw_config import cw_conf_keys as config_keys
from cw2.cw_slurm import cw_slurm


def _slurm_config(slurm_values, tmp_path):
    config = SimpleNamespace(
        exp_configs=[{config_keys.i_BASIC_PATH: str(tmp_path)}],
    )
    parsed = cw_slurm.SlurmConfig.__new__(cw_slurm.SlurmConfig)
    parsed.conf = config
    parsed.slurm_conf = {
        "time": 60,
        "sbatch_args": {},
        **slurm_values,
    }
    return parsed


def test_top_level_mem_becomes_sbatch_mem(tmp_path):
    parsed = _slurm_config({"mem": 20000}, tmp_path)

    parsed._complete_optionals()

    assert parsed.slurm_conf["sbatch_args"]["mem"] == 20000


def test_top_level_mem_rejects_conflicting_sbatch_mem(tmp_path):
    parsed = _slurm_config(
        {
            "mem": 20000,
            "sbatch_args": {"mem": "30000M"},
        },
        tmp_path,
    )

    with pytest.raises(cw_error.ConfigKeyError, match="Conflicting Slurm mem"):
        parsed._complete_optionals()


def test_mem_and_mem_per_cpu_are_mutually_exclusive(tmp_path):
    parsed = _slurm_config(
        {
            "mem": 20000,
            "mem-per-cpu": 2000,
        },
        tmp_path,
    )

    with pytest.raises(cw_error.ConfigKeyError, match="either mem or mem-per-cpu"):
        parsed._complete_optionals()
