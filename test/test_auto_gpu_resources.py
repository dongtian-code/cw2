from types import SimpleNamespace

import pytest

from cw2 import cw_error
from cw2 import scheduler as scheduler_module
from cw2 import job as job_module
from cw2.cw_config import cw_conf_keys as config_keys
from cw2.cw_slurm import cw_slurm
from cw2.scheduler import (
    GPUDistributingLocalScheduler,
    HOREKAAffinityGPUDistributingLocalScheduler,
    MPGPUDistributingLocalScheduler,
)


def _config(
    reps_per_gpu=1,
    node_counts=(1, 2, 4),
    fallback_count=2,
):
    return SimpleNamespace(
        slurm_config={
            "num_gpus": "auto",
            "reps_per_gpu": reps_per_gpu,
            "auto_gpu_node_counts": list(node_counts),
            "auto_gpu_models": ["H100", "A100"],
            "auto_gpu_fallback_count": fallback_count,
            "auto_gpu_subtract_pending_priority": True,
            "num_parallel_jobs": 120,
            "partition": "allgpu",
            "cpus-per-task": 64,
            "cpus_per_rep": 16,
            "ntasks": 1,
            "sbatch_args": {},
        }
    )


def _job(task_count, n_parallel=None):
    return SimpleNamespace(
        tasks=[object() for _ in range(task_count)],
        n_parallel=task_count if n_parallel is None else n_parallel,
    )


def test_auto_gpu_assignment_uses_only_current_idle_nodes_then_fallback():
    conf = _config()
    jobs = [_job(4) for _ in range(9)]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 2, 2: 3, 1: 1},
    )

    assert assignment == {
        4: [0, 1],
        2: [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [5, 16],
    }


def test_auto_gpu_assignment_respects_reps_per_gpu():
    conf = _config(reps_per_gpu=2)
    jobs = [_job(4), _job(8), _job(2)]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 3, 2: 3, 1: 3},
    )

    assert assignment == {
        4: [0],
        2: [1],
        1: [2],
    }


def test_auto_gpu_assignment_replaces_static_reps_in_parallel():
    conf = _config()
    jobs = [_job(task_count=4, n_parallel=2)]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 1, 2: 1, 1: 1},
    )

    assert assignment == {4: [0]}


def test_auto_gpu_count_rejects_node_that_cannot_be_filled():
    conf = _config(reps_per_gpu=2)

    with pytest.raises(cw_error.ConfigKeyError, match="cannot fully occupy"):
        cw_slurm.resolve_auto_gpu_resources(
            conf,
            [_job(1)],
            idle_node_counts={1: 1, 2: 1, 4: 1},
        )


def test_fixed_gpu_count_is_unchanged():
    conf = _config()
    conf.slurm_config["num_gpus"] = 2

    selected = cw_slurm.resolve_auto_gpu_resources(conf, [_job(4)])

    assert selected is None


def test_idle_node_query_filters_gpu_models(monkeypatch):
    conf = _config()

    monkeypatch.setattr(
        cw_slurm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="\n".join(
                [
                    "GPUx4,H100",
                    "GPUx4,MI250",
                    "GPUx2,A100",
                    "GPUx1,H100",
                    "GPUx2,H100",
                ]
            )
        ),
    )

    assert cw_slurm.query_idle_auto_gpu_nodes(
        conf,
        [1, 2, 4],
        ["H100", "A100"],
    ) == {1: 1, 2: 2, 4: 1}


def test_pending_priority_query_expands_arrays_and_filters_jobs(monkeypatch):
    conf = _config()
    observed_command = []

    def fake_run(command, **_kwargs):
        observed_command.extend(command)
        return SimpleNamespace(
            stdout="\n".join(
                [
                    "1\tGPUx4&(H100|A100)\tPriority\t1000",
                    "1\tGPUx4&(MI250)\tPriority\t1200",
                    "2\tGPUx2&(H100)\tPriority\t1100",
                    "1\tGPUx2&(A100)\tResources\t900",
                    "1\tGPUx1\tPriority\t800",
                    "1\tH100\tPriority\t700",
                ]
            )
        )

    monkeypatch.setattr(cw_slurm.subprocess, "run", fake_run)

    assert cw_slurm.query_pending_priority_auto_gpu_demand(
        conf,
        [1, 2, 4],
        ["H100", "A100"],
    ) == {1: 1, 2: 2, 4: 1}
    assert "-r" in observed_command


def test_auto_gpu_assignment_subtracts_pending_priority_demand():
    conf = _config()
    jobs = [_job(4) for _ in range(6)]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 2, 2: 4, 1: 1},
        pending_node_counts={4: 2, 2: 0, 1: 0},
    )

    assert assignment == {
        2: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
        1: [4, 12],
    }


def test_auto_gpu_assignment_can_disable_pending_subtraction():
    conf = _config()
    conf.slurm_config["auto_gpu_subtract_pending_priority"] = False

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(4)],
        idle_node_counts={4: 1, 2: 0, 1: 0},
        pending_node_counts={4: 1, 2: 0, 1: 0},
    )

    assert assignment == {4: [0]}


def test_auto_gpu_job_capacities_cover_every_run_exactly():
    conf = _config()

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(100)],
        idle_node_counts={4: 1, 2: 17, 1: 9},
        pending_node_counts={4: 2, 2: 0, 1: 0},
        return_job_capacities=True,
    )

    assert len(assignment[2]) == 45
    assert len(assignment[1]) == 10
    assert 4 not in assignment
    assert sum(capacities) == 100
    assert all(
        capacities[job_idx] == gpu_count
        for gpu_count, job_indices in assignment.items()
        for job_idx in job_indices
    )


def test_job_factory_applies_dynamic_reps_per_job_and_parallelism():
    tasks = [
        {
            config_keys.NAME: "experiment",
            config_keys.REPS_P_JOB: 4,
            config_keys.REPS_PARALL: 4,
        }
        for _ in range(7)
    ]
    factory = job_module.JobFactory(
        exp_cls=None,
        logger=None,
        read_only=True,
        job_capacities=[4, 2, 1],
    )

    divided = factory._divide_tasks(tasks)

    assert [len(group) for group in divided] == [4, 2, 1]
    for group, capacity in zip(divided, [4, 2, 1]):
        assert all(
            task[config_keys.REPS_P_JOB] == capacity
            and task[config_keys.REPS_PARALL] == capacity
            for task in group
        )


def test_auto_gpu_submission_keeps_global_array_indices(monkeypatch):
    conf = _config()
    assignment = {4: [0, 1], 2: [2, 3, 6], 1: [4, 5]}
    submitted = []
    conf.slurm_config["sbatch_args"] = "#SBATCH --signal B:USR1@300"
    monkeypatch.setattr(
        cw_slurm.subprocess,
        "check_output",
        lambda command: submitted.append(command),
    )

    cw_slurm._submit_auto_gpu_arrays(
        conf,
        assignment,
        "/tmp/sbatch.sh",
        ["H100", "A100"],
        reps_per_gpu=1,
        cpus_per_rep=16,
    )

    assert submitted == [
        [
            "sbatch",
            "--array=0-1%2",
            "--constraint=GPUx4&(H100|A100)",
            "--cpus-per-task=64",
            (
                "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx4&(H100|A100),"
                "MPRL_RESUBMIT_CPUS_PER_TASK=64"
            ),
            "/tmp/sbatch.sh",
        ],
        [
            "sbatch",
            "--array=2-3,6%3",
            "--constraint=GPUx2&(H100|A100)",
            "--cpus-per-task=32",
            (
                "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx2&(H100|A100),"
                "MPRL_RESUBMIT_CPUS_PER_TASK=32"
            ),
            "/tmp/sbatch.sh",
        ],
        [
            "sbatch",
            "--array=4-5%2",
            "--constraint=GPUx1&(H100|A100)",
            "--cpus-per-task=16",
            (
                "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx1&(H100|A100),"
                "MPRL_RESUBMIT_CPUS_PER_TASK=16"
            ),
            "/tmp/sbatch.sh",
        ],
    ]


def test_auto_cpu_request_respects_reps_per_gpu():
    assert cw_slurm._auto_cpus_per_task(
        gpu_count=2,
        reps_per_gpu=2,
        cpus_per_rep=16,
    ) == 64


def test_auto_cpu_per_rep_can_be_inferred_from_legacy_config():
    conf = _config()
    del conf.slurm_config["cpus_per_rep"]

    assert cw_slurm._auto_cpus_per_rep(
        conf,
        [_job(4)],
        reps_per_gpu=1,
        node_counts=[1, 2, 4],
    ) == 16


def test_auto_gpu_group_throttles_preserve_global_limit():
    assignment = {
        4: list(range(0, 50)),
        2: list(range(50, 100)),
        1: list(range(100, 150)),
    }

    throttles = cw_slurm._auto_gpu_group_throttles(
        assignment,
        max_parallel=12,
    )

    assert sum(throttles.values()) == 12
    assert all(value >= 1 for value in throttles.values())


def test_runtime_gpu_detection_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "GPU-first,GPU-second,GPU-third",
    )

    assert GPUDistributingLocalScheduler.detect_available_gpu_count() == 3


def test_auto_runtime_parallelism_uses_gpu_queue_capacity():
    scheduler = MPGPUDistributingLocalScheduler.__new__(
        MPGPUDistributingLocalScheduler
    )
    scheduler._auto_num_gpus = True
    scheduler._queue_elements = 4
    scheduler._total_num_gpus = 2
    scheduler._reps_per_gpu = 2
    scheduler.joblist = [_job(4, n_parallel=8)]

    assert scheduler._gpu_num_parallel() == 4


def test_auto_runtime_parallelism_fails_if_node_cannot_be_filled():
    scheduler = MPGPUDistributingLocalScheduler.__new__(
        MPGPUDistributingLocalScheduler
    )
    scheduler._auto_num_gpus = True
    scheduler._queue_elements = 4
    scheduler._total_num_gpus = 4
    scheduler._reps_per_gpu = 1
    scheduler.joblist = [_job(2, n_parallel=4)]

    with pytest.raises(RuntimeError, match="contains only 2 runs"):
        scheduler._gpu_num_parallel()


def test_horeka_scheduler_honors_explicit_cpus_per_rep(monkeypatch):
    conf = _config()
    conf.slurm_config["auto_gpu_node_counts"] = [1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        scheduler_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(32)),
        raising=False,
    )

    scheduler = HOREKAAffinityGPUDistributingLocalScheduler(conf)

    assert scheduler._queue_elements == 2
    assert scheduler._cpus_per_rep == 16
    assert scheduler._usable_cpus == list(range(32))
