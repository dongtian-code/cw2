from types import SimpleNamespace

import pytest

from cw2 import cw_error
from cw2.cw_slurm import cw_slurm
from cw2.scheduler import (
    GPUDistributingLocalScheduler,
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
            "num_parallel_jobs": 120,
            "partition": "allgpu",
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
        2: [2, 3, 4, 6, 7, 8],
        1: [5],
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
        4: [1],
        2: [0],
        1: [2],
    }


def test_auto_gpu_assignment_respects_reps_in_parallel():
    conf = _config()
    jobs = [_job(task_count=4, n_parallel=2)]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 1, 2: 1, 1: 1},
    )

    assert assignment == {2: [0]}


def test_auto_gpu_count_rejects_node_that_cannot_be_filled():
    conf = _config(reps_per_gpu=2)

    with pytest.raises(cw_error.ConfigKeyError, match="fully occupied"):
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
                    "GPUx4,V100",
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

    cw_slurm._submit_auto_gpu_arrays(conf, assignment, "/tmp/sbatch.sh")

    assert submitted == [
        [
            "sbatch",
            "--array=0-1%2",
            "--constraint=GPUx4&(H100|A100)",
            "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx4&(H100|A100)",
            "/tmp/sbatch.sh",
        ],
        [
            "sbatch",
            "--array=2-3,6%3",
            "--constraint=GPUx2&(H100|A100)",
            "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx2&(H100|A100)",
            "/tmp/sbatch.sh",
        ],
        [
            "sbatch",
            "--array=4-5%2",
            "--constraint=GPUx1&(H100|A100)",
            "--export=ALL,MPRL_RESUBMIT_CONSTRAINT=GPUx1&(H100|A100)",
            "/tmp/sbatch.sh",
        ],
    ]


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
