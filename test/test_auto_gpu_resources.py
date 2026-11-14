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
        2: [2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        1: [5, 13, 14, 15, 16, 17, 18, 19],
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
                    "node-h100-4\tGPUx4,H100\tidle",
                    "node-h100-4\tGPUx4,H100\tidle",
                    "node-drain\tGPUx4,H100\tdrain",
                    "node-p100\tGPUx4,P100\tidle",
                    "node-mi250\tGPUx4,MI250\tidle",
                    "node-a100-2\tGPUx2,A100\tidle",
                    "node-h100-1\tGPUx1,H100\tidle",
                    "node-h100-2\tGPUx2,H100\tidle",
                ]
            )
        ),
    )

    assert cw_slurm.query_idle_auto_gpu_nodes(
        conf,
        [1, 2, 4],
        ["H100", "A100"],
    ) == {1: 1, 2: 2, 4: 1}


def test_idle_node_query_accepts_p100_and_idle_state_suffixes(monkeypatch):
    conf = _config()
    observed_command = []

    def fake_run(command, **_kwargs):
        observed_command.extend(command)
        return SimpleNamespace(
            stdout="\n".join(
                [
                    "node-p100-sleep\tGPU,GPUx1,P100\tidle~",
                    "node-p100-resume\tGPU,GPUx1,P100\tidle#",
                    "node-p100-busy\tGPU,GPUx1,P100\tallocated",
                    "node-v100\tGPU,GPUx1,V100\tidle",
                ]
            )
        )

    monkeypatch.setattr(cw_slurm.subprocess, "run", fake_run)

    assert cw_slurm.query_idle_auto_gpu_nodes(
        conf,
        [1, 2, 4],
        ["P100"],
    ) == {1: 2, 2: 0, 4: 0}
    assert "\t".join(["%N", "%f", "%T"]) in observed_command


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
        2: [0, 1, 2, 3, 5, 6, 7, 8, 9],
        1: [4, 10, 11, 12, 13, 14],
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

    assert len(assignment[2]) == 36
    assert len(assignment[1]) == 28
    assert 4 not in assignment
    assert sum(capacities) == 100
    assert all(
        capacities[job_idx] == gpu_count
        for gpu_count, job_indices in assignment.items()
        for job_idx in job_indices
    )


def test_auto_gpu_fallback_balances_all_node_sizes_when_exactly_possible():
    conf = _config(fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(42)],
        idle_node_counts={4: 0, 2: 0, 1: 0},
        return_job_capacities=True,
    )

    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 6,
        2: 6,
        4: 6,
    }
    assert sum(capacities) == 42


def test_auto_gpu_fallback_balances_only_tasks_beyond_idle_capacity():
    conf = _config(fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(21)],
        idle_node_counts={4: 2, 2: 0, 1: 0},
        return_job_capacities=True,
    )

    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 1,
        2: 2,
        4: 4,
    }
    assert sum(capacities) == 21


def test_auto_gpu_fallback_does_not_rebalance_available_node_assignments():
    conf = _config(fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(200)],
        idle_node_counts={4: 11, 2: 3, 1: 24},
        return_job_capacities=True,
    )

    # The first 11/3/24 jobs consume currently idle nodes. Only the remaining
    # 126 runs are balanced, adding exactly 18 jobs for each node size.
    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 42,
        2: 21,
        4: 29,
    }
    assert sum(capacities) == 200


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


def test_runtime_config_is_persisted_next_to_current_config(tmp_path):
    current_path = tmp_path / "relative_config.yaml"
    written = []
    conf = SimpleNamespace(
        config_path=str(current_path),
        to_yaml=lambda directory, relpath: (
            written.append((directory, relpath))
            or str(tmp_path / "relative_updated.yaml")
        ),
    )

    runtime_path = cw_slurm._persist_runtime_config(conf)

    assert written == [(str(tmp_path), False)]
    assert runtime_path == str(tmp_path / "relative_updated.yaml")
    assert conf.config_path == runtime_path


def test_auto_cpu_per_rep_can_be_inferred_from_legacy_config():
    conf = _config()
    del conf.slurm_config["cpus_per_rep"]

    assert cw_slurm._auto_cpus_per_rep(
        conf,
        [_job(4)],
        reps_per_gpu=1,
        base_gpu_count=4,
    ) == 16


def test_auto_cpu_per_rep_inference_ignores_larger_node_sizes():
    conf = _config(node_counts=(1, 2, 4, 10), fallback_count=4)
    del conf.slurm_config["cpus_per_rep"]

    # 64 / 10 would not divide; the value describes the fallback GPUx4 node.
    assert cw_slurm._auto_cpus_per_rep(
        conf,
        [_job(40)],
        reps_per_gpu=1,
        base_gpu_count=4,
    ) == 16


def test_node_cpu_limit_query_takes_largest_matching_node(monkeypatch):
    conf = _config(node_counts=(1, 2, 4, 10))
    conf.slurm_config["auto_gpu_models"] = ["A100", "RTX8000", "L40S"]

    monkeypatch.setattr(
        cw_slurm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="\n".join(
                [
                    "node-rtx8000-a\tEPYC,GPU,GPUx10,RTX8000\t128",
                    "node-rtx8000-b\tEPYC,GPU,GPUx10,RTX8000\t128",
                    "node-l40s-10\tINTEL,GPU,GPUx10,L40S\t96",
                    "node-h200-10\tEPYC,GPU,GPUx10,H200\t512",
                    "node-a100-4a\tEPYC,GPU,GPUx4,A100\t72",
                    "node-a100-4b\tEPYC,GPU,GPUx4,A100\t192",
                    "node-a100-1\tEPYC,GPU,GPUx1,A100\t40",
                ]
            )
        ),
    )

    assert cw_slurm.query_auto_gpu_node_cpu_limits(
        conf,
        [1, 2, 4, 10],
        ["A100", "RTX8000", "L40S"],
    ) == {10: 128, 4: 192, 1: 40}


def test_auto_cpu_request_is_capped_to_largest_node():
    assert cw_slurm._auto_cpus_per_task(
        gpu_count=10,
        reps_per_gpu=2,
        cpus_per_rep=16,
        cpu_limit=128,
    ) == 128
    assert cw_slurm._auto_cpus_per_task(
        gpu_count=4,
        reps_per_gpu=1,
        cpus_per_rep=16,
        cpu_limit=192,
    ) == 64
    with pytest.raises(cw_error.ConfigKeyError, match="at most 8 CPUs"):
        cw_slurm._auto_cpus_per_task(
            gpu_count=10,
            reps_per_gpu=1,
            cpus_per_rep=16,
            cpu_limit=8,
        )


def test_auto_gpu_assignment_fills_idle_ten_gpu_nodes_first():
    conf = _config(node_counts=(1, 2, 4, 10), fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(28)],
        idle_node_counts={10: 2, 4: 1, 2: 0, 1: 0},
        pending_node_counts={},
        return_job_capacities=True,
    )

    # Idle GPUx10 nodes take 20 runs, the idle GPUx4 4 more; the last 4 go to
    # the fallback balancing, which never picks the larger GPUx10 size.
    assert assignment == {10: [0, 1], 4: [2, 3]}
    assert capacities == [10, 10, 4, 4]


def test_auto_gpu_submission_caps_ten_gpu_cpus(monkeypatch):
    conf = _config(node_counts=(1, 2, 4, 10))
    submitted = []
    monkeypatch.setattr(
        cw_slurm.subprocess,
        "check_output",
        lambda command: submitted.append(command),
    )

    cw_slurm._submit_auto_gpu_arrays(
        conf,
        {10: [0, 1], 4: [2]},
        "/tmp/sbatch.sh",
        ["A100", "RTX8000"],
        reps_per_gpu=1,
        cpus_per_rep=16,
        cpu_limits={10: 128, 4: 192},
    )

    assert [command[1:4] for command in submitted] == [
        [
            "--array=0-1%2",
            "--constraint=GPUx10&(A100|RTX8000)",
            "--cpus-per-task=128",
        ],
        [
            "--array=2%1",
            "--constraint=GPUx4&(A100|RTX8000)",
            "--cpus-per-task=64",
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


def test_horeka_scheduler_shrinks_cpus_per_rep_on_ten_gpu_node(monkeypatch):
    conf = _config(node_counts=(1, 2, 4, 10), fallback_count=4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(10))))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
    monkeypatch.setattr(
        scheduler_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(128)),
        raising=False,
    )

    scheduler = HOREKAAffinityGPUDistributingLocalScheduler(conf)

    assert scheduler._queue_elements == 10
    assert scheduler._cpus_per_rep == 12
    assert scheduler._usable_cpus == list(range(120))


def test_horeka_scheduler_uses_slurm_cpus_for_inferred_cpus_per_rep(monkeypatch):
    conf = _config(node_counts=(1, 2, 4, 10), fallback_count=4)
    del conf.slurm_config["cpus_per_rep"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(10))))
    # The persisted config says 64 (the GPUx4 fallback); this array got 128.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
    monkeypatch.setattr(
        scheduler_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(128)),
        raising=False,
    )

    scheduler = HOREKAAffinityGPUDistributingLocalScheduler(conf)

    assert scheduler._cpus_per_rep == 12
    assert scheduler._usable_cpus == list(range(128))


def test_auto_gpu_fallback_is_proportional_to_usable_node_counts():
    conf = _config(fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(216)],
        idle_node_counts={4: 0, 2: 0, 1: 0},
        total_node_counts={1: 62, 2: 11, 4: 33},
        return_job_capacities=True,
    )

    # 62 + 2*11 + 4*33 = 216 runs: exactly one array task per usable node.
    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 62,
        2: 11,
        4: 33,
    }
    assert sum(capacities) == 216


def test_auto_gpu_fallback_proportional_rounds_close_to_node_ratio():
    conf = _config(fallback_count=4)

    assignment, capacities = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(196)],
        idle_node_counts={4: 0, 2: 0, 1: 0},
        total_node_counts={1: 62, 2: 11, 4: 33},
        return_job_capacities=True,
    )

    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 56,
        2: 10,
        4: 30,
    }
    assert sum(capacities) == 196


def test_auto_gpu_fallback_skips_sizes_without_usable_nodes():
    conf = _config(fallback_count=4)

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(16)],
        idle_node_counts={4: 0, 2: 0, 1: 0},
        total_node_counts={1: 5, 2: 0, 4: 5},
    )

    # Equal GPUx1/GPUx4 node counts -> equal task counts, none on GPUx2.
    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 4,
        4: 3,
    }


def test_auto_gpu_fallback_proportion_accumulates_across_groups():
    conf = _config(fallback_count=4)
    jobs = [
        SimpleNamespace(
            tasks=[{config_keys.NAME: "exp_{}".format(group)} for _ in range(4)],
            n_parallel=4,
        )
        for group in range(12)
    ]

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        jobs,
        idle_node_counts={4: 0, 2: 0, 1: 0},
        total_node_counts={1: 2, 2: 0, 4: 1},
    )

    # Each group of 4 runs fits one GPUx4 task or four GPUx1 tasks. A
    # per-group rounding would pick the same size every time; accumulated
    # weighting keeps GPUx1:GPUx4 tasks near 2:1 overall.
    n1 = len(assignment.get(1, []))
    n4 = len(assignment.get(4, []))
    assert n1 + 4 * n4 == 48
    assert n4 > 0 and n1 > 0
    assert abs(n1 / 2 - n4) <= 2


def test_auto_gpu_fallback_balanced_strategy_ignores_node_counts():
    conf = _config(fallback_count=4)
    conf.slurm_config["auto_gpu_fallback_strategy"] = "balanced"

    assignment = cw_slurm.resolve_auto_gpu_resources(
        conf,
        [_job(42)],
        idle_node_counts={4: 0, 2: 0, 1: 0},
        total_node_counts={1: 62, 2: 11, 4: 33},
    )

    assert {count: len(indices) for count, indices in assignment.items()} == {
        1: 6,
        2: 6,
        4: 6,
    }


def test_total_node_query_excludes_p100_v100_and_unusable_states(monkeypatch):
    conf = _config()
    conf.slurm_config["auto_gpu_models"] = []

    monkeypatch.setattr(
        cw_slurm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="\n".join(
                [
                    "n1\tGPU,GPUx1,A100\tallocated",
                    "n1\tGPU,GPUx1,A100\tallocated",
                    "n2\tGPU,GPUx1,H200\treserved",
                    "n3\tGPU,GPUx1,P100\tidle",
                    "n4\tGPU,GPUx1,V100\tallocated",
                    "n5\tGPU,GPUx2,A6000\tidle~",
                    "n6\tGPU,GPUx2,H100\tdrained*",
                    "n7\tGPU,GPUx4,L40S\tmixed",
                    "n8\tGPU,GPUx4,A100\tdown*",
                    "n9\tGPU,GPUx4,V100\tallocated",
                ]
            )
        ),
    )

    assert cw_slurm.query_total_auto_gpu_nodes(
        conf,
        [1, 2, 4],
        [],
        cw_slurm._auto_gpu_fallback_exclude_models(conf),
    ) == {1: 2, 2: 1, 4: 1}
