from cw2 import job
from cw2.cw_config import cw_conf_keys as KEYS


def test_job_injects_exact_task_membership_for_checkpoint_barrier(tmp_path):
    tasks = []
    for rep in (4, 5, 6):
        tasks.append(
            {
                KEYS.PATH: "experiment",
                KEYS.LOG_PATH: "experiment/log",
                KEYS.i_REP_LOG_PATH: f"experiment/log/rep_{rep:02d}",
                KEYS.i_REP_IDX: rep,
            }
        )

    created_job = job.Job(
        tasks=tasks,
        exp_cls=None,
        logger=None,
        root_dir=str(tmp_path),
    )

    assert len(created_job.tasks) == 3
    for task_index, task in enumerate(created_job.tasks):
        assert task["_cw2_job_task_id"] == task_index
        assert task["_cw2_job_task_ids"] == [0, 1, 2]
        assert task["_cw2_job_task_count"] == 3
        assert task["_cw2_job_rep_ids"] == [4, 5, 6]
        assert task["_cw2_job_rep_count"] == 3


def test_job_task_membership_is_unique_when_sweep_reps_are_all_zero(tmp_path):
    tasks = []
    for task_index in range(8):
        tasks.append(
            {
                KEYS.PATH: f"experiment_{task_index}",
                KEYS.LOG_PATH: f"experiment_{task_index}/log",
                KEYS.i_REP_LOG_PATH: (
                    f"experiment_{task_index}/log/rep_00"
                ),
                KEYS.i_REP_IDX: 0,
            }
        )

    created_job = job.Job(
        tasks=tasks,
        exp_cls=None,
        logger=None,
        root_dir=str(tmp_path),
    )

    for task_index, task in enumerate(created_job.tasks):
        assert task["_cw2_job_task_id"] == task_index
        assert task["_cw2_job_task_ids"] == list(range(8))
        assert task["_cw2_job_task_count"] == 8
        assert task["_cw2_job_rep_ids"] == [0] * 8
