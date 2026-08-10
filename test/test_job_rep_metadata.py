from cw2 import job
from cw2.cw_config import cw_conf_keys as KEYS


def test_job_injects_exact_rep_membership_for_checkpoint_barrier(tmp_path):
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
    for task in created_job.tasks:
        assert task["_cw2_job_rep_ids"] == [4, 5, 6]
        assert task["_cw2_job_rep_count"] == 3
