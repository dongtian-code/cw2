from types import SimpleNamespace

import pytest

from cw2.cw_data.cw_wandb_logger import WandBLogger


def _logger(tmp_path, sync_on_finish=True):
    run_dir = tmp_path / "wandb" / "run-test"
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True)

    events = []
    logger = WandBLogger()
    logger.run = SimpleNamespace(
        id="run-id",
        dir=str(files_dir),
        finish=lambda: events.append("finish"),
    )
    logger.sync_on_finish = sync_on_finish
    logger.sync_on_finish_timeout = 12.0
    logger.sync_on_finish_max_attempts = 3
    logger.sync_on_finish_retry_initial_delay = 2.0
    logger.sync_on_finish_retry_max_delay = 3.0
    logger.write_wandb_metadata = lambda: events.append("metadata")
    logger.log_model = lambda: events.append("model")
    return logger, run_dir, events


def test_completion_sync_runs_after_wandb_finish(tmp_path, monkeypatch):
    logger, run_dir, events = _logger(tmp_path)
    calls = []

    monkeypatch.setattr("shutil.which", lambda executable: "/env/bin/wandb")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        events.append("sync")
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    logger.finalize()

    assert events == ["metadata", "model", "finish", "sync"]
    command, kwargs = calls[0]
    assert command == [
        "/env/bin/wandb",
        "sync",
        "--include-online",
        "--include-synced",
        "--no-sync-tensorboard",
        "--append",
        "--id",
        "run-id",
        str(run_dir),
    ]
    assert kwargs["timeout"] == 12.0


def test_disabled_completion_sync_preserves_normal_finalize(tmp_path, monkeypatch):
    logger, _, events = _logger(tmp_path, sync_on_finish=False)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("sync must remain disabled"),
    )

    logger.finalize()

    assert events == ["metadata", "model", "finish"]


def test_completion_sync_failure_only_warns(tmp_path, monkeypatch):
    logger, _, events = _logger(tmp_path)
    calls = []
    monkeypatch.setattr("shutil.which", lambda executable: "/env/bin/wandb")

    def fail_sync(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="network unavailable",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fail_sync,
    )
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.sleep",
        lambda delay: None,
    )

    with pytest.warns(UserWarning, match="network unavailable"):
        logger.finalize()

    assert events == ["metadata", "model", "finish"]
    assert len(calls) == 3


def test_completion_sync_retries_with_capped_exponential_backoff(
    tmp_path,
    monkeypatch,
):
    logger, _, events = _logger(tmp_path)
    outcomes = iter([1, 1, 0])
    delays = []

    monkeypatch.setattr("shutil.which", lambda executable: "/env/bin/wandb")

    def eventually_succeed(*args, **kwargs):
        events.append("sync")
        return SimpleNamespace(
            returncode=next(outcomes),
            stdout="",
            stderr="temporary network failure",
        )

    monkeypatch.setattr("subprocess.run", eventually_succeed)
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.sleep",
        delays.append,
    )

    logger.finalize()

    assert events == [
        "metadata",
        "model",
        "finish",
        "sync",
        "sync",
        "sync",
    ]
    assert delays == [2.0, 3.0]


def test_finish_failure_still_attempts_completion_sync(tmp_path, monkeypatch):
    logger, _, events = _logger(tmp_path)

    def fail_finish():
        events.append("finish")
        raise RuntimeError("service stopped")

    logger.run.finish = fail_finish
    monkeypatch.setattr("shutil.which", lambda executable: "/env/bin/wandb")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="done",
            stderr="",
        ),
    )

    with pytest.warns(UserWarning, match="run finish failed"):
        logger.finalize()

    assert events == ["metadata", "model", "finish"]
