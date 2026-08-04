import fcntl
import json
import os
from types import SimpleNamespace

from cw2.cw_data.cw_wandb_logger import WandBLogger


def _config(tmp_path, seed=0, resume_same_run=True):
    return {
        "_experiment_name": "experiment__sam.arg.eimetaworld/task-v2",
        "_rep_idx": 0,
        "iterations": 5000,
        "params": {
            "sampler": {
                "args": {
                    "env_id": "metaworld/task-v2",
                    "seed": seed,
                }
            }
        },
        "resume_model_dir": str(
            tmp_path / "resume" / "scope" / "rep_00" / "model"
        ),
        "resume_scope_name": "scope",
        "seed": seed,
        "wandb": {
            "enabled": True,
            "entity": "entity",
            "group": "group",
            "log_model": False,
            "project": "project",
            "resume_same_run": resume_same_run,
            "sync_on_finish": False,
        },
    }


def _fake_wandb_init(tmp_path, calls):
    def initialize(**kwargs):
        calls.append(kwargs)
        files_dir = tmp_path / f"run-{kwargs.get('id', 'new')}" / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            dir=str(files_dir),
            finish=lambda: None,
            id=kwargs.get("id", "new-run"),
        )

    return initialize


def _initialize_logger(tmp_path, monkeypatch, config):
    calls = []
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.wandb.init",
        _fake_wandb_init(tmp_path, calls),
    )
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.wandb.util.generate_id",
        lambda: "stable123",
    )
    logger = WandBLogger()
    logger.initialize(config, 0, str(tmp_path / "log" / "rep_00"))
    return logger, calls


def test_resume_same_run_reuses_persisted_id(tmp_path, monkeypatch):
    config = _config(tmp_path)
    first_logger, first_calls = _initialize_logger(
        tmp_path,
        monkeypatch,
        config,
    )

    assert first_calls[0]["id"] == "stable123"
    assert first_calls[0]["resume"] == "allow"
    first_name = first_calls[0]["name"]
    first_logger.finalize()

    second_logger, second_calls = _initialize_logger(
        tmp_path,
        monkeypatch,
        _config(tmp_path),
    )

    assert second_calls[0]["id"] == "stable123"
    assert second_calls[0]["resume"] == "allow"
    assert second_calls[0]["name"] == first_name
    second_logger.finalize()


def test_different_seed_uses_different_persistent_record(tmp_path, monkeypatch):
    generated_ids = iter(["seed-zero", "seed-one"])
    calls = []
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.wandb.init",
        _fake_wandb_init(tmp_path, calls),
    )
    monkeypatch.setattr(
        "cw2.cw_data.cw_wandb_logger.wandb.util.generate_id",
        lambda: next(generated_ids),
    )

    first_logger = WandBLogger()
    first_logger.initialize(
        _config(tmp_path, seed=0),
        0,
        str(tmp_path / "log-0"),
    )
    first_logger.finalize()

    second_logger = WandBLogger()
    second_logger.initialize(
        _config(tmp_path, seed=1),
        0,
        str(tmp_path / "log-1"),
    )
    second_logger.finalize()

    assert [call["id"] for call in calls] == ["seed-zero", "seed-one"]
    record_dir = tmp_path / "resume" / "scope" / ".wandb_runs"
    records = list(record_dir.glob("*.json"))
    assert len(records) == 2
    assert {
        json.loads(record.read_text())["seed"]
        for record in records
    } == {0, 1}


def test_disabled_resume_preserves_original_wandb_init(tmp_path, monkeypatch):
    logger, calls = _initialize_logger(
        tmp_path,
        monkeypatch,
        _config(tmp_path, resume_same_run=False),
    )

    assert "id" not in calls[0]
    assert "resume" not in calls[0]
    logger.finalize()


def test_active_duplicate_does_not_open_same_wandb_run(tmp_path, monkeypatch):
    config = _config(tmp_path)
    first_logger, first_calls = _initialize_logger(
        tmp_path,
        monkeypatch,
        config,
    )
    assert len(first_calls) == 1

    active_lock_path = (
        tmp_path / "resume" / "scope" / "rep_00" / "active.lock"
    )
    active_lock_path.parent.mkdir(parents=True, exist_ok=True)
    active_lock_fd = os.open(active_lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(active_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        second_logger, second_calls = _initialize_logger(
            tmp_path,
            monkeypatch,
            _config(tmp_path),
        )
        assert second_calls == []
        assert second_logger.run is None
        second_logger.finalize()
    finally:
        fcntl.flock(active_lock_fd, fcntl.LOCK_UN)
        os.close(active_lock_fd)
        first_logger.finalize()


def test_persistent_identity_is_written_to_metadata(tmp_path, monkeypatch):
    config = _config(tmp_path)
    logger, _ = _initialize_logger(tmp_path, monkeypatch, config)

    metadata_path = (
        tmp_path / "run-stable123" / "files" / "wandb-metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())

    assert metadata["persistent_wandb_run_id"] == "stable123"
    assert len(metadata["wandb_resume_identity"]) == 64
    logger.finalize()
