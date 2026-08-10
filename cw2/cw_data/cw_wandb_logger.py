import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import warnings
from random import random
from time import sleep

# To prevent conflicts between wandb and the joblib scheduler
# see https://github.com/wandb/client/issues/1525 for reference
os.environ["WANDB_START_METHOD"] = "thread"

from itertools import groupby
from typing import Dict, Iterable, List, Optional

import pandas as pd
import wandb

from cw2.cw_data import cw_logging
from cw2.util import convert_param_names, get_file_names_in_directory


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_DATA_DIR",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def group_parameters(list_of_strings: List[str]):
    """groups different strings that start with a common substring (using "." as delimiter)
        and outputs a single, more concise string.
    Example:
        outstring = group_parameters['local', 'mod.enc.tidentity', 'mod.hea.nhl5', 'mod.hea.ioFalse', 'mod.enc.hd64']
        % outstring will be 'local,mod_[enc_[hd64,tidentity],hea_[ioFalse,nhl5]]'
    """
    groups = []
    uniquekeys = []
    num_subgroups = 0
    substring = ""

    for k, g in groupby(sorted(list_of_strings), lambda string: string.split(".")[0]):
        groups.append(list(g))
        uniquekeys.append(k)

        if len(groups[-1]) == 1:
            substring += groups[-1][0] + ","
            num_subgroups += 1
        else:
            remainder = [s.replace(k, "", 1) for s in groups[-1]]
            remainder = [s.replace(".", "", 1) for s in remainder]
            if len(remainder) > 0:
                subgroups, num_subs = group_parameters(remainder)
                if num_subs > 1:
                    substring += k + "_[" + subgroups + "],"
                else:
                    substring += k + "_" + subgroups + ","
                num_subgroups += num_subs
    return substring[:-1], len(groups)


def build_sweep_group_name(
    base_group: Optional[str],
    experiment_name: str,
    max_parameter_length: int = 96,
    excluded_parameter_paths: Optional[Iterable[str]] = None,
    parameters: Optional[Dict] = None,
) -> Optional[str]:
    """Build one readable W&B group for each expanded sweep configuration."""
    experiment_name = str(experiment_name or "")
    experiment_base, separator, sweep_parameters = experiment_name.partition("__")
    sweep_parameters = sweep_parameters.strip("_")
    if not separator:
        return base_group

    parameters = parameters or {}
    for parameter_path in excluded_parameter_paths or ():
        value = _nested_parameter(parameters, parameter_path)
        if value is _MISSING_PARAMETER:
            continue
        abbreviated_parameter = convert_param_names(
            [str(parameter_path)],
            [value],
        )
        sweep_parameters = (
            f"_{sweep_parameters}_"
            .replace(f"_{abbreviated_parameter}_", "_")
            .strip("_")
        )

    group_prefix = str(base_group or experiment_base).strip()
    if not sweep_parameters:
        return group_prefix or None

    max_parameter_length = int(max_parameter_length)
    if max_parameter_length < 16:
        raise ValueError(
            "wandb.sweep_group_max_parameter_length must be at least 16."
        )

    parameter_tokens = [
        token for token in sweep_parameters.split("_") if token
    ]
    short_parameters = group_parameters(parameter_tokens)[0]
    if len(short_parameters) > max_parameter_length:
        digest = hashlib.sha256(
            sweep_parameters.encode("utf-8")
        ).hexdigest()[:10]
        prefix_length = max_parameter_length - len(digest) - 1
        short_parameters = (
            short_parameters[:prefix_length] + "~" + digest
        )

    if not group_prefix:
        return short_parameters
    return f"{group_prefix} | {short_parameters}"


_MISSING_PARAMETER = object()


def _nested_parameter(parameters: Dict, parameter_path: str):
    value = parameters
    for key in str(parameter_path).split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING_PARAMETER
        value = value[key]
    return value


class WandBLogger(cw_logging.AbstractLogger):
    def __init__(
        self,
        ignore_keys: Optional[Iterable] = None,
        allow_keys: Optional[Iterable] = None,
    ):
        super(WandBLogger, self).__init__(
            ignore_keys=ignore_keys, allow_keys=allow_keys
        )
        self.log_path = ""
        self.run = None
        self._wandb_resume_lock_fd = None
        self.wandb_run_id = None
        self.wandb_resume_identity = None

    def initialize(self, config: Dict, rep: int, rep_log_path: str) -> None:
        if "wandb" in config.keys():
            self.init_fields(config, rep, rep_log_path)
            try:
                self.connect_to_wandb()
            except Exception:
                self._release_wandb_resume_lock()
                raise

        else:
            warnings.warn("No 'wandb' field in yaml - Ignoring Weights & Biases Logger")

    def init_fields(self, config: Dict, rep: int, rep_log_path: str):
        self.run = None
        self._wandb_resume_lock_fd = None
        self.wandb_run_id = None
        self.wandb_resume_identity = None
        self.log_path = rep_log_path
        self.rep_idx = rep
        self.rep = config.get("seed", rep)
        self.config = config["wandb"]
        self.cw2_config = config
        reset_wandb_env()
        self.job_name = config["_experiment_name"].replace("__", "_")
        self.use_group_parameters = self.config.get("use_group_parameters", False)
        if self.use_group_parameters:
            self.job_name = group_parameters(self.job_name.split("_"))[0]
        try:
            rep_label = f"{int(self.rep):02d}"
        except (TypeError, ValueError):
            rep_label = str(self.rep)
        self.runname = f"{self.job_name}_rep_{rep_label}"

        # optional: change the job_type to a fixed alias if the option is present
        if "job_type" in self.config:
            self.job_name = self.config["job_type"]
        # have entity and group config entry optional
        self.entity = self.config.get("entity", None)
        self.group = self.config.get("group", None)
        self.group_by_sweep = self._bool_config_value(
            self.config.get("group_by_sweep", False)
        )
        self.sweep_group_resolved = self._bool_config_value(
            self.config.get("sweep_group_resolved", False)
        )
        if self.group_by_sweep and not self.sweep_group_resolved:
            self.group = build_sweep_group_name(
                base_group=self.group,
                experiment_name=config.get("_experiment_name", ""),
                max_parameter_length=self.config.get(
                    "sweep_group_max_parameter_length",
                    96,
                ),
                excluded_parameter_paths=self.config.get(
                    "sweep_group_exclude_parameters",
                    (),
                ),
                parameters=config.get("params", {}),
            )
        if self.group_by_sweep or self.sweep_group_resolved:
            print(f"[wandb] Sweep group: {self.group}", flush=True)
        self.wandb_local_dir = self._optional_path(
            os.environ.get("MPRL_WANDB_DIR", None)
            or self.config.get("local_dir", None)
        )
        self.wandb_cache_dir = self._optional_path(
            os.environ.get("MPRL_WANDB_CACHE_DIR", None)
            or self.config.get("cache_dir", None)
        )
        self.wandb_local_dir = self._ensure_optional_dir(
            self.wandb_local_dir, "wandb local_dir"
        )
        self.wandb_cache_dir = self._ensure_optional_dir(
            self.wandb_cache_dir, "wandb cache_dir"
        )
        if self.wandb_cache_dir is not None:
            os.environ["WANDB_CACHE_DIR"] = self.wandb_cache_dir
        # Get the model logging directory
        self.wandb_log_model = self.config.get("log_model", False)
        self.sync_on_finish = self._bool_config_value(
            self.config.get("sync_on_finish", False)
        )
        sync_timeout = self.config.get("sync_on_finish_timeout", 600)
        self.sync_on_finish_timeout = (
            None
            if sync_timeout is None or float(sync_timeout) <= 0
            else float(sync_timeout)
        )
        self.sync_on_finish_max_attempts = max(
            1,
            int(self.config.get("sync_on_finish_max_attempts", 3)),
        )
        self.sync_on_finish_retry_initial_delay = max(
            0.0,
            float(
                self.config.get(
                    "sync_on_finish_retry_initial_delay",
                    5.0,
                )
            ),
        )
        self.sync_on_finish_retry_max_delay = max(
            self.sync_on_finish_retry_initial_delay,
            float(
                self.config.get(
                    "sync_on_finish_retry_max_delay",
                    60.0,
                )
            ),
        )
        self.resume_same_run = self._bool_config_value(
            self.config.get("resume_same_run", False)
        )
        sync_retry_wait_budget = sum(
            min(
                self.sync_on_finish_retry_initial_delay * (2**retry_index),
                self.sync_on_finish_retry_max_delay,
            )
            for retry_index in range(
                self.sync_on_finish_max_attempts - 1
            )
        )
        sync_attempt_budget = (
            (self.sync_on_finish_timeout or 600)
            * self.sync_on_finish_max_attempts
        )
        resume_lock_timeout = self.config.get(
            "resume_lock_timeout",
            sync_attempt_budget + sync_retry_wait_budget + 60,
        )
        self.resume_lock_timeout = max(0.0, float(resume_lock_timeout))
        self.model_artifact_exclude = set(
            self.config.get("model_artifact_exclude", [])
        )
        if self.wandb_log_model:
            self.save_model_dir = os.path.join(self.log_path, "model")
            self.cw2_config["save_model_dir"] = self.save_model_dir
            self.model_name = self.config.get("model_name", "model")
        else:
            self.save_model_dir = None

    @staticmethod
    def _bool_config_value(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _optional_path(path):
        if path is None:
            return None
        path = str(path).strip()
        if path == "" or path.lower() in {"none", "null", "false"}:
            return None
        path = os.path.expanduser(os.path.expandvars(path))
        if "$" in path:
            return None
        return os.path.abspath(path)

    @staticmethod
    def _ensure_optional_dir(path, name):
        if path is None:
            return None
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError as error:
            warnings.warn(
                f"Could not create {name} at {path}: {error}. "
                "Falling back to the run log directory."
            )
            return None

    def connect_to_wandb(self):
        if self.resume_same_run and not self._prepare_persistent_run():
            return

        last_error = None
        for i in range(10):
            try:
                init_kwargs = dict(
                    project=self.cw2_config["wandb"]["project"],
                    entity=self.entity,
                    group=self.group,
                    job_type=self.job_name[:63],
                    name=self.runname[:63],
                    config=self.cw2_config["params"],
                    dir=self.wandb_local_dir or self.log_path,
                    settings=wandb.Settings(
                        _disable_stats=self.cw2_config["wandb"].get(
                            "disable_stats", False
                        )
                    ),
                    mode="online"
                    if self.cw2_config["wandb"].get("enabled", True)
                    else "disabled",
                )
                if self.wandb_run_id is not None:
                    init_kwargs.update(
                        id=self.wandb_run_id,
                        resume="allow",
                    )
                self.run = wandb.init(**init_kwargs)
                self.write_wandb_metadata()
                return  # if starting the run is successful, exit the loop (and in this case the function)
            except Exception as e:
                last_error = e
                # implement a simple randomized exponential backoff if starting a run fails
                waiting_time = ((random() / 50) + 0.01) * (2**i)
                # wait between 0.01 and 10.24 seconds depending on the random seed and the iteration of the exponent

                warnings.warn(
                    "Problem with starting wandb: {}. Trying again in {} seconds".format(
                        e, waiting_time
                    )
                )
                sleep(waiting_time)
        warnings.warn("wandb init failed several times.")
        raise last_error

    def _prepare_persistent_run(self):
        resume_model_dir = self._optional_path(
            self.cw2_config.get("resume_model_dir")
        )
        if resume_model_dir is None:
            warnings.warn(
                "wandb.resume_same_run is enabled, but resume_model_dir is "
                "unavailable. Starting a normal W&B run without persistent "
                "resume identity."
            )
            return True

        identity_payload = {
            "entity": self.entity,
            "experiment": self.cw2_config.get("_experiment_name"),
            "group": self.group,
            "iterations": self.cw2_config.get("iterations"),
            "params": self.cw2_config.get("params"),
            "project": self.cw2_config["wandb"]["project"],
            "resume_scope_name": self.cw2_config.get("resume_scope_name"),
            "seed": self.cw2_config.get("seed"),
        }
        identity_json = json.dumps(
            identity_payload,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
        identity = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        identity_dir = os.path.join(
            os.path.dirname(os.path.dirname(resume_model_dir)),
            ".wandb_runs",
        )
        os.makedirs(identity_dir, exist_ok=True)

        lock_path = os.path.join(identity_dir, f"{identity}.lock")
        if not self._acquire_wandb_resume_lock(
            lock_path=lock_path,
            resume_model_dir=resume_model_dir,
        ):
            return False

        record_path = os.path.join(identity_dir, f"{identity}.json")
        record = self._read_persistent_run_record(record_path, identity)
        checkpoint_preflight = self.cw2_config.get(
            "_checkpoint_resume_preflight",
            {},
        )
        checkpoint_status = (
            checkpoint_preflight.get("status")
            if isinstance(checkpoint_preflight, dict)
            else None
        )
        if checkpoint_status == "complete":
            print(
                "[wandb] Matching checkpoint is already complete; skipping W&B "
                "initialization for this duplicate task.",
                flush=True,
            )
            self._release_wandb_resume_lock()
            return False

        replaced_run_id = None
        non_resumable_statuses = {"fresh", "disabled", "explicit_load"}
        if record is not None and checkpoint_status in non_resumable_statuses:
            replaced_run_id = str(record["run_id"])
            record = None
            print(
                "[wandb] No compatible resumable checkpoint was selected; "
                "refusing to "
                f"resume stale W&B run {replaced_run_id}.",
                flush=True,
            )
        created = record is None
        if created:
            record = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entity": self.entity,
                "group": self.group,
                "identity": identity,
                "project": self.cw2_config["wandb"]["project"],
                "run_id": wandb.util.generate_id(),
                "run_name": self.runname,
                "seed": self.cw2_config.get("seed"),
            }
            if replaced_run_id is not None:
                record["replaces_run_id"] = replaced_run_id
                record["replacement_reason"] = "no_compatible_checkpoint"
            self._write_persistent_run_record(record_path, record)

        self.wandb_run_id = str(record["run_id"])
        self.wandb_resume_identity = identity
        self.runname = str(record.get("run_name") or self.runname)
        self.cw2_config["wandb_run_id"] = self.wandb_run_id
        self.cw2_config["wandb_resume_identity"] = self.wandb_resume_identity
        if replaced_run_id is not None:
            action = "Created replacement"
        else:
            action = "Created" if created else "Resuming"
        print(
            f"[wandb] {action} persistent run {self.wandb_run_id} "
            f"using {record_path}",
            flush=True,
        )
        return True

    def _acquire_wandb_resume_lock(self, lock_path, resume_model_dir):
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.resume_lock_timeout
        active_lock_path = os.path.join(
            os.path.dirname(resume_model_dir),
            "active.lock",
        )
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    os.close(lock_fd)
                    raise
                if self._file_lock_is_held(active_lock_path):
                    os.close(lock_fd)
                    print(
                        "[wandb] Matching experiment is already active; "
                        "skipping duplicate W&B initialization.",
                        flush=True,
                    )
                    return False
                if time.monotonic() >= deadline:
                    os.close(lock_fd)
                    raise TimeoutError(
                        "Timed out waiting for the previous process to finish "
                        f"the persistent W&B run lock: {lock_path}"
                    )
                sleep(0.25)
                continue

            self._wandb_resume_lock_fd = lock_fd
            return True

    @staticmethod
    def _file_lock_is_held(lock_path):
        if not os.path.isfile(lock_path):
            return False
        lock_fd = os.open(lock_path, os.O_RDWR)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    return True
                raise
            else:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                return False
        finally:
            os.close(lock_fd)

    @staticmethod
    def _read_persistent_run_record(record_path, identity):
        if not os.path.isfile(record_path):
            return None
        try:
            with open(record_path, "r") as record_file:
                record = json.load(record_file)
        except (OSError, json.JSONDecodeError) as error:
            warnings.warn(
                f"Ignoring unreadable persistent W&B run record "
                f"{record_path}: {error}"
            )
            return None
        if (
            not isinstance(record, dict)
            or record.get("identity") != identity
            or not record.get("run_id")
        ):
            warnings.warn(
                f"Ignoring invalid persistent W&B run record: {record_path}"
            )
            return None
        return record

    @staticmethod
    def _write_persistent_run_record(record_path, record):
        tmp_path = f"{record_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as record_file:
                json.dump(record, record_file, indent=2, sort_keys=True)
                record_file.flush()
                os.fsync(record_file.fileno())
            os.replace(tmp_path, record_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _release_wandb_resume_lock(self):
        lock_fd = self._wandb_resume_lock_fd
        if lock_fd is None:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            self._wandb_resume_lock_fd = None

    def process(self, data: dict) -> None:
        if self.run is not None:
            log_step = data.get("num_iterations", data.get("iter", None))
            final_iteration = (
                self.cw2_config.get("iterations") is not None
                and log_step is not None
                and log_step >= self.cw2_config["iterations"] - 1
            )

            # Skip logging if interval is defined but not satisfied.
            # Keep the first point and final point even when they are not exact
            # multiples of log_interval.
            log_interval = self.config.get("log_interval", None)
            first_iteration = log_step is not None and log_step <= 1
            if (
                log_interval is not None
                and log_step is not None
                and not first_iteration
                and not final_iteration
                and log_step % log_interval != 0
            ):
                return

            step = (
                self.cw2_config["iterations"]
                if final_iteration
                else log_step
            )

            if "histogram" in self.config:
                for el in self.config["histogram"]:
                    if el in data:
                        self.run.log(
                            {el: wandb.Histogram(np_histogram=data[el])},
                            step=step,
                        )
            filtered_data = self.filter(data)
            self.run.log(filtered_data, step=step)

    def finalize(self) -> None:
        try:
            if self.run is not None:
                run_dir = self._local_run_dir()
                run_id = getattr(self.run, "id", None)
                for operation_name, operation in (
                    ("metadata update", self.write_wandb_metadata),
                    ("model artifact upload", self.log_model),
                    ("run finish", self.run.finish),
                ):
                    try:
                        operation()
                    except Exception as error:
                        warnings.warn(
                            f"W&B {operation_name} failed during finalization: {error}"
                        )

                if self.sync_on_finish:
                    self._sync_local_run(run_dir=run_dir, run_id=run_id)
        finally:
            self.run = None
            self._release_wandb_resume_lock()

    def _local_run_dir(self):
        run_files_dir = getattr(self.run, "dir", None)
        if not run_files_dir:
            return None
        run_files_dir = os.path.abspath(run_files_dir)
        if os.path.basename(run_files_dir) == "files":
            return os.path.dirname(run_files_dir)
        return run_files_dir

    def _sync_local_run(self, run_dir, run_id=None):
        if run_dir is None or not os.path.isdir(run_dir):
            warnings.warn(
                "W&B completion sync was requested, but the local run directory "
                f"is unavailable: {run_dir}"
            )
            return

        wandb_executable = shutil.which("wandb")
        if wandb_executable is None:
            warnings.warn(
                "W&B completion sync was requested, but the wandb executable "
                "was not found."
            )
            return

        command = [
            wandb_executable,
            "sync",
            "--include-online",
            "--include-synced",
            "--no-sync-tensorboard",
            "--append",
        ]
        if run_id:
            command.extend(["--id", str(run_id)])
        command.append(run_dir)

        for attempt in range(1, self.sync_on_finish_max_attempts + 1):
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.sync_on_finish_timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                details = str(error)
            else:
                if result.returncode == 0:
                    print(
                        "[wandb] Completion sync succeeded on attempt "
                        f"{attempt}/{self.sync_on_finish_max_attempts}: "
                        f"{run_dir}",
                        flush=True,
                    )
                    return True
                output = (result.stderr or result.stdout or "").strip()
                details = (
                    f"exit code {result.returncode}"
                    + (f": {output}" if output else "")
                )

            if attempt >= self.sync_on_finish_max_attempts:
                warnings.warn(
                    "W&B completion sync failed after "
                    f"{self.sync_on_finish_max_attempts} attempt(s): "
                    f"{details}. Local data remains available at {run_dir}"
                )
                return False

            delay = min(
                self.sync_on_finish_retry_initial_delay
                * (2 ** (attempt - 1)),
                self.sync_on_finish_retry_max_delay,
            )
            print(
                f"[wandb] Completion sync attempt {attempt}/"
                f"{self.sync_on_finish_max_attempts} failed: {details}. "
                f"Retrying in {delay:g} seconds.",
                flush=True,
            )
            sleep(delay)

        return False

    def _git_metadata_payload(self):
        git_repos = self.cw2_config.get("git_repos")
        git_snapshot = self.cw2_config.get("git_snapshot")

        payload = {}
        if isinstance(git_repos, dict):
            payload["git_repos"] = git_repos
            for repo_key, commit in git_repos.items():
                payload[f"git_commit_{repo_key}"] = commit

        if isinstance(git_snapshot, dict):
            payload["git_snapshot"] = git_snapshot
            for key in (
                "copy_manifest_sha256",
                "copy_manifest_num_files",
                "source_path",
                "copy_path",
                "copied_at",
            ):
                if key in git_snapshot:
                    payload[key] = git_snapshot[key]

            snapshot_repos = git_snapshot.get("git_repos")
            if isinstance(snapshot_repos, dict):
                payload.setdefault("git_repos", snapshot_repos)
                for repo_key, commit in snapshot_repos.items():
                    payload.setdefault(f"git_commit_{repo_key}", commit)

        return payload

    def _wandb_metadata_path(self):
        if self.run is None or getattr(self.run, "dir", None) is None:
            return None
        return os.path.join(self.run.dir, "wandb-metadata.json")

    def write_wandb_metadata(self):
        payload = self._git_metadata_payload()
        if self.wandb_run_id is not None:
            payload["persistent_wandb_run_id"] = self.wandb_run_id
            payload["wandb_resume_identity"] = self.wandb_resume_identity
        if not payload:
            return

        metadata_path = self._wandb_metadata_path()
        if metadata_path is None:
            return

        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata = {}
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                metadata = {}

        metadata.update(payload)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def load(self):
        pass

    def log_model(self):
        """
        Log model as an Artifact

        Returns:
            None
        """
        if self.wandb_log_model is False:
            return

        # Initialize wandb artifact
        model_artifact = wandb.Artifact(name=self.model_name, type="model")

        # Get all file names in log dir
        file_names = get_file_names_in_directory(self.save_model_dir)

        if file_names is None:
            warnings.warn("save model dir is not available or empty.")
            return

        # Add files into artifact
        logged_file_count = 0
        for file in file_names:
            if not self._should_log_model_file(file):
                continue
            model_artifact.add_file(os.path.join(self.save_model_dir, file))
            logged_file_count += 1

        if logged_file_count == 0:
            warnings.warn("save model dir has no files selected for wandb upload.")
            return

        aliases = ["latest", f"finished-rep-{self.rep}"]

        # Log and upload
        self.run.log_artifact(model_artifact, aliases=aliases)

    def _should_log_model_file(self, file_name):
        base_name = os.path.basename(file_name)
        if base_name in self.model_artifact_exclude:
            return False
        if base_name.endswith(".tmp") or ".tmp." in base_name:
            return False
        if (
            base_name == "checkpoint_state"
            or base_name.startswith("checkpoint_state_")
        ):
            return False
        return True

    def log_plot(self, x, y, column_names=("x", "y"), plot_id="plot", title="Plot"):
        data = [list(i) for i in zip(x, y)]
        table = wandb.Table(data=data, columns=column_names)
        self.run.log(
            {
                plot_id: wandb.plot.line(
                    table, column_names[0], column_names[0], title=title
                )
            }
        )

    def log_table(self, data, table_id="table"):
        assert type(data) is pd.DataFrame
        table = wandb.Table(dataframe=data)
        self.run.log({table_id: table})
