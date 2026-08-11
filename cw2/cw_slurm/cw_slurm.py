import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import __main__

import cw2.cw_config.cw_conf_keys as CKEYS
import cw2.cw_slurm.cw_slurm_keys as SKEYS
from cw2 import cli_parser, cw_error, util
from cw2.cw_config import cw_config
from cw2.cw_data import cw_logging


class SlurmConfig:
    def __init__(self, conf: cw_config.Config) -> None:
        self.conf = conf
        self.slurm_conf = conf.slurm_config

        if self.slurm_conf is None:
            raise cw_error.MissingConfigError(
                "No SLURM configuration found in {}".format(self.conf.config_path)
            )

        self._check_template()

    def _check_template(self):
        """check if an sbatch.sh template is present.
        If no costum template has been specified, the default will be used.
        """

        if SKEYS.TEMPLATE_PATH not in self.slurm_conf:
            self.slurm_conf[SKEYS.TEMPLATE_PATH] = os.path.join(
                os.path.dirname(__file__), "../default_sbatch.sh"
            )

        if not os.path.exists(self.slurm_conf[SKEYS.TEMPLATE_PATH]):
            raise cw_error.ConfigKeyError(
                "Could not find default sbatch template. Please specify your own 'path_to_template'."
            )

    def _complete_optionals(self):
        """Fill in any optional values."""

        sc: dict = self.slurm_conf

        exp_output_path = self.conf.exp_configs[0][CKEYS.i_BASIC_PATH]

        # CREATE OPTIONAL COLLECTIONS
        # Must be done first:
        sc.setdefault(SKEYS.SBATCH_ARGS, {})

        # SET DEFAULT VALUES
        sc.setdefault(SKEYS.SLURM_LOG, os.path.join(exp_output_path, "slurmlog"))
        sc.setdefault(SKEYS.SLURM_OUT, os.path.join(exp_output_path, "sbatch.sh"))
        sc[SKEYS.SLURM_LOG] = os.path.abspath(sc[SKEYS.SLURM_LOG])
        sc[SKEYS.SLURM_OUT] = os.path.abspath(sc[SKEYS.SLURM_OUT])
        sc.setdefault(SKEYS.ACCOUNT, "")

        # COMPLEX CONVERSIONS
        if isinstance(sc[SKEYS.TIME], int):
            sc[SKEYS.TIME] = "{:d}:{:d}:00".format(
                sc[SKEYS.TIME] // 60, sc[SKEYS.TIME] % 60
            )

        if SKEYS.MEM in sc and SKEYS.CPU_MEM in sc:
            raise cw_error.ConfigKeyError(
                "Slurm memory must be configured with either mem or "
                "mem-per-cpu, not both."
            )

        for memory_key in (SKEYS.MEM, SKEYS.CPU_MEM):
            if memory_key not in sc:
                continue
            configured_memory = sc[memory_key]
            existing_memory = sc[SKEYS.SBATCH_ARGS].get(memory_key)
            if (
                existing_memory is not None
                and existing_memory != configured_memory
            ):
                raise cw_error.ConfigKeyError(
                    "Conflicting Slurm {} values: top-level {!r} and "
                    "sbatch_args {!r}.".format(
                        memory_key,
                        configured_memory,
                        existing_memory,
                    )
                )
            sc[SKEYS.SBATCH_ARGS][memory_key] = configured_memory

        # DEFAULT OR COMPLEX CONVERSION
        if SKEYS.VENV in sc:
            sc[SKEYS.VENV] = "source activate {}".format(sc[SKEYS.VENV])
        else:
            sc[SKEYS.VENV] = ""

        if SKEYS.SH_LINES in sc:
            sc[SKEYS.SH_LINES] = "\n".join(sc[SKEYS.SH_LINES])
        else:
            sc[SKEYS.SH_LINES] = ""

    def _complete_cli_args(self):
        """identify and process the relevant CLI flags from the original call."""
        sc = self.slurm_conf
        cw_options = cli_parser.Arguments().get()

        sc[SKEYS.CW_ARGS] = ""
        if cw_options["overwrite"]:
            sc[SKEYS.CW_ARGS] += " -o"
        if cw_options["experiments"] is not None:
            sc[SKEYS.CW_ARGS] += " -e " + " ".join(cw_options["experiments"])

    def _complete_sbatch_args(self):
        """if optional SBATCH arguments are present, build a corresponding string."""
        sc = self.slurm_conf

        if SKEYS.SBATCH_ARGS not in sc:  # Check if empty
            sc[SKEYS.SBATCH_ARGS] = ""
            return
        else:  # Else build String
            sbatch_args = sc.get(SKEYS.SBATCH_ARGS)

            args_list = ["#SBATCH --{} {}".format(k, v) for k, v in sbatch_args.items()]
            sc[SKEYS.SBATCH_ARGS] = "\n".join(args_list)

    def finalize(self, num_jobs: int):
        """enrich slurm configuration with dynamically computed values

        Args:
            num_jobs (int): total number of defined jobs
        """

        # counting starts at 0
        self.slurm_conf[SKEYS.LAST_IDX] = num_jobs - 1

        # Order is important!
        self._complete_optionals()
        self._complete_cli_args()
        self._complete_sbatch_args()


class SlurmDirectoryManager:
    MODE_COPY = "COPY"
    MODE_MULTI = "MULTI"
    MODE_NOCOPY = "NOCOPY"
    MODE_ZIP = "ZIP"
    RUNTIME_COPY_EXCLUDES = (
        ".idea/",
        ".vscode/",
        ".cw2_git_repos.json",
    )
    GIT_SNAPSHOT_FILE = ".cw2_git_repos.json"
    GIT_REPO_MARKERS = (
        ("mprl", "mprl"),
        ("mp_pytorch", "mp_pytorch"),
        ("fancy_gym", "fancy_gym"),
        ("metaworld", "metaworld"),
        ("cw2", "cw2"),
        ("git_repos_tracker", "git_repos_tracker"),
    )

    def __init__(self, sc: SlurmConfig, conf: cw_config.Config) -> None:
        self.slurm_config = sc
        self.conf = conf
        self.m = self.set_mode()
        os.makedirs(sc.slurm_conf[SKEYS.SLURM_LOG], exist_ok=True)

    def set_mode(self):
        """find which code-copy mode is configured

        Raises:
            cw_error.ConfigKeyError: if incomplete definition

        Returns:
            code-copy mode
        """
        sc = self.slurm_config.slurm_conf

        # COUNT MISSING ARGS
        cp_error_count = 0
        missing_arg = ""
        if SKEYS.EXP_CP_AUTO not in sc and SKEYS.EXP_CP_DST not in sc:
            cp_error_count += 1
            missing_arg = SKEYS.EXP_CP_DST

        if SKEYS.EXP_CP_SRC not in sc:
            cp_error_count += 1
            missing_arg = SKEYS.EXP_CP_SRC

        # MODE SWITCH
        if cp_error_count == 1:
            raise cw_error.ConfigKeyError(
                "Incomplete SLURM experiment copy config. Missing key: {}".format(
                    missing_arg
                )
            )

        cw_options = cli_parser.Arguments().get()
        if cw_options.get("zip"):
            return self.MODE_ZIP

        if cw_options.get("multicopy"):
            if cp_error_count == 0:
                return self.MODE_MULTI
            else:
                raise cw_error.ConfigKeyError(
                    "Incomplete SLURM experiment copy config. Please define SRC and DST for --multicopy"
                )

        if cp_error_count == 0:
            return self.MODE_COPY
        return self.MODE_NOCOPY

    def dir_size_validation(self, src):
        """validates that the SRC for code copy is below 200MB in size

        Args:
            src: src path

        Raises:
            cw_error.ConfigKeyError: if directory is greater than 200MB
        """
        cw_options = cli_parser.Arguments().get()
        if cw_options.get("skipsizecheck"):
            return

        dirsize = self._copy_manifest_size(src)
        if dirsize > 200.0:
            cw_logging.getLogger().warning(
                "SourceDir {} is greater than 200MByte".format(src)
            )
            msg = (
                "Directory {} is greater than 200MByte."
                " If you are sure you want to copy/zip this dir, use --skipsizecheck."
                "\nElse check experiment_copy__ configuration keys".format(src)
            )
            raise cw_error.ConfigKeyError(msg)

    def get_exp_src(self) -> str:
        """retrieves the code-copy src.
        Uses CWD as default unless specified

        Returns:
            src path
        """
        sc = self.slurm_config.slurm_conf
        return sc.get(SKEYS.EXP_CP_SRC, os.getcwd())

    def get_exp_dst(self):
        """retrieves the code-copy dst.
        Uses CWD as default unless specified

        Returns:
            src path
        """
        sc = self.slurm_config.slurm_conf
        if SKEYS.EXP_CP_AUTO in sc and SKEYS.EXP_CP_DST not in sc:
            sc[SKEYS.EXP_CP_DST] = os.path.join(
                sc.get(SKEYS.EXP_CP_AUTO),
                datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            )
        if SKEYS.EXP_CP_DST in sc:
            return sc[SKEYS.EXP_CP_DST]
        else:
            exp_output_path = self.conf.exp_configs[0][CKEYS.i_BASIC_PATH]
            return os.path.join(exp_output_path, "code")

    @staticmethod
    def _git_root(src: str):
        try:
            return subprocess.check_output(
                ["git", "-C", src, "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @staticmethod
    def _git_output(git_root: str, *args: str):
        try:
            return subprocess.check_output(
                ["git", "-C", git_root, *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @classmethod
    def _repo_key_from_root(cls, git_root: str):
        base_name = os.path.basename(os.path.abspath(git_root))
        if base_name == "dt_rl" and os.path.isdir(os.path.join(git_root, "mprl")):
            return "mprl"

        for repo_key, package_dir in cls.GIT_REPO_MARKERS:
            if base_name == package_dir or os.path.isdir(os.path.join(git_root, package_dir)):
                return repo_key

        return base_name

    def _candidate_git_roots(self, src: str):
        candidates = [
            src,
            os.getcwd(),
            os.path.dirname(__file__),
        ]
        candidates.extend([path for path in sys.path if path])
        candidates.extend(
            path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
        )

        src_git_root = self._git_root(src)
        if src_git_root is not None:
            for deps_root in (
                os.path.join(src_git_root, ".deps"),
                os.path.join(os.path.dirname(src_git_root), ".deps"),
            ):
                if not os.path.isdir(deps_root):
                    continue
                for item in os.listdir(deps_root):
                    candidates.append(os.path.join(deps_root, item))

        git_roots = []
        seen = set()
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            git_root = self._git_root(path)
            if git_root is None:
                continue
            git_root = os.path.abspath(git_root)
            if git_root in seen:
                continue
            seen.add(git_root)
            git_roots.append(git_root)

        return git_roots

    def _repo_snapshot(self, git_root: str):
        commit = self._git_output(git_root, "rev-parse", "HEAD")
        if commit is None:
            return None

        branch = self._git_output(git_root, "rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            branch = None
        status = self._git_output(git_root, "status", "--porcelain") or ""
        status_lines = [line for line in status.splitlines() if line]

        return {
            "path": git_root,
            "branch": branch,
            "commit": commit,
            "clean": len(status_lines) == 0,
            "num_changed": len(status_lines),
            "num_untracked": sum(line.startswith("??") for line in status_lines),
        }

    @staticmethod
    def _manifest_sha256(src: str, manifest):
        if manifest is None:
            return None

        src = os.path.abspath(src)
        digest = hashlib.sha256()
        for rel_path in manifest:
            path = os.path.join(src, *rel_path.split("/"))
            digest.update(rel_path.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            if os.path.islink(path):
                digest.update(b"symlink\0")
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
                digest.update(b"\0")
                continue
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            digest.update(b"\0")

        return digest.hexdigest()

    def _write_git_snapshot(self, src: str, dst: str, manifest):
        repos = {}
        for git_root in self._candidate_git_roots(src):
            repo_info = self._repo_snapshot(git_root)
            if repo_info is None:
                continue
            repos[self._repo_key_from_root(git_root)] = repo_info

        if not repos:
            return

        snapshot = {
            "copied_at": datetime.datetime.now().isoformat(),
            "source_path": os.path.abspath(src),
            "copy_path": os.path.abspath(dst),
            "copy_manifest_sha256": self._manifest_sha256(dst, manifest),
            "copy_manifest_num_files": len(manifest) if manifest is not None else None,
            "git_repos": {
                repo_key: repo_info["commit"]
                for repo_key, repo_info in repos.items()
            },
            "git_snapshot": repos,
        }

        with open(os.path.join(dst, self.GIT_SNAPSHOT_FILE), "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)

    def _copy_manifest(self, src):
        src = os.path.abspath(src)
        if not os.path.isdir(src):
            return None

        git_root = self._git_root(src)
        if git_root is None:
            return None

        try:
            raw_paths = subprocess.check_output(
                [
                    "git",
                    "-C",
                    git_root,
                    "ls-files",
                    "-z",
                    "--cached",
                    "--modified",
                    "--others",
                    "--exclude-standard",
                ],
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        rel_src = os.path.relpath(src, git_root)
        rel_src_posix = "" if rel_src == "." else rel_src.replace(os.sep, "/").rstrip("/") + "/"
        manifest = set()
        for raw_path in raw_paths.split(b"\0"):
            if not raw_path:
                continue
            rel_git_path = raw_path.decode(sys.getfilesystemencoding(), errors="surrogateescape")
            if rel_src_posix:
                if not rel_git_path.startswith(rel_src_posix):
                    continue
                rel_path = rel_git_path[len(rel_src_posix):]
            else:
                rel_path = rel_git_path

            if self._runtime_copy_excluded(rel_path):
                continue

            src_path = os.path.join(src, *rel_path.split("/"))
            if os.path.isfile(src_path) or os.path.islink(src_path):
                manifest.add(rel_path)

        return sorted(manifest)

    def _runtime_copy_excluded(self, rel_path):
        rel_path = rel_path.replace(os.sep, "/")
        return any(
            rel_path == pattern.rstrip("/") or rel_path.startswith(pattern)
            for pattern in self.RUNTIME_COPY_EXCLUDES
        )

    def _copy_manifest_size(self, src):
        manifest = self._copy_manifest(src)
        if manifest is None:
            return util.get_size(src)

        total_size = 0
        src = os.path.abspath(src)
        for rel_path in manifest:
            src_path = os.path.join(src, *rel_path.split("/"))
            total_size += os.path.getsize(src_path)
        return total_size / 1000000.0

    def _copy_manifest_files(self, src, dst, manifest):
        src = os.path.abspath(src)
        for rel_path in manifest:
            s = os.path.join(src, *rel_path.split("/"))
            d = os.path.join(dst, *rel_path.split("/"))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)

    @staticmethod
    def _copy_all_files(src, dst):
        ign = shutil.ignore_patterns("*.pyc", "tmp*", ".git*", ".idea", ".vscode")
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, ignore=ign)
            else:
                shutil.copy2(s, d)

    def zip_exp(self):
        """procedure for creating a zip backup"""
        src = self.get_exp_src()
        dst = self.get_exp_dst()
        self.dir_size_validation(src)

        shutil.make_archive(dst, "zip", src)

    def create_single_copy(self):
        """creates a copy of the exp for slurm execution"""
        src = self.get_exp_src()
        dst = self.get_exp_dst()
        self._copy_files(src, dst)

    def create_multi_copy(self, num_jobs: int):
        """creates multiple copies of the exp, one for each slurm job

        Args:
            num_jobs (int): number of total jobs
        """
        src = self.get_exp_src()
        dst_base = self.get_exp_dst()

        for i in range(num_jobs):
            dst = os.path.join(dst_base, str(i))
            self._copy_files(src, dst)

        # Add MultiCopy ChangeDir to Slurmconf
        self.slurm_config.slurm_conf[SKEYS.SH_LINES] += "\ncd {} \n".format(
            os.path.join(self.get_exp_dst(), "$SLURM_ARRAY_TASK_ID")
        )

    def _copy_files(self, src, dst):
        """copies files from src to dst

        Args:
            src: source directory
            dst: destination directory

        Raises:
            cw_error.ConfigKeyError: if the dst is inside the source. Recursive copying!
            cw_error.ConfigKeyError: if the dst already exists and overwrite is not forced.
        """
        self.dir_size_validation(src)

        # Check Filesystem
        if util.check_subdir(src, dst):
            raise cw_error.ConfigKeyError(
                "experiment_copy_dst is a subdirectory of experiment_copy_src. Recursive Copying is bad."
            )
        try:
            os.makedirs(dst, exist_ok=cli_parser.Arguments().get()["overwrite"])
        except FileExistsError:
            raise cw_error.ConfigKeyError(
                "{} already exists. Please define a different 'experiment_copy_dst', use '-o' to overwrite or '--nocodecopy' to skip."
            )

        # Copy the same file set Git would sync: tracked files plus untracked
        # files that are not ignored by .gitignore/.git/info/exclude.
        manifest = self._copy_manifest(src)
        if manifest is None:
            self._copy_all_files(src, dst)
        else:
            self._copy_manifest_files(src, dst, manifest)
        self._write_git_snapshot(src, dst, manifest)

    def move_files(self, num_jobs: int):
        """moves exp files according to detected copy mode
        Args:
            num_jobs: number of slurm jobs for multi-copy
        """
        # Check Skip Flag
        cw_options = cli_parser.Arguments().get()
        if cw_options.get("nocodecopy"):
            print("Skipping Code Copy")
            return

        if self.m == self.MODE_COPY:
            self.create_single_copy()

        if self.m == self.MODE_MULTI:
            self.create_multi_copy(num_jobs)

        if self.m == self.MODE_ZIP:
            self.zip_exp()

    def get_exp_exec_dir(self) -> str:
        """retrieves the experiment execution dir.
        This dir depends on the exp_copy_dst

        Returns:
            str: experiment execution directory
        """
        if self.m == self.MODE_COPY:
            return self._map_path_into_copy(os.path.abspath(self.get_exp_dst()))

        if self.m == self.MODE_MULTI:
            return self._map_path_into_copy(
                os.path.join(os.path.abspath(self.get_exp_dst()), "$SLURM_ARRAY_TASK_ID")
            )

        return self.get_exp_src()

    def _map_path_into_copy(self, copied_root: str) -> str:
        """Map the original working directory into the copied source tree."""
        src = os.path.abspath(self.get_exp_src())
        cwd = os.path.abspath(os.getcwd())
        if not util.check_subdir(src, cwd):
            return copied_root

        rel_cwd = os.path.relpath(cwd, src)
        if rel_cwd == ".":
            return copied_root
        return os.path.join(copied_root, rel_cwd)

    def get_config_exec_path(self, config_path: str) -> str:
        """Copy runtime-generated config into CODE_COPY and return its exec path."""
        if self.m in [self.MODE_NOCOPY, self.MODE_ZIP]:
            return config_path

        src = os.path.abspath(self.get_exp_src())
        config_path = os.path.abspath(config_path)
        try:
            if os.path.commonpath([src, config_path]) != src:
                return config_path
        except ValueError:
            return config_path

        rel_config_path = os.path.relpath(config_path, src)
        if self.m == self.MODE_COPY:
            copied_config_path = os.path.join(
                os.path.abspath(self.get_exp_dst()), rel_config_path
            )
            self._copy_runtime_config(config_path, copied_config_path)
            return copied_config_path

        if self.m == self.MODE_MULTI:
            dst_base = os.path.abspath(self.get_exp_dst())
            for idx in range(self.slurm_config.slurm_conf[SKEYS.LAST_IDX] + 1):
                self._copy_runtime_config(
                    config_path,
                    os.path.join(dst_base, str(idx), rel_config_path),
                )
            return os.path.join(dst_base, "$SLURM_ARRAY_TASK_ID", rel_config_path)

        return config_path

    @staticmethod
    def _copy_runtime_config(src: str, dst: str):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    def freeze_slurm_script(self, script_path: str):
        """Store this submission's sbatch script inside CODE_COPY for exact resubmission."""
        if self.m in [self.MODE_NOCOPY, self.MODE_ZIP]:
            return

        if self.m == self.MODE_COPY:
            frozen_script_path = os.path.join(
                os.path.abspath(self.get_exp_dst()), "sbatch.sh"
            )
            self._copy_runtime_config(script_path, frozen_script_path)
            return

        if self.m == self.MODE_MULTI:
            dst_base = os.path.abspath(self.get_exp_dst())
            for idx in range(self.slurm_config.slurm_conf[SKEYS.LAST_IDX] + 1):
                self._copy_runtime_config(
                    script_path,
                    os.path.join(dst_base, str(idx), "sbatch.sh"),
                )

    def get_py_path(self) -> str:
        """computes a modified python path, depending on the experiment_copy procedure

        Returns:
            str: python path setting
        """
        if self.m in [self.MODE_NOCOPY, self.MODE_ZIP]:
            return ""

        pypath = sys.path.copy()

        src = self.get_exp_src()
        dst = self.get_exp_dst()

        if self.m == self.MODE_MULTI:
            dst = os.path.join(dst, "$SLURM_ARRAY_TASK_ID")

        new_path = [
            x.replace(os.path.abspath(src), os.path.abspath(dst)) for x in pypath
        ]
        copied_root = os.path.abspath(dst)
        if copied_root not in new_path:
            new_path.insert(0, copied_root)
        return "export PYTHONPATH=" + ":".join(new_path) + ":$PYTHONPATH"


def _job_concurrent_task_capacity(cw_job) -> int:
    task_count = len(cw_job.tasks)
    configured_parallelism = cw_job.n_parallel
    if isinstance(configured_parallelism, str):
        if configured_parallelism.strip().lower() == "auto":
            return task_count
        raise cw_error.ConfigKeyError(
            "reps_in_parallel must be a positive integer or 'auto'."
        )

    configured_parallelism = int(configured_parallelism)
    if configured_parallelism < 1:
        raise cw_error.ConfigKeyError(
            "reps_in_parallel must be at least 1."
        )
    return min(task_count, configured_parallelism)


def _auto_gpu_settings(conf: cw_config.Config):
    slurm_conf = conf.slurm_config
    reps_per_gpu = int(slurm_conf.get("reps_per_gpu", 1))
    if reps_per_gpu < 1:
        raise cw_error.ConfigKeyError("reps_per_gpu must be at least 1.")

    raw_node_counts = slurm_conf.get("auto_gpu_node_counts", [1, 2, 4])
    try:
        node_counts = sorted({int(count) for count in raw_node_counts})
    except (TypeError, ValueError) as exc:
        raise cw_error.ConfigKeyError(
            "auto_gpu_node_counts must be a list of positive integers."
        ) from exc
    if not node_counts or node_counts[0] < 1:
        raise cw_error.ConfigKeyError(
            "auto_gpu_node_counts must contain positive integers."
        )

    raw_models = slurm_conf.get("auto_gpu_models", [])
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if not isinstance(raw_models, list):
        raise cw_error.ConfigKeyError(
            "auto_gpu_models must be a list of Slurm node feature names."
        )
    gpu_models = [str(model).strip() for model in raw_models]
    if any(
        not model
        or not all(char.isalnum() or char in "_.-" for char in model)
        for model in gpu_models
    ):
        raise cw_error.ConfigKeyError(
            "auto_gpu_models contains an invalid Slurm node feature."
        )

    fallback_count = int(slurm_conf.get("auto_gpu_fallback_count", 2))
    if fallback_count not in node_counts:
        raise cw_error.ConfigKeyError(
            "auto_gpu_fallback_count must be present in auto_gpu_node_counts."
        )

    sbatch_args = slurm_conf.get("sbatch_args", {})
    if not isinstance(sbatch_args, dict):
        raise cw_error.ConfigKeyError(
            "num_gpus=auto requires sbatch_args to be a dictionary."
        )
    fixed_gpu_args = {
        "gres",
        "gpus",
        "gpus-per-node",
        "gpus-per-task",
    }.intersection(sbatch_args)
    if fixed_gpu_args:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto cannot be combined with fixed GPU sbatch arguments: "
            "{}.".format(", ".join(sorted(fixed_gpu_args)))
        )

    return reps_per_gpu, node_counts, gpu_models, fallback_count


def _auto_gpu_constraint(gpu_count: int, gpu_models: list) -> str:
    gpu_feature = "GPUx{}".format(gpu_count)
    if not gpu_models:
        return gpu_feature
    return "{}&({})".format(gpu_feature, "|".join(gpu_models))


def query_idle_auto_gpu_nodes(
    conf: cw_config.Config,
    node_counts: list,
    gpu_models: list,
):
    partition = conf.slurm_config.get("partition")
    if not partition:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto requires a Slurm partition."
        )
    delimiter = "\t"
    command = [
        "sinfo",
        "-h",
        "-N",
        "-t",
        "idle",
        "-p",
        partition,
        "-o",
        delimiter.join(["%N", "%f", "%T"]),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto could not inspect idle Slurm nodes with: {}."
            .format(" ".join(command))
        ) from exc

    idle_counts = {count: 0 for count in node_counts}
    model_features = set(gpu_models)
    counted_nodes = set()
    for line in result.stdout.splitlines():
        fields = line.split(delimiter)
        if len(fields) != 3:
            raise cw_error.ConfigKeyError(
                "Could not parse idle Slurm node data: {!r}.".format(line)
            )
        node_name, raw_features, state = (
            field.strip() for field in fields
        )
        # Slurm may append state flags (for example ``idle~`` for a
        # powered-down idle node). Such nodes still satisfy ``-t idle`` and
        # are valid candidates for scheduling.
        if not state.lower().startswith("idle"):
            continue
        features = {
            feature.strip()
            for feature in raw_features.split(",")
            if feature.strip()
        }
        if model_features and not features.intersection(model_features):
            continue
        if node_name in counted_nodes:
            continue
        for gpu_count in node_counts:
            if "GPUx{}".format(gpu_count) in features:
                idle_counts[gpu_count] += 1
                counted_nodes.add(node_name)
                break
    return idle_counts


def _constraint_has_feature(constraint: str, feature: str) -> bool:
    feature_pattern = r"(?<![A-Za-z0-9_.-]){}(?![A-Za-z0-9_.-])"
    return re.search(feature_pattern.format(re.escape(feature)), constraint) is not None


def _pending_constraint_matches_models(
    constraint: str,
    gpu_models: list,
) -> bool:
    if not gpu_models:
        return True
    if any(
        _constraint_has_feature(constraint, model)
        for model in gpu_models
    ):
        return True

    # Auto-generated constraints put the accepted GPU models immediately
    # after GPUxN. A disjoint group cannot consume one of our eligible nodes.
    model_group = re.search(
        r"GPUx\d+\s*&\s*\(([^)]*)\)",
        constraint,
    )
    return model_group is None


def query_pending_priority_auto_gpu_demand(
    conf: cw_config.Config,
    node_counts: list,
    gpu_models: list,
):
    partition = conf.slurm_config.get("partition")
    if not partition:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto requires a Slurm partition."
        )

    delimiter = "\t"
    command = [
        "squeue",
        "-r",
        "-h",
        "-p",
        partition,
        "-t",
        "PENDING",
        "-o",
        delimiter.join(["%D", "%f", "%r", "%Q"]),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto could not inspect pending Slurm jobs with: {}."
            .format(" ".join(command))
        ) from exc

    pending_counts = {count: 0 for count in node_counts}
    allowed_counts = set(node_counts)
    for line in result.stdout.splitlines():
        fields = line.split(delimiter)
        if len(fields) != 4:
            raise cw_error.ConfigKeyError(
                "Could not parse pending Slurm job data: {!r}.".format(line)
            )
        raw_nodes, constraint, reason, _priority = (
            field.strip() for field in fields
        )
        if reason.lower() != "priority":
            continue

        gpu_count_match = re.search(
            r"(?<![A-Za-z0-9_.-])GPUx(\d+)(?![A-Za-z0-9_.-])",
            constraint,
        )
        if gpu_count_match is None:
            continue
        gpu_count = int(gpu_count_match.group(1))
        if gpu_count not in allowed_counts:
            continue
        if not _pending_constraint_matches_models(constraint, gpu_models):
            continue

        try:
            requested_nodes = int(raw_nodes)
        except ValueError as exc:
            raise cw_error.ConfigKeyError(
                "Could not parse pending Slurm node count {!r}."
                .format(raw_nodes)
            ) from exc
        if requested_nodes < 1:
            raise cw_error.ConfigKeyError(
                "Pending Slurm node count must be positive, got {}."
                .format(requested_nodes)
            )
        pending_counts[gpu_count] += requested_nodes
    return pending_counts


def _eligible_auto_gpu_counts(
    cw_job,
    reps_per_gpu: int,
    node_counts: list,
):
    concurrent_capacity = _job_concurrent_task_capacity(cw_job)
    gpu_capacity = concurrent_capacity // reps_per_gpu
    eligible_counts = [
        count for count in node_counts if count <= gpu_capacity
    ]
    if not eligible_counts:
        raise cw_error.ConfigKeyError(
            "No GPU node size can be fully occupied: Slurm array task has "
            "{} concurrent runs, reps_per_gpu={}, and allowed node sizes are "
            "{}.".format(concurrent_capacity, reps_per_gpu, node_counts)
        )
    return eligible_counts


def _auto_cpus_per_rep(
    conf: cw_config.Config,
    jobs: list,
    reps_per_gpu: int,
    node_counts: list,
) -> int:
    configured_cpus_per_rep = conf.slurm_config.get("cpus_per_rep")
    if configured_cpus_per_rep is not None:
        try:
            cpus_per_rep = int(configured_cpus_per_rep)
        except (TypeError, ValueError) as exc:
            raise cw_error.ConfigKeyError(
                "cpus_per_rep must be a positive integer."
            ) from exc
        if cpus_per_rep < 1 or cpus_per_rep != configured_cpus_per_rep:
            raise cw_error.ConfigKeyError(
                "cpus_per_rep must be a positive integer."
            )
        return cpus_per_rep

    max_concurrent_reps = max(node_counts) * reps_per_gpu
    configured_cpus = int(conf.slurm_config["cpus-per-task"])
    if configured_cpus < max_concurrent_reps:
        raise cw_error.ConfigKeyError(
            "cpus-per-task is smaller than the maximum number of concurrent "
            "runs. Set cpus_per_rep explicitly."
        )
    if configured_cpus % max_concurrent_reps != 0:
        raise cw_error.ConfigKeyError(
            "Cannot infer an integer cpus_per_rep from cpus-per-task={} and "
            "{} concurrent runs. Set cpus_per_rep explicitly.".format(
                configured_cpus,
                max_concurrent_reps,
            )
        )
    return configured_cpus // max_concurrent_reps


def _auto_cpus_per_task(
    gpu_count: int,
    reps_per_gpu: int,
    cpus_per_rep: int,
) -> int:
    return gpu_count * reps_per_gpu * cpus_per_rep


def _persist_runtime_config(conf: cw_config.Config) -> str:
    config_dir = os.path.dirname(os.path.abspath(conf.config_path))
    # The runtime config is executed from the code-copy directory. Keeping
    # result paths relative here would redirect logs and checkpoints into the
    # code copy instead of the experiment's configured output root.
    runtime_config_path = conf.to_yaml(config_dir, relpath=False)
    conf.config_path = runtime_config_path
    return runtime_config_path


def _balanced_fallback_job_counts(
    remaining_tasks: int,
    reps_per_gpu: int,
    node_counts: list,
    fallback_count: int,
):
    eligible_counts = sorted(
        (count for count in node_counts if count <= fallback_count),
        reverse=True,
    )
    if remaining_tasks == 0:
        return {count: 0 for count in eligible_counts}
    if not eligible_counts or remaining_tasks % reps_per_gpu != 0:
        return None

    remaining_gpu_units = remaining_tasks // reps_per_gpu
    allocation = {count: 0 for count in eligible_counts}
    best_score = None
    best_allocation = None

    def consider_candidate():
        nonlocal best_score, best_allocation
        # Idle-node assignments are already runnable and must not distort the
        # balancing of jobs that exceed currently available capacity.
        values = list(allocation.values())
        spread = max(values) - min(values)
        pairwise_imbalance = sum(
            (values[left] - values[right]) ** 2
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
        score = (
            spread,
            pairwise_imbalance,
            sum(allocation.values()),
            tuple(-allocation[count] for count in eligible_counts),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_allocation = dict(allocation)

    def search(position, units_left):
        gpu_count = eligible_counts[position]
        if position == len(eligible_counts) - 1:
            if units_left % gpu_count != 0:
                return
            allocation[gpu_count] = units_left // gpu_count
            consider_candidate()
            allocation[gpu_count] = 0
            return

        for number_of_jobs in range(units_left // gpu_count + 1):
            allocation[gpu_count] = number_of_jobs
            search(
                position + 1,
                units_left - number_of_jobs * gpu_count,
            )
        allocation[gpu_count] = 0

    search(0, remaining_gpu_units)
    return best_allocation


def resolve_auto_gpu_resources(
    conf: cw_config.Config,
    jobs: list,
    idle_node_counts=None,
    pending_node_counts=None,
    return_job_capacities=False,
):
    configured_num_gpus = conf.slurm_config.get("num_gpus", 0)
    if not (
        isinstance(configured_num_gpus, str)
        and configured_num_gpus.strip().lower() == "auto"
    ):
        return None
    if not jobs:
        raise cw_error.ConfigKeyError(
            "num_gpus=auto requires at least one expanded Slurm job."
        )

    reps_per_gpu, node_counts, gpu_models, fallback_count = (
        _auto_gpu_settings(conf)
    )
    grouped_task_counts = {}
    for cw_job in jobs:
        for task in cw_job.tasks:
            task_name = (
                task.get(CKEYS.NAME, "__auto_gpu_default__")
                if hasattr(task, "get")
                else "__auto_gpu_default__"
            )
            grouped_task_counts[task_name] = (
                grouped_task_counts.get(task_name, 0) + 1
            )
    queried_idle_nodes = idle_node_counts is None
    if queried_idle_nodes:
        idle_node_counts = query_idle_auto_gpu_nodes(
            conf,
            node_counts,
            gpu_models,
        )
    try:
        idle_by_count = {
            count: max(0, int(idle_node_counts.get(count, 0)))
            for count in node_counts
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise cw_error.ConfigKeyError(
            "idle_node_counts must map GPU node sizes to non-negative counts."
        ) from exc

    subtract_pending = conf.slurm_config.get(
        "auto_gpu_subtract_pending_priority",
        False,
    )
    if not isinstance(subtract_pending, bool):
        raise cw_error.ConfigKeyError(
            "auto_gpu_subtract_pending_priority must be true or false."
        )
    if subtract_pending and pending_node_counts is None and queried_idle_nodes:
        pending_node_counts = query_pending_priority_auto_gpu_demand(
            conf,
            node_counts,
            gpu_models,
        )
    if pending_node_counts is None:
        pending_node_counts = {}
    try:
        pending_by_count = {
            count: max(0, int(pending_node_counts.get(count, 0)))
            if subtract_pending
            else 0
            for count in node_counts
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise cw_error.ConfigKeyError(
            "pending_node_counts must map GPU node sizes to non-negative "
            "counts."
        ) from exc
    available_by_count = {
        count: max(0, idle_by_count[count] - pending_by_count[count])
        for count in node_counts
    }
    remaining_idle_by_count = dict(available_by_count)

    assignment = {count: [] for count in node_counts}
    job_capacities = []

    def add_jobs(gpu_count, number_of_jobs):
        capacity = gpu_count * reps_per_gpu
        for _ in range(number_of_jobs):
            job_idx = len(job_capacities)
            job_capacities.append(capacity)
            assignment[gpu_count].append(job_idx)

    for task_count in grouped_task_counts.values():
        remaining_tasks = task_count
        for gpu_count in sorted(node_counts, reverse=True):
            capacity = gpu_count * reps_per_gpu
            number_of_jobs = min(
                remaining_idle_by_count[gpu_count],
                remaining_tasks // capacity,
            )
            add_jobs(gpu_count, number_of_jobs)
            remaining_idle_by_count[gpu_count] -= number_of_jobs
            remaining_tasks -= number_of_jobs * capacity

        fallback_jobs = _balanced_fallback_job_counts(
            remaining_tasks=remaining_tasks,
            reps_per_gpu=reps_per_gpu,
            node_counts=node_counts,
            fallback_count=fallback_count,
        )
        if fallback_jobs is None:
            raise cw_error.ConfigKeyError(
                "Expanded experiment group with {} run(s) cannot fully "
                "occupy any configured GPU node with reps_per_gpu={}. "
                "Adjust repetitions or reps_per_gpu.".format(
                    task_count,
                    reps_per_gpu,
                )
            )
        for gpu_count in sorted(fallback_jobs, reverse=True):
            add_jobs(gpu_count, fallback_jobs[gpu_count])

    assignment = {
        count: indices
        for count, indices in assignment.items()
        if indices
    }
    print(
        "[slurm] Idle GPU nodes matching configured models: {}."
        .format(idle_by_count)
    )
    if subtract_pending:
        print(
            "[slurm] Pending Priority GPU demand reserved before submission: "
            "{}.".format(pending_by_count)
        )
        print(
            "[slurm] Effective idle GPU nodes after pending-demand "
            "subtraction: {}.".format(available_by_count)
        )
    for gpu_count in sorted(assignment, reverse=True):
        print(
            "[slurm] GPUx{}: {} array task(s), {} run(s) per task, global "
            "indices {}.".format(
                gpu_count,
                len(assignment[gpu_count]),
                gpu_count * reps_per_gpu,
                _compress_array_indices(assignment[gpu_count]),
            )
        )
    if return_job_capacities:
        return assignment, job_capacities
    return assignment


def _auto_gpu_base_count(conf: cw_config.Config, jobs: list) -> int:
    _, _, _, fallback_count = _auto_gpu_settings(conf)
    return fallback_count


def _compress_array_indices(indices: list) -> str:
    sorted_indices = sorted(indices)
    ranges = []
    start = previous = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(
            str(start) if start == previous else "{}-{}".format(start, previous)
        )
        start = previous = index
    ranges.append(
        str(start) if start == previous else "{}-{}".format(start, previous)
    )
    return ",".join(ranges)


def _auto_gpu_group_throttles(assignment: dict, max_parallel: int):
    group_sizes = {
        gpu_count: len(indices)
        for gpu_count, indices in assignment.items()
    }
    total_jobs = sum(group_sizes.values())
    if total_jobs <= max_parallel:
        return group_sizes
    if max_parallel < len(group_sizes):
        raise cw_error.ConfigKeyError(
            "num_parallel_jobs must be at least the number of auto GPU "
            "submission groups."
        )

    throttles = {gpu_count: 1 for gpu_count in group_sizes}
    remaining_slots = max_parallel - len(group_sizes)
    while remaining_slots > 0:
        candidates = [
            gpu_count
            for gpu_count, size in group_sizes.items()
            if throttles[gpu_count] < size
        ]
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda count: (
                group_sizes[count] / throttles[count],
                group_sizes[count],
                count,
            ),
        )
        throttles[selected] += 1
        remaining_slots -= 1
    return throttles


def _submit_auto_gpu_arrays(
    conf: cw_config.Config,
    assignment: dict,
    slurm_script: str,
    gpu_models: list,
    reps_per_gpu: int,
    cpus_per_rep: int,
):
    max_parallel = int(conf.slurm_config["num_parallel_jobs"])
    throttles = _auto_gpu_group_throttles(assignment, max_parallel)
    for gpu_count in sorted(assignment, reverse=True):
        constraint = _auto_gpu_constraint(gpu_count, gpu_models)
        cpus_per_task = _auto_cpus_per_task(
            gpu_count,
            reps_per_gpu,
            cpus_per_rep,
        )
        array_spec = "{}%{}".format(
            _compress_array_indices(assignment[gpu_count]),
            throttles[gpu_count],
        )
        command = [
            "sbatch",
            "--array={}".format(array_spec),
            "--constraint={}".format(constraint),
            "--cpus-per-task={}".format(cpus_per_task),
            (
                "--export=ALL,MPRL_RESUBMIT_CONSTRAINT={},"
                "MPRL_RESUBMIT_CPUS_PER_TASK={}"
            ).format(constraint, cpus_per_task),
            slurm_script,
        ]
        print(" ".join(command))
        subprocess.check_output(command)


def run_slurm(conf: cw_config.Config, jobs) -> None:
    """starts slurm execution

    Args:
        conf (cw_config.Config): config object
        jobs: expanded cw2 jobs mapped to Slurm array tasks. An integer job
            count remains supported for fixed-GPU callers.
    """
    if isinstance(jobs, int):
        if (
            isinstance(conf.slurm_config.get("num_gpus"), str)
            and conf.slurm_config["num_gpus"].strip().lower() == "auto"
        ):
            raise cw_error.ConfigKeyError(
                "num_gpus=auto requires expanded jobs, not only a job count."
            )
        num_jobs = jobs
    else:
        num_jobs = len(jobs)
        auto_gpu_plan = resolve_auto_gpu_resources(
            conf,
            jobs,
            return_job_capacities=True,
        )
        auto_gpu_assignment = (
            auto_gpu_plan[0] if auto_gpu_plan is not None else None
        )
        if auto_gpu_assignment is not None:
            auto_job_capacities = auto_gpu_plan[1]
            num_jobs = len(auto_job_capacities)
            (
                auto_reps_per_gpu,
                auto_node_counts,
                auto_gpu_models,
                _,
            ) = _auto_gpu_settings(conf)
            auto_cpus_per_rep = _auto_cpus_per_rep(
                conf,
                jobs,
                auto_reps_per_gpu,
                auto_node_counts,
            )
            auto_base_gpu_count = _auto_gpu_base_count(conf, jobs)
            sbatch_args = conf.slurm_config.setdefault("sbatch_args", {})
            sbatch_args["constraint"] = _auto_gpu_constraint(
                auto_base_gpu_count,
                auto_gpu_models,
            )
            sbatch_args.pop("prefer", None)
            conf.slurm_config["cpus-per-task"] = _auto_cpus_per_task(
                auto_base_gpu_count,
                auto_reps_per_gpu,
                auto_cpus_per_rep,
            )
            conf.slurm_config["auto_gpu_job_capacities"] = (
                auto_job_capacities
            )
            _persist_runtime_config(conf)

    # Finalize Configs
    sc = SlurmConfig(conf)
    sc.finalize(num_jobs)

    # Create Code Copies
    dir_mgr = SlurmDirectoryManager(sc, conf)
    dir_mgr.move_files(num_jobs)

    # Write and call slurm script
    slurm_script = write_slurm_script(sc, dir_mgr)
    if not isinstance(jobs, int) and auto_gpu_assignment is not None:
        _submit_auto_gpu_arrays(
            conf,
            auto_gpu_assignment,
            slurm_script,
            auto_gpu_models,
            auto_reps_per_gpu,
            auto_cpus_per_rep,
        )
        return
    command = ["sbatch", slurm_script]
    print(" ".join(command))
    subprocess.check_output(command)


def write_slurm_script(slurm_conf: SlurmConfig, dir_mgr: SlurmDirectoryManager) -> str:
    """write the sbatch.sh script for slurm to disk

    Args:
        slurm_conf (SlurmConfig): Slurm configuration object

    Returns:
        str: path to the written script
    """
    sc = slurm_conf.slurm_conf
    conf = slurm_conf.conf

    template_path = sc[SKEYS.TEMPLATE_PATH]
    output_path = sc[SKEYS.SLURM_OUT]

    exp_main_file = os.path.relpath(__main__.__file__, os.getcwd())
    config_exec_path = dir_mgr.get_config_exec_path(conf.config_path)
    gpu_env_selector_path = os.path.join(
        os.path.dirname(__file__), "../gpu_env_selector.sh"
    )
    with open(gpu_env_selector_path, "r") as selector_file:
        gpu_env_selector = selector_file.read().rstrip()

    fid_in = open(template_path, "r")
    fid_out = open(output_path, "w")

    tline = fid_in.readline()

    while tline:
        tline = tline.replace("%%partition%%", sc["partition"])
        tline = tline.replace("%%account%%", sc[SKEYS.ACCOUNT])
        tline = tline.replace("%%job-name%%", sc["job-name"])

        tline = tline.replace("%%last_job_idx%%", "{:d}".format(sc[SKEYS.LAST_IDX]))
        tline = tline.replace(
            "%%num_parallel_jobs%%", "{:d}".format(sc["num_parallel_jobs"])
        )

        tline = tline.replace(
            "%%experiment_execution_dir%%", dir_mgr.get_exp_exec_dir()
        )

        tline = tline.replace("%%slurm_log%%", sc[SKEYS.SLURM_LOG])

        tline = tline.replace("%%ntasks%%", "{:d}".format(sc["ntasks"]))
        tline = tline.replace("%%cpus-per-task%%", "{:d}".format(sc["cpus-per-task"]))
        tline = tline.replace("%%time%%", sc[SKEYS.TIME])

        tline = tline.replace("%%sh_lines%%", sc[SKEYS.SH_LINES])
        tline = tline.replace("%%gpu_env_selector%%", gpu_env_selector)

        tline = tline.replace("%%venv%%", sc[SKEYS.VENV])
        tline = tline.replace("%%pythonpath%%", dir_mgr.get_py_path())

        tline = tline.replace("%%python_script%%", exp_main_file)
        tline = tline.replace("%%path_to_yaml_config%%", config_exec_path)

        tline = tline.replace("%%cw_args%%", sc[SKEYS.CW_ARGS])
        tline = tline.replace("%%sbatch_args%%", sc[SKEYS.SBATCH_ARGS])

        fid_out.write(tline)

        tline = fid_in.readline()
    fid_in.close()
    fid_out.close()
    dir_mgr.freeze_slurm_script(output_path)
    return output_path
