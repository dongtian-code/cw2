import abc
import os
import concurrent.futures
import multiprocessing
import signal
import socket
import subprocess
import warnings
import math
from typing import List

from joblib import Parallel, delayed

from cw2 import cw_error, job
from cw2.cw_config import cw_conf_keys as KEYS
from cw2.cw_config import cw_config
from cw2.cw_slurm import cw_slurm


class AbstractScheduler(abc.ABC):
    def __init__(self, conf: cw_config.Config = None):
        self.joblist = None
        self.config = conf

    def assign(self, joblist: List[job.Job]) -> None:
        """assigns the scheduler a list of jobs to execute

        Arguments:
            joblist {List[job.AbstractJob]} -- list of configured and implemented jobs
        """
        self.joblist = joblist

    @abc.abstractmethod
    def run(self, overwrite=False):
        """the scheduler begins to execute all assigned jobs

        Args:
            overwrite (bool, optional): overwrite flag. can be passed to the job. Defaults to False.
        """
        raise NotImplementedError


class GPUDistributingLocalScheduler(AbstractScheduler):
    def __init__(self, conf: cw_config.Config = None):
        super(GPUDistributingLocalScheduler, self).__init__(conf=conf)
        self._auto_num_gpus = self.is_auto_gpu_count(
            conf.slurm_config.get("num_gpus", 0)
        )
        self._total_num_gpus = self.get_num_requested_gpus(conf)
        self._reps_per_gpu = int(conf.slurm_config.get("reps_per_gpu", 1))
        assert self._reps_per_gpu >= 1, "reps_per_gpu must be >= 1"

        if "reps_per_gpu" in conf.slurm_config:
            self._gpus_per_rep = 1.0 / self._reps_per_gpu
            configured_gpus_per_rep = conf.slurm_config.get("gpus_per_rep", None)
            if configured_gpus_per_rep is not None and not math.isclose(
                float(configured_gpus_per_rep),
                self._gpus_per_rep,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                warnings.warn(
                    "Both reps_per_gpu and gpus_per_rep are set. "
                    "Using reps_per_gpu={} (equivalent gpus_per_rep={}).".format(
                        self._reps_per_gpu,
                        self._gpus_per_rep,
                    )
                )
        else:
            self._gpus_per_rep = float(conf.slurm_config["gpus_per_rep"])

        assert self._gpus_per_rep > 0, "gpus_per_rep must be > 0"
        queue_elements = self._total_num_gpus / self._gpus_per_rep
        assert math.isclose(
            queue_elements,
            round(queue_elements),
            rel_tol=0.0,
            abs_tol=1e-12,
        ), "gpus_per_rep / reps_per_gpu must divide the requested GPUs evenly"
        self._queue_elements = int(round(queue_elements))

        print(
            "GPUDistributingLocalScheduler: {} GPUs available, {} GPUs per rep, {} reps per GPU, {} queue elements".format(
                self._total_num_gpus,
                self._gpus_per_rep,
                self._reps_per_gpu,
                self._queue_elements,
            )
        )

        if self._gpus_per_rep >= 1.0:
            assert self._gpus_per_rep == int(
                self._gpus_per_rep
            ), "gpus_per_rep must be integer"

    @staticmethod
    def get_num_requested_gpus(conf: cw_config.Config) -> int:
        sbatch_args = conf.slurm_config.get("sbatch_args", {})
        if isinstance(sbatch_args, dict) and "gres" in sbatch_args:
            return int(str(sbatch_args["gres"]).rsplit(":", 1)[1])
        configured_num_gpus = conf.slurm_config.get("num_gpus", 0)
        if GPUDistributingLocalScheduler.is_auto_gpu_count(configured_num_gpus):
            return GPUDistributingLocalScheduler.detect_available_gpu_count()
        return int(configured_num_gpus)

    @staticmethod
    def is_auto_gpu_count(value) -> bool:
        return isinstance(value, str) and value.strip().lower() == "auto"

    @staticmethod
    def detect_available_gpu_count() -> int:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            visible_devices = visible_devices.strip()
            if visible_devices in ("", "-1", "NoDevFiles"):
                raise RuntimeError(
                    "num_gpus=auto, but CUDA_VISIBLE_DEVICES exposes no GPUs."
                )
            return len(
                [device for device in visible_devices.split(",") if device.strip()]
            )

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "num_gpus=auto could not detect GPUs. CUDA_VISIBLE_DEVICES is "
                "unset and nvidia-smi did not return the node GPU list."
            ) from exc

        gpu_count = len(
            [line for line in result.stdout.splitlines() if line.strip()]
        )
        if gpu_count < 1:
            raise RuntimeError(
                "num_gpus=auto detected zero GPUs from nvidia-smi."
            )
        return gpu_count

    def _gpu_num_parallel(self) -> int:
        if self._auto_num_gpus:
            for j in self.joblist:
                if len(j.tasks) < self._queue_elements:
                    raise RuntimeError(
                        "Auto GPU scheduling selected {} concurrent run slots, "
                        "but a Slurm task contains only {} runs. Reduce the "
                        "selected node GPU count or reps_per_gpu.".format(
                            self._queue_elements,
                            len(j.tasks),
                        )
                    )
            print(
                "Auto GPU scheduling: using {} worker processes from {} detected "
                "GPUs and {} reps per GPU.".format(
                    self._queue_elements,
                    self._total_num_gpus,
                    self._reps_per_gpu,
                ),
                flush=True,
            )
            return self._queue_elements

        num_parallel = self.joblist[0].n_parallel
        for j in self.joblist:
            assert (
                j.n_parallel == num_parallel
            ), "All jobs in list must have same n_parallel"
            assert j.n_parallel == self._queue_elements, (
                "Mismatch between GPUs Queue Elements and Jobs executed in"
                "parallel. Fix for optimal resource usage!!"
            )
        return num_parallel

    @staticmethod
    def use_distributed_gpu_scheduling(conf: cw_config.Config) -> bool:
        if conf.slurm_config is None:
            return False
        # Use if GPU allocation is explicitly controlled by either the legacy
        # gpus_per_rep setting or the newer reps_per_gpu setting.
        num_gpus_requested = GPUDistributingLocalScheduler.get_num_requested_gpus(conf)
        gpus_requested = num_gpus_requested > 0
        gpus_per_rep_specified = "gpus_per_rep" in conf.slurm_config
        reps_per_gpu_specified = "reps_per_gpu" in conf.slurm_config

        use_distributed_gpu_scheduling = (
            gpus_requested
            and (
                (
                    gpus_per_rep_specified
                    and num_gpus_requested != conf.slurm_config["gpus_per_rep"]
                )
                or reps_per_gpu_specified
            )
        )

        if not use_distributed_gpu_scheduling:
            on_horeka_gpu = (
                "hkn" in socket.gethostname()
                and conf.slurm_config["partition"] == "accelerated"
            )
            # FIXME, DISABLE THIS TO ALLOW ONE GPU USAGE IN HOREKA
            # if on_horeka_gpu:
            #     assert (
            #         num_gpus_requested == 4
            #     ), "On HoreKA, you must request 4 GPUs (gres=gpu:4)"
            # assert (
            #     not on_horeka_gpu
            # ), "You are on HoreKA and not using the GPU scheduler, don't! "

        return use_distributed_gpu_scheduling

    @staticmethod
    def get_gpu_str(queue_idx: int, gpus_per_rep: float) -> str:
        if gpus_per_rep >= 1:
            assert (
                int(gpus_per_rep) == gpus_per_rep
            ), "gpus_per_rep must be integer if >= 1"
            gpus_per_rep = int(gpus_per_rep)
            return ("{}," * gpus_per_rep).format(
                *[queue_idx * gpus_per_rep + i for i in range(gpus_per_rep)]
            )[:-1]
        else:
            return str(int(queue_idx * gpus_per_rep))

    @staticmethod
    def _pool_worker_pids(pool) -> list:
        if pool is None:
            return []
        processes = getattr(pool, "_processes", None)
        if isinstance(processes, dict):
            return [
                pid for pid, process in processes.items()
                if process is not None and process.is_alive()
            ]
        workers = getattr(pool, "_pool", None)
        if workers is not None:
            return [
                process.pid for process in workers
                if process is not None
                and process.pid is not None
                and process.is_alive()
            ]
        return []

    @staticmethod
    def _install_worker_signal_forwarding(get_pool):
        previous_handlers = {}

        def _forward_signal(signum, _frame):
            worker_pids = GPUDistributingLocalScheduler._pool_worker_pids(
                get_pool()
            )
            print(
                f"[scheduler] Received signal {signum}; "
                f"forwarding to worker processes {worker_pids}",
                flush=True,
            )
            for pid in worker_pids:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _forward_signal)
            except (AttributeError, ValueError):
                pass
        return previous_handlers

    @staticmethod
    def _restore_signal_handlers(previous_handlers):
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (AttributeError, ValueError):
                pass


class MPGPUDistributingLocalScheduler(GPUDistributingLocalScheduler):
    def run(self, overwrite: bool = False):
        num_parallel = self._gpu_num_parallel()

        active_pool = None
        previous_handlers = self._install_worker_signal_forwarding(
            lambda: active_pool
        )
        try:
            with multiprocessing.Pool(processes=num_parallel) as pool:
                active_pool = pool
                # setup gpu resource queue
                m = multiprocessing.Manager()
                gpu_queue = m.Queue(maxsize=self._queue_elements)
                for i in range(self._queue_elements):
                    gpu_queue.put(i)

                for j in self.joblist:
                    for c in j.tasks:
                        pool.apply_async(
                            MPGPUDistributingLocalScheduler._execute_task,
                            (j, c, gpu_queue, self._gpus_per_rep, overwrite),
                        )
                pool.close()
                pool.join()
        finally:
            self._restore_signal_handlers(previous_handlers)

    @staticmethod
    def _execute_task(
        j: job.Job,
        c: dict,
        q: multiprocessing.Queue,
        gpus_per_rep: int,
        overwrite: bool = False,
    ):
        queue_idx = q.get()
        gpu_str = MPGPUDistributingLocalScheduler.get_gpu_str(queue_idx, gpus_per_rep)
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            j.run_task(c, overwrite)
        except cw_error.ExperimentSurrender as _:
            return
        finally:
            q.put(queue_idx)


class HOREKAAffinityGPUDistributingLocalScheduler(GPUDistributingLocalScheduler):
    def __init__(self, conf: cw_config.Config = None):
        super(HOREKAAffinityGPUDistributingLocalScheduler, self).__init__(conf=conf)

        total_cpus = conf.slurm_config["cpus-per-task"] * conf.slurm_config["ntasks"]
        self._allowed_cpus = sorted(os.sched_getaffinity(0))
        usable_cpu_count = min(total_cpus, len(self._allowed_cpus))
        self._usable_cpus = self._allowed_cpus[:usable_cpu_count]
        self._cpus_per_rep = usable_cpu_count // self._queue_elements

        assert (
            self._cpus_per_rep > 0
        ), "Not enough CPUs for the number of GPUs requested"

    def run(self, overwrite: bool = False):
        print("Seeing CPUs:", os.sched_getaffinity(0), flush=True)
        num_parallel = self._gpu_num_parallel()

        active_pool = None
        previous_handlers = self._install_worker_signal_forwarding(
            lambda: active_pool
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_parallel,
            ) as pool:
                active_pool = pool
                # setup gpu resource queue
                m = multiprocessing.Manager()
                gpu_queue = m.Queue(maxsize=self._queue_elements)
                for i in range(self._queue_elements):
                    gpu_queue.put(i)

                futures = []
                for j in self.joblist:
                    for c in j.tasks:
                        futures.append(
                            pool.submit(
                                HOREKAAffinityGPUDistributingLocalScheduler._execute_task,
                                j,
                                c,
                                gpu_queue,
                                self._gpus_per_rep,
                                self._usable_cpus,
                                self._cpus_per_rep,
                                overwrite,
                            )
                        )
                for future in futures:
                    future.result()
        finally:
            self._restore_signal_handlers(previous_handlers)

    @staticmethod
    def _execute_task(
        j: job.Job,
        c: dict,
        q: multiprocessing.Queue,
        gpus_per_rep: int,
        usable_cpus: list,
        cpus_per_rep: int,
        overwrite: bool = False,
    ):
        print("Seeing CPUs:", os.sched_getaffinity(0), flush=True)
        queue_idx = q.get()
        gpu_str = HOREKAAffinityGPUDistributingLocalScheduler.get_gpu_str(
            queue_idx, gpus_per_rep
        )
        cpu_start = queue_idx * cpus_per_rep
        cpu_end = (queue_idx + 1) * cpus_per_rep
        cpus = set(usable_cpus[cpu_start:cpu_end])
        print("Job {}: Using GPUs: {} and CPUs: {}".format(queue_idx, gpu_str, cpus), flush=True)
        try:
            os.sched_setaffinity(0, cpus)
            c[KEYS.i_CPU_CORES] = cpus
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            j.run_task(c, overwrite)
        except cw_error.ExperimentSurrender as _:
            return
        finally:
            q.put(queue_idx)


class KlusterThreadLimitingScheduler(GPUDistributingLocalScheduler):
    def __init__(self, conf: cw_config.Config = None):
        super(KlusterThreadLimitingScheduler, self).__init__(conf=conf)
        total_cpus = conf.slurm_config["cpus-per-task"] * conf.slurm_config["ntasks"]
        self._num_threads = total_cpus // self._queue_elements
        print("Using {} threads per Rep".format(self._num_threads))

    def run(self, overwrite: bool = False):
        num_parallel = self._gpu_num_parallel()

        with multiprocessing.Pool(processes=num_parallel) as pool:
            # setup gpu resource queue
            m = multiprocessing.Manager()
            gpu_queue = m.Queue(maxsize=self._queue_elements)
            for i in range(self._queue_elements):
                gpu_queue.put(i)

            for j in self.joblist:
                for c in j.tasks:
                    args = (
                        j,
                        c,
                        gpu_queue,
                        self._gpus_per_rep,
                        self._num_threads,
                        overwrite,
                    )
                    pool.apply_async(KlusterThreadLimitingScheduler._execute_task, args)
            pool.close()
            pool.join()

    @staticmethod
    def _execute_task(
        j: job.Job,
        c: dict,
        q: multiprocessing.Queue,
        gpus_per_rep: int,
        num_threads: int,
        overwrite: bool = False,
    ):
        queue_idx = q.get()
        gpu_str = KlusterThreadLimitingScheduler.get_gpu_str(queue_idx, gpus_per_rep)
        try:
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            # Ok, that's not so nice, but I did not find better way yet
            try:
                import torch

                torch.set_num_threads(num_threads)
            except ImportError:
                pass

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            j.run_task(c, overwrite)
        except cw_error.ExperimentSurrender as _:
            return
        finally:
            q.put(queue_idx)


def get_gpu_scheduler_cls(scheduler: str):
    if scheduler == "mp":
        return MPGPUDistributingLocalScheduler
    elif scheduler == "horeka":
        return HOREKAAffinityGPUDistributingLocalScheduler
    elif scheduler == "kluster":
        return KlusterThreadLimitingScheduler
    else:
        raise NotImplementedError


class CpuDistributingLocalScheduler(AbstractScheduler):
    def __init__(self, conf: cw_config.Config = None):
        super(CpuDistributingLocalScheduler, self).__init__(conf=conf)
        self._total_num_cpus = (
            conf.slurm_config["cpus-per-task"] * conf.slurm_config["ntasks"]
        )
        self._cpus_per_rep = conf.slurm_config["cpus_per_rep"]
        assert self._cpus_per_rep == int(
            self._cpus_per_rep
        ), "cpus_per_rep must be integer"
        self._queue_elements = int(self._total_num_cpus / self._cpus_per_rep)
        print(
            "CPUDistributingLocalScheduler: {} CPUs available, {} CPUs per rep, {} queue elements".format(
                self._total_num_cpus, self._cpus_per_rep, self._queue_elements
            )
        )

    def run(self, overwrite: bool = False):
        print("Seeing CPUs:", os.sched_getaffinity(0))
        num_parallel = self.joblist[0].n_parallel
        for j in self.joblist:
            assert (
                j.n_parallel == num_parallel
            ), "All jobs in list must have same n_parallel"
            assert j.n_parallel == self._queue_elements, (
                "Mismatch between CPUs Queue Elements and Jobs executed in"
                "parallel. Fix for optimal resource usage!!"
            )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_parallel,
        ) as pool:
            # setup gpu resource queue
            m = multiprocessing.Manager()
            cpu_queue = m.Queue(maxsize=self._queue_elements)
            for i in range(self._queue_elements):
                cpu_queue.put(i)

            for j in self.joblist:
                for c in j.tasks:
                    pool.submit(
                        CpuDistributingLocalScheduler._execute_task,
                        j,
                        c,
                        cpu_queue,
                        self._cpus_per_rep,
                        overwrite,
                    )

    @staticmethod
    def _execute_task(
        j: job.Job,
        c: dict,
        q: multiprocessing.Queue,
        cpus_per_rep: int,
        overwrite: bool = False,
    ):
        print("Seeing CPUs:", os.sched_getaffinity(0))
        queue_idx = q.get()
        cpus = set(range(queue_idx * cpus_per_rep, (queue_idx + 1) * cpus_per_rep))
        print("Job {}: Using CPUs: {}".format(queue_idx, cpus))
        try:
            os.sched_setaffinity(0, cpus)
            c[KEYS.i_CPU_CORES] = cpus
            j.run_task(c, overwrite)
        except cw_error.ExperimentSurrender as _:
            return
        finally:
            q.put(queue_idx)

    @staticmethod
    def use_distributed_cpu_scheduling(conf: cw_config.Config) -> bool:
        if conf.slurm_config is None:
            return False
        else:
            scheduler = conf.slurm_config.get("scheduler", None)
            return scheduler == "cpu_distribute"


class LocalScheduler(AbstractScheduler):
    def run(self, overwrite: bool = False):
        for j in self.joblist:
            Parallel(n_jobs=j.n_parallel)(
                delayed(self.execute_task)(j, c, overwrite) for c in j.tasks
            )

    def execute_task(self, j: job.Job, c: dict, overwrite: bool = False):
        try:
            j.run_task(c, overwrite)
        except cw_error.ExperimentSurrender as _:
            return


class SlurmScheduler(AbstractScheduler):
    def run(self, overwrite: bool = False):
        cw_slurm.run_slurm(self.config, self.joblist)
