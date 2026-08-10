import os
from typing import Dict, List, Type

from cw2 import cw_error, experiment
from cw2.cw_config import cw_conf_keys as KEYS
from cw2.cw_data import cw_logging


class Job:
    """Class defining a computation job.
    Can contain 1..n tasks. Each job should encapsulate all information necessary for execution.
    A task is an experiment configuration with unique repetition idx.
    """

    def __init__(
        self,
        tasks: List[Dict],
        exp_cls: experiment.AbstractExperiment.__class__,
        logger: cw_logging.AbstractLogger,
        delete_old_files: bool = False,
        root_dir: str = "",
        read_only: bool = False,
    ):
        self.tasks = tasks
        task_rep_ids = [
            task.get(KEYS.i_REP_IDX, task_index)
            for task_index, task in enumerate(tasks)
        ]
        for task in tasks:
            # The worker processes use this exact job membership for the
            # preemption checkpoint barrier. It must reflect dynamic auto-GPU
            # grouping, not the static YAML reps_per_job value.
            task["_cw2_job_rep_ids"] = list(task_rep_ids)
            task["_cw2_job_rep_count"] = len(task_rep_ids)

        if exp_cls is not None:
            self.exp = exp_cls()
        self.logger = logger

        self.n_parallel = 1
        if KEYS.REPS_PARALL in tasks[0]:
            self.n_parallel = tasks[0][KEYS.REPS_PARALL]

        self._root_dir = root_dir

        if not read_only:
            self.__create_experiment_directory(tasks, delete_old_files, root_dir)

    def __create_experiment_directory(
        self, tasks: List[Dict], delete_old_files=False, root_dir=""
    ):
        """internal function creating the directories in which the job will write its data.

        Args:
            task (List[attrdict.Attrdict]): a list of experiment tasks
            delete_old_files (bool, optional): Should the directory be emptied beforehand?. Defaults to False.
            root_dir (str, optional): [description]. Defaults to "".
        """
        for conf in tasks:
            # create experiment path and subdir
            os.makedirs(os.path.join(root_dir, conf[KEYS.PATH]), exist_ok=True)

            # create a directory for the log path
            os.makedirs(os.path.join(root_dir, conf[KEYS.LOG_PATH]), exist_ok=True)

            # create log path for each repetition
            rep_path = os.path.join(root_dir, conf[KEYS.i_REP_LOG_PATH])

            # XXX: Disable Delete for now
            """
            if delete_old_files:
                pass
            """
            os.makedirs(rep_path, exist_ok=True)

    def run_task(self, c: Dict, overwrite: bool):
        """Execute a single task of the job.

        Args:
            c (attrdict.AttrDict): task configuration
        """
        rep_path = c[KEYS.i_REP_LOG_PATH]
        r = c[KEYS.i_REP_IDX]
        print(rep_path)

        if not overwrite and self._check_task_exists(c, r):
            cw_logging.getLogger().warning(
                "Skipping run, as {} is not empty. Use -o to overwrite.".format(
                    rep_path
                )
            )
            return

        surrender = None
        crash = False

        self.exp.prepare_logger(c, r)
        self.logger.initialize(c, r, rep_path)
        try:
            self.exp.initialize(c, r, self.logger)
            self.exp.run(c, r, self.logger)
        except cw_error.ExperimentSurrender as s:
            cw_logging.getLogger().info("SURRENDER: {}".format(rep_path))
            surrender = s
        except:
            crash = True
            cw_logging.getLogger().exception("EXCEPTION: {}".format(rep_path))

        self.exp.finalize(surrender, crash)
        self.logger.finalize()

    def load_task(self, c: Dict) -> Dict:
        """Load the results of a single task.

        Args:
            c (attrdict.AttrDict): task configuration

        Returns:
            dict: the loaded data
        """
        rep_path = os.path.join(self._root_dir, c[KEYS.i_REP_LOG_PATH])
        r = c[KEYS.i_REP_IDX]
        self.logger.initialize(c, r, rep_path)
        return self.logger.load()

    def _check_task_exists(self, c: Dict, r: int) -> bool:
        """internal function. checks if the task has already been run in the past.

        Args:
            c (attrdict.AttrDict): task configuration

        Returns:
            bool: True if the repetition was already run
        """
        rep_path = c[KEYS.i_REP_LOG_PATH]
        return len(os.listdir(rep_path)) != 0


class JobFactory:
    """Facotry class to create single jobs from experiment configuration.
    Specifially used to map experiment repetitions to Jobs.
    """

    def __init__(
        self,
        exp_cls: Type[experiment.AbstractExperiment],
        logger: cw_logging.AbstractLogger,
        delete_old_files: bool = False,
        root_dir: str = "",
        read_only: bool = False,
        job_capacities=None,
    ):
        self.exp_cls = exp_cls
        self.logger = logger
        self.delete_old_files = delete_old_files
        self.root_dir = root_dir
        self.read_only = read_only
        self.job_capacities = job_capacities

    def _group_exp_tasks(self, task_confs: List[Dict]) -> Dict:
        """group tasks by experiment to access common attributes like reps_per_job

        Args:
            task_confs (List[attrdict.AttrDict]): list of all task configurations

        Returns:
            dict: dictionary of task configurations grouped by name.
        """
        grouped_exps = {}
        for t in task_confs:
            name = t[KEYS.NAME]
            if name not in grouped_exps:
                grouped_exps[name] = []
            grouped_exps[name].append(t)
        return grouped_exps

    def _divide_tasks(self, task_confs: List[Dict]) -> List[List[Dict]]:
        """internal function to divide experiment repetitions into sets of repetitions.
        Dependent on configured reps_per_job attribute. Each set of repetitions will be one job.

        Args:
            task_confs (List[attrdict.AttrDict]): List of task configurations

        Returns:
            List[List[attrdict.AttrDict]]: a list containing all subpackages of tasks as lists
        """
        grouped_exps = self._group_exp_tasks(task_confs)
        if self.job_capacities is not None:
            return self._divide_tasks_by_capacities(grouped_exps)

        tasks = []

        for exp_name in grouped_exps:
            exp_group = grouped_exps[exp_name]

            max_rep = len(exp_group)

            # Use 1 Repetition per job if not defined otherwise
            rep_portion = 1
            if KEYS.REPS_P_JOB in exp_group[0]:
                rep_portion = exp_group[0][KEYS.REPS_P_JOB]

            for start_rep in range(0, max_rep, rep_portion):
                tasks.append(exp_group[start_rep : start_rep + rep_portion])
        return tasks

    def _divide_tasks_by_capacities(
        self,
        grouped_exps: Dict[str, List[Dict]],
    ) -> List[List[Dict]]:
        try:
            capacities = [int(value) for value in self.job_capacities]
        except (TypeError, ValueError) as exc:
            raise cw_error.ConfigKeyError(
                "auto_gpu_job_capacities must be a list of positive integers."
            ) from exc
        if not capacities or any(value < 1 for value in capacities):
            raise cw_error.ConfigKeyError(
                "auto_gpu_job_capacities must be a non-empty list of "
                "positive integers."
            )

        tasks = []
        capacity_idx = 0
        for exp_group in grouped_exps.values():
            task_idx = 0
            while task_idx < len(exp_group):
                if capacity_idx >= len(capacities):
                    raise cw_error.ConfigKeyError(
                        "auto_gpu_job_capacities does not cover every "
                        "expanded experiment run."
                    )
                capacity = capacities[capacity_idx]
                task_end = task_idx + capacity
                if task_end > len(exp_group):
                    raise cw_error.ConfigKeyError(
                        "auto_gpu_job_capacities crosses an experiment-group "
                        "boundary."
                    )
                task_group = exp_group[task_idx:task_end]
                for task_config in task_group:
                    task_config[KEYS.REPS_P_JOB] = capacity
                    task_config[KEYS.REPS_PARALL] = capacity
                tasks.append(task_group)
                task_idx = task_end
                capacity_idx += 1

        if capacity_idx != len(capacities):
            raise cw_error.ConfigKeyError(
                "auto_gpu_job_capacities contains more entries than expanded "
                "experiment runs require."
            )
        return tasks

    def create_jobs(self, exp_configs: List[Dict]) -> List[Job]:
        """creates a list of all jobs.

        Args:
            exp_configs (List[attrdict.AttrDict]): list of all defined experiment configurations.

        Returns:
            List[Job]: list of configured jobs.
        """
        task_list = self._divide_tasks(exp_configs)
        joblist = []
        for task in task_list:
            j = Job(
                task,
                self.exp_cls,
                self.logger,
                self.delete_old_files,
                self.root_dir,
                self.read_only,
            )
            joblist.append(j)
        return joblist
