#!/bin/bash
#SBATCH -p %%partition%%
# #SBATCH -A %%account%%
#SBATCH -J %%job-name%%
#SBATCH --array 0-%%last_job_idx%%%%%num_parallel_jobs%%

# Please use the complete path details :
#SBATCH -D %%experiment_execution_dir%%
#SBATCH -o %%slurm_log%%/out_%A_%a.log
#SBATCH -e %%slurm_log%%/err_%A_%a.log

# Cluster Settings
#SBATCH -n %%ntasks%%         # Number of tasks
#SBATCH -c %%cpus-per-task%%  # Number of cores per task
#SBATCH --mem-per-cpu=%%mem-per-cpu%% # Main memory in MByte per MPI task
#SBATCH -t %%time%%             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

%%sbatch_args%%
# -------------------------------

# Activate the virtualenv / conda environment
%%venv%%

# Make pip-installed CUDA split libraries visible to PyTorch in batch jobs.
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
fi

# Export Pythonpath
%%pythonpath%%

# Additional Instructions from CONFIG.yml
%%sh_lines%%

python3 %%python_script%% %%path_to_yaml_config%% -j $SLURM_ARRAY_TASK_ID %%cw_args%%

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE
