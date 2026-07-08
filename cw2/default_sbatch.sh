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
#SBATCH -t %%time%%             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

%%sbatch_args%%
# -------------------------------

# Activate the virtualenv / conda environment
%%venv%%

# Make PyTorch and pip-installed CUDA split libraries visible in batch jobs.
if [ -n "$CONDA_PREFIX" ]; then
    PY_SITE_PACKAGES="$CONDA_PREFIX/lib/python3.11/site-packages"
    for lib_dir in "$PY_SITE_PACKAGES/torch/lib" "$PY_SITE_PACKAGES"/nvidia/*/lib; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        fi
    done
fi

# Export Pythonpath
%%pythonpath%%

# Additional Instructions from CONFIG.yml
%%sh_lines%%

CW2_PYTHON_PID=""

forward_signal_to_python() {
    signal_name="$1"
    if [ -n "$CW2_PYTHON_PID" ] && kill -0 "$CW2_PYTHON_PID" 2>/dev/null; then
        echo "[slurm] Forwarding ${signal_name} to python process ${CW2_PYTHON_PID}"
        kill -s "$signal_name" "$CW2_PYTHON_PID" 2>/dev/null || true
    fi
}

trap 'forward_signal_to_python USR1' USR1
trap 'forward_signal_to_python TERM' TERM
trap 'forward_signal_to_python INT' INT

python3 %%python_script%% %%path_to_yaml_config%% -j $SLURM_ARRAY_TASK_ID %%cw_args%% &
CW2_PYTHON_PID=$!

while true; do
    wait "$CW2_PYTHON_PID"
    exit_code=$?
    if kill -0 "$CW2_PYTHON_PID" 2>/dev/null; then
        continue
    fi
    exit "$exit_code"
done

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE
