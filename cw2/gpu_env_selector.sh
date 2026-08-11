# GPU-aware Conda selection embedded into generated Slurm scripts.
MPRL_PYTHON_BIN="${MPRL_PYTHON_BIN:-python3}"

_mprl_conda_base() {
    if [ -n "${MPRL_CONDA_BASE:-}" ]; then
        printf '%s\n' "$MPRL_CONDA_BASE"
    elif [ -n "${CONDA_PREFIX:-}" ] && [[ "$CONDA_PREFIX" == */envs/* ]]; then
        printf '%s\n' "${CONDA_PREFIX%%/envs/*}"
    elif [ -n "${CONDA_EXE:-}" ]; then
        cd "$(dirname "$CONDA_EXE")/.." && pwd -P
    elif command -v conda >/dev/null 2>&1; then
        conda info --base
    fi
}

_mprl_env_library_path() {
    local env_prefix="$1"
    local env_library_path=""
    local site_packages
    local lib_dir

    for site_packages in "$env_prefix"/lib/python*/site-packages; do
        [ -d "$site_packages" ] || continue
        for lib_dir in "$site_packages/torch/lib" "$site_packages"/nvidia/*/lib; do
            [ -d "$lib_dir" ] || continue
            env_library_path="${env_library_path}${env_library_path:+:}${lib_dir}"
        done
    done
    printf '%s\n' "$env_library_path"
}

_mprl_env_prefix() {
    local candidate_env="$1"
    local candidate_prefix

    if [[ "$candidate_env" == /* ]] && [ -d "$candidate_env" ]; then
        printf '%s\n' "$candidate_env"
        return
    fi

    for candidate_prefix in \
        "$MPRL_CONDA_BASE/envs/$candidate_env" \
        "$HOME/miniconda3/envs/$candidate_env" \
        "/gpfs/petra3/scratch/$USER/miniconda3/envs/$candidate_env"; do
        if [ -d "$candidate_prefix" ]; then
            printf '%s\n' "$candidate_prefix"
            return
        fi
    done

    if [ -x "$MPRL_CONDA_BASE/bin/conda" ]; then
        "$MPRL_CONDA_BASE/bin/conda" env list 2>/dev/null |
            awk -v env_name="$candidate_env" '$1 == env_name { print $NF; exit }'
    fi
}

_mprl_remove_conda_envs_from_path() {
    local variable_name="$1"
    local current_value="${!variable_name:-}"
    local cleaned_value=""
    local path_entry
    local -a path_entries

    IFS=':' read -r -a path_entries <<< "$current_value"
    for path_entry in "${path_entries[@]}"; do
        [ -n "$path_entry" ] || continue
        case "$path_entry" in
            "$MPRL_CONDA_BASE"/envs/*)
                continue
                ;;
        esac
        cleaned_value="${cleaned_value}${cleaned_value:+:}${path_entry}"
    done
    printf -v "$variable_name" '%s' "$cleaned_value"
    export "$variable_name"
}

_mprl_probe_gpu_env() {
    local candidate_env="$1"
    local candidate_prefix
    local candidate_python
    local candidate_ld_library_path

    candidate_prefix="$(_mprl_env_prefix "$candidate_env")"
    candidate_python="$candidate_prefix/bin/python3"
    if [ ! -x "$candidate_python" ]; then
        MPRL_GPU_ENV_PROBE_OUTPUT="environment '$candidate_env' is not installed"
        return 1
    fi

    candidate_ld_library_path="$(_mprl_env_library_path "$candidate_prefix")"
    if MPRL_GPU_ENV_PROBE_OUTPUT="$(
        env -u PYTHONPATH \
            PYTHONNOUSERSITE=1 \
            LD_LIBRARY_PATH="$candidate_ld_library_path" \
            "$candidate_python" -c '
import torch

if not torch.cuda.is_available():
    print("torch.cuda.is_available() is false")
    raise SystemExit(2)

supported = set(torch.cuda.get_arch_list())
missing = []
for index in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(index)
    suffix = f"{major}{minor}"
    if f"sm_{suffix}" not in supported and f"compute_{suffix}" not in supported:
        missing.append(f"GPU{index}=sm_{suffix}")

if missing:
    print(
        "unsupported " + ", ".join(missing)
        + "; wheel architectures=" + " ".join(sorted(supported))
    )
    raise SystemExit(3)

print("compatible architectures=" + " ".join(sorted(supported)))
' 2>&1
    )"; then
        MPRL_GPU_ENV_PROBE_PREFIX="$candidate_prefix"
        return 0
    fi
    return 1
}

_mprl_name_contains_p100() {
    local value
    value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" == *p100* ]]
}

_mprl_configured_gpu_envs() {
    local candidate_spec="${MPRL_GPU_ENV_CANDIDATES:-}"
    local -a candidate_envs

    if [ -n "$candidate_spec" ]; then
        # Conda environment names cannot contain commas. Accept whitespace too
        # so the same variable remains convenient in shell and YAML configs.
        candidate_spec="${candidate_spec//,/ }"
        read -r -a candidate_envs <<< "$candidate_spec"
        printf '%s\n' "${candidate_envs[@]}"
        return
    fi

    # Backward compatibility for existing Maxwell configurations.
    printf '%s\n' \
        "${MPRL_GPU_ENV_DEFAULT:-policy_chunking_transformer_official}" \
        "${MPRL_GPU_ENV_P100:-policy_chunking_transformer_p100}"
}

_mprl_order_gpu_env_candidates() {
    local gpu_names="$1"
    shift
    local target_uses_p100=0
    local candidate_env
    local -a matching_envs
    local -a fallback_envs

    if _mprl_name_contains_p100 "$gpu_names"; then
        target_uses_p100=1
    fi

    for candidate_env in "$@"; do
        [ -n "$candidate_env" ] || continue
        if _mprl_name_contains_p100 "$candidate_env"; then
            if [ "$target_uses_p100" -eq 1 ]; then
                matching_envs+=("$candidate_env")
            else
                fallback_envs+=("$candidate_env")
            fi
        elif [ "$target_uses_p100" -eq 0 ]; then
            matching_envs+=("$candidate_env")
        else
            fallback_envs+=("$candidate_env")
        fi
    done

    printf '%s\n' "${matching_envs[@]}" "${fallback_envs[@]}"
}

_mprl_select_gpu_conda_env() {
    local gpu_names
    local selected_env=""
    local selected_prefix=""
    local tested_envs=""
    local candidate_env
    local selected_ld_library_path
    local -a configured_envs
    local -a candidate_envs

    MPRL_CONDA_BASE="$(_mprl_conda_base)"
    if [ -z "$MPRL_CONDA_BASE" ] || [ ! -f "$MPRL_CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "[slurm] Cannot locate Conda base for GPU-aware environment selection." >&2
        return 1
    fi
    export MPRL_CONDA_BASE

    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null)"
    if [ -z "$gpu_names" ]; then
        echo "[slurm] No visible NVIDIA GPU found; cannot select a CUDA environment." >&2
        return 1
    fi

    while IFS= read -r candidate_env; do
        [ -n "$candidate_env" ] && configured_envs+=("$candidate_env")
    done < <(_mprl_configured_gpu_envs)
    if [ "${#configured_envs[@]}" -eq 0 ]; then
        echo "[slurm] MPRL_GPU_ENV_CANDIDATES does not contain any environments." >&2
        return 1
    fi

    # Environment roles are inferred from their names. This keeps the
    # selector project-independent: each project only supplies its candidate
    # list, and an environment containing "p100" is preferred exactly for a
    # P100 worker. Keep the other class as a compatibility fallback; every
    # candidate is still validated against the visible GPU before use.
    while IFS= read -r candidate_env; do
        [ -n "$candidate_env" ] && candidate_envs+=("$candidate_env")
    done < <(_mprl_order_gpu_env_candidates "$gpu_names" "${configured_envs[@]}")

    for candidate_env in "${candidate_envs[@]}"; do
        case " $tested_envs " in
            *" $candidate_env "*) continue ;;
        esac
        tested_envs="${tested_envs}${tested_envs:+ }${candidate_env}"
        if _mprl_probe_gpu_env "$candidate_env"; then
            selected_env="$candidate_env"
            selected_prefix="$MPRL_GPU_ENV_PROBE_PREFIX"
            break
        fi
        echo "[slurm] Conda env '$candidate_env' rejected: $MPRL_GPU_ENV_PROBE_OUTPUT" >&2
    done

    if [ -z "$selected_env" ]; then
        echo "[slurm] None of the configured Conda environments supports: $gpu_names" >&2
        return 1
    fi

    # The generated PYTHONPATH and inherited LD_LIBRARY_PATH may still point
    # at the submission environment. Remove those entries before activation.
    _mprl_remove_conda_envs_from_path PYTHONPATH
    _mprl_remove_conda_envs_from_path LD_LIBRARY_PATH
    # shellcheck source=/dev/null
    source "$MPRL_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$selected_prefix"

    selected_ld_library_path="$(_mprl_env_library_path "$CONDA_PREFIX")"
    if [ -n "$selected_ld_library_path" ]; then
        export LD_LIBRARY_PATH="${selected_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    MPRL_PYTHON_BIN="$CONDA_PREFIX/bin/python3"
    export MPRL_PYTHON_BIN
    echo "[slurm] GPU model(s): ${gpu_names//$'\n'/,}"
    echo "[slurm] Selected Conda env '$selected_env' at '$selected_prefix' ($MPRL_GPU_ENV_PROBE_OUTPUT)"
}

if [ "${MPRL_GPU_ENV_AUTO_SELECT:-0}" = "1" ]; then
    _mprl_select_gpu_conda_env || exit 86
elif [ -n "${CONDA_PREFIX:-}" ]; then
    selected_ld_library_path="$(_mprl_env_library_path "$CONDA_PREFIX")"
    if [ -n "$selected_ld_library_path" ]; then
        export LD_LIBRARY_PATH="${selected_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    MPRL_PYTHON_BIN="$CONDA_PREFIX/bin/python3"
fi
