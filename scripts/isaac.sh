#!/usr/bin/env bash
# Run a command in the Isaac Sim 6.1 environment on an ORCD GPU node.
#
#   scripts/isaac.sh uv sync --locked          # install the pinned environment (once, ~25 GB)
#   scripts/isaac.sh uv run pytest -m isaac    # anything else, exactly as on a workstation
#   scripts/isaac.sh uv run sp infer data-set/scene017 --num-worlds 64
#   scripts/isaac.sh bash                      # interactive shell inside the container
#
# Why: the cluster nodes are Rocky 8 (glibc 2.28) and the Isaac Sim 6.1 wheels are
# manylinux_2_35, so they cannot run on the host. The container (scripts/isaac-sim.def)
# only supplies glibc 2.39 and the shared libraries Kit dlopens; the environment itself
# is the normal uv venv from uv.lock, and the GPU comes from the host driver via --nv.
#
# The venv, the uv cache and Kit's shader caches live under $SP_POOL (default
# /orcd/pool/006/$USER) because a full Isaac Sim environment is ~25 GB and the home
# quota is 200 GB. Overridable: SP_POOL, SP_SIF, UV_PROJECT_ENVIRONMENT, UV_CACHE_DIR.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
POOL=${SP_POOL:-/orcd/pool/006/$USER}
SIF=${SP_SIF:-$POOL/containers/isaac-sim-6.1.sif}

if [[ ! -f $SIF ]]; then
    echo "isaac.sh: container image not found: $SIF" >&2
    echo "  module load apptainer && env -u APPTAINER_BINDPATH apptainer build \\" >&2
    echo "      $SIF $REPO/scripts/isaac-sim.def" >&2
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
fi
command -v apptainer >/dev/null 2>&1 || { echo "isaac.sh: apptainer not found (module load apptainer)" >&2; exit 1; }

UV=${UV:-$(command -v uv || true)}
[[ -x ${UV:-} ]] || { echo "isaac.sh: uv not found on PATH" >&2; exit 1; }

ENV_DIR=${UV_PROJECT_ENVIRONMENT:-$POOL/envs/scene-physics-isaac}
CACHE_DIR=${UV_CACHE_DIR:-$POOL/uv_cache}          # same filesystem as the venv, so uv hardlinks
KIT_CACHE=${SP_KIT_CACHE:-$POOL/cache}             # ~/.cache/{ov,warp,nv} can reach several GB
mkdir -p "$ENV_DIR" "$CACHE_DIR" "$KIT_CACHE" "$POOL/nvidia-omniverse"

# --cleanenv so the host's conda/module PATH cannot leak in and shadow the container's
# python3.12 (the host interpreters are glibc 2.28 binaries and would break uv's
# platform resolution). Everything the run needs is passed explicitly.
exec apptainer exec --nv --cleanenv \
    --env "PATH=$(dirname "$UV"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "UV_PROJECT_ENVIRONMENT=$ENV_DIR" \
    --env "UV_CACHE_DIR=$CACHE_DIR" \
    --env "UV_PYTHON=/usr/bin/python3.12" \
    --env "UV_PYTHON_DOWNLOADS=never" \
    --env "XDG_CACHE_HOME=$KIT_CACHE" \
    --env "OMNI_USER_CACHE_DIR=$KIT_CACHE/ov" \
    --env "OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}" \
    --env "SCENE_PHYSICS_DATA=${SCENE_PHYSICS_DATA:-}" \
    --env "SCENE_PHYSICS_RESULTS=${SCENE_PHYSICS_RESULTS:-}" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "NUM_WORLDS=${NUM_WORLDS:-}" \
    --env "NUM_EPOCHS=${NUM_EPOCHS:-}" \
    --env "TERM=${TERM:-dumb}" \
    --bind "$REPO" --bind "$POOL" --bind "$HOME" \
    --pwd "$REPO" \
    "$SIF" "$@"
