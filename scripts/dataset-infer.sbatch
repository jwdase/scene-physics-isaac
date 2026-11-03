#!/bin/bash
# Run the model on every scene of one or more datasets under data/, in one Isaac Sim session.
#
#   sbatch scripts/dataset-infer.sbatch data-set-3
#   sbatch scripts/dataset-infer.sbatch data-set-3 data-set-4
#   sbatch --time=1:00:00 scripts/dataset-infer.sbatch data-set-4 --num-worlds 32 --iterations 100
#   SP_TAG=rerun sbatch scripts/dataset-infer.sbatch data-set-3   # ...-<tag> on the results directory
#
# Arguments that name a directory under data/ are datasets, every other argument is passed on to
# `sp infer` after the defaults below, so repeating an option (e.g. --num-worlds) overrides it.
#
# Results land where `sp infer` puts them, one directory per scene:
# results/<experiment>/<scene>/<timestamp>-<tag>/, where the experiment comes from the scene itself
# (data-set-3 scenes carry experiment "data-set-3"), so each scene's results stay in its own dataset
# folder even when one job covers several datasets. They are written as each scene finishes.
#
# Partition: ou_bcs_normal, because ou_bcs_low is PreemptMode=REQUEUE and a preempted sweep would
# restart from the first scene and write a second set of result directories. Submit from the
# repository root; `sbatch --partition=... --time=...` overrides anything below.
#SBATCH --job-name=sp-dataset-infer
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sp-dataset-infer-%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO=$PWD
mkdir -p logs

WORLDS=${SP_WORLDS:-64}            # 64 worlds is what the H100 timings in docs/DESIGN.md were measured at
ITERATIONS=${SP_ITERS:-50}
TAG=${SP_TAG:-job${SLURM_JOB_ID:-local}}

datasets=()
extra=()
for a in "$@"; do
    if [[ -d data/$a ]]; then datasets+=("$a"); else extra+=("$a"); fi
done
if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "usage: sbatch scripts/dataset-infer.sbatch <dataset-under-data> [...] [sp infer options]" >&2
    exit 2
fi

scenes=()
for ds in "${datasets[@]}"; do
    for d in data/"$ds"/*/; do
        [[ -f ${d}scene.usd ]] && scenes+=("${d%/}")
    done
done
if [[ ${#scenes[@]} -eq 0 ]]; then
    echo "dataset-infer: no scene.usd under data/{${datasets[*]}}" >&2
    exit 2
fi

echo "[job] ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is)"
echo "[job] ${#scenes[@]} scenes from ${datasets[*]}: $WORLDS worlds, $ITERATIONS iterations, tag $TAG"
printf '[job]   %s\n' "${scenes[@]}"
module load apptainer 2>/dev/null || true
nvidia-smi -L || true

# `uv run --no-sync` with PYTHONPATH rather than a plain `uv run sp`: the environment under $SP_POOL
# has this package installed editable at whichever spelling of $HOME the last `uv sync` used
# (/home/... or /orcd/home/...), only one of which is bound into the container, and --no-sync also
# keeps a batch job from rewriting that shared environment under interactive sessions using it.
sp() { scripts/isaac.sh env PYTHONPATH="$REPO/src" uv run --no-sync sp "$@"; }

# Readiness report first: roles, priors, meshes, segmentation ids. CPU only, no Kit, ~1 s a scene.
sp validate "${scenes[@]}" || echo "[job] warning: scenes above marked INVALID will fail; running the rest"

status=0
sp infer "${scenes[@]}" --num-worlds "$WORLDS" --iterations "$ITERATIONS" --tag "$TAG" \
    ${extra[@]+"${extra[@]}"} || status=$?
echo "[job] finished at $(date -Is) with status $status"
exit $status
