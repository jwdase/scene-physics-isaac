#!/bin/bash
# Batch inference on an ORCD GPU node. Everything after the script is passed to `sp infer`:
#
#   sbatch scripts/infer.sbatch data-set/scene017 --num-worlds 64 --iterations 50
#   sbatch scripts/infer.sbatch 'data-set/scene0*' --num-worlds 64        # one Kit session, many scenes
#   SBATCH_PARTITION=ou_bcs_high sbatch --time=2:00:00 scripts/infer.sbatch ...
#
# Submit from the repository root; results land in results/<exp>/<scene>/<timestamp>/ as usual.
#SBATCH --job-name=sp-infer
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sp-infer-%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load apptainer
nvidia-smi -L

# Quote-expand scene globs here rather than in the submit shell, so one job can take a whole dataset.
args=()
for a in "$@"; do
    case "$a" in
        *[*?]*) for m in $a; do args+=("$m"); done ;;
        *)      args+=("$a") ;;
    esac
done

exec scripts/isaac.sh uv run sp infer "${args[@]}"
