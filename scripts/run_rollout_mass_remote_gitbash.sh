#!/usr/bin/env bash
# Run from **Git Bash** on the Windows GPU box (not cmd/PowerShell).
# Repo root = parent of this file; venv = sibling ../venv (pick14-project layout).
#
# Pipeline: git pull → 20k stem-only (branch-free) mass rollout → 40-epoch timed Q training
# on the generated shards. Checkpoints under train_runs_remote/; CUDA used when available.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."
ROOT="${PWD}"

if [[ ! -d "${ROOT}/.git" ]]; then
  echo "error: ${ROOT} is not a git checkout (missing .git)" >&2
  exit 1
fi

git fetch origin
git checkout evolve-play-strategy
git pull origin evolve-play-strategy

PYTHON="${ROOT}/../venv/Scripts/python.exe"
if [[ ! -f "${PYTHON}" ]]; then
  echo "error: missing venv python: ${PYTHON}" >&2
  exit 1
fi

SESSION_LOG="${ROOT}/rollout_train_20k_40ep.log"
OUTPUT_ROLL="rollout_data_20k"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(ts)] Starting session log" >> "${SESSION_LOG}"

echo "[$(ts)] === 20k stem rollout (branch-free) → ${OUTPUT_ROLL} ===" >> "${SESSION_LOG}"
set +e
"${PYTHON}" -u "${ROOT}/scripts/05_rollout_mass.py" \
  --deck-configs 20000 \
  --reps-per-deck 1 \
  --max-games 20000 \
  --shard-every-games 500 \
  --output-dir "${OUTPUT_ROLL}" \
  --clear-output-dir \
  >> "${SESSION_LOG}" 2>&1
ec=$?
set -e
echo "[$(ts)] rollout_mass exit=${ec}" >> "${SESSION_LOG}"
if [[ "${ec}" -ne 0 ]]; then
  exit "${ec}"
fi

mkdir -p "${ROOT}/train_runs_remote"
echo "[$(ts)] === 40-epoch PickQ train (rollout-dir ${OUTPUT_ROLL}) ===" >> "${SESSION_LOG}"
set +e
"${PYTHON}" -u "${ROOT}/scripts/05_train_timed.py" \
  --rollout-dir "${OUTPUT_ROLL}" \
  --epochs 40 \
  --batch 64 \
  --lr 1e-3 \
  --seed 1 \
  --log-every 5 \
  --opp-weight 0.1 \
  --out-dir train_runs_remote \
  --log-file train_runs_remote/train_pickq_20k_40ep.log \
  >> "${SESSION_LOG}" 2>&1
ec2=$?
set -e
echo "[$(ts)] train_timed exit=${ec2}" >> "${SESSION_LOG}"
exit "${ec2}"
