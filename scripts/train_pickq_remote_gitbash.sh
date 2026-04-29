#!/usr/bin/env bash
# Train PickQNet from rollout shards — run inside **Git Bash** on Windows.
# Sync git manually when GitHub is reachable (GPU server may be offline).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."
ROOT="${PWD}"

if [[ ! -d "${ROOT}/.git" ]]; then
  echo "error: ${ROOT} is not a git checkout (missing .git)" >&2
  exit 1
fi

PYTHON="${ROOT}/../venv/Scripts/python.exe"
if [[ ! -f "${PYTHON}" ]]; then
  echo "error: missing venv python: ${PYTHON}" >&2
  exit 1
fi

OUT_ROOT="${PICK14_TRAIN_OUT_DIR:-train_runs_remote}"
mkdir -p "${ROOT}/${OUT_ROOT}"

"${PYTHON}" -u "${ROOT}/scripts/05_train_timed.py" \
  --rollout-dir rollout_data \
  --epochs 30 \
  --log-every 1 \
  --batch 64 \
  --lr 1e-3 \
  --opp-weight 0.1 \
  --out-dir "${OUT_ROOT}" \
  --log-file "${OUT_ROOT}/train_pickq_latest.log"
