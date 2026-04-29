#!/usr/bin/env bash
# Run from **Git Bash** on Windows (not cmd/PowerShell).
# Repo root = parent of this file; venv = sibling ../venv (pick14-project layout).
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

LOG="${ROOT}/rollout_mass_80k.log"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(ts)] Starting rollout_mass (10k games, stem greedy–stingy)" >> "${LOG}"
set +e
"${PYTHON}" -u "${ROOT}/scripts/05_rollout_mass.py" \
  --deck-configs 10000 \
  --reps-per-deck 1 \
  --max-games 10000 \
  --shard-every-games 500 \
  --output-dir rollout_data \
  --clear-output-dir \
  >> "${LOG}" 2>&1
ec=$?
set -e
echo "[$(ts)] Finished rollout_mass exit=${ec}" >> "${LOG}"
exit "${ec}"
