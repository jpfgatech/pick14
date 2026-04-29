#!/usr/bin/env bash
# Launch Git Bash on the Windows GPU box and run mass rollout (git pull + python).
#
# Requires:
#   - Repo at ~/Documents/projects/pick14 with .git (clone evolve-play-strategy).
#   - Git for Windows (bash.exe).
#
# Usage:
#   export PICK14_REMOTE_HOST=your.windows.host
#   optional: export PICK14_REMOTE_USER=Administrator
#   ./scripts/launch_rollout_mass_remote.sh

set -euo pipefail

HOST="${PICK14_REMOTE_HOST:-100.74.144.124}"
USER="${PICK14_REMOTE_USER:-administrator}"

echo "SSH ${USER}@${HOST} → Git Bash → scripts/run_rollout_mass_remote_gitbash.sh"
# shellcheck disable=SC2029
ssh "${USER}@${HOST}" '"C:/Program Files/Git/bin/bash.exe" -lc "cd /c/Users/Administrator/Documents/projects/pick14 && exec bash scripts/run_rollout_mass_remote_gitbash.sh"'
