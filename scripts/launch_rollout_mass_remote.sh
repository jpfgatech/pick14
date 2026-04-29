#!/usr/bin/env bash
# Launch scripts/run_rollout_mass_remote.bat on the Windows GPU box via SSH.
#
# Prerequisites:
#   - Repo synced on server (git pull / matching branch).
#   - Windows OpenSSH with Administrator login.
#
# The rollout runs for a long time; keep this SSH session open or run ssh from tmux/screen.
#
# Usage:
#   export PICK14_REMOTE_HOST=your.windows.host   # hostname or IP
#   optional: export PICK14_REMOTE_USER=Administrator
#   ./scripts/launch_rollout_mass_remote.sh

set -euo pipefail

HOST="${PICK14_REMOTE_HOST:-}"
if [[ -z "${HOST}" ]]; then
  echo "Set PICK14_REMOTE_HOST to the Windows machine hostname or IP, then re-run." >&2
  echo "Example: export PICK14_REMOTE_HOST=203.0.113.50 && ./scripts/launch_rollout_mass_remote.sh" >&2
  exit 1
fi

USER="${PICK14_REMOTE_USER:-Administrator}"

echo "SSH ${USER}@${HOST} → run_rollout_mass_remote.bat"
ssh "${USER}@${HOST}" 'cmd.exe /c "C:\Users\Administrator\Documents\projects\pick14\scripts\run_rollout_mass_remote.bat"'
