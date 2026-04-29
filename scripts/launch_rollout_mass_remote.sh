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

# Default GPU server — see project_rule.md (override with PICK14_REMOTE_HOST).
HOST="${PICK14_REMOTE_HOST:-100.74.144.124}"
USER="${PICK14_REMOTE_USER:-administrator}"

echo "SSH ${USER}@${HOST} → run_rollout_mass_remote.bat"
ssh "${USER}@${HOST}" 'cmd.exe /c "C:\Users\Administrator\Documents\projects\pick14\scripts\run_rollout_mass_remote.bat"'
