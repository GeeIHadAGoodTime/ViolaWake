#!/usr/bin/env bash
# Install (or re-install) the ViolaWake backend deploy reconciler on this host.
#
# Idempotent: re-running it re-renders the units from the repo copies and
# restarts the timer. That is the point -- the repo is the source of truth for
# what the host runs, so a drifted hand-edit is repaired by running this again
# rather than by editing /etc.
#
# Usage (as root, from the deploy checkout):
#     sudo infra/deploy/install.sh
#     sudo infra/deploy/install.sh --uninstall
#
# Secrets: none. This script never writes a credential. Host-specific settings
# (notably where operator alerts are appended) live in /etc/violawake-deploy.env
# which is off-VCS and created here only as a commented template if absent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR=/etc/systemd/system
STATE_DIR=/var/lib/violawake-deploy
ENV_FILE=/etc/violawake-deploy.env
SERVICE=violawake-deploy.service
TIMER=violawake-deploy.timer

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl disable --now "$TIMER" 2>/dev/null || true
    rm -f "$UNIT_DIR/$TIMER" "$UNIT_DIR/$SERVICE"
    systemctl daemon-reload
    echo "removed $TIMER and $SERVICE (state in $STATE_DIR and $ENV_FILE left alone)"
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "install.sh must run as root (it writes to $UNIT_DIR)" >&2
    exit 1
fi

PYTHON="$(command -v python3)"
if [[ -z "$PYTHON" ]]; then
    echo "python3 not found on PATH" >&2
    exit 1
fi

for required in "$REPO_ROOT/scripts/deploy_backend.py" "$REPO_ROOT/scripts/check_in_flight_jobs.py"; do
    [[ -f "$required" ]] || { echo "missing $required -- is $REPO_ROOT the deploy checkout?" >&2; exit 1; }
done
command -v flock >/dev/null || { echo "flock not found (util-linux)" >&2; exit 1; }

mkdir -p "$STATE_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" <<'TEMPLATE'
# Host-specific settings for the ViolaWake backend deploy reconciler.
# This file is off-VCS on purpose: it is where a host says where its operator
# alert inbox lives, so the ViolaWake repo never hard-codes another project's
# paths.
#
# VIOLAWAKE_DEPLOY_ALERT_SINK=/path/to/red_alerts.jsonl
# VIOLAWAKE_DEPLOY_STATE_DIR=/var/lib/violawake-deploy
TEMPLATE
    chmod 600 "$ENV_FILE"
    echo "created template $ENV_FILE (no alert sink configured yet)"
fi

render() {
    sed -e "s#@REPO_ROOT@#${REPO_ROOT}#g" -e "s#@PYTHON@#${PYTHON}#g" "$1" > "$2"
}

render "$REPO_ROOT/infra/deploy/$SERVICE" "$UNIT_DIR/$SERVICE"
render "$REPO_ROOT/infra/deploy/$TIMER" "$UNIT_DIR/$TIMER"
chmod 644 "$UNIT_DIR/$SERVICE" "$UNIT_DIR/$TIMER"

systemctl daemon-reload
systemctl enable --now "$TIMER"

echo "installed:"
systemctl list-timers "$TIMER" --no-pager || true
echo
echo "one tick on demand:  systemctl start $SERVICE"
echo "what it would do:    $PYTHON $REPO_ROOT/scripts/deploy_backend.py --dry-run"
echo "logs:                journalctl -u $SERVICE -n 100"
