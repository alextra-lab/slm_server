#!/bin/bash
# Install the SLM Server LaunchAgent so the stack starts at login (FRE-241 AC-4).
#
#   ./scripts/install-launchagent.sh            install (or reinstall) and start
#   ./scripts/install-launchagent.sh --uninstall stop and remove
#
# The agent runs as your user, not as root: the models sit on an external volume
# under your account and llama-server needs no privileges.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="com.slm-server"
TEMPLATE="$REPO_DIR/config/launchd/$LABEL.plist.example"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "${1:-}" = "--uninstall" ]; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$TARGET"
    echo "✅ Removed $LABEL"
    exit 0
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "❌ Missing template: $TEMPLATE" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPO_DIR/logs"

# Render the template with this checkout's path.
sed "s|__SLM_SERVER_DIR__|$REPO_DIR|g" "$TEMPLATE" > "$TARGET"
plutil -lint "$TARGET" >/dev/null

# bootout first so a reinstall replaces cleanly rather than erroring on a
# label that is already loaded.
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$TARGET"
launchctl enable "$DOMAIN/$LABEL"

echo "✅ Installed $LABEL"
echo "   plist:   $TARGET"
echo "   repo:    $REPO_DIR"
echo "   logs:    $REPO_DIR/logs/launchd.{out,err} and $REPO_DIR/logs/watchdog.jsonl"
echo ""
echo "Status:   launchctl print $DOMAIN/$LABEL | head -20"
echo "Restart:  launchctl kickstart -k $DOMAIN/$LABEL"
