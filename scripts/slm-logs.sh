#!/bin/bash
# Live-tail SLM Server logs, readably.
#
#   ./scripts/slm-logs.sh              # server output (router + backends)
#   ./scripts/slm-logs.sh watchdog     # restart decisions only
#   ./scripts/slm-logs.sh all          # both, interleaved
#   ./scripts/slm-logs.sh raw          # server output, unformatted
#
# Structlog emits dense JSON lines; they are flattened to
# `HH:MM:SS LEVEL event key=value ...` here, and non-JSON lines pass through
# untouched so uvicorn and start.sh banners still read normally.
#
# The stack is started by hand (./start.sh), so its stdout goes wherever you
# started it — a tmux pane, usually. This reads the files instead, which is
# what you want when that pane is elsewhere or has scrolled past.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/logs"
MODE="${1:-server}"

# start.sh tees its whole run here. This used to name four candidates, three of
# which nothing has ever written — two belonged to the withdrawn LaunchAgent and
# one was a scratch file from a test run — so the default mode could not display
# anything. It also expanded an empty array under `set -u`, which on macOS's
# /bin/bash 3.2 aborts with "unbound variable" rather than degrading. Verified:
# this host reports 3.2.57 and does raise it.
SERVER_LOG="$LOG_DIR/start.out"

require_server_log() {
    if [ ! -f "$SERVER_LOG" ]; then
        echo "No server log yet: $SERVER_LOG" >&2
        echo "It is written by ./start.sh — start the server, or use 'watchdog' mode." >&2
        exit 1
    fi
}

prettify() {
    python3 -u -c '
import json, sys

LEVELS = {"debug": "DBG", "info": "INF", "warning": "WRN", "error": "ERR", "critical": "CRT"}

for line in sys.stdin:
    line = line.rstrip("\n")
    stripped = line.lstrip()
    if not stripped.startswith("{"):
        print(line, flush=True)
        continue
    try:
        doc = json.loads(stripped)
    except json.JSONDecodeError:
        print(line, flush=True)
        continue
    ts = str(doc.pop("timestamp", None) or doc.pop("ts", ""))[11:19]
    level = LEVELS.get(str(doc.pop("level", "")).lower(), "   ")
    event = doc.pop("event", "")
    rest = " ".join(f"{k}={v}" for k, v in doc.items() if v is not None)
    print(f"{ts} {level} {event:<28} {rest}".rstrip(), flush=True)
'
}

case "$MODE" in
    watchdog)
        echo "→ tailing $LOG_DIR/watchdog.jsonl"
        touch "$LOG_DIR/watchdog.jsonl"
        tail -n 50 -F "$LOG_DIR/watchdog.jsonl" | prettify
        ;;
    raw)
        require_server_log
        tail -n 100 -F "$SERVER_LOG"
        ;;
    all)
        require_server_log
        touch "$LOG_DIR/watchdog.jsonl"
        tail -n 50 -F "$SERVER_LOG" "$LOG_DIR/watchdog.jsonl" | prettify
        ;;
    server|*)
        require_server_log
        tail -n 100 -F "$SERVER_LOG" | prettify
        ;;
esac
