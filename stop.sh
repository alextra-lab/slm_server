#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${BLUE}🛑 Stopping SLM Server...${NC}"

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping $name on port $port (PID: $pid)...${NC}"
        kill $pid 2>/dev/null || true
        sleep 1
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo -e "${RED}Force killing $name (PID: $pid)...${NC}"
            kill -9 $pid 2>/dev/null || true
        fi
        echo -e "${GREEN}✅ Stopped $name${NC}"
    else
        echo -e "${YELLOW}ℹ️  $name not running on port $port${NC}"
    fi
}

# Function to kill process by name pattern
kill_pattern() {
    local pattern=$1
    local name=$2
    local pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Stopping $name processes...${NC}"
        echo "$pids" | while read pid; do
            if [ -n "$pid" ]; then
                kill $pid 2>/dev/null || true
            fi
        done
        sleep 1
        
        # Force kill any remaining
        pids=$(pgrep -f "$pattern" 2>/dev/null)
        if [ -n "$pids" ]; then
            echo -e "${RED}Force killing remaining $name processes...${NC}"
            echo "$pids" | while read pid; do
                if [ -n "$pid" ]; then
                    kill -9 $pid 2>/dev/null || true
                fi
            done
        fi
        echo -e "${GREEN}✅ Stopped $name${NC}"
    else
        echo -e "${YELLOW}ℹ️  No $name processes found${NC}"
    fi
}

# Stop router service
echo ""
echo -e "${BLUE}🔄 Stopping routing service...${NC}"
kill_port 8000 "Router"

# Stop backend model servers. Ports and names are read from models.yaml so this
# block never goes stale when the model lineup changes. If the read fails (e.g. uv
# unavailable), the catch-all kill_pattern calls below still reap every backend.
echo ""
echo -e "${BLUE}📦 Stopping backend model servers...${NC}"
BACKEND_PORTS=""
PORT_LINES=$(uv run python -c "
from pathlib import Path
import yaml
config = Path('$SCRIPT_DIR/config/models.yaml')
data = yaml.safe_load(config.read_text()) if config.exists() else {}
for role, model in (data.get('models') or {}).items():
    port = model.get('port')
    if port:
        print(f\"{port}|{role} ({model.get('id', role)})\")
" 2>/dev/null)
OLD_IFS=$IFS
IFS='
'
for line in $PORT_LINES; do
    port=${line%%|*}
    label=${line#*|}
    case "$port" in
        ''|*[!0-9]*) continue ;;  # skip blank/non-numeric (e.g. stray output)
    esac
    kill_port "$port" "$label"
    BACKEND_PORTS="$BACKEND_PORTS $port"
done
IFS=$OLD_IFS

# Clean up any remaining mlx-openai-server processes
echo ""
echo -e "${BLUE}🧹 Cleaning up any remaining processes...${NC}"
kill_pattern "mlx-openai-server" "mlx-openai-server"

# Clean up uvicorn processes for this project
kill_pattern "uvicorn.*slm_server" "uvicorn (slm_server)"

# Clean up orphaned slm_server launcher processes. These are the detached
# `uv run ... python -m slm_server backends/router` wrappers (and their children)
# that survive when start.sh is terminated early — kill_port only reaps whatever
# holds the listening socket, leaving these behind.
kill_pattern "slm_server backends" "slm_server backends launcher"
kill_pattern "slm_server router" "slm_server router launcher"

# Clean up native llama.cpp model servers (orphans that lost their port survive
# the port-based kills above). This project owns all llama-server instances.
kill_pattern "llama-server" "llama.cpp model servers"

# Verify all stopped
echo ""
echo -e "${BLUE}🔍 Verifying all services stopped...${NC}"

all_stopped=true

# Check ports
for port in 8000 $BACKEND_PORTS; do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${RED}❌ Port $port still in use${NC}"
        all_stopped=false
    fi
done

# Check processes
if pgrep -f "mlx-openai-server" > /dev/null 2>&1; then
    echo -e "${RED}❌ mlx-openai-server processes still running${NC}"
    all_stopped=false
fi

if pgrep -f "uvicorn.*slm_server" > /dev/null 2>&1; then
    echo -e "${RED}❌ uvicorn processes still running${NC}"
    all_stopped=false
fi

if pgrep -f "slm_server (backends|router)" > /dev/null 2>&1; then
    echo -e "${RED}❌ slm_server launcher processes still running${NC}"
    all_stopped=false
fi

if pgrep -f "llama-server" > /dev/null 2>&1; then
    echo -e "${RED}❌ llama-server processes still running${NC}"
    all_stopped=false
fi

if [ "$all_stopped" = true ]; then
    echo -e "${GREEN}✅ All services stopped cleanly${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may still be running${NC}"
    echo -e "${YELLOW}Run with 'sudo' or manually check: ps aux | grep -E 'mlx-openai-server|uvicorn'${NC}"
fi

echo ""
echo -e "${GREEN}🎉 SLM Server stopped${NC}"
