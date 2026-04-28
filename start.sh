#!/bin/bash
# Start script for SLM Server
#
# This script starts:
# 1. Backend model servers (MLX/llama.cpp)
# 2. FastAPI routing service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting SLM Server..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    exit 1
fi

# Function to check if a port is listening
# Prefer lsof on macOS (Darwin) for explicit LISTEN state; nc can be flaky with BSD netcat
check_port() {
    local port=$1
    if [ "$(uname -s)" = "Darwin" ] && command -v lsof &> /dev/null; then
        lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v nc &> /dev/null; then
        nc -z localhost "$port" 2>/dev/null
    elif command -v lsof &> /dev/null; then
        lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1
    else
        # Fallback: try to connect using bash's built-in TCP
        (exec 3<>/dev/tcp/localhost/"$port") 2>/dev/null
        local result=$?
        exec 3>&-
        [ $result -eq 0 ]
    fi
}

# Function to verify all configured backend ports are listening
verify_backend_ports() {
    echo "🔍 Verifying backend servers are ready..."
    
    # Get all configured ports from models.yaml using Python
    local config_file="$SCRIPT_DIR/config/models.yaml"
    local ports
    ports=$(CONFIG_FILE="$config_file" uv run python -c "
from pathlib import Path
import yaml
import sys
import os

config_path = Path(os.environ.get('CONFIG_FILE', 'config/models.yaml'))
if not config_path.exists():
    print(f'Error: config file not found: {config_path}', file=sys.stderr)
    sys.exit(1)

with config_path.open() as f:
    data = yaml.safe_load(f)

ports = []
for role, model in data.get('models', {}).items():
    backend = model.get('backend', '')
    enabled = model.get('enabled', True)  # Default to True if not specified
    port = model.get('port')
    if port and enabled:  # Skip disabled models
        ports.append(str(port))

print(' '.join(ports))
" 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to read model configuration"
        echo "$ports"
        return 1
    fi
    
    if [ -z "$ports" ]; then
        echo "⚠️  Warning: No backend ports found in configuration"
        return 0
    fi
    
    local max_attempts=30
    local attempt=0
    local failed_ports=()
    
    while [ $attempt -lt $max_attempts ]; do
        failed_ports=()
        for port in $ports; do
            if ! check_port "$port"; then
                failed_ports+=("$port")
            fi
        done
        
        if [ ${#failed_ports[@]} -eq 0 ]; then
            echo "✅ All backend servers are ready"
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -lt $max_attempts ]; then
            sleep 1
        fi
    done
    
    echo "❌ Error: Backend servers failed to start on the following ports:"
    for port in "${failed_ports[@]}"; do
        echo "   - Port $port"
    done
    echo ""
    echo "Please check the backend server logs for errors."
    return 1
}

# Start backend servers in background (--extra mlx/llamacpp so mlx-openai-server etc. are available)
echo "📦 Starting backend model servers..."
uv run --extra mlx --extra llamacpp python -m slm_server backends &
BACKEND_PID=$!

# Wait for backend processes to spawn and for mlx-openai-server to load models and bind to ports
sleep 10

# Ensure the backend launcher itself is still running (port checks alone can be false positives
# if stale processes are already bound to configured ports).
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "❌ Error: Backend launcher process exited unexpectedly (pid $BACKEND_PID)"
    exit 1
fi

# Verify all backend servers are ready
if ! verify_backend_ports; then
    echo "🛑 Shutting down backend servers..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start routing service in background
echo "🔄 Starting routing service..."
uv run python -m slm_server router &
ROUTER_PID=$!

# Wait a bit for router to start
sleep 2

# Ensure router process did not die immediately (e.g., port already in use).
if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    echo "❌ Error: Routing service process exited unexpectedly (pid $ROUTER_PID)"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Verify router is ready
if ! check_port 8000; then
    echo "❌ Error: Routing service failed to start on port 8000"
    kill $BACKEND_PID $ROUTER_PID 2>/dev/null || true
    exit 1
fi

echo "✅ SLM Server running on http://localhost:8000"

# Trap signals to cleanup
trap "echo '🛑 Shutting down...'; kill $BACKEND_PID $ROUTER_PID 2>/dev/null; exit" INT TERM

# Wait for processes
wait
