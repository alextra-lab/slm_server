#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Stop backend model servers
echo ""
echo -e "${BLUE}📦 Stopping backend model servers...${NC}"
kill_port 8500 "Router Model (LFM2.5)"
kill_port 8501 "Standard Model (Qwen3-4B)"
kill_port 8502 "Reasoning Model (Qwen3-8B)"
kill_port 8503 "Coding Model (Devstral)"

# Clean up any remaining mlx-openai-server processes
echo ""
echo -e "${BLUE}🧹 Cleaning up any remaining processes...${NC}"
kill_pattern "mlx-openai-server" "mlx-openai-server"

# Clean up uvicorn processes for this project
kill_pattern "uvicorn.*slm_server" "uvicorn (slm_server)"

# Verify all stopped
echo ""
echo -e "${BLUE}🔍 Verifying all services stopped...${NC}"

all_stopped=true

# Check ports
for port in 8000 8500 8501 8502 8503; do
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

if [ "$all_stopped" = true ]; then
    echo -e "${GREEN}✅ All services stopped cleanly${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may still be running${NC}"
    echo -e "${YELLOW}Run with 'sudo' or manually check: ps aux | grep -E 'mlx-openai-server|uvicorn'${NC}"
fi

echo ""
echo -e "${GREEN}🎉 SLM Server stopped${NC}"
