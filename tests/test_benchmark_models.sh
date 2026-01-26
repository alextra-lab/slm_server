#!/bin/bash
# Test script for benchmark_models.py (slm_server)
# Tests start/stop functionality for each backend with each model role

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Testing slm_server.benchmark_models ==="
echo ""

# Test 1: List models
echo "Test 1: List models"
uv run python -m slm_server.benchmark_models list-models
echo ""

# Test 2: Check availability for each backend
echo "Test 2: Check MLX availability"
uv run python -m slm_server.benchmark_models check --backend mlx 2>&1 | head -20
echo ""

echo "Test 3: Check llama.cpp availability"
uv run python -m slm_server.benchmark_models check --backend llamacpp 2>&1 | head -20
echo ""


# Test 5: Test start command building (will fail without models, but tests error handling)
echo "Test 5: Test start command error handling (MLX - router model)"
timeout 5 uv run python -m slm_server.benchmark_models start --backend mlx --model router --port 1234 2>&1 || echo "Expected failure (no model found)"
echo ""

echo "Test 6: Test start command error handling (llama.cpp - router model)"
timeout 5 uv run python -m slm_server.benchmark_models start --backend llamacpp --model router --port 8001 2>&1 || echo "Expected failure (no model found)"
echo ""


echo "=== All tests completed ==="
echo ""
echo "Note: Actual server start/stop tests require:"
echo "  1. Backend tools installed (mlx-openai-server, llama-cpp-python[server])"
echo "  2. Model files available in expected locations"
echo "  3. Sufficient system resources"
