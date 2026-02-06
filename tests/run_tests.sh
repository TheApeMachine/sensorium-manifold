#!/bin/bash
# Test runner script for Sensorium Manifold
# Run this before and after making core system changes

set -e

echo "=========================================="
echo "Sensorium Manifold Test Suite"
echo "=========================================="
echo ""

# Run unit tests
echo "1. Running unit tests..."
uv run pytest tests/ -v --tb=short

# Run benchmarks and compare to baseline
echo ""
echo "2. Running benchmarks..."
if [ -f tests/baseline_v1.json ]; then
    echo "   Comparing against baseline..."
    uv run python -m tests.benchmark_suite --compare tests/baseline_v1.json
else
    echo "   No baseline found. Creating initial baseline..."
    uv run python -m tests.benchmark_suite --save tests/baseline_v1.json
fi

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
