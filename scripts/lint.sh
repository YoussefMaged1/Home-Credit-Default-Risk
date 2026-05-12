#!/bin/bash
set -e

echo "Running ruff check..."
uv run ruff check src/

echo "Running ruff format check..."
uv run ruff format --check src/

echo "All checks passed!"