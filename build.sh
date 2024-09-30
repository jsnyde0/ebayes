#!/usr/bin/env bash
# exit on error
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
export PATH="/root/.cargo/bin:$PATH"

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Run Django migrations
uv run python manage.py migrate

# Collect static files
uv run python manage.py collectstatic --no-input