#!/usr/bin/env bash
# exit on error
set -o errexit

# Create a directory for uv
mkdir -p $HOME/.local/bin

# Install uv to the local directory
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir $HOME/.local/bin

# Add the local bin directory to PATH
export PATH="$HOME/.local/bin:$PATH"

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Run Django migrations
uv run python manage.py migrate

# Collect static files
uv run python manage.py collectstatic --no-input