#!/usr/bin/env bash
# exit on error
set -o errexit

# Create a directory for uv
mkdir -p $HOME/.local/bin

# Download uv binary
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-linux-x64 -o $HOME/.local/bin/uv

# Make uv executable
chmod +x $HOME/.local/bin/uv

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