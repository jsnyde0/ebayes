#!/usr/bin/env bash
# exit on error
set -o errexit

# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Find the actual installation directory
UV_INSTALL_DIR=$(dirname $(dirname $(which uv)))

# Add the installation directory to PATH
export PATH="$UV_INSTALL_DIR:$PATH"

# Find and source the environment file
ENV_FILE=$(find $UV_INSTALL_DIR -name "env" | head -n 1)
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "Environment file not found"
    exit 1
fi

# Verify uv installation
which uv

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Run Django migrations
uv run python manage.py migrate

# Collect static files
uv run python manage.py collectstatic --no-input