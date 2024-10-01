#!/usr/bin/env bash
# exit on error
set -o errexit

# Create a directory for uv
mkdir -p /opt/render/.local/bin/bin

# Set UV_INSTALL_DIR to the correct directory
export UV_INSTALL_DIR="/opt/render/.local/bin/bin"

# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add the installation directory to PATH
export PATH="/opt/render/.local/bin/bin:$PATH"

# Source the environment file created by the installer
source /opt/render/.local/bin/env

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