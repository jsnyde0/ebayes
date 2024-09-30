#!/usr/bin/env bash
# exit on error
set -o errexit

# Activate virtual environment
source .venv/bin/activate

# Start Gunicorn
gunicorn a_core.wsgi:application