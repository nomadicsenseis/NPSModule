#!/bin/bash

# Script to start services in the development container

echo "Starting development environment..."

# Load environment variables if they exist
if [ -f /app/.devcontainer/.env ]; then
    echo "Loading environment variables from .devcontainer/.env"
    source /app/.devcontainer/.env
fi

if [ -f /app/.env ]; then
    echo "Loading environment variables from .env"
    source /app/.env
fi

# Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH

echo "Development environment ready!"
echo "Python path: $PYTHONPATH"
echo "Current directory: $(pwd)"

# Keep container running
exec "$@" 