#!/bin/bash

# Create environment
VENV_NAME=".cls_venv"
PYTHON_RW_VERSION="3.10"
echo "Starting MacOS setup..."

if ! command -v python$PYTHON_RW_VERSION &> /dev/null; then
    echo "Error: Python $PYTHON_RW_VERSION not found."
    exit 1
fi
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating venv: $VENV_NAME..."
    python$PYTHON_RW_VERSION -m venv $VENV_NAME
fi
source $VENV_NAME/bin/activate

# Installing Dependencies
echo "Upgrading pip and installing requirements (MPS/Metal support)..."
pip install --upgrade pip==26.0.1
if [ -f "requirements_mac.txt" ]; then
    pip install -r requirements_mac.txt
fi

# WandB config
echo "Setting wandb to offline mode (default)..."
export WANDB_MODE=offline
wandb offline

source $VENV_NAME/bin/activate

echo "Setup complete! Activate your environment, if not already activated, with: source $VENV_NAME/bin/activate"