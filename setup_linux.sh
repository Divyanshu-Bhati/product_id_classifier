#!/bin/bash

# Create environment
VENV_NAME=".cls_venv"
PYTHON_RW_VERSION="3.10"
echo "Starting Linux setup..."

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
echo "Upgrading pip and installing requrements (CUDA support)..."
pip install --upgrade pip==26.0.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if [ -f "requirements_linux.txt" ]; then
    pip install -r requirements_linux.txt
fi

# WandB config
echo "Setting wandb to offline mode (default)..."
export WANDB_MODE=offline
wandb offline

source $VENV_NAME/bin/activate

echo "Setup complete! Activate your environment, if not already activated, with: source $VENV_NAME/bin/activate"