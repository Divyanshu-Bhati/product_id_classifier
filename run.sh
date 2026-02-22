#!/bin/bash

# 1. Configuration
VENV_NAME=".cls_venv"
PYTHON_RW_VERSION="3.10"

echo "--- Starting Environment Setup ---"

wandb setup --relogin

# 2. Check if Python 3.10 is installed on the system
if ! command -v python$PYTHON_RW_VERSION &> /dev/null; then
    echo "Python Version Error: Python $PYTHON_RW_VERSION not found. Please install it first."
    exit 1
fi

# 3. Create Virtual Environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME using Python $PYTHON_RW_VERSION..."
    python$PYTHON_RW_VERSION -m venv $VENV_NAME
else
    echo "✅ Virtual environment '$VENV_NAME' already exists."
fi

# 4. Activate the environment
source $VENV_NAME/bin/activate

# 5. Upgrade pip to a specific version
echo "Upgrading pip..."
pip install --upgrade pip==26.0.1

# 6. Install Torch with CUDA 12.6
echo "Installing PyTorch for CUDA 12.6..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 7. Install remaining requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
fi

echo "--- Setup Complete. Environment '$VENV_NAME' is active. ---"