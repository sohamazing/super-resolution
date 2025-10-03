#!/bin/bash

# Automates Conda environment setup for the super-resolution project

ENV_NAME="super-res"
PYTHON_VERSION="3.10"

echo "--- Super-Resolution Project Setup ---"

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in your PATH."
    echo "Please install Miniconda or Anaconda and try again."
    exit 1
fi

# 2. Create the Conda environment if it doesn't exist
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Conda environment."
        exit 1
    fi
fi

# 3. Activate the environment and install packages
echo "Activating environment and installing packages from requirements.txt..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

pip install --upgrade -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install packages from requirements.txt."
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo "To activate this environment in the future, run:"
echo "conda activate $ENV_NAME"

