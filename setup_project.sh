#!/bin/bash

# Initialize and update submodules script for Linux

# Activate venv to prevent accidentally installing into global space
source ./venv/bin/activate

if [ $? -eq 0 ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-pytorch.txt

    echo ""
    echo "Installing submodules and their dependencies..."
    git submodule update --init --remote --recursive
    git submodule foreach --recursive "pip install -r requirements.txt"

    echo ""
    echo "Setup complete!"
else
    echo "Please install a virtual environment in the directory 'venv', at the project root directory"
fi
