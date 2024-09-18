# Initialize the project for Windows

# Activate venv to prevent accidentally installing into global space
./venv/Scripts/Activate.ps1

if($?) {
    # If successfully activated venv
    "Installing project dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-pytorch.txt
    
    ""
    "Installing submodules and their dependencies..."
    git submodule update --init --remote --recursive
    git submodule foreach --recursive "pip install -r requirements.txt"
    
    deactivate
    ""
    "Seutp complete!"
} else {
    "Please install a virtual environment in the directory 'venv', at the project root directory"
}
