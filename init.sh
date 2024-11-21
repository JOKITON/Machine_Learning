#!/bin/bash

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Select the interpreter in VSCode
code --install-extension ms-python.python
code -r . --command "python.setInterpreter" --args "$(pwd)/.venv/bin/python"