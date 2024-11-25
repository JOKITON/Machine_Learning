#!/bin/bash

# Create a virtual environment
if [ -d ".venv" ]; then
	echo "Virtual environment already exists..."
	continue # Skip the virtual environment creation
else
	python3 -m venv .venv
fi

if [ -f ".venv/activate.lock" ]; then
	echo "Virtual enviroment already running..."
	continue # Skip the virtual environment activation
else
	# Create a lock file
	touch .venv/activate.lock
	# Activate the virtual environment
	source .venv/bin/activate
	# Upgrade pip
	pip install --upgrade pip

	# Install the required packages
	pip install -r requirements.txt
fi

