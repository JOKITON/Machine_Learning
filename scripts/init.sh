#!/bin/bash

# Create a virtual environment
if [ -d ".venv" ]; then
	echo "Virtual environment already exists..."
else
	python3 -m venv .venv
fi

if [ -f ".venv/activate.lock" ]; then
	echo "Virtual enviroment already running..."
else
	# Create a lock file
	touch .venv/activate.lock
	# Activate the virtual environment
	source .venv/bin/activate
	# Upgrade pip
	pip install --upgrade pip

	export os_choice=1

	# Set the appropriate environment variables based on the user's choice
	if [ $os_choice -eq 1 ]; then
		echo "You chose Mac."
		pip install -r requirements-mac.txt
	elif [ $os_choice -eq 2 ]; then
		echo "You chose Windows/Linux."
		pip install -r requirements-nvidia.txt
	else
		echo "Invalid choice. Exiting."
		exit(1)
	fi
fi

