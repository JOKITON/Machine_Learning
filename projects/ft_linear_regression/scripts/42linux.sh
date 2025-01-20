#!/bin/bash

# Create a virtual environment
if [ -d ".venv" ]; then
	echo "Virtual environment already exists..."
else
	virtualenv -p /sgoinfre/students/jaizpuru/homebrew/bin/python3.9 .venv
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

	export os_choice=3

	# Set the appropriate environment variables based on the user's choice
	if [ $os_choice -eq 1 ]; then
		echo "You chose Mac."
		pip install -r requirements-mac.txt
	elif [ $os_choice -eq 2 ]; then
		echo "You chose Windows/Linux."
		pip install -r requirements-nvidia.txt
    elif [ $os_choice -eq 3 ]; then
		echo "You chose 42 Linux."
		.venv/bin/pip install -r requirements.txt
	else
		echo "Invalid choice. Exiting."
	fi
fi
