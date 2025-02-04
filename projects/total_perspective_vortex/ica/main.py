""" Plots the alpha band (8-12 Hz) power spectral density (PSD) of EEG/MEG data and empty room noise. """

import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from eeg_model import EEGData
import json

script_path = Path(__file__).resolve().parent
folder = (script_path / "../").resolve()

JSON_PATH = script_path / "config.json"
EVENTS_PATH = script_path / "events.json"

with open(JSON_PATH, "r") as f:
    config = json.load(f)

with open(EVENTS_PATH, "r") as f:
    event_dict = json.load(f)

def	plot_clean_eeg(config, event_dict, l_freq, h_freq):

	# Get the sample data,
	sample_data_raw_file = folder / config["eeg_path"]
	sample_data_raw_noise = folder / config["noise_path"]
	# sample_data_raw_events = folder / config["event_path"]

	eeg_data = EEGData(sample_data_raw_file, sample_data_raw_noise, l_freq=l_freq, h_freq=h_freq)	
	eeg_data.load_event_dict(event_dict)

	raw_data = eeg_data.get_raw_data()
	events, event_dict = eeg_data.get_events()
	event_values = list(event_dict.values())


	eeg_data.filter_data(tmax=None)

	# Plot difference between EEG & Room Noise
	""" eeg_data.compute_psd()
	eeg_data.plot_psd() """

	# Compute ICA
	clean_data = eeg_data.compute_ica(plot_comp=True, plot_arts=False)

	# Plot cleaned EEG after ICA
	eeg_data.plot_clean_eeg()

	epochs = mne.Epochs(
	clean_data, events, event_id=event_values,
	tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True
	)
	epochs.plot()
	plt.show()

plot_clean_eeg(config, event_dict, l_freq=0, h_freq=75)
