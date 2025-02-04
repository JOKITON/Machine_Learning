""" Plots the alpha band (8-12 Hz) power spectral density (PSD) of EEG/MEG data and empty room noise. """

import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from eeg_model import EEGData
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

def	plot_clean_eeg(l_freq, h_freq):

	# Get the sample data,
	script_path = Path(__file__).resolve().parent
	sample_data_folder = (script_path / "../sample_data").resolve()

	sample_data_raw_file = (
		sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
	)

	sample_data_raw_noise = (
		sample_data_folder / "MEG" / "sample" / "ernoise_raw.fif"
	)

	eeg_data = EEGData(sample_data_raw_file, sample_data_raw_noise, l_freq=l_freq, h_freq=h_freq)	
	raw_data, raw_noise_data = eeg_data.get_data()

	eeg_data.filter_data(tmax=60.0)

	# Plot difference between EEG & Room Noise
	""" eeg_data.compute_psd()
	eeg_data.plot_psd() """

	# Compute ICA
	clean_data = eeg_data.compute_ica()
	# Plot cleaned EEG after ICA
	eeg_data.plot_clean_eeg()

plot_clean_eeg(0, 75)
