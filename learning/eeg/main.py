import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import os

# Get the sample data, if there isn't any, it fetches automatically
pwd = os.getcwd()
sample_data_folder = Path(pwd + "/sample_data")

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

# Apply band-pass filtering (1–40 Hz)
raw.filter(1, 40, fir_design="firwin", verbose=False)

# Get the info out of the Raw object
# The high-range is high-filtered to 150.1 Hz
info = mne.io.read_info(sample_data_raw_file, verbose=False)

ssp_proj = raw.info["projs"]
print("Projections:\n", ssp_proj)
raw.del_proj()

# Number of channels of each type
meg_channels = mne.pick_types(raw.info, meg=True)
eeg_channels = mne.pick_types(raw.info, eeg=True)
eog_channels = mne.pick_types(raw.info, eog=True)
ecg_channels = mne.pick_types(raw.info, ecg=True)

print()
print(f"EMG Channels: {len(meg_channels)}")
print(f"EMG Channels: {len(eeg_channels)}")
print(f"EOG Channels: {len(eog_channels)}")
print(f"ECG Channels: {len(ecg_channels)}")
print()

# Perform spectral analysis on the data
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)

import matplotlib.pyplot as plt

# Some configuration for plotting the raw data
raw.plot(duration=5, n_channels=20)

# Set up and fit the Independent Component Analysis
ica = ICA(n_components=20, random_state=97, max_iter=800, verbose=False)
ica.fit(raw)

# Exclude the EOG and ECG channels
ica.exclude = [1, 2]
ica.plot_properties(raw, picks=ica.exclude, verbose=False)

# Get the ICA components
data_clean = ica.apply(raw.copy())
# print(data_clean)

plt.show()
