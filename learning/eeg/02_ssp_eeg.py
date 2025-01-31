""" Removes Signal Space Projection artifact removal from EEG data """

import numpy as np
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import os

# Get the sample data, if there isn't any, it fetches automatically
pwd = os.getcwd()
sample_data_folder = Path(pwd + "/sample_data")

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)

raw_non_ssp = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
raw_ssp = raw_non_ssp.copy()

# Remove SSP that are present in the 'raw' data
raw_non_ssp.del_proj()

# Prints the SSP that we just removed
ssp_proj = raw_non_ssp.info["projs"]
print("Projections:\n", ssp_proj)

# print(raw_non_ssp.ch_names)

# We only pick MAG channels to plot
eeg_channels = mne.pick_types(raw_non_ssp.info, eeg=True)
raw_non_ssp.plot(order=eeg_channels, n_channels=len(eeg_channels), remove_dc=False)

eeg_channels = mne.pick_types(raw_ssp.info, eeg=True)
raw_ssp.plot(order=eeg_channels, n_channels=len(eeg_channels), remove_dc=False)

# Plot raw data with SSP
plt.show()
