""" Removes Signal Space Projection artifact removal from MEG data """

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

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
raw2 = raw.copy()

# Remove SSP that are present in the 'raw' data
raw.del_proj()

# Prints the SSP that we just removed
ssp_proj = raw.info["projs"]
print("Projections:\n", ssp_proj)

# We only pick MAG channels to plot
mag_channels = mne.pick_types(raw.info, meg="mag")
raw.plot(order=mag_channels, n_channels=len(mag_channels), remove_dc=False)

mag_channels = mne.pick_types(raw2.info, meg="mag")
raw2.plot(order=mag_channels, n_channels=len(mag_channels), remove_dc=False)

# Plot raw data with SSP
plt.show()
