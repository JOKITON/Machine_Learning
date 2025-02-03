""" Repairs common artifacts like power-line noise, heartbeats or eye movements from EEG data. """

import os

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mne
from mne.preprocessing import (
    compute_proj_ecg,
    compute_proj_eog,
    create_ecg_epochs,
    create_eog_epochs,
)

pwd = os.getcwd()
sample_data_folder = Path(pwd + "/sample_data")

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)

# here we crop and resample just for speed
raw = mne.io.read_raw_fif(sample_data_raw_file).crop(0, 60)
print()
raw.load_data().resample(100)
print()

system_projs = raw.info["projs"]
raw.del_proj()
empty_room_file = os.path.join(sample_data_folder, "MEG", "sample", "ernoise_raw.fif")
# cropped to 60 s just for speed
empty_room_raw = mne.io.read_raw_fif(empty_room_file).crop(0, 30)

# Remove SSP projectors from empty room recordings
empty_room_raw.del_proj()

# Choose excluded MEG channels
raw.info["bads"] = ["MEG 2443"]

# Get the spectrum of the empty room data
spectrum = empty_room_raw.compute_psd()

# Plot the average and individual spectrum of the empty room data
for average in (False, True):
    spectrum.plot(
        average=average,
        dB=False,
        amplitude=True,
        xscale="log",
        picks="data",
        exclude="bads",
    )

plt.show()
