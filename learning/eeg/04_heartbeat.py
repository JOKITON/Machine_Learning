""" Removes Power Line artifact removal from MEG data """

import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import os

# Get the sample data, if there isn't any, it fetches automatically
pwd = os.getcwd()
sample_data_folder = Path(pwd + "/sample_data")

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=True)

print(raw.info)

# Fetch heartbeat artifacts from raw data
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)

# Apply baseline to see heartbeat artifacts easier
avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))

# Plot the heartbeat artifacts
ecg_epochs.plot_image(combine="mean")

# See the spatial pattern relative to the peak of the peak
avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))

# an ERP/F plot to see the average ECG response
avg_ecg_epochs.plot_joint()
     
plt.show()
