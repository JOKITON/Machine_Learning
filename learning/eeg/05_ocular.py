""" Analyzes the ocular artifacts in the EEG data """

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

eog_epochs = mne.preprocessing.create_eog_epochs(raw)

eog_epochs.plot_image(combine="mean")

eog_epochs.average().plot_joint()
     
plt.show()
