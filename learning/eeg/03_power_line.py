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

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

print(raw.info['sfreq'])  # Check the actual sampling frequency

fig = raw.compute_psd(tmax=np.inf, fmax=75).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

print(raw.info)

# add some arrows at 60 Hz and its harmonics:
for ax in fig.axes[1:]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (50, 60, 70):  # Focus on power line noise
        idx = np.searchsorted(freqs, freq)
        ax.arrow(
            x=freqs[idx],
            y=psds[idx] + 18,
            dx=0,
            dy=-12,
            color="red",
            width=0.1,
            head_width=3,
            length_includes_head=True,
        )
        
plt.show()
