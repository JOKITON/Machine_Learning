import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

class EEGData:
    def __init__(self, eeg_path, noise_path, l_freq=None, h_freq=None):
        """Initialize EEGData object with file paths and optional filter settings."""
        self.eeg_path = eeg_path
        self.noise_path = noise_path
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.load_data()

    def load_data(self):
        """Loads EEG and noise data."""
        self.raw_eeg = mne.io.read_raw_fif(self.eeg_path, preload=True, verbose=False)
        # Cut down duration to 60s
        # self.raw_eeg.crop(tmax=60.0).pick(picks=["mag", "eeg", "stim", "eog"])

        self.raw_noise = mne.io.read_raw_fif(self.noise_path, preload=True, verbose=False)
    
    def get_data(self):
        """Returns EEG and noise data."""
        return self.raw_eeg, self.raw_noise

    def filter_data(self, tmax=None):
        """Applies bandpass filter to EEG and noise data. Can also crop data."""
        if tmax:
            print(f"Cropping data to {tmax:.2f} seconds.")
            self.raw_eeg.crop(tmax=tmax)
            self.raw_noise.crop(tmax=tmax)
        self.raw_eeg_filtered = self.raw_eeg.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin")
        self.raw_noise_filtered = self.raw_noise.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin")

    def compute_ica(self):
        ica = ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(self.raw_eeg_filtered)  # Fit ICA on raw EEG data
        
        # Plot ICA components
        ica.plot_components()

        # Detect ECG artifacts
        ecg_inds, ecg_scores = ica.find_bads_ecg(inst=self.raw_eeg_filtered, method='correlation')
        print("\nPlotting scores of ECG components...")
        ica.plot_scores(ecg_scores)

        # Detect EOG artifacts
        eog_inds, eog_scores = ica.find_bads_eog(inst=self.raw_eeg_filtered, ch_name='EOG 061')
        print("\nPlotting scores of EOG components...")
        ica.plot_scores(eog_scores)

        # Flatten lists in case they are nested
        ecg_inds = [comp for sublist in ecg_inds for comp in sublist] if ecg_inds and isinstance(ecg_inds[0], list) else ecg_inds
        eog_inds = [comp for sublist in eog_inds for comp in sublist] if eog_inds and isinstance(eog_inds[0], list) else eog_inds

        # Visualize the identified components
        if ecg_inds:
            ica.plot_sources(self.raw_eeg_filtered, picks=ecg_inds)
        if eog_inds:
            ica.plot_sources(self.raw_eeg_filtered, picks=eog_inds)

        plt.show()
        # Exclude the identified components
        ica.exclude = ecg_inds + eog_inds
        print(f"Excluding ICA components: {ica.exclude}")

        # Apply ICA only if there are components to exclude
        if ica.exclude:
            self.raw_clean = ica.apply(self.raw_eeg_filtered.copy())  # Apply ICA on a copy
        else:
            print("No components to exclude, skipping ICA application.")
            self.raw_clean = self.raw_eeg_filtered.copy()

        return self.raw_clean  # Return cleaned data

    def compute_psd(self):
        """Computes Power Spectral Density (PSD) for EEG and noise data."""
        data_eeg = self.raw_eeg_filtered.get_data()
        data_noise = self.raw_noise_filtered.get_data()
        sfreq = self.raw_eeg.info["sfreq"]

        self.psd_eeg, self.freqs = mne.time_frequency.psd_array_welch(data_eeg, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq)
        self.psd_noise, _ = mne.time_frequency.psd_array_welch(data_noise, fmin=self.l_freq, fmax=self.h_freq, sfreq=sfreq)

        self.psd_eeg_mean = np.mean(self.psd_eeg, axis=0)
        self.psd_noise_mean = np.mean(self.psd_noise, axis=0)

        # Convert to dB
        #? Add a small value to avoid log(0)
        self.psd_eeg_db = 10 * np.log10(self.psd_eeg_mean + 1e-10)
        self.psd_noise_db = 10 * np.log10(self.psd_noise_mean + 1e-10)

    def plot_psd(self):
        """Plots EEG and noise PSD."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.freqs, self.psd_eeg_db, label="EEG/MEG Data", color="blue")
        plt.plot(self.freqs, self.psd_noise_db, label="Empty Room Noise", color="red", linestyle="dashed")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.title("PSD Comparison: EEG vs. Noise")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_clean_eeg(self, title="Cleaned EEG Data"):
        """Plots cleaned EEG data."""
        self.raw_clean.plot(title=title)
        plt.show()
