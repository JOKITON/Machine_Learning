import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from mne import create_info
from mne.io import RawArray
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne.decoding import CSP

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from scipy.signal import butter, filtfilt

from utils.csp import butter_bandpass, apply_bandpass_filter, compute_csp
from utils.data import read_data, save_data, fetch_data
from utils.ica import remove_eog

class EEGData:

	def __init__(self, config, config_csp, folder, verbose=False):
		"""Initialize EEGData object with file paths and optional filter settings."""
		self.IS_FILTERED = False
		self.IS_NORMALIZED = False
		self.IS_ICA = False

		self.raw_h, self.raw_hf = None, None
		self.raw_filt_h, self.raw_filt_hf = None, None
		self.norm_raw_h, self.norm_raw_hf = None, None
		self.clean_raw_h, self.clean_raw_hf = None, None
		self.ica_h = None
		self.ica_hf = None
		self.epochs = None
		self.lda, self.svm, self.rf = None, None, None

		# Initialize basic variables to fill later
		self.config = config
		self.csp_config = config_csp

		self.subject = np.arange(config["n_subjects"]) + 1

		# Create a montage object to store the channel positions
		self.montage = mne.channels.make_standard_montage(config["montage"])

		self.load_data(folder, verbose=verbose)

		# Ensure raw_h is initialized before accessing its attributes
		if self.raw_h is not None:
			# Get the picks for the EEG channels
			self.picks = mne.pick_types(self.raw_h.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
		else:
			raise ValueError("raw_h is not initialized. Please check the data loading process.")

		# Store the frequencies for later on filtering/plotting
		self.l_freq = config["l_freq"]
		self.h_freq = config["h_freq"]

	def load_data(self, folder, verbose=False):
		"""Loads EEG data from files."""
		fast_start = self.config['fast_start'].lower() == 'true'
		is_raw_local = self.config['is_raw_local'].lower() == 'true'
		is_raw_filt_local = self.config['is_raw_filt_local'].lower() == 'true'
		is_event_local = self.config['is_event_local'].lower() == 'true'
		is_ica_local = self.config['is_ica_local'].lower() == 'true'
		is_epoch_local = self.config['is_epoch_local'].lower() == 'true'

		if is_raw_local is False: # In case data is not stored locally
			data1, _ = fetch_data(self.subject, self.config["run_exec_h"],
				{1:'rest', 2: 'do/left_hand', 3: 'do/right_hand'}, self.montage, verbose=verbose)
			data2, _ = fetch_data(self.subject, self.config["run_img_h"],
				{1:'rest', 2: 'imagine/left_hand', 3: 'imagine/right_hand'}, self.montage, verbose=verbose)
			self.raw_h = mne.concatenate_raws(raws=[data1, data2])

			data1, _ = fetch_data(self.subject, self.config["run_exec_hf"],
				{1:'rest', 2: 'do/feet', 3: 'do/hands'}, self.montage, verbose=verbose)
			data2, _ = fetch_data(self.subject, self.config["run_img_hf"],
				{1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}, self.montage, verbose=verbose)
			self.raw_hf = mne.concatenate_raws(raws=[data1, data2])

		elif is_raw_local is True: # In case data is stored locally
			print("Loaded data:")
			self.raw_h, self.raw_hf = read_data(
        			type="raw", config=self.config, base_path=folder, verbose=verbose)
			print(self.raw_h, self.raw_hf)

		if is_event_local == True:
			self.events_h, self.events_hf = read_data(
				type="events", config=self.config, base_path=folder, verbose=verbose)
		else:
			self.events_h, _ = mne.events_from_annotations(self.raw_h, verbose=verbose)
			self.events_hf, _ = mne.events_from_annotations(self.raw_hf, verbose=verbose)

		if is_raw_filt_local == True and fast_start is False: # In case filtered data is stored locally
			self.raw_filt_h, self.raw_filt_hf = read_data(
				type="filtered", config=self.config, base_path=folder, verbose=verbose)
			print(self.raw_filt_h, self.raw_filt_hf)

			self.IS_FILTERED = True

		if is_ica_local == True: # In case filtered data is stored locally
			self.ica_h, self.ica_hf = read_data(
				type="ica", config=self.config, base_path=folder, verbose=verbose)
			print(self.ica_h, self.ica_hf)

			self.clean_raw_h, self.clean_raw_hf = read_data(
				type="clean", config=self.config, base_path=folder, verbose=verbose)
			print(self.ica_h, self.ica_hf)

			self.IS_ICA = True

		if is_epoch_local is True:
			self.epochs = read_data(
				type="epochs", config=self.config, base_path=folder, verbose=verbose)
			print(self.epochs)
			

	def get_raw(self):
		"""Returns EEG and noise data."""
		return self.raw_h, self.raw_hf

	def get_filt(self):
		"""Returns noise data."""
		return self.raw_filt_h, self.raw_filt_hf

	def get_ica(self):
		"""Returns noise data."""
		return self.ica_h, self.ica_hf

	def get_clean(self):
		"""Returns noise data."""
		return self.clean_raw_h, self.clean_raw_hf

	def get_events(self):
		"""Returns events data."""
		return self.events_h, self.events_hf

	def filter_data(self, tmax=None, verbose=False):
		"""Applies bandpass filter to EEG and noise data. Can also crop data."""
		if self.IS_FILTERED:
			raise(ValueError("Data has already been filtered..."))
		if self.raw_h is None or self.raw_hf is None:
			raise(ValueError("Data has not been loaded. Call `load_data()` before filtering."))
		if tmax:
			print(f"Cropping data to {tmax:.2f} seconds.")
			if isinstance(self.raw_h, mne.io.Raw):
				self.raw_h.crop(tmax=tmax)
			if isinstance(self.raw_hf, mne.io.Raw):
				self.raw_hf.crop(tmax=tmax)

		self.raw_filt_h = self.raw_h.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)
		self.raw_filt_hf = self.raw_hf.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin", verbose=verbose)

		self.IS_FILTERED = True

	def plot_psd_ba_filt(self, verbose=False):
		"""Computes Power Spectral Density (PSD) for EEG and noise data
		before and after filtering."""

		# Plot the power spectral density (PSD) of the raw data
		fig = self.raw_h.compute_psd(picks=None).plot()

		if self.IS_FILTERED:
			title = 'PSD of concatenated Raw data after filtering'
		else:
			title = 'PSD of concatenated Raw data before filtering'
		fig.axes[0].set_title(title)
		plt.show()

		if self.IS_FILTERED:
			print("The data is already filtered, skipping filtering step.")
		else:
			self.filter_data(verbose=verbose)

		fig = self.raw_filt_h.compute_psd(picks=None).plot()
		fig.axes[0].set_title('PSD of concatenated Raw data after filtering')
		plt.show()

	def save_type_data(self, type, folder_path, verbose=False):
		"""Saves cleaned data to a given filepath."""
		if type == "raw":
			save_data(self.raw_h, type, 1, self.config, folder_path, verbose=verbose)
			save_data(self.raw_hf, type, 2, self.config, folder_path, verbose=verbose)
		elif type == "events":
			save_data(self.events_h, type, 1, self.config, folder_path, verbose=verbose)
			save_data(self.events_hf, type, 2, self.config, folder_path, verbose=verbose)
		elif type == "filtered" and self.IS_FILTERED:
			save_data(self.raw_filt_h, type, 1, self.config, folder_path, verbose=verbose)
			save_data(self.raw_filt_hf, type, 2, self.config, folder_path, verbose=verbose)
		elif type == "ica" and self.IS_ICA:
			save_data(self.ica_h, type, 1, self.config, folder_path, verbose=verbose)
			save_data(self.ica_hf, type, 2, self.config, folder_path, verbose=verbose)
		elif type == "clean" and self.IS_ICA:
			save_data(self.clean_raw_h, type, 1, self.config, folder_path, verbose=verbose)
			save_data(self.clean_raw_hf, type, 2, self.config, folder_path, verbose=verbose)
		elif type == "epochs" and self.IS_ICA:
			save_data(self.epochs, type, 1, self.config, folder_path, verbose=verbose)
		else:
			raise ValueError("Data has not been proccessed correctly. Check the type and the data.")

	def decomp_ica(self, plt_show=False, n_components=None, verbose=False):
		if self.IS_FILTERED is False:
			raise ValueError("Data has not been filtered. Call `filter_data()` before applying ICA.")
		if self.IS_ICA is True:
			raise ValueError("Data has already been processed with ICA...")

		# Create and fit ICA for the first dataset
		self.ica_h = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica')
		self.ica_h.fit(self.raw_filt_h)

		# Create and fit ICA for the second dataset
		self.ica_hf = mne.preprocessing.ICA(n_components=n_components, random_state=42, method='fastica')
		self.ica_hf.fit(self.raw_filt_hf)

		self.ica_h = remove_eog(self.ica_h, self.raw_filt_h, plt=plt_show, title="EOG artifacts in individual Hands", verbose=False)
		self.ica_hf = remove_eog(self.ica_hf, self.raw_filt_hf, plt=plt_show, title="EOG artifacts in both Hands & Feet", verbose=False)

		self.clean_raw_h = self.ica_h.apply(self.raw_filt_h.copy())
		self.clean_raw_hf =self.ica_hf.apply(self.raw_filt_hf.copy())

		self.IS_ICA = True

	def crt_epochs(self, data, events, event_dict, group_type, verbose=False):
		""" Cretates epochs to later apply CSP, ICA and feed the ML algorimth selected. """
		if self.IS_ICA is False:
			raise(ValueError("ICA has not been applied to the data. Please check the data."))
		if self.epochs is not None:
			print("Epochs have already been created... Re-creating epochs.")

		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		if group_type == 'hands':
			groupeve_dict = self.csp_config["event_dict_h"]
			freq_bands = self.csp_config["freq_exec_hands"]
		else:
			groupeve_dict = self.csp_config["event_dict_hf"]
			freq_bands = self.csp_config["freq_exec_hf"]

		event_dict = {key: value for key, value in groupeve_dict.items() if value in event_dict[0]}
		print("Event dict. : ", event_dict)

		self.epochs = mne.Epochs(data, events, event_dict, tmin=tmin, tmax=tmax, baseline=None, verbose=verbose)

		return self.epochs, freq_bands

	def csp(self, data, labels, freq_bands, epochs_info, verbose=False):
		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		n_components = self.csp_config["n_components"]
		fs = self.csp_config["frequency_sample"]

		features, csp = compute_csp(data, labels, freq_bands, n_components, fs, epochs_info, verbose=verbose)

		return features, csp

	def two_step_csp(self, epochs1, epochs2, freq_bands, verbose=False):
		""" Performs a two-step binary classification using CSP """
		from utils.csp import truncate_csp
		labels1 = epochs1.events[:, -1]
		labels2 = epochs2.events[:, -1]

		# **Step 1: Extract features for the first class**
		features_csp1, _ = self.csp(epochs1.get_data(), labels1, freq_bands, verbose=verbose)

		# **Step 2: Extract features for the second class**
		features_csp2 = self.csp(epochs2.get_data(), labels2, freq_bands, verbose=verbose)

		features_csp1, features_csp2, min_samples = truncate_csp(features_csp1, features_csp2)

		# **Final Step: Stack CSP1 and CSP2 features together**
		all_features = np.hstack([features_csp1, features_csp2])

		return all_features, labels2

	def csp_performance(self, epochs, labels, clf_type='lda', verbose=False):
		sfreq = self.csp_config["frequency_sample"]
		w_length = int(sfreq * 0.5)  # Running classifier: window length
		w_step = int(sfreq * 0.1)    # Running classifier: window step size
		w_start = np.arange(0, epochs.get_data().shape[2] - w_length, w_step)

		data = epochs.get_data()
		cv = ShuffleSplit(10, test_size=0.2, random_state=42)

		if clf_type == 'lda':
			clf = LDA()
		elif clf_type == 'svm':
			clf = SVC(kernel='rbf', C=100, gamma=2, probability=False)
		elif clf_type == 'rf':
			clf = RandomForestClassifier(n_estimators=200, random_state=42)
		else:
			raise(ValueError("Classifier type not recognized. Please use 'lda', 'svm' or 'rf'."))

		csp = CSP(n_components=self.csp_config["n_components"], reg='ledoit_wolf', log=True, norm_trace=False)

		scores_windows = []

		for train_idx, test_idx in cv.split(data):
			X_train, X_test = data[train_idx], data[test_idx]
			y_train, y_test = labels[train_idx], labels[test_idx]

			score_this_window = []

			for n in w_start:
				# Extract sliding window segment
				X_train_win = X_train[:, :, n : n + w_length]
				X_test_win = X_test[:, :, n : n + w_length]

				# Apply CSP and reshape features
				csp.fit(X_train_win, y_train)
				X_train_csp = csp.transform(X_train_win)
				X_test_csp = csp.transform(X_test_win)

				# Fit LDA and compute score
				clf.fit(X_train_csp, y_train)
				score_this_window.append(clf.score(X_test_csp, y_test))

			scores_windows.append(score_this_window)

		# Plot scores over time
		w_times = (w_start + w_length / 2.0) / sfreq + 0.3  # Adjusted time axis

		plt.figure()
		plt.plot(w_times, np.mean(scores_windows, axis=0), label="Score")
		plt.axvline(0, linestyle="--", color="k", label="Onset")
		plt.axhline(0.5, linestyle="-", color="k", label="Chance Level")
		plt.xlabel("Time (s)")
		plt.ylabel("Classification Accuracy")
		plt.title("Classification Score Over Time")
		plt.legend(loc="lower right")
		plt.show()

	def count_events(self, events, groupeve_dict, ev_dict):
		""" Extract discriminative features for binary classification tasks """

		# Filter event_dict to only keep specified event indices
		event_dict = {key: value for key, value in groupeve_dict.items() if value in ev_dict[0]}

		#* Count the number of events of each type
		for val, ev_name in zip(event_dict.values(), event_dict.items()):
			event_count = 0
			for event in events:	
				if event[2] == val:
					# print(event[2])
					event_count += 1
			print(f"Number of events of type {ev_name}: {event_count}")
		print()

	def normalize(self, epochs, verbose=False):
		# Initialize StandardScaler and PCA
		scaler = StandardScaler()

		# Standardize and apply PCA to each epoch individually
		std_epochs = np.empty((epochs.shape[0], epochs.shape[1], epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_std = scaler.fit_transform(epoch)  # Standardize
			std_epochs[i, :, :] = epoch_std

		# Verify the shape
		print("Shape after standardization:", std_epochs.shape)

		return std_epochs

	def pca(self, epochs, verbose=False):
		# Initialize StandardScaler and PCA
		pca = PCA(n_components=32)

		# Standardize and apply PCA to each epoch individually
		pca_epochs = np.empty((epochs.shape[0], 32, epochs.shape[2]))

		for i in range(epochs.shape[0]):
			epoch = epochs[i, :, :]  # Shape: (n_channels, n_times)
			epoch_pca = pca.fit_transform(epoch.T).T  # Apply PCA
			pca_epochs[i, :, :] = epoch_pca

		# Verify the shape
		print("Shape after PCA:", pca_epochs.shape)

		return pca_epochs

	def cross_val(self, X, y, pipeline, n_splits=5):
		# Evaluate the pipeline using cross-validation
		cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		scores = cross_val_score(pipeline, X, y, cv=cv)

		print(scores)

		return scores

	def cross_validate(self, X, y, pipeline, n_splits=5):
		# Evaluate the pipeline using cross-validation
		cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
		scores = cross_validate(pipeline, X, y, cv=cv, return_train_score=True)
  
		print(scores)

	def train_model(self, X, y, pipeline=None):

		self.lda = LDA(solver="eigen")
		self.lda.fit(X, y)

		self.svm = SVC(kernel='rbf', C=100, gamma=2, probability=False)
		self.svm.fit(X, y)

		self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
		self.rf.fit(X, y)


		print("Train Accuracy LDA:", accuracy_score(y, self.lda.predict(X)))
		print("Train Accuracy SVM:", accuracy_score(y, self.svm.predict(X)))
		print("Train Accuracy RF:", accuracy_score(y, self.rf.predict(X)))
		print()


	def pred(self, X, y, pipeline, n_preds=20, prt_matrix=False):
		if self.lda and self.svm and self.rf is None:
			raise ValueError("Model has not been trained. Call `train_model()` before predicting.")

		y_pred_lda = self.lda.predict(X)
		y_pred_svm = self.svm.predict(X)
		y_pred_rf = self.rf.predict(X)

		print("epoch nb: [prediction] [truth] equal?")
		for i, (pred, real) in enumerate(zip(y_pred_svm, y)):
			is_correct = "True" if pred == real else "False"
			if i > n_preds:
				break
			print(f"epoch {i:03}:\t[{pred}]\t\t[{real}]  {is_correct}")
		print()

		print("LDA Accuracy:", accuracy_score(y, y_pred_lda))
		print("SVM Accuracy:", accuracy_score(y, y_pred_svm))
		print("Random Forest Accuracy:", accuracy_score(y, y_pred_rf))

		if prt_matrix:
			print("Confusion Matrix LDA:\n", confusion_matrix(y, y_pred_lda))
			print("Confusion Matrix SVM:\n", confusion_matrix(y, y_pred_svm))
			print("Confusion Matrix RF:\n", confusion_matrix(y, y_pred_rf))