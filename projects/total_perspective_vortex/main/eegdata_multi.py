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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from scipy.signal import butter, filtfilt

from csp import butter_bandpass, apply_bandpass_filter, compute_csp
from data import read_data, save_data, fetch_data
from ica import remove_eog

class EEGData:

	def __init__(self, config, config_csp, folder, verbose=False):
		"""Initialize EEGData object with file paths and optional filter settings."""
		self.IS_FILTERED = False
		self.IS_NORMALIZED = False
		self.IS_ICA = False

		self.raw_h, self.raw_hf = None, None
		self.raw_filt_h, self.raw_filt_hf = None, None
		self.norm_raw_h, self.norm_raw_hf = None, None
		self.ica_h = None
		self.ica_hf = None

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
		is_raw_local = self.config['is_raw_local'].lower() == 'true'
		is_raw_filt_local = self.config['is_raw_filt_local'].lower() == 'true'
		is_ica_local = self.config['is_ica_local'].lower() == 'true'
		is_raw_norm_local = self.config['is_norm_local'].lower() == 'true'
		is_csp_local = self.config['is_csp_local'].lower() == 'true'
		is_event_local = self.config['is_event_local'].lower() == 'true'
		fast_start = self.config['fast_start'].lower() == 'true'

		if is_raw_local is False: # In case data is not stored locally
			data1, _ = fetch_data(self.subject, self.config["run_exec_h"],
				{1:'rest', 2: 'do/left_hand', 3: 'do/right_hand'}, self.montage, verbose=verbose)
			data2, _ = fetch_data(self.subject, self.config["run_img_h"],
				{1:'rest', 2: 'imagine/left_hand', 3: 'imagine/right_hand'}, self.montage, verbose=verbose)

			data1, _ = fetch_data(self.subject, self.config["run_exec_hf"],
				{1:'rest', 2: 'do/feet', 3: 'do/hands'}, self.montage, verbose=verbose)
			data2, _ = fetch_data(self.subject, self.config["run_img_hf"],
				{1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}, self.montage, verbose=verbose)

			self.raw_h = mne.concatenate_raws(raws=[data1, data2])
			self.raw_hf = mne.concatenate_raws(raws=[data1, data2])

		else: # In case data is stored locally
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

		if is_ica_local == True:
			self.clean_raw_h, self.clean_raw_hf = read_data(
				type="filtered", config=self.config, base_path=folder, verbose=verbose)
			print(self.clean_raw_h, self.clean_raw_hf)

			self.IS_ICA = True

		if is_raw_norm_local == True and fast_start is False: # In case normalized data is stored locally

			self.norm_raw_h, self.norm_raw_hf = read_data(
				type="norm", config=self.config, base_path=folder, verbose=verbose)
			print(self.norm_raw_h, self.norm_raw_hf)

			self.IS_NORMALIZED = True

	def get_raw(self):
		"""Returns EEG and noise data."""
		return self.raw_h, self.raw_hf

	def get_filt(self):
		"""Returns noise data."""
		return self.raw_filt_h, self.raw_filt_hf

	def get_events(self):
		"""Returns events data."""
		return self.events_h, self.events_hf

	def filter_data(self, tmax=None, verbose=False):
		"""Applies bandpass filter to EEG and noise data. Can also crop data."""
		if self.IS_FILTERED:
			raise(ValueError("Data has already been filtered..."))
		if self.raw_h or self.raw_hf is None:
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
		elif type == "epochs" and self.IS_FILTERED:
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
		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		if group_type == 'hands':
			groupeve_dict = self.csp_config["event_dict_h"]
			freq_bands = self.csp_config["freq_exec_hands"]
		else:
			groupeve_dict = self.csp_config["event_dict_hf"]
			freq_bands = self.csp_config["freq_exec_hf"]

		print("Before : ", event_dict)
		event_dict = {key: value for key, value in groupeve_dict.items() if value in event_dict[0]}
		print("After : ", event_dict)

		self.epochs = mne.Epochs(data, events, event_dict, tmin=tmin, tmax=tmax, baseline=None, verbose=verbose)
		# data = self.epochs.get_data()

		return self.epochs, freq_bands

	def simple_csp(self, epochs, freq_bands, verbose=False):
		tmin = self.csp_config["tmin"]
		tmax = self.csp_config["tmax"]

		n_components = self.csp_config["n_components"]
		fs = self.csp_config["frequency_sample"]

		epochs_data = epochs.get_data()
		labels = epochs.events[:, -1]

		features = compute_csp(epochs_data, labels, freq_bands, n_components, fs, verbose=verbose)

		return features, labels

	def two_step_csp(self, data, events, event_dict1, event_dict2, freq_bands, verbose=False):
		""" Performs a two-step binary classification using CSP """
		from csp import truncate_csp

		# **Step 1: Extract features for the first class**
		features_csp1, labels_csp1 = self.simple_csp(data, events, event_dict1, freq_bands, verbose=verbose)

		# **Step 2: Extract features for the second class**
		features_csp2, labels_csp2 = self.simple_csp(data, events, event_dict2, freq_bands, verbose=verbose)

		features_csp1, features_csp2, min_samples = truncate_csp(features_csp1, features_csp2)
		epochs = mne.Epochs(data, events, event_dict2, tmin=self.csp_config["tmin"], tmax=self.csp_config["tmax"], baseline=None, verbose=verbose)

		# **Final Step: Stack CSP1 and CSP2 features together**
		all_features = np.hstack([features_csp1, features_csp2])
		labels = epochs.events[:min_samples, -1]

		return all_features, labels

	def class_csp(self, data, events, groupeve_dict, freq_bands, ev_dict, verbose=False):
		""" Extract discriminative features for binary classification tasks """

		# Filter event_dict to only keep specified event indices
		event_dict1 = {key: value for key, value in groupeve_dict.items() if value in ev_dict[0]}
		#* Count the number of events of each type
		for val, ev_name in zip(event_dict1.values(), event_dict1.items()):
			event_count = 0
			for event in events:	
				if event[2] == val:
					# print(event[2])
					event_count += 1
			print(f"Number of events of type {ev_name}: {event_count}")
		print()

		if len(ev_dict) == 2: # In case of two-step binary classification
			event_dict2 = {key: value for key, value in groupeve_dict.items() if value in ev_dict[1]}
			all_features, labels = self.two_step_csp(data, events, event_dict1, event_dict2, freq_bands, verbose=verbose)
		elif (len(ev_dict) == 1):
			all_features, labels = self.simple_csp(data, events, event_dict1, freq_bands, verbose=verbose)
		else:
			raise ValueError("Invalid combination specified. Choose from 'two_step'.")

		return all_features, labels

	""" def normalize_data(self, data, verbose=False):
		if self.IS_NORMALIZED:
			raise(ValueError("Data has already been normalized..."))
		if self.IS_FILTERED is False:
			raise ValueError("Data has not been filtered. Call `filter_data()` before normalizing.")
		if self.IS_ICA is False:
			raise ValueError("Data hasn't been processed with ICA... Call `compute_ica()` before applying CSP.")
	
		normalized_h = StandardScaler().fit_transform(data1)
		normalized_hf = StandardScaler().fit_transform(data2)

		self.IS_NORMALIZED = True """

	def normalize_data(self, data, verbose=False):
		"""Normalizes the data."""
	
		normalized = StandardScaler().fit_transform(data)

		return normalized

	def train_model(self, features, labels):
		from sklearn.model_selection import train_test_split
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
		from sklearn.svm import SVC
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

		# Dividir datos en train y test
		X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

		# Probar LDA
		lda = LinearDiscriminantAnalysis()
		lda.fit(X_train, y_train)
		y_pred_lda = lda.predict(X_test)

		# Probar SVM
		svm = SVC(kernel='rbf', C=100, gamma=2, probability=True)
		svm.fit(X_train, y_train)
		y_pred_svm = svm.predict(X_test)

		# Probar Random Forest

		rf = RandomForestClassifier(n_estimators=100, random_state=42)
		rf.fit(X_train, y_train)
		y_pred_rf = rf.predict(X_test)

		print("Train Accuracy LDA:", accuracy_score(y_train, lda.predict(X_train)))
		print("Train Accuracy SVM:", accuracy_score(y_train, svm.predict(X_train)))
		print("Train Accuracy RF:", accuracy_score(y_train, rf.predict(X_train)))

		""" for pred, real in zip(y_pred_lda, y_test):
			print(f"Predicted: {pred} - Real: {real}") """

		# Evaluar rendimiento
		print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
		print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
		print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

		# Matriz de confusión
		print("Confusion Matrix LDA:\n", confusion_matrix(y_test, y_pred_lda))
		print("Confusion Matrix SVM:\n", confusion_matrix(y_test, y_pred_svm))
		print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred_rf))
