
import mne

def remove_eog(ica, data, plt=False, title="", verbose=False):
	eog_arts, scores = ica.find_bads_eog(data, ch_name='Fpz', verbose=verbose)
	if plt is True:
		#* Plot the scores and mark the artifacts to be excluded
		ica.plot_scores(scores, exclude=eog_arts, title=title)
		# ica.plot_scores(scores, exclude=ecg_arts)

	#* Finally, exclude the artifacts from the ICA Object
	ica.exclude = eog_arts

	return ica

def plot_ica_comp(self, path, verbose=False):
	print(path)
	ica = mne.preprocessing.read_ica(path)
	ica.plot_components()
