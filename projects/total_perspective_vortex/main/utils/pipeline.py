from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def crt_pipeline(clf=False, voting='soft', clf_type='svm'):
	""" Creates a pipeline for classification tasks """

	svm_clf = SVC(kernel='rbf', C=100, gamma=2, probability=True)
	rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
	if clf is True:
		ensemble = VotingClassifier(estimators=[('svm', svm_clf), ('rf', rf_clf)], voting=voting)
		pipeline = Pipeline([
			('voting_cs', ensemble)
		])
	else:
		if clf_type is 'svm':
			pipeline = Pipeline([
				('svm', svm_clf)
			])
		elif clf_type is 'rf':
			pipeline = Pipeline([
				('rf', rf_clf)
			])
		elif clf_type is 'lda':
			pipeline = Pipeline([
				('lda', LDA())
			])
		else:
			raise(ValueError('Invalid classifier type'))

	return pipeline


