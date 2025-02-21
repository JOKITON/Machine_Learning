from sklearn.model_selection import GridSearchCV

def wide_svm_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_00"]
	grid["svm__C"] = json_grid["svm__C_00"]
	grid["svm__gamma"] = json_grid["svm__gamma_00"]

	return grid

def default_svm_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_01"]
	grid["svm__C"] = json_grid["svm__C_01"]
	grid["svm__gamma"] = json_grid["svm__gamma_01"]

	return grid

def narrow_svm_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_02"]
	grid["svm__C"] = json_grid["svm__C_02"]
	grid["svm__gamma"] = json_grid["svm__gamma_02"]

	return grid
 
def grid_finder(json_grid, ml_type, grid_type):
	if ml_type == 'svm':
		if grid_type == 'default':
			return default_svm_grid(json_grid)
		elif grid_type == 'wide':
			return wide_svm_grid(json_grid)
		elif grid_type == 'narrow':
			return narrow_svm_grid(json_grid)
		else:
			print("Invalid grid type")
	else:
		print("Invalid ML type")

def grid_search(X, y, pipeline, param_grid):
	grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=0)
	grid_search.fit(X, y)

	print("Best parameters for SVM:", grid_search.best_params_)
	print("Best SVM accuracy:", grid_search.best_score_)