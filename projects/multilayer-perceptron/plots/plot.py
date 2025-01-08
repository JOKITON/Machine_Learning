import config
from config import N_LAYERS
from loss import f_loss, f_mse, f_mae, f_r2score, f_cross_entropy

class Plot:
	acc_train = None
	acc_test = None

	mse_train = None
	mse_test = None

	mae_train = None
	mae_test = None

	def __init__(self, X_train, X_test, y_train, y_test, epochs):
		self.acc_train = [epochs]
		self.acc_test = [epochs]

		self.mse_train = [epochs]
		self.mse_test = [epochs]

		self.mae_train = [epochs]
		self.mae_test = [epochs]

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
  
	def append_preds(self, layers):
		# Append the data
		self.append_f_mae(layers)
		self.append_f_mse(layers)
		self.append_f_r2score(layers)
  
		return self.acc_train[-1], self.mse_train[-1], self.mae_train[-1]

	def append_f_mse(self, layers):
		mse_train_data = self.get_plot_data("train", layers, "mse")
		mse_test_data = self.get_plot_data("test", layers, "mse")
		self.mse_train.append(mse_train_data)
		self.mse_test.append(mse_test_data)

	def append_f_mae(self, layers):
		mae_train_data = self.get_plot_data("train", layers, "mae")
		mae_test_data = self.get_plot_data("test", layers, "mae")
		self.mae_train.append(mae_train_data)
		self.mae_test.append(mae_test_data)

	def append_f_r2score(self, layers):
		r2_train_data = self.get_plot_data("train", layers, "r2")
		r2_test_data = self.get_plot_data("test", layers, "r2")
		self.acc_train.append(r2_train_data)
		self.acc_test.append(r2_test_data)

	def get_plot_data(self, data_type, layers, loss_type):
		activations = [None] * N_LAYERS

		if (data_type == "train"):
			X = self.X_train
			y = self.y_train
		elif (data_type == "test"):
			X = self.X_test
			y = self.y_test
		train_input = X

		for i in range(N_LAYERS):
			activations[i], output = layers[i].forward(train_input)
			train_input = activations[i]
		if (loss_type == "r2"):
			loss = f_r2score(y, activations[-1])
		elif (loss_type == "mse"):
			loss = f_mse(y, activations[-1])
		elif(loss_type == "mae"):
			loss = f_mae(y, activations[-1])
		return loss
