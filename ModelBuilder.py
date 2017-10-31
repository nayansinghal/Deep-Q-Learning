from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

class ModelBuilder:

	def __init__(self, input_dim, output_dim, lr):
		self.input_dim = (1, input_dim)
		self.output_dim = output_dim
		self.model = None
		self.lr = lr

	def create_model(self, weights_path=None):

		model = Sequential()
		model.add(Dense(24, input_shape=self.input_dim, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.output_dim, activation='linear'))
		model.add(Flatten())
		print(model.summary())
		self.model = model

	def compile_model(self):
		optimizer = Adam(lr=self.lr, decay=1e-6)
		self.model.compile(loss='mse', optimizer=optimizer)

	def save(self, f_name):
		if f_name is not None:
			print('Model save: {}'.format(f_name))
			self.model.save_weights(f_name)

	def load(self, f_name):
		if f_name is not None:
			print('Model load from : {}'.format(f_name))
			self.model.load_weights(f_name)