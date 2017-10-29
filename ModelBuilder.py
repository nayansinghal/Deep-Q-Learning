from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class ModelBuilder:

	def __init__(self, input_dim, output_dim, lr):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.model = None
		self.lr = lr

	def create_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.input_dim, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.output_dim, activation='linear'))
		self.model = model

	def compile_model(self):
		optimizer = Adam(lr=self.lr, decay=1e-6)
		self.model.compile(loss='mse', optimizer=optimizer)