from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, Permute
from keras.optimizers import Adam
import keras.backend as K
from util import *

class AtariModel:
	def __init__(self, input_dim, output_dim, lr):
		self.input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
		self.output_dim = output_dim
		self.model = None
		self.lr = lr

	def create_model(self, weights_path=None):

		model = Sequential()
		if K.image_dim_ordering() == 'tf':
			# (width, height, channels)
			model.add(Permute((2, 3, 1), input_shape=self.input_shape))
		elif K.image_dim_ordering() == 'th':
			# (channels, width, height)
			model.add(Permute((1, 2, 3), input_shape=self.input_shape))
		else:
			raise RuntimeError('Unknown image_dim_ordering.')

		model.add(Conv2D(32, (8, 8), strides=(4, 4)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (4, 4), strides=(2, 2)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1)))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(self.output_dim))
		model.add(Activation('linear'))
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


