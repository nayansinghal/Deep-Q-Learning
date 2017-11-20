import numpy as np
from util import *
from keras.initializations import normal, identity
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
from keras.layers import LSTM
import tensorflow as tf
import keras.backend as K

class CriticNet(object):

	def __init__(self, sess, lr, state_size, action_dim, target_mode, isRecurrent=True):
		self.sess = sess
		self.lr = lr
		self.state_size = state_size
		self.action_dim = action_dim
		self.isRecurrent = isRecurrent
		self.h1units = CRITIC_NET_HIDDEN1_UNITS
		self.h2units = CRITIC_NET_HIDDEN2_UNITS
		K.set_session(sess)
		self.model, self.action, self.state = self.create_model()
		self.compile_model()
		if target_mode is False:
			self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
			self.sess.run(tf.initialize_all_variables())
	
	def gradients(self, states, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.state: states,
			self.action: actions
		})[0]

	def create_model(self):

		if self.isRecurrent:
			state_input = Input(shape=(WINDOW_LENGTH, self.state_size))  
			w1 = LSTM(self.h1units)(state_input)
		else:
			state_input = Input(shape=[self.state_size])
			w1 = Dense(self.h1units, activation='relu')(state_input)

		action_input = Input(shape=[self.action_dim],name='action2')
		a1 = Dense(self.h2units, activation='linear')(action_input)
		h1 = Dense(self.h2units, activation='linear')(w1) 
		h2 = merge([h1,a1],mode='sum')    
		h3 = Dense(self.h2units, activation='relu')(h2)
		q_val = Dense(self.action_dim,activation='linear')(h3)   
		model = Model(input=[state_input,action_input],output=q_val)
		# print(model.summary())
		return model, action_input, state_input

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