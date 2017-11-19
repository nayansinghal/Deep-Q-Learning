import numpy as np
from util import *
import math
from keras.initializations import normal, identity
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class ActorNet(object):

	def __init__(self, sess, lr, state_size, action_dim, target_mode):
		self.sess = sess
		self.lr = lr
		self.state_size = state_size
		self.action_dim = action_dim
		self.h1units = ACTOR_NET_HIDDEN1_UNITS
		self.h2units = ACTOR_NET_HIDDEN2_UNITS
		
		K.set_session(sess)
		self.model, self.weights, self.state = self.create_model()
		if target_mode is False:
			self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
			self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
			grads = zip(self.params_grad, self.weights)
			self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
			self.sess.run(tf.initialize_all_variables())
		else:
			self.compile_model()

	def train(self, states, action_grads):
		self.sess.run(self.optimize, feed_dict={
			self.state: states,
			self.action_gradient: action_grads
		})

	def create_model(self):
		state_in = Input(shape=[self.state_size])  
		h1 = Dense(self.h1units, activation='relu')(state_in)
		h2 = Dense(self.h2units, activation='relu')(h1)
		act_out = Dense(self.action_dim,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2)
		#act_steer = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2)
		#act_acclr = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2)
		#act_brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2) 
		#act_out = merge([act_steer, act_acclr, act_brake],mode='concat')
		model = Model(input=state_in,output=act_out)
		#print(model.summary())
		return model, model.trainable_weights, state_in

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