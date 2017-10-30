import numpy as np
from collections import deque
import random
from ModelBuilder import ModelBuilder

class DQNAgent:
	def __init__(self, state_size, action_size, epsilon=1.0, model_path=None):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = epsilon
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.995
		self.model = ModelBuilder(state_size, action_size, 0.001)
		self.model.create_model()
		self.model.compile_model()
		self.model.model.summary()
		self.model.load(model_path)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def get_Act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		state = np.expand_dims(state, axis=0)
		return np.argmax(self.model.model.predict(state)[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		state, action, reward, next_state, done = zip(*minibatch)
		state = np.reshape(state, (batch_size, 1, 4))
		next_state = np.reshape(next_state, (batch_size, 1, 4))

		target = reward + self.gamma * np.amax(self.model.model.predict(next_state), axis=1)
		target_f = self.model.model.predict(state)
		
		for idx, (target_, target, R, action, done) in enumerate(zip(target_f, target, reward, action, done)):
			if not done:
				target_[action] = target.astype('float32')
			else:
				target_[action] = R

		self.model.model.fit(state, target_f, verbose=0, epochs=1)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay