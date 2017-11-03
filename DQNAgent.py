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
		self.target_model = ModelBuilder(state_size, action_size, 0.001)
		self.model.create_model()
		self.model.compile_model()
		self.target_model.create_model()
		self.target_model.compile_model()
		self.model.load(model_path)
		self.target_model.load(model_path)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def get_Act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		return np.argmax(self.model.model.predict(state)[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				temp_q = self.model.model.predict(next_state)[0]
				temp_action = np.argmax(temp_q)
				target_q = self.target_model.model.predict(next_state)[0]
				_q = target_q[temp_action]

				target = reward + self.gamma * _q #temp_q[temp_action]
			target_f = self.model.model.predict(state)
			target_f[0][action] = target
			self.model.model.fit(state, target_f, verbose=0, epochs=1)
			self.target_model.model.fit(state, target_f, verbose=0, epochs=1)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay