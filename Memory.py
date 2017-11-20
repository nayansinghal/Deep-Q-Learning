from collections import deque
import numpy as np

class Memory:
	def __init__(self, limit=100000, window_length=4):
		self.memory = deque(maxlen=limit)
		self.window_length = window_length
		self.current_state = deque(maxlen=window_length)

	def append(self, obs):
		obs = obs.reshape(np.shape(obs)[1])
		self.current_state.append(obs)

	def get_recent_state(self, curr_obs):
		
		curr_obs = curr_obs.reshape(np.shape(curr_obs)[1])
		state = [curr_obs]
		idx = len(self.current_state) - 1

		for offset in range(0, self.window_length - 1):
			current_idx = idx - offset
			if current_idx < 0:
				break
			state.insert(0, self.current_state[current_idx])

		while len(state) < self.window_length:
			state.insert(0, np.zeros(np.shape(curr_obs)))

		return state

