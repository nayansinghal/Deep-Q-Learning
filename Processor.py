from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from util import *

class AtariProcessor:
	def process_observation(self, obs):
		obs = Image.fromarray(obs)
		obs = obs.resize(INPUT_SHAPE).convert('L')
		obs = np.array(obs)
		return obs.astype('uint8')

		# Karpathy version of processing image: obs = curr_state - prev_state

		# obs = obs[35:195] # crop
		# obs = obs[::2,::2,0] # downsample by factor of 2
		# obs[obs == 144] = 0 # erase background (background type 1)
		# obs[obs == 109] = 0 # erase background (background type 2)
		# obs[obs != 0] = 1 # everything else (paddles, ball) just set to 1
		# return obs.astype(np.float)

	def process_reward(self, reward):
		return np.clip(reward, -1., 1.)
