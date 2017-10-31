import gym
import numpy as np
import argparse
from DQNAgent import DQNAgent
from Processor import AtariProcessor
from util import *

def main(epsilon, episodes, render=False, model_path=None, save_path=None):
	env = gym.make('Pong-v0')
	state_size= env.observation_space.shape[0]
	action_size = env.action_space.n
	processor = AtariProcessor()
	agent = DQNAgent(processor, state_size, action_size, epsilon, model_path)

	batch_size = 32
	done = False
	running_reward = None
	reward_sum = 0

	for e in range(episodes):
		state = env.reset()
		state = processor.process_observation(state)

		while True:
			if render:
				env.render()
			action = agent.get_Act(state)
			next_state, reward, done, _ = env.step(action)
			reward_sum += reward
			next_state = processor.process_observation(next_state)
			agent.remember(state, action, reward, next_state, done)
			state = next_state

			if done:
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print 'episode: %5d/%d, Total Reward: %.4f, running mean: %.4f, e:%.4f' %(e, episodes, reward_sum, running_reward, agent.epsilon)
				reward_sum = 0
				break

		if len(agent.memory) > batch_size:
			agent.replay(batch_size)

	if save_path is not None:
		agent.model.save(save_path)

def parseArguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("--load_path", dest="load_path",
						help="Load Model Path",
						required = False, default=None, type=str)

	parser.add_argument("--save_path", dest="save_path",
						help="Model Save Path",
						required = False, default=None, type=str)

	parser.add_argument("--epsilon", dest="epsilon",
						help="Exploration starting value",
						required = False, default=0.1, type=float)

	parser.add_argument("--episodes", dest="episodes",
						help="Total episodes to run",
						required = False, default=1000, type=int)

	parser.add_argument("--render", dest="render",
						help="Render Environment",
						required = False, default='f', type=str2bool)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parseArguments()
	print args.render
	main(args.epsilon, args.episodes, args.render, args.load_path, args.save_path)