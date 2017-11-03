import gym
import numpy as np
import argparse
from DQNAgent import DQNAgent

def main(epsilon, render=True, model_path=None, save_path=None):
	env = gym.make('CartPole-v1')
	state_size= env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size, epsilon, model_path)

	batch_size = 32
	done = False

	for e in range(1000):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			if render:
				env.render()
			action = agent.get_Act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state =np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state

			if done:
				print("episode: {}/{}, score:{}, e {:.2}".format(e, 1000, time, agent.epsilon))
				break

		if len(agent.memory) > batch_size:
			agent.replay(batch_size)

	if save_path is not None:
		agent.model.save(save_path)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

	parser.add_argument("--render", dest="render",
						help="Render Environment",
						required = False, default='t', type=str2bool)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parseArguments()
	print args.render
	main(args.epsilon, args.render, args.load_path, args.save_path)