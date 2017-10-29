import gym
import numpy as np
from DQNAgent import DQNAgent

def main():
	env = gym.make('CartPole-v1')
	state_size= env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)

	batch_size = 32
	done = False

	for e in range(1000):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			# env.render()
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

if __name__ == "__main__":
	main()