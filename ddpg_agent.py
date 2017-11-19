import sys
import gym
from gym.spaces import Box, Discrete
import numpy as np
import cv2
from ActorNet import ActorNet
from CriticNet import CriticNet
from Memory import Memory
import random
from keras import backend as K
import tensorflow as tf
from ou_noise import OUNoise

class DDPG_Agent(object):

	def __init__(self, env, experiment):
		self.env = env
		self.experiment = experiment
		self.episodes = 150000
		self.max_steps = env.spec.timestep_limit
		self.batch_size = 32  
		self.gamma = 0.99
		self.memory_capacity = 100000  
		self.action_dim = env.action_space.shape[0]
		self.state_dim = env.observation_space.shape[0]
		self.exploration_noise = OUNoise(self.action_dim)
		self.tau = 0.001
		self.lra = 0.0001
		self.lrc = 0.001
		self.isTrain = True
		self.explore = 100000
		self.epsilon = 1
		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		self.sess = tf.Session()
		K.set_session(self.sess)
		self.actor_net = ActorNet(self.sess, self.lra, self.state_dim, self.action_dim, False)
		self.target_actor_net = ActorNet(self.sess, self.lra, self.state_dim, self.action_dim, True)
		self.critic_net = CriticNet(self.sess, self.lrc, self.state_dim, self.action_dim, False)
		self.target_critic_net = CriticNet(self.sess, self.lrc, self.state_dim, self.action_dim, True)
		self.load_weights()
		self.memory = Memory(limit=self.memory_capacity)
		
	def load_weights(self):
		try:
			self.actor_net.load("cv/" + self.experiment + "/actormodel.h5")
			self.critic_net.load("cv/" + self.experiment + "/criticmodel.h5")
			self.target_actor_net.load("cv/" + self.experiment + "/actormodel.h5")
			self.target_critic_net.load("cv/" + self.experiment + "/criticmodel.h5")
			print("Weight load successfully")
		except:
			print("Cannot find the weight")

	def noise_OU_function(self, x, mu, theta, sigma):
		return theta * (mu - x) + sigma * np.random.randn(1)

	def add_action_noise(self, action_t):
		noise = self.exploration_noise.noise()
		action_t = action_t + noise
		# noise_t = np.zeros([1,self.action_dim])
		# for i in range(self.action_dim):
		# 	noise_t[0][i] = self.isTrain * max(self.epsilon, 0) * self.noise_OU_function(action_t[0][i],  0.0 , 0.15, 0.30) 
		# 	action_t[0][i] += noise_t[0][i]
		return action_t

	def update_target_net(self):

		# Update target actor network
		actor_weights = self.actor_net.model.get_weights()
		actor_target_weights = self.target_actor_net.model.get_weights()

		# Update target weight using discount factor
		for i in xrange(len(actor_weights)):
			actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
		self.target_actor_net.model.set_weights(actor_target_weights)

		# Update target critic network
		critic_weights = self.critic_net.model.get_weights()

		critic_target_weights = self.target_critic_net.model.get_weights()
		# Update target weight using discount factor
		for i in xrange(len(critic_weights)):
			critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
		self.target_critic_net.model.set_weights(critic_target_weights)
		
	def get_Act(self, state_t):
		action_t = np.zeros([1,self.action_dim])
		#print(state_t.shape)
		#print(self.state_dim)
		action_t = self.actor_net.model.predict(state_t)
		action_t = self.add_action_noise(action_t)
		# if(random.random()<0.2):
		# 	action_t[0][0] = random.uniform(-1,1)
		# if(random.random()<0.1):
		# 	action_t[0][1] = random.uniform(0,1)
		# if(random.random()<0.1):	
		# 	action_t[0][2] = random.uniform(0,1)	
		return action_t

	def replay(self):
		if(not(len(self.memory.memory)>self.batch_size)):
			return

		# Get random sample
		batch = random.sample(self.memory.memory, self.batch_size)

		# retrieve states, actions, rewards, new_states and dones from batch
		states = np.asarray([e[0] for e in batch])
		actions = np.asarray([e[1] for e in batch])
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.asarray([e[3] for e in batch])
		dones = np.asarray([e[4] for e in batch])
		y_t = np.asarray([e[1] for e in batch])
		#print(new_states.shape)
		states = np.reshape(states, (self.batch_size, self.state_dim))
		new_states = np.reshape(new_states, (self.batch_size, self.state_dim))
		#print(new_states.shape)
		#raw_input()
		temp = self.target_actor_net.model.predict(new_states)
		#print(temp.shape)
		target_q_values = self.target_critic_net.model.predict([new_states, temp])
		for k in range(len(batch)):
			if dones[k]:
				y_t[k] = rewards[k]
			else:
				y_t[k] = rewards[k] + self.gamma*target_q_values[k]

		# Train critic network
		self.loss += self.critic_net.model.train_on_batch([states,actions], y_t) 

		# Train Actor Network
		a_for_grad = self.actor_net.model.predict(states)
		grads = self.critic_net.gradients(states, a_for_grad)
		self.actor_net.train(states, grads)

		# Update target network
		self.update_target_net()

	def save_models(self):
		print("Saving Models...")
		self.actor_net.save("cv/" + self.experiment + "/actormodel.h5")
		self.critic_net.save("cv/" + self.experiment + "/criticmodel.h5")
		print("Models Successfully Saved...")

	def remember(self, state_t, action_t, reward_t, state_t1, done):
		self.memory.memory.append((state_t, action_t[0], reward_t, state_t1, done))

	def play(self):
		env = self.env		
		for i in range(self.episodes):
			#print("Episode : " + str(i))
			total_reward = 0
			observ = env.reset()

			for step in range(self.max_steps):
				#env.render()
				state_t = np.reshape(observ,[1,self.state_dim])

				self.loss = 0
				self.epsilon -= 1.0 / self.explore
				action_t = self.get_Act(state_t)

				observ, reward_t, done, info = env.step(action_t[0])
				state_t1 = np.reshape(observ,[1,self.state_dim])

				self.remember(state_t, action_t, reward_t, state_t1, done)				
				total_reward += reward_t
				state_t = state_t1
				if(self.isTrain):
					self.replay()
				#print("Episode", i, "Step", step, "Action", action_t, "Reward", reward_t, "Loss", self.loss)
				step += 1
				if done:
					break
			
			if np.mod(i, 1000) == 0 and self.isTrain:
				self.save_models()
			
			print(str(i) +"-th Episode  : Reward " + str(total_reward))
			#print("Total Steps : " + str(step))
			#print("")
		
		print("Finish.")

if __name__ == '__main__':
	experiment = sys.argv[1] #'Humanoid-v1' #specify environments here
	env = gym.make(experiment)
	print 'environment make successfully'
	ddpg_agent = DDPG_Agent(env, experiment)
	ddpg_agent.play()
	env.end() 
