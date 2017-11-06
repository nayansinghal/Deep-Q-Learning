import numpy as np
from keras import backend as K
import tensorflow as tf
from gym_torcs import TorcsEnv
from ActorNet import ActorNet
from CriticNet import CriticNet
from Memory import Memory
import random

class DDPG_Agent(object):

	def __init__(self):
		self.episodes = 1000
		self.max_steps = 100000
		self.batch_size = 32  
		self.gamma = 0.99
		self.memory_capacity = 100000  
		self.vision = True 
		self.throttle = True
		self.gear_change = False
		self.action_dim = 3
		self.state_dim = 29
		self.tau = 0.001
		self.lra = 0001
		self.lrc = 0.01
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
		self.memory = Memory(limit=self.memory_capacity)
		
	def load_weights(self):
		try:
			self.actor_net.load("actormodel.h5")
			self.critic_net.load("criticmodel.h5")
			self.target_actor_net.load("actormodel.h5")
			self.target_critic_net.load("criticmodel.h5")
			print("Weight load successfully")
		except:
			print("Cannot find the weight")

	def noise_OU_function(self, x, mu, theta, sigma):
		return theta * (mu - x) + sigma * np.random.randn(1)

	def add_action_noise(self, action_t):
		noise_t = np.zeros([1,self.action_dim])
		noise_t[0][0] = self.isTrain * max(self.epsilon, 0) * self.noise_OU_function(action_t[0][0],  0.0 , 0.60, 0.30)
		noise_t[0][1] = self.isTrain * max(self.epsilon, 0) * self.noise_OU_function(action_t[0][1],  0.5 , 1.00, 0.10)
		noise_t[0][2] = self.isTrain * max(self.epsilon, 0) * self.noise_OU_function(action_t[0][2], -0.1 , 1.00, 0.05)
		action_t[0][0] += noise_t[0][0]
		action_t[0][1] += noise_t[0][1]
		action_t[0][2] += noise_t[0][2]
		return action_t

	def update_target_net(self):
		actor_weights = self.actor_net.model.get_weights()
		actor_target_weights = self.target_actor_net.model.get_weights()
		for i in xrange(len(actor_weights)):
			actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
		self.target_actor_net.model.set_weights(actor_target_weights)
		critic_weights = self.critic_net.model.get_weights()
		critic_target_weights = self.target_critic_net.model.get_weights()
		for i in xrange(len(critic_weights)):
			critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
		self.target_critic_net.model.set_weights(critic_target_weights)
		
	def get_Act(self, state_t):
		action_t = np.zeros([1,self.action_dim])
		action_t = self.actor_net.model.predict(state_t.reshape(1, state_t.shape[0]))
		action_t = self.add_action_noise(action_t)
		return action_t

	def replay(self):
		if(not(len(self.memory.memory)>self.batch_size)):
			return
		batch = random.sample(self.memory.memory, self.batch_size)
		states = np.asarray([e[0] for e in batch])
		actions = np.asarray([e[1] for e in batch])
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.asarray([e[3] for e in batch])
		dones = np.asarray([e[4] for e in batch])
		y_t = np.asarray([e[1] for e in batch])
		target_q_values = self.target_critic_net.model.predict([new_states, self.target_actor_net.model.predict(new_states)])
		for k in range(len(batch)):
			if dones[k]:
				y_t[k] = rewards[k]
			else:
				y_t[k] = rewards[k] + self.gamma*target_q_values[k]
		self.loss += self.critic_net.model.train_on_batch([states,actions], y_t) 
		a_for_grad = self.actor_net.model.predict(states)
		grads = self.critic_net.gradients(states, a_for_grad)
		self.actor_net.train(states, grads)
		self.update_target_net()

	def save_models(self):
		print("Saving Models...")
		self.actor_net.model.save("actormodel.h5")
		self.critic_net.model.save_weights("criticmodel.h5")
		print("Models Successfully Saved...")

	def remember(self, state_t, action_t, reward_t, state_t1, done):
		self.memory.memory.append((state_t, action_t[0], reward_t, state_t1, done))

	def play(self):
		env = TorcsEnv(vision=self.vision, throttle=self.throttle,gear_change=self.gear_change)
		for i in range(self.episodes):
			print("Episode : " + str(i))
			total_reward = 0
			if np.mod(i, 3) == 0:
				observ = env.reset(relaunch=True)
			else:
				observ = env.reset()
			state_t = np.hstack((observ.angle, observ.track, observ.trackPos, observ.speedX, observ.speedY,  observ.speedZ, observ.wheelSpinVel/100.0, observ.rpm))
			for step in range(self.max_steps):
				self.loss = 0
				self.epsilon -= 1.0 / self.explore
				action_t = self.get_Act(state_t)
				observ, reward_t, done, info = env.step(action_t[0])
				state_t1 = np.hstack((observ.angle, observ.track, observ.trackPos, observ.speedX, observ.speedY, observ.speedZ, observ.wheelSpinVel/100.0, observ.rpm))
				self.remember(state_t, action_t, reward_t, state_t1, done)				
				total_reward += reward_t
				state_t = state_t1
				if(self.isTrain):
					self.replay()
				print("Episode", i, "Step", step, "Action", action_t, "Reward", reward_t, "Loss", self.loss)
				step += 1
				if done:
					break
			if np.mod(i, 3) == 0 and self.isTrain:
				self.save_models()
			print("Total Reward: " + str(i) +"-th Episode  : Reward " + str(total_reward))
			print("Total Steps : " + str(step))
			print("")
		env.end() 
		print("Finish.")

if __name__ == '__main__':
	ddpg_agent = DDPG_Agent()
	ddpg_agent.play()
