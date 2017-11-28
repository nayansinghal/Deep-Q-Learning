import os
import numpy as np
import glob

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
ACTOR_NET_HIDDEN1_UNITS = 300
ACTOR_NET_HIDDEN2_UNITS = 600
CRITIC_NET_HIDDEN1_UNITS = 300
CRITIC_NET_HIDDEN2_UNITS = 600


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

# Read rewards from log and append it to the list
def getLogs(filename):
	values = []
	file = open(filename,'r')
	for line in file:
		try:
			values.append(float(line))
		except:
			print(line)
	file.close()
	return values

# Read rewards for all corresponding dir
def getRewardsFromLogs():
	dic = {}
	for filename in glob.glob(os.path.join('logs/*/*_cut.txt')):
		dic[filename.split('/')[1]] = getLogs(filename)
	return dic

# Calulate the moving average
def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

# plot graph using dictionary and two environments
def plotgraph(dic, env1, env2, moving_avg=False, moving_value=0):
	data_env1 = dic[env1]
	data_env2 = dic[env2]
	if moving_avg:
		data_env1 = moving_average(data_env1, moving_value)
		data_env1 = moving_average(data_env1, moving_value)
	line_up, = plt.plot(data_env1, label=env1)
	line_down, = plt.plot(data_env2, 'r', label=env2)
	plt.legend(handles=[line_up, line_down])
	plt.show()