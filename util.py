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