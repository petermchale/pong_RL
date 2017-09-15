import numpy as np
import os


def vectorize(image):
    image = image.astype(np.float).ravel()  # flatten
    length, = image.shape
    return image.reshape(length, 1)  # return column vector


def compute_discounted_rewards(R, gamma):
    """ take bare rewards over a trajectory R^t;
    compute discounted rewards R_{disc}^t """

    R_disc = np.zeros_like(R, dtype=float)
    game_reward = None
    for t in reversed(xrange(0, R.size)):
        if R[t] != 0:
            game_reward = float(R[t])
        else:
            game_reward *= gamma
        R_disc[t] = game_reward
    return R_disc


def test_compute_discounted_rewards():

    print 'testing compute_discounted_rewards...'
    rewards = np.array([0, 0, 1, 0, 0, -1], dtype=float)
    gamma = 0.99
    print 'rewards = ', rewards
    print 'gamma = ', gamma
    print 'discounted rewards = ', compute_discounted_rewards(rewards, gamma)


def initialize_successive_images(env):
    # calls `reset` on whatever object `env` refers to    
    return env.reset(), None


def append_log_file(log_file_name, string):
    """ Write diagnostic information to a log file"""

    with open(log_file_name, 'a') as flog:
        flog.write(string)
        flog.flush()  # flush the program buffer
        os.fsync(flog.fileno())  # flush the OS buffer


if __name__ == '__main__':

    test_compute_discounted_rewards()
