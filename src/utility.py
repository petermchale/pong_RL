import numpy as np 

def vectorize(image):
    image = image.astype(np.float).ravel() # flatten
    length, = image.shape
    return image.reshape(length,1) # return column vector

def rewards_partial_sum(R, gamma):
    """ take bare rewards over a trajectory;
    compute partial sum of future discounted rewards, R_{part}^t """
    R_disc = np.zeros_like(R, dtype=float)
    for t in reversed(xrange(0, R.size)):
        if R[t] != 0:
            game_reward = float(R[t])
        else:
            game_reward *= gamma
        R_disc[t] = game_reward
    R_part = R_disc[::-1].cumsum()[::-1]
    length, = R_part.shape
    return R_part.reshape(length,1) # return column vector

def initialize_successive_images(env): 
    # calls `reset` on whatever object `env` refers to    
    return env.reset(), None 
