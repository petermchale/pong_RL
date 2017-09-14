# from src import policy
#
# gamma = 0.99 # discount factor for reward
# alpha = 1e-4 # learning rate
# weights_filename = 'weights.cPickle'
#
# policy.train(gamma, alpha, weights_filename, render=False, print_game_outcomes=False)

# weights_filename = 'weights.cPickle'
weights_filename = 'save.p'
number_time_points = 1000

from src import draw as dr
from src import simulate as sim

sim.simulate_with_trained_policy(weights_filename, number_time_points)