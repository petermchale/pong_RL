
from src import policy

gamma = 0.95
alpha = 1e-3
weights_filename = 'weights.cPickle'

policy.train(gamma, alpha, weights_filename, render=True, print_game_outcomes=False)
