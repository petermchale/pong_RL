import numpy as np
import policy
import utility as ut
import cPickle
from datetime import datetime


def initialize_time_course():
    return {key: [] for key in ['s', 'h', 'delta3', 'reward', 'pi']}


def initialize_gradient_likelihood_minibatch(weights):
    return {layer: np.zeros_like(weights[layer]) for layer in weights}


def vstack(time_course):
    for key, value in time_course.items():
        time_course[key] = np.vstack(value)


def train(gamma, alpha, weights_filename, beta, batch_size, num_hidden_neurons,
          initialize_weights_using_checkpoint=False, render=True, print_diagnostics=True, random_seed=None):

    prng, env, weights, image, image_processed_previous = \
        policy.initialize_training(weights_filename, random_seed, num_hidden_neurons,
                                   initialize_weights_using_checkpoint)
    if print_diagnostics:
        print 'gamma = ', gamma
        print 'alpha = ', alpha
        print 'beta = ', beta
        print 'batch_size = ', batch_size
        print 'num_hidden_neurons = ', num_hidden_neurons
        print 'initialize_weights_using_checkpoint = ', initialize_weights_using_checkpoint
        print 'random_seed = ', random_seed

    trajectory_reward__moving_average = None
    time_course = initialize_time_course()
    gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)
    trajectory = 1

    rmsprop_cache = {k: np.zeros_like(v) for k, v in weights.iteritems()}  # rmsprop memory

    while trajectory < 1000:
        image, image_processed_previous, s, h, delta3, pi, reward, trajectory_finished = \
            policy.step_with_neural_network(image, image_processed_previous, env, weights, prng, render)

        time_course['s'].append(s.T)
        time_course['h'].append(h.T)
        time_course['delta3'].append(delta3)
        time_course['reward'].append(reward)
        time_course['pi'].append(pi)

        if trajectory_finished:
            vstack(time_course)

            time_course_discounted_reward = ut.compute_discounted_rewards(time_course['reward'], gamma)

            # # standardize discounted rewards to have zero mean and unit variance
            # # (helps control the gradient estimator variance?)
            # time_course_discounted_reward -= np.mean(time_course_discounted_reward)
            # time_course_discounted_reward /= np.std(time_course_discounted_reward)

            delta3_hat = time_course['delta3'] * time_course_discounted_reward
            gradient_likelihood = \
                policy.compute_gradient_likelihood(delta3_hat, time_course['h'], time_course['s'], weights)

            if print_diagnostics:
                trajectory_reward = np.sum(time_course['reward'])
                # exponential moving average
                trajectory_reward__moving_average = trajectory_reward if trajectory_reward__moving_average is None \
                    else beta * trajectory_reward + (1.0 - beta) * trajectory_reward__moving_average
                H = time_course['h']
                hidden_dead_neuron_fraction = len(H[~(H > 0.0)]) / float(H.size)
                print 'trajectory = %3d' % trajectory, \
                    '| reward = %d' % trajectory_reward, \
                    '| <reward> = %.3f' % trajectory_reward__moving_average, \
                    '| L = %.3f' % np.sum(np.log(time_course['pi']) * time_course_discounted_reward), \
                    '| frac. of hidden neurons that are dead = ', hidden_dead_neuron_fraction

            time_course = initialize_time_course()

            for layer in gradient_likelihood:
                gradient_likelihood_minibatch[layer] += gradient_likelihood[layer]

            if trajectory % batch_size == 0:  # update weights using a batch of trajectories
                for layer in weights:
                    # decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
                    # learning_rate = 1e-3
                    # g = gradient_likelihood_minibatch[layer]
                    # rmsprop_cache[layer] = decay_rate * rmsprop_cache[layer] + (1.0 - decay_rate) * g ** 2
                    # weights[layer] += learning_rate * g / (np.sqrt(rmsprop_cache[layer]) + 1e-5)
                    weights[layer] += alpha * gradient_likelihood_minibatch[layer] / float(batch_size)

                gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)

                print datetime.now()
                print 'batch: trajectory = ', trajectory, \
                    'mean W1 = ', weights[1].mean(), 'mean W2 = ', weights[2].mean()
                print 'std dev W1 = ', weights[1].std(), 'std dev W2 = ', weights[2].std()
                print 'min W2 = ', weights[2].min(), 'max W2 = ', weights[2].max()
                print 'batch: p = ', policy.forward_propagate(s, weights)[0]

            if trajectory % 100 == 0:  # save weights
                with open(weights_filename, 'wb') as file_out:
                    cPickle.dump(weights, file_out)

            trajectory += 1


if __name__ == '__main__':

        train(gamma=0.99, alpha=1e-4, weights_filename='weights.cPickle',
              beta=0.5, batch_size=10, num_hidden_neurons=200,
              initialize_weights_using_checkpoint=False, render=False, random_seed=0)
