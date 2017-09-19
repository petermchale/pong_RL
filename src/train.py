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


def train(gamma, alpha, eta, weights_filename, beta, num_hidden_neurons,
          initialize_weights_using_checkpoint=False, render=True, log_file_name=None, random_seed=None):

    prng, env, weights, image, image_processed_previous = \
        policy.initialize_training(weights_filename, random_seed, num_hidden_neurons,
                                   initialize_weights_using_checkpoint)

    if log_file_name:
        ut.append_log_file(log_file_name, 'gamma = %f \n' % gamma)
        ut.append_log_file(log_file_name, 'alpha = %f \n' % alpha)
        ut.append_log_file(log_file_name, 'beta = %f \n' % beta)
        ut.append_log_file(log_file_name, 'eta = %f \n' % eta)
        ut.append_log_file(log_file_name, 'num_hidden_neurons = %d \n' % num_hidden_neurons)
        ut.append_log_file(log_file_name, 'initialize_weights_using_checkpoint = %d \n' %
                           initialize_weights_using_checkpoint)
        ut.append_log_file(log_file_name, 'random_seed = %f \n' % random_seed)
        ut.append_log_file(log_file_name, '\n')

    trajectory_reward__moving_average = None
    time_course = initialize_time_course()
    gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)
    g2__moving_average = {k: np.zeros_like(v) for k, v in weights.iteritems()}
    trajectory = 1

    while True:
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

            delta3_hat = time_course['delta3'] * time_course_discounted_reward
            gradient_likelihood = \
                policy.compute_gradient_likelihood(delta3_hat, time_course['h'], time_course['s'], weights)

            trajectory_reward = np.sum(time_course['reward'])
            # exponential moving average
            trajectory_reward__moving_average = trajectory_reward if trajectory_reward__moving_average is None \
                else beta * trajectory_reward + (1.0 - beta) * trajectory_reward__moving_average
            L = np.sum(np.log(time_course['pi']) * time_course_discounted_reward)
            H = time_course['h']
            fraction_of_hidden_neurons_that_are_dead = len(H[~(H > 0.0)]) / float(H.size)
            print '%6d\t' % trajectory, \
                '%d\t' % trajectory_reward, \
                '%.3f\t' % trajectory_reward__moving_average, \
                '%4.3f\t' % L, \
                '%.3f' % fraction_of_hidden_neurons_that_are_dead

            time_course = initialize_time_course()

            for layer in gradient_likelihood:
                gradient_likelihood_minibatch[layer] += gradient_likelihood[layer]

            if trajectory % 10 == 0:  # update weights using a batch of trajectories
                for layer in weights:
                    g = gradient_likelihood_minibatch[layer]
                    g2__moving_average[layer] = eta * (g ** 2) + (1.0 - eta) * g2__moving_average[layer]
                    weights[layer] += alpha * g / (np.sqrt(g2__moving_average[layer]) + 1e-5)

                gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)

                if log_file_name:
                    ut.append_log_file(log_file_name, str(datetime.now()) + '\n')
                    ut.append_log_file(log_file_name, 'trajectory = %d \n' % trajectory)
                    ut.append_log_file(log_file_name, 'mean W1 = %f \n' % weights[1].mean())
                    ut.append_log_file(log_file_name, 'mean W2 = %f \n' % weights[2].mean())
                    ut.append_log_file(log_file_name, 'std dev W1 = %f \n' % weights[1].std())
                    ut.append_log_file(log_file_name, 'std dev W2 = %f \n' % weights[2].std())
                    ut.append_log_file(log_file_name, 'min W2 = %f \n' % weights[2].min())
                    ut.append_log_file(log_file_name, 'max W2 = %f \n' % weights[2].max())
                    ut.append_log_file(log_file_name, 'p = %f \n' % policy.forward_propagate(s, weights)[0])
                    ut.append_log_file(log_file_name, '\n')

            if trajectory % 100 == 0:  # save weights
                with open(weights_filename, 'wb') as file_out:
                    cPickle.dump(weights, file_out)

            trajectory += 1


if __name__ == '__main__':

        train(gamma=0.99, alpha=1e-3, eta=0.01,
              weights_filename='data/weights.cPickle', beta=0.01, num_hidden_neurons=200,
              initialize_weights_using_checkpoint=False, render=False, log_file_name='data/train.log', random_seed=0)
