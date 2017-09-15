import numpy as np
import gym  # Uses OpenAI Gym
import cPickle
import utility as ut
import draw as dr
# from datetime import datetime


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # https://stackoverflow.com/a/25164452/6674256


def forward_propagate(s, weights):
    h = sigmoid(np.dot(weights[1], s))
    p = sigmoid(np.dot(weights[2], h))
    return float(p), float(h)  # return probability of taking action 2 (up), and hidden state


def compute_gradient_likelihood(delta3_hat, H, S, weights):
    gradient_likelihood_W2 = np.dot(delta3_hat.T, H)
    delta2_hat = np.outer(delta3_hat, weights[2]) * H * (1 - H)
    gradient_likelihood_W1 = np.dot(delta2_hat.T, S)
    return {1: gradient_likelihood_W1, 2: gradient_likelihood_W2}


def initialize(weights_filename, random_seed,
               initialize_weights_using_checkpoint=True):

    env = gym.make('Pong-v0')
    env.seed(random_seed)

    if initialize_weights_using_checkpoint:
        with open(weights_filename, 'rb') as file_in:
            weights = cPickle.load(file_in)
    else:
        num_input_neurons = len(ut.vectorize(dr.process(env.render(mode='rgb_array'))))
        num_hidden_neurons = 200
        num_output_neurons = 1
        weights = {
            1: np.random.randn(num_hidden_neurons, num_input_neurons) / np.sqrt(num_input_neurons),
            2: np.random.randn(num_output_neurons, num_hidden_neurons) / np.sqrt(num_hidden_neurons)
        }

    image, image_processed_previous = ut.initialize_successive_images(env)

    return env, weights, image, image_processed_previous


def step_with_neural_network(image, image_processed_previous, env, weights, prng,
                             render=True, training=True):
    if render:
        env.render()

    image_processed_current = ut.vectorize(dr.process(image))
    s = (image_processed_current - image_processed_previous
         if image_processed_previous is not None
         else np.zeros_like(image_processed_current))
    image_processed_previous = image_processed_current
    p, h = forward_propagate(s, weights)

    action = 2 if prng.uniform() < p else 3
    image, reward, trajectory_finished, info = env.step(action)

    if trajectory_finished:
        image, image_processed_previous = ut.initialize_successive_images(env)

    if training:
        y, pi = (1, p) if (action == 2) else (0, 1 - p)
        delta3 = y - p
        return image, image_processed_previous, s, h, delta3, pi, reward, trajectory_finished
    else:
        return image, image_processed_previous


def initialize_time_course():
    return dict.fromkeys(['s', 'h', 'delta3', 'reward', 'pi'], [])


def initialize_gradient_likelihood_minibatch(weights):
    return {layer: np.zeros_like(weights[layer]) for layer in weights}


def vstack(time_course):
    for key, value in time_course.items():
        time_course[key] = np.vstack(value)


def train(gamma, alpha, weights_filename, beta,
          initialize_weights_using_checkpoint=False, render=True, print_info=True, random_seed=None):

    env, weights, image, image_processed_previous = initialize(weights_filename, random_seed,
                                                               initialize_weights_using_checkpoint)
    prng = np.random.RandomState(random_seed)
    trajectory_reward__moving_average = None
    time_course = initialize_time_course()
    gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)
    trajectory = 1

    # log_file_name = 'training.log'
    # ut.append_log_file(log_file_name, str(datetime.now()) + '\n')

    while True:
        image, image_processed_previous, s, h, delta3, pi, reward, trajectory_finished = \
            step_with_neural_network(image, image_processed_previous, env, weights, prng, render)

        time_course['s'].append(s.T)
        time_course['h'].append(h.T)
        time_course['delta3'].append(delta3)
        time_course['reward'].append(reward)
        time_course['pi'].append(pi)

        if print_info and reward != 0:
            print 'trajectory = ', trajectory, 'game reward = ', reward, ('' if reward == -1 else ' !!!!!!!!')

        if trajectory_finished:
            vstack(time_course)

            time_course_discounted_reward = ut.compute_discounted_rewards(time_course['reward'], gamma)
            delta3_hat = time_course['delta3'] * time_course_discounted_reward
            gradient_likelihood = compute_gradient_likelihood(delta3_hat, time_course['h'], time_course['s'], weights)

            if print_info:
                trajectory_reward = np.sum(time_course['reward'])
                # exponential moving average
                trajectory_reward__moving_average = trajectory_reward if trajectory_reward__moving_average is None \
                    else beta * trajectory_reward + (1.0 - beta) * trajectory_reward__moving_average
                print 'trajectory = ', trajectory, \
                    '; reward = ', trajectory_reward, \
                    '; <reward> = ', trajectory_reward__moving_average, \
                    '; L = ', np.sum(np.log(time_course['pi']) * time_course_discounted_reward)

            time_course = initialize_time_course()

            for layer in gradient_likelihood:
                gradient_likelihood_minibatch[layer] += gradient_likelihood[layer]

            if trajectory % 10 == 0:  # update weights using a batch of trajectories
                for layer in weights:
                    weights[layer] += alpha * gradient_likelihood_minibatch[layer]

                gradient_likelihood_minibatch = initialize_gradient_likelihood_minibatch(weights)

                # print 'batch: trajectory = ', trajectory, \
                #     'mean W1 = ', weights[1].mean(), 'mean W2 = ', weights[2].mean()
                # print 'std dev W1 = ', weights[1].std(), 'std dev W2 = ', weights[2].std()
                # print 'min W2 = ', weights[2].min(), 'max W2 = ', weights[2].max()
                # print 'batch: p = ', forward_propagate(s, weights)[0]

            if trajectory % 100 == 0:  # save weights
                with open(weights_filename, 'wb') as file_out:
                    cPickle.dump(weights, file_out)

        trajectory += 1


if __name__ == '__main__':

        train(gamma=0.99, alpha=1e-4, weights_filename='weights.cPickle', beta=0.5, random_seed=0)
