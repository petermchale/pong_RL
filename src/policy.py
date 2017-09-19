import numpy as np
import gym  # Uses OpenAI Gym
import cPickle
import utility as ut
import draw as dr


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # https://stackoverflow.com/a/25164452/6674256


def forward_propagate(s, weights):
    h = np.dot(weights[1], s)
    h[h < 0.0] = 0.0  # ReLU
    p = sigmoid(np.dot(weights[2], h))
    return float(p), h  # return probability of taking action 2 (up), and hidden state


def compute_gradient_likelihood(delta3_hat, H, S, weights):
    gradient_likelihood_W2 = np.dot(delta3_hat.T, H)
    delta2_hat = np.outer(delta3_hat, weights[2])
    delta2_hat[~(H > 0.0)] = 0.0  # back-propagate through ReLU
    gradient_likelihood_W1 = np.dot(delta2_hat.T, S)
    return {1: gradient_likelihood_W1, 2: gradient_likelihood_W2}


def initialize_weights_using_checkpoint(weights_filename):
    with open(weights_filename, 'rb') as file_in:
        return cPickle.load(file_in)


def initialize_weights_without_checkpoint(prng, env, num_hidden_neurons):
    num_input_neurons = len(ut.vectorize(dr.process(env.render(mode='rgb_array'))))
    num_output_neurons = 1
    return {
        1: prng.randn(num_hidden_neurons, num_input_neurons) / np.sqrt(num_input_neurons),
        2: prng.randn(num_output_neurons, num_hidden_neurons) / np.sqrt(num_hidden_neurons)
    }


def initialize_rest(random_seed):
    prng = np.random.RandomState(random_seed)

    env = gym.make('Pong-v0')
    env.seed(random_seed)

    image, image_processed_previous = ut.initialize_successive_images(env)

    return prng, env, image, image_processed_previous


def initialize_training(weights_filename, random_seed, num_hidden_neurons,
                        use_checkpoint=True):
    prng, env, image, image_processed_previous = initialize_rest(random_seed)

    if use_checkpoint:
        weights = initialize_weights_using_checkpoint(weights_filename)
    else:
        weights = initialize_weights_without_checkpoint(prng, env, num_hidden_neurons)

    return prng, env, weights, image, image_processed_previous


def initialize_testing(weights_filename, random_seed):
    prng, env, image, image_processed_previous = initialize_rest(random_seed)

    weights = initialize_weights_using_checkpoint(weights_filename)

    return prng, env, weights, image, image_processed_previous


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
