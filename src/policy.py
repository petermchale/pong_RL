import numpy as np
import gym  # Uses OpenAI Gym
import cPickle
import utility as ut
import draw as dr
from datetime import datetime


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # https://stackoverflow.com/a/25164452/6674256


def forward_propagate(s, weights):
    h = sigmoid(np.dot(weights[1], s))
    p = sigmoid(np.dot(weights[2], h))
    return float(p), h  # return probability of taking action 2 (up), and hidden state


def gradient(delta3_hat, H, S, weights):
    GW2 = np.dot(delta3_hat.T, H)
    delta2_hat = np.outer(delta3_hat, weights[2]) * H * (1 - H)
    GW1 = np.dot(delta2_hat.T, S)
    return {1: GW1, 2: GW2}


def initialize(weights_filename, my_seed, initialize_weights_using_checkpoint=True):
    env = gym.make("Pong-v0")
    env.seed(my_seed)

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


def step_with_neural_network(image, image_processed_previous, env, weights, r, render=True, training=True):
    if render:
        env.render()

    image_processed_current = ut.vectorize(dr.process(image))
    s = (image_processed_current - image_processed_previous
         if image_processed_previous is not None
         else np.zeros_like(image_processed_current))
    image_processed_previous = image_processed_current
    p, h = forward_propagate(s, weights)

    action = 2 if r.uniform() < p else 3
    y = 1 if action == 2 else 0
    delta3 = y - p
    image, reward, trajectory_finished, info = env.step(action)

    if trajectory_finished:
        image, image_processed_previous = ut.initialize_successive_images(env)

    if training:
        return image, image_processed_previous, s, h, delta3, reward, trajectory_finished
    else:
        return image, image_processed_previous


def train(gamma, alpha, weights_filename, initialize_weights_using_checkpoint=False, render=True,
          print_game_outcomes=True):
    trajectory_filename = 'trajectories.log'
    reward_filename = 'expected_initial_discounted_reward.log'
    ut.append_log_file(trajectory_filename, str(datetime.now()) + '\n')
    ut.append_log_file(reward_filename, str(datetime.now()) + '\n')

    env, weights, image, image_processed_previous = initialize(weights_filename, initialize_weights_using_checkpoint)

    s_list, h_list, delta3_list, reward_list = [], [], [], []
    GW_sum = {layer: np.zeros_like(weights[layer]) for layer in weights.keys()}
    initial_discounted_reward_sum, number_items_in_batch = 0.0, 0
    trajectory = 0

    print 'start: traj = ', trajectory, 'mean W1 = ', weights[1].mean(), 'mean W2 = ', weights[2].mean()
    print 'std dev W1 = ', weights[1].std(), 'std dev W2 = ', weights[2].std()
    print 'min W2 = ', weights[2].min(), 'max W2 = ', weights[2].max()

    while True:
        image, image_processed_previous, s, h, delta3, reward, trajectory_finished = \
            step_with_neural_network(image, image_processed_previous, env, weights, render, training=True)

        s_list.append(s.T)
        h_list.append(h.T)
        delta3_list.append(delta3)
        reward_list.append(reward)

        if print_game_outcomes and reward != 0:
            print ('trajectory %d: game finished, reward: %f' % (trajectory, reward)) + (
                '' if reward == -1 else ' !!!!!!!!')

        if trajectory_finished:
            # partialSum_discounted_rewards = ut.rewards_partial_sum(np.vstack(reward_list), gamma)
            number_items_in_batch += 1
            # delta3_hat = np.vstack(delta3_list) * partialSum_discounted_rewards
            discounted_rewards = ut.compute_discounted_rewards(np.vstack(reward_list), gamma)
            initial_discounted_reward_sum += discounted_rewards[0,0]
            delta3_hat = np.vstack(delta3_list) * discounted_rewards
            GW = gradient(delta3_hat, np.vstack(h_list), np.vstack(s_list), weights)
            s_list, h_list, delta3_list, reward_list = [], [], [], []
            for layer in weights:
                GW_sum[layer] += GW[layer]

            trajectory += 1
            if trajectory % 10 == 0:
                for layer in weights.keys():
                    weights[layer] += alpha * GW_sum[layer]
                    GW_sum[layer] = np.zeros_like(GW_sum[layer])
                ut.append_log_file(reward_filename, str(initial_discounted_reward_sum/number_items_in_batch) + '\n')
                initial_discounted_reward_sum, number_items_in_batch = 0.0, 0

                print 'batch: traj = ', trajectory, 'mean W1 = ', weights[1].mean(), 'mean W2 = ', weights[2].mean()
                print 'std dev W1 = ', weights[1].std(), 'std dev W2 = ', weights[2].std()
                print 'min W2 = ', weights[2].min(), 'max W2 = ', weights[2].max()
                print 'batch: p = ', forward_propagate(s, weights)[0]
            if trajectory % 100 == 0:
                with open(weights_filename, 'wb') as file_out:
                    cPickle.dump(weights, file_out)
                ut.append_log_file(trajectory_filename, 'trajectory = ' + str(trajectory) + '\n')

