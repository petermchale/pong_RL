import policy
import numpy as np


def simulate_with_random_policy(env, number_time_points):

    env.reset()
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode='rgb_array'))
        env.step(env.action_space.sample())
    env.render(close=True)
    return frames


def simulate_with_trained_policy(weights_filename, number_time_points,
                                 random_seed=None):

    prng = np.random.RandomState(random_seed)

    env, weights, image, image_processed_previous = policy.initialize(weights_filename, random_seed)
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode='rgb_array'))
        image, image_processed_previous = \
            policy.step_with_neural_network(image, image_processed_previous, env, weights, prng,
                                            render=True, training=False)
    env.render(close=True)
    return frames
