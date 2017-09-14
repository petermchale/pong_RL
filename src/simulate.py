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


def simulate_with_trained_policy(weights_filename, number_time_points):
    my_seed = 0
    r = np.random.RandomState(my_seed)

    env, weights, image, image_processed_previous = policy.initialize(weights_filename, my_seed)
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode='rgb_array'))
        image, image_processed_previous = \
            policy.step_with_neural_network(image, image_processed_previous, env, weights, r, render=True, training=False)
    # env.render(close=True)
    return frames
