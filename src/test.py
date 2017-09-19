import policy
import gym  # Uses OpenAI Gym


def simulate_with_random_policy(number_time_points,
                                render=False, random_seed=None):

    env = gym.make('Pong-v0')
    env.seed(random_seed)
    env.reset()
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode='rgb_array'))
        if render:
            env.render()
        env.step(env.action_space.sample())
    env.render(close=True)
    return frames


def simulate_with_trained_policy(weights_filename, number_time_points, render,
                                 random_seed=None):

    prng, env, weights, image, image_processed_previous = \
        policy.initialize_testing(weights_filename, random_seed)

    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode='rgb_array'))
        image, image_processed_previous = \
            policy.step_with_neural_network(image, image_processed_previous, env, weights, prng,
                                            render, training=False)
    env.render(close=True)
    return frames


if __name__ == '__main__':

    simulate_with_trained_policy(weights_filename='data/weights.cPickle',
                                 number_time_points=50000,
                                 render=True,
                                 random_seed=0)
