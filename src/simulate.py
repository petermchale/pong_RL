def simulate_with_random_policy(env, number_time_points):
    env.reset() 
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode = 'rgb_array'))
        env.step(env.action_space.sample())
    env.render(close=True)
    return frames

import policy 

def simulate_with_trained_policy(weights_filename, number_time_points): 

    env, weights, image, image_proc_previous = policy.initialize(weights_filename)
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode = 'rgb_array'))
        image, image_proc_previous = \
        policy.step_with_neural_network(image, image_proc_previous, env, weights, render=False, train=False)
    env.render(close=True)    
    return frames