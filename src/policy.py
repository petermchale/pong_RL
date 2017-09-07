import numpy as np

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z)) 

def forward_propagate(s, weights):
    h = sigmoid(np.dot(weights[1], s))
    p = sigmoid(np.dot(weights[2], h))
    return float(p), h # return probability of taking action 2 (up), and hidden state

def gradient(delta3_hat, H, S, weights):
    GW2 = np.dot(delta3_hat.T, H)
    delta2_hat = np.outer(delta3_hat, weights[2])*H*(1-H)
    GW1 = np.dot(delta2_hat.T, S)
    return {1:GW1, 2:GW2}

import gym # Uses OpenAI Gym
import cPickle 
import utility as ut 
import draw as dr 

def initialize(weights_filename, resume_from_checkpoint=True):
    env = gym.make("Pong-v0")

    if resume_from_checkpoint:
        with open(weights_filename, 'rb') as file_in:
            weights = cPickle.load(file_in)
    else: 
        num_input_neurons = len(ut.vectorize(dr.process(env.render(mode = 'rgb_array'))))
        num_hidden_neurons = 200 
        num_output_neurons = 1
        weights = {
            1: np.random.randn(num_hidden_neurons, num_input_neurons) / np.sqrt(num_input_neurons),
            2: np.random.randn(num_output_neurons, num_hidden_neurons) / np.sqrt(num_hidden_neurons)
        }  

    image, image_proc_previous = ut.initialize_successive_images(env)

    return env, weights, image, image_proc_previous

def step_with_neural_network(image, image_proc_previous, env, weights, render=True, train=True):    
    if render: env.render()

    image_proc_current = ut.vectorize(dr.process(image))
    s = (image_proc_current - image_proc_previous 
         if image_proc_previous is not None 
         else np.zeros_like(image_proc_current))
    image_proc_previous = image_proc_current

    p, h = forward_propagate(s, weights)

    action = 2 if np.random.uniform() < p else 3 
    y = 1 if action == 2 else 0 
    delta3 = y - p
    image, reward, trajectory_finished, info = env.step(action)
    
    if trajectory_finished: 
        image, image_proc_previous = ut.initialize_successive_images(env)
    
    if train:
        return image, image_proc_previous, s, h, delta3, reward, trajectory_finished
    else: 
        return image, image_proc_previous

def train(gamma, alpha, weights_filename, resume_from_checkpoint=False, render=True): 

    env, weights, image, image_proc_previous = initialize(weights_filename, resume_from_checkpoint)

    s_list, h_list, delta3_list, reward_list = [], [], [], []
    GW_sum = { layer : np.zeros_like(weights[layer]) for layer in weights.keys() } 
    trajectory = 0

    while True:    
        image, image_proc_previous, s, h, delta3, reward, trajectory_finished = \
        step_with_neural_network(image, image_proc_previous, env, weights, render, train=True)

        s_list.append(s.T) 
        h_list.append(h.T) 
        delta3_list.append(delta3) 
        reward_list.append(reward) 
        
        if reward != 0: 
            print ('trajectory %d: game finished, reward: %f' % (trajectory, reward)) + ('' if reward == -1 else ' !!!!!!!!')

        if trajectory_finished:
            delta3_hat = np.vstack(delta3_list) * ut.rewards_partial_sum(np.vstack(reward_list), gamma)
            GW = gradient(delta3_hat, np.vstack(h_list), np.vstack(s_list), weights)
            s_list, h_list, delta3_list, reward_list = [], [], [], []
            for layer in weights: GW_sum[layer] += GW[layer] 

            trajectory += 1
            if trajectory % 10 == 0:
                for layer in weights.keys():                
                    weights[layer] += alpha * GW_sum[layer] 
                    GW_sum[layer] = np.zeros_like(GW_sum[layer]) 
            if trajectory % 1 == 0: 
                with open(weights_filename, 'wb') as file_out:                
                    cPickle.dump(weights, file_out)