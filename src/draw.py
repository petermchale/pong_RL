def generate_frames(env, number_time_points):
    env.reset() 
    frames = []
    for t in range(number_time_points):
        frames.append(env.render(mode = 'rgb_array'))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.render(close=True)
    return frames


import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from JSAnimation.IPython_display import display_animation
'''
Install Jake Vanderplas' [JSAnimation](https://github.com/jakevdp/JSAnimation) package as follows:
git clone https://github.com/jakevdp/JSAnimation
cd JSAnimation
python setup.py install
[this installs in the currently active conda env]
'''

def display_frames(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='once'))
