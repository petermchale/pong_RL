import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from JSAnimation.IPython_display import display_animation

# Install Jake Vanderplas' [JSAnimation](https://github.com/jakevdp/JSAnimation) package as follows:
# git clone https://github.com/jakevdp/JSAnimation
# cd JSAnimation
# python setup.py install
# [this installs in the currently active conda env]


def display_frames(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(display_animation(anim, default_mode='once'))


def process(image):
    """ convert the 210x160x3 uint8 frame into a 80 x 80 matrix"""
    image = image[35:195]  # crop
    image = image[::2, ::2, :]  # down sample
    image = image[:, :, 0]  # remove_color
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image
