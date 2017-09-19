<img src="images/pong animation.gif">

# Playing Pong using Reinforcement Learning
Reinforcement Learning is a hot field right now, with exciting applications in robotics and self-driving cars. 
I present a rigorous mathematical derivation of a reinforcement learning algorithm that learns to beat a computer at Pong. 
This AI does not rely on hand-engineered rules or features; 
instead, it masters the environment by looking at raw pixels and learning from experience, just as humans do.
A full description is contained in the Jupyter notebook `analysis.ipynb`. 
To see the videos embedded in the notebook, please use nbviewer 
[here](http://nbviewer.jupyter.org/github/petermchale/pong_RL/blob/master/analysis.ipynb).

# Installation 
On Mac OS X 10.11.6 and 10.12.6, one can set up an environment to simulate Pong by issuing the commands

```
brew install cmake boost boost-python sdl2 swig wget
conda create -n openai python=2 ipython-notebook --yes 
source activate openai
pip install 'gym[all]' 
```

I use Jake Vanderplas' [JSAnimation](https://github.com/jakevdp/JSAnimation) package to render movies in the browser. Install the package in the currently active conda environment as follows:

```
git clone https://github.com/jakevdp/JSAnimation
cd JSAnimation
python setup.py install
```

Additionally, you may need to do 

```
conda install matplotlib 
conda install pandas
``` 

in the `openai` conda environment. 

# Acknowledgements
Inspired by [Andrej Karpathy's blog post](http://karpathy.github.io/2016/05/31/rl/). 


