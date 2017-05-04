Introduction to Restricted Boltzmann Machines using Tensorflow
====
Material
--------
The notebook tutorial reconstructs some digits doing unsupervised learning.

There is also an accompanying presentation I gave for my group at ENS.
The second part of the [presentation](https://drive.google.com/file/d/0B8w_D99ccMLYc2lEczlPZlhKNnM/view?usp=sharing)
has some derivations for classic RBMs and the Contrastive Divergence algorithm.

**Installation notes**

I suggest to create a special [environment](https://conda.io/docs/using/envs.html) for any Tensorflow related work using [Anaconda](Anaconda-Navigator:https://docs.continuum.io/anaconda/navigator). All dependencies get automatically installed (i.e. Python/Jupyter/numpy)

```
conda create --name tensorflow-env
source activate tensorflow-env
jupyter notebook
```
[Tensorflow 1.0.0](https://www.tensorflow.org/)

[Python 3.5.2](https://www.python.org/)

[Matplotlib 2.0.0](http://matplotlib.org/)

[Numpy 1.12.0](www.numpy.org)

[Jupyter Notebook](http://jupyter.org/)

Outline
-------

Restricted Boltzmann Machines are a class of undirected probabilistic graphical models of joint probability distributions (Markov Random Fields), where the nodes in the graph are random variables. 
The latter are well known and extensively studied in the physics literature, with the ferromagnetic Ising spin model from statistical mechanics being the best example. Atoms are fixed on a 2-D (or 1-D) lattice and neighbours interact with each other. We consider the energy associated with a spin state of +/- 1 and we are interested in the possible states the system takes. It turns out that the joint probability of such a system is modelled by the Boltzmann (Gibbs) distribution.  

Similarly, the joint probability of a restricted boltzmann machine can be modelled by the gibbs distribution. Furthermore, an RBM can be considered a stochastic neural network where you have a set of visible nodes that take some data as input and a set of hidden nodes that encode a lower dimensional representation of that data. Because you can think of your input as a high dimension probability distribution, the goal is to learn the joint probability of the ensemble (visible-hidden). 

Model and Learning 
------
The goal is to learn the joint probability distribution that maximizes the probability over the data, also known as likelihood. 


