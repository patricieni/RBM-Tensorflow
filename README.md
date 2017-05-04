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

* [RBM Model](#rbm-model)
* [Binary / Gaussian RBM on BAS(bars-as-stripes) dataset](#binary--gaussian-rbm-on-basbars-as-stripes-dataset)
* [MNIST reconstruction](#MNIST-reconstruction)

Restricted Boltzmann Machines are a class of undirected probabilistic graphical models of joint probability distributions (Markov Random Fields), where the nodes in the graph are random variables. 
The latter are well known and extensively studied in the physics literature, with the ferromagnetic Ising spin model from statistical mechanics being the best example. Atoms are fixed on a 2-D (or 1-D) lattice and neighbours interact with each other. We consider the energy associated with a spin state of +/- 1 and we are interested in the possible states the system takes. It turns out that the joint probability of such a system is modelled by the Boltzmann (Gibbs) distribution.  

Similarly, the joint probability of a restricted boltzmann machine can be modelled by the gibbs distribution. Furthermore, an RBM can be considered a stochastic neural network where you have a set of visible nodes that take some data as input and a set of hidden nodes that encode a lower dimensional representation of that data. Because you can think of your input as a high dimension probability distribution, the goal is to learn the joint probability of the ensemble (visible-hidden). 


### RBM Model
The model is defined in the **rbm** folder, together with methods for computing the probabilities and free energy of the system as well as sampling. The goal is to learn the joint probability distribution that maximizes the probability over the data, also known as likelihood. 
![RBM Energy](https://github.com/patricieni/RBM-Tensorflow/blob/master/img/rbm_energy.png)
![RBM Likelihood](https://github.com/patricieni/RBM-Tensorflow/blob/master/img/rbm_likelihood.png)

### Binary / Gaussian RBM on BAS(bars-as-stripes) dataset 
The BAS dataset is a dummy dataset that consists of a n by n dataset of binary values where rows have either 1 or 0. Same for columns. For a 4 by 4 dataset you would have 32 options. We use a binary and gaussian RBM (hidden units are gaussian not binary) to try and reconstruct the input as well as partial input with 16 hidden units for the 4 by 4 case. 

Training the binary RBM for 3000 epochs we see it reconstructs partial input with 70% accuracy. 
Training the gaussian RBM is slightly better for 1000 epochs with 86% accuracy. 

### MNIST reconstruction 
The other two notebooks show how to use the RBM for learning a lower dimensional representation of the MNIST dataset. 
You can see the reconstructions in both cases and how it's slightly better in the gaussian scenario. 

Bare in mind this is the simplest example of RBM that uses Contrastive Divergence 1 (only 1 step of MCMC simulation) without weight cost or temperature [Tieleman 08]. Of course there are better performing variants of the model. 


 


