# rl-minicourse

This mini-course provides materials for [AMSI Winterschool23](https://ws.amsi.org.au/timetable/) in Brisbane, Australia:

## _From Theory to Practice: Reinforcement Learning for Realistic Ecosystem Management_

My goal during that over the next six sessions students will understand the basic objectives of Sequential Decision Problems in the context of conservation and resource management. We will explore both classic and emerging model-free deep reinforcement learning techniques to such problems to better understand the potential and limitations of both. 

We will begin with a quick review of classic optimal control through the lens of fisheries management, from foundation work of [Gordon (1954)](https://doi.org/10.1086/257497) & [Schaefer (1954)](http://hdl.handle.net/1834/21257) though dynamic (e.g. [Clark 1973](https://www.jstor.org/stable/1831136)) and stochastic ([Reed 1979](https://doi.org/10.1016/0095-0696(79)90014-7)) theory.  This will give us the foundations to discuss an important class of sequential decision problems known as Markov Decision Processes (MDPs).  We will explore exact computational solutions using Bellman recursion through dynamic programming, and encounter the curse of dimensionality when trying to add complexity to our initial models.  

We will then fast-forward some 40-50 years to consider modern techniques of deep reinforcement learning (RL) to tackle high-dimensional MDPs (including partially observed MDPs, called POMDPs).  These methods are quite different from both exact methods like dynamic programming, and from other machine learning methods like supervised or unsupervised learning, but share aspects of each.

While I shall introduce the basic premises of each approach, our emphasis will be on hands-on manipulation of algorithms to build intuition and application mastery.  While there is a wealth of high quality material on the methods and theory of RL, we learn best by being able to interact and experiment directly with these algorithms and environments -- just like our AI agents do.

## Software

We will use python throughout this module.  I will assume some familiarity with programming in R or python, and I'll try to fill in the gaps. Implementation details and high quality abstractions are essential, as provided by familiarity with high-quality open source frameworks.  We will focus on [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) using [PyTorch](https://pytorch.org/), for clean, performant and well-documented reference implementations.  Time permitting, we will also introduce the industry grade [ray rllib](https://docs.ray.io/en/latest/rllib/index.html) framework which was used to train ChatGPT and other large-scale applications.

## Getting started

### Option A: Codespaces

The simplest way to get started running code is to open this project in GitHub Codespaces. Codespaces provides a VSCode environment running on a virtual machine on Azure cloud, making this a viable choice from any machine with a web browser and a quick way to get started without any local installation.  We have access to free-tier educational account instances, which are sufficient for most exploration, including RL agent training, though keep in mind these have more limited computational capacity than most laptops. 

### Option B: VSCode

Microsoft's [VSCode editor](https://code.visualstudio.com/) is a widely used and extensible integrated development environment that can be installed on any machine. It provides good support for remote access as well. 


### Option C: RStudio

Users already familiar with RStudio will find that it makes nearly as good an integrated development environment for python as it does for R.  Opening this project in RStudio (new project->from GitHub->clone this repo), will activate `renv`.  Run`renv::restore()` to automatically install necessary python packages on your machine. (Tested with system python on most recent RStudio.)  RStudio wil not render `.ipynb` notebooks, but provides good integration for running python interactively in `.py` scripts or quarto notebooks. 

To access [tensorboard](https://www.tensorflow.org/tensorboard) from RStudio, run `tensorflow::tensorboard(log_dir = "~/ray_results")` from the R console first.  (For monitoring RL training)

### Option D: JupyterLab

A local or cloud-based JupyterLab instance provides an industry-standard platform for machine learning applications in python.  Developer features are more limited than a fully fledged IDE and familiarity with the bash shell may be helpful.


## Installation

In the _Codespaces_ environment the necessary python libaries are already installed.  On other platforms I recommend setting up a virtual environment and installing the `requirements.txt` packages there:

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```


## Module 1: An Introduction

Run: 

```bash
python fishing_game.py
```

(or if you prefer, copy into a notebook for interactive execution).  Here we encounter all the basic elements of a reinforcement learning task. We are presented with an _observation_: the current estimate of total fish stock, and asked to choose an _action_ (fishing mortality) between 0 & 1.  Each sucessive harvest increases our running score or _reward_, while the stock size adjusts in response to harvest and recruitment.  

How then do we select our sequence of actions?  What are the sources of uncertainty in this problem? Try various strategies.  Examine the `fishing_game.py` code, and try adjusting the initial state or number of timesteps.  

## Optimal control

Recalling Gordon or Schaefer, maybe we should try searching for the optimal mortality. Let's write a simple experiment to sweep through possible values of our action.  See `msy.py` if you need help impementing this.

 What is the optimal mortality? How well does this policy do? Is this really optimal? Consider different initial conditions for starting state.

**A more exhaustive search**. Can we search for the truly optimal sequence of actions given the fact that we may choose a different action at each timestep?  Recall Clark and Reed, consider Bellman recursion.  See `value_iteration/sdp.py` for an example implementation.  

## Reinforcement Learning

Consider adjusting the value iteration example to a more interesting model formulation -- maybe multiple species, each subject to harvest or conservation actions?  Varying environments? Partially observed states? Is there a better way forward?

Let's train our first RL agent by executing 

```bash
python sb3_train.py
```


