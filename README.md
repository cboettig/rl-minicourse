# rl-minicourse

This mini-course provides materials for [Quantitative Ecology Training Program]() at the Serrapilheira Institute in Brazil.


# Quickstart

1. You should have already used the 'magic link' provided by the instructor to create this repository inside the [github.com/espm-157](https://github.com/espm-157) organization.  
2. Go to <https://espm-157.nrp-nautilus.io/> and select "Login with GitHub"
3. Select "GPU image" (the default) and hit the **Start** button.

_This can take a few minutes to pull the software image before it will start._

4. Clone the repository from GitHub as shown by the instructor.


## I. Sequential decision problems in ecology: a hands-on introduction

Monday, Feb 24, 2 - 3:30p

## II. Exact and heuristic algorithms and the curse of dimensionality

Tuesday, Feb 25, 9 - 10:30a

## III. Saving the caribou ecosystem: a human vs AI competition

Wednesday Feb 26, 11 - 12:30p


# Overview

My goal during that over the next three sessions students will understand the basic objectives of Sequential Decision Problems in the context of conservation and resource management. We will explore both classic and emerging model-free deep reinforcement learning techniques to such problems to better understand the potential and limitations of both. 

We will begin with a quick review of classic optimal control through the lens of fisheries management, from foundation work of [Gordon (1954)](https://doi.org/10.1086/257497) & [Schaefer (1954)](http://hdl.handle.net/1834/21257) though dynamic (e.g. [Clark 1973](https://www.jstor.org/stable/1831136)) and stochastic ([Reed 1979](https://doi.org/10.1016/0095-0696(79)90014-7)) theory.  This will give us the foundations to discuss an important class of sequential decision problems known as Markov Decision Processes (MDPs).  We will explore exact computational solutions using Bellman recursion through dynamic programming, and encounter the curse of dimensionality when trying to add complexity to our initial models.  

We will then fast-forward some 40-50 years to consider modern techniques of deep reinforcement learning (RL) to tackle high-dimensional MDPs (including partially observed MDPs, called POMDPs).  These methods are quite different from both exact methods like dynamic programming, and from other machine learning methods like supervised or unsupervised learning, but share aspects of each.

While I shall introduce the basic premises of each approach, our emphasis will be on hands-on manipulation of algorithms to build intuition and application mastery.  While there is a wealth of high quality material on the methods and theory of RL, we learn best by being able to interact and experiment directly with these algorithms and environments -- just like our AI agents do.  [Layeperolerie et al ]() provides a strong introduction.

## Software

We will use python throughout this module.  Some familiarity with R or python will be helpful (creating variables, calling functions, basic mathematical operations), but all exercises should be accessible without substantial prior experience.  We will run pre-prepared examples in Jupyter notebooks, making simple edits. The heavy-lifting of computation is performed by existing high-quality, user friendly open source frameworks.  We will focus on [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) using [PyTorch](https://pytorch.org/), for clean, performant and well-documented reference implementations.




## Module 1: An Introduction

Run: 

```bash
python fishing_game.py
```

(or if you prefer, copy into a notebook for interactive execution).  Here we encounter all the basic elements of a reinforcement learning task. We are presented with an _observation_: the current estimate of total fish stock, and asked to choose an _action_ (fishing mortality) between 0 & 1.  Each sucessive harvest increases our running score or _reward_, while the stock size adjusts in response to harvest and recruitment.  

How then do we select our sequence of actions?  What are the sources of uncertainty in this problem? Try various strategies.  Examine the `fishing_game.py` code, and try adjusting the initial state or number of timesteps.  



