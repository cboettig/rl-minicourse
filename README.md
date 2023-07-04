# rl-minicourse

This mini-course provides materials for [AMSI Winterschool23](https://ws.amsi.org.au/timetable/) in Brisbane, Australia:

## _From Theory to Practice: Reinforcement Learning for Realistic Ecosystem Management_

My goal during that over the next six sessions students will understand the basic objectives of Sequential Decision Problems in the context of conservation and resource management. We will explore both classic and emerging model-free deep reinforcement learning techniques to such problems to better understand the potential and limitations of both. 

We will begin with a quick review of classic optimal control through the lens of fisheries management, from foundation work of Gordon & Schaefer (1954) though dynamic (e.g. Clark 1974) and stochastic (Reed 1979) theory.  This will give us the foundations to discuss an important class of sequential decision problems known as Markov Decision Processes (MDPs).  We will explore exact computational solutions using Bellman recursion through dynamic programming, and encounter the curse of dimensionality when trying to add complexity to our initial models.  

We will then fast-forward some 40-50 years to consider modern techniques of deep reinforcement learning (RL) to tackle high-dimensional MDPs (including partially observed MDPs, called POMDPs).  These methods are quite different from both exact methods like dynamic programming, and from other machine learning methods like supervised or unsupervised learning, but share aspects of each.

While I shall introduce the basic premises of each approach, our emphasis will be on hands-on manipulation of algorithms to build intuition and application mastery.  While there is a wealth of high quality material on the methods and theory of RL, we learn best by being able to interact and experiment directly with these algorithms and environments -- just like our AI agents do.

## Software

We will use python throughout this module.  I will assume some familiarity with programming in R or python, and I'll try to fill in the gaps.  

## Getting started

### Option A: Codespaces

The simplest way to get started running code is to open this project in GitHub Codespaces. Codespaces provides a VSCode environment running on a virtual machine on Azure cloud, making this a viable choice from any machine with a web browser.  We have access to free-tier educational account instances, which are sufficient for most exploration, including RL agent training, but will provide more limited computational capacity than most laptops. 


### Option B: RStudio

Users already familiar with RStudio will find that it makes nearly as good an integrated development environment for python as it does for R.  Opening this project in RStudio (new project->from GitHub->clone this repo), will activate `renv`.  Agree to `renv` restoring the environment to automatically install necessary python packages on your machine. (Tested with system python on most recent RStudio.)

To access [tensorboard](https://www.tensorflow.org/tensorboard) from RStudio, run `tensorflow::tensorboard(log_dir = "~/ray_results")` from the R console first.  (For monitoring RL training)

### Option C: JupyterLab or VSCode environment

A Jupyter instance 

Microsoft's [VSCode editor](https://code.visualstudio.com/) is another polished, extensible and widely used integrated development environment.


### Installation

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Module 1: An Introduction to Optimal Control

Open `fishing_game.py`

- The deterministic problem
- Adding uncertainty
- Stochastic Dynamic Programming

## Module 2: RL Environments & Human Agents



## Module 3: Training AI Agents, Human-Agent competitions

