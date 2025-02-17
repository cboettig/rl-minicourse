
import numpy as np
def utility(pop, effort):
    q0 = 0.1 # catchability / restoration coefficients

    benefits = effort[0] * pop[0] * q0
    # small cost to any harvesting
    costs = .00001 * sum(effort) # cost to culling

    # extinction penalty
    if np.any(pop <= 0.001):
        benefits -= 10
    return benefits - costs

def harvest(pop, effort):
    q0 = 0.1 # catchability / restoration coefficients
    pop[0] = pop[0] * (1 - effort[0] * q0) # pop 0, salmon
    return pop

initial_pop = [0.5]


parameters = {
"r_x": np.float32(0.13),
"K": np.float32(1),
"sigma_x": np.float32(0.05),
}

# pop = elk, caribou, wolves
# Caribou Scenario
def dynamics(pop, effort, harvest_fn, p, timestep=1):

    pop = harvest_fn(pop, effort)        
    X = pop[0]
    
    ## env fluctuations
    K = p["K"] # - 0.2 * np.sin(2 * np.pi * timestep / 3200)

    X += (p["r_x"] * X * (1 - X / K)
            + p["sigma_x"] * X * np.random.normal()
            )
    
    
    pop = np.array([X], dtype=np.float32)
    pop = np.clip(pop, [0], [np.Inf])
    return(pop)


import gymnasium as gym
class fish(gym.Env):
    def __init__(self, config=None):
        config = config or {}
                                
        ## these parameters may be specified in config                                  
        self.Tmax = config.get("Tmax", 800)
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", initial_pop)
        self.parameters = config.get("parameters", parameters)
        self.dynamics = config.get("dynamics", dynamics)
        self.harvest = config.get("harvest", harvest)
        self.utility = config.get("utility", utility)
        self.observe = config.get("observe", lambda state: state) # default to perfectly observed case
        self.bound = 2 * self.parameters["K"]
        
        self.action_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype = np.float32
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )        
        self.reset(seed=config.get("seed", None))


    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.initial_pop += np.multiply(self.initial_pop, np.float32(self.init_sigma * np.random.normal(size=1)))
        self.state = self.state_units(self.initial_pop)
        info = {}
        return self.observe(self.state), info


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population_units(self.state) # current state in natural units
        effort = (action + 1.) / 2

        # harvest and recruitment
        reward = self.utility(pop, effort)
        nextpop = self.dynamics(pop, effort, self.harvest, self.parameters, self.timestep)
        
        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)
        
        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward -= 50/self.timestep
        
        self.state = self.state_units(nextpop) # transform into [-1, 1] space
        observation = self.observe(self.state) # same as self.state
        return observation, reward, terminated, False, {}
    
    def state_units(self, pop):
        self.state = 2 * pop / self.bound - 1
        self.state = np.clip(self.state,  
                             np.repeat(-1, self.state.__len__()), 
                             np.repeat(1, self.state.__len__()))
        return np.float32(self.state)
    
    def population_units(self, state):
        pop = (state + 1) * self.bound /2
        return np.clip(pop, 
                       np.repeat(0, pop.__len__()),
                       np.repeat(np.Inf, pop.__len__()))
        
    def time_step(self, effort):
        action = effort * 2 - 1
        observation, reward, terminated, done, info = self.step(action)
        obs = self.population_units(observation)
        return obs, reward, terminated
    
# SMOKE-TEST verify that the environment is defined correctly    
from stable_baselines3.common.env_checker import check_env
env = fish()
check_env(env, warn=True)    
