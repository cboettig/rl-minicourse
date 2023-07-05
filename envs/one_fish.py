# NB: It is typical to use float32 precision to benefit from enhanced GPU speeds

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class one_fish(gym.Env):
    """A 1-species fisheries model"""
    def __init__(self, config=None):
        config = config or {}
        parameters = {
         "r": np.float32(0.1),
         "K": np.float32(1.0),
         "sigma": np.float32(0.01),
         "cost": np.float32(0.0)
        }
        initial_pop = np.array([0.8],
                                dtype=np.float32)
                                
        ## these parameters may be specified in config                                  
        self.Tmax = config.get("Tmax", 200)
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", initial_pop)
        self.parameters = config.get("parameters", parameters)
        
        self.bound = 2 * self.parameters["K"]
        
        self.action_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )        
        self.reset(seed=config.get("seed", None))

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.update_state(self.initial_pop)
        self.state += np.float32(self.parameters["sigma"] * np.random.normal(size=1) )
        info = {}
        return self.state, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population() # current state in natural units
        effort = (action + 1.) / 2
        # harvest and recruitment
        pop, reward = self.harvest(pop, effort)
        pop = self.population_growth(pop)
        
        self.timestep += 1
        self.state = self.update_state(pop) # transform into [-1, 1] space
        observation = self.observation() # for now, same as self.state
        
        terminated = bool(self.timestep >= self.Tmax)
        truncated = bool(self.state <= -1.0)
        info = {}

        return observation, reward, terminated, truncated, info

    
    def harvest(self, pop, effort): 
        harvest = effort[0] * pop[0]
        pop[0] = pop[0] - harvest
        
        reward = np.max(harvest,0) - self.parameters["cost"] * effort[0]
        return pop, np.float64(reward)
      
    def population_growth(self, pop):
        X = pop[0]
        p = self.parameters
        X += p["r"] * X * (1 - X / p["K"]) + p["sigma"] * X * np.random.normal()
        pop = np.array([X], dtype=np.float32)
        return(pop)

    def observation(self): # perfectly observed case
        return self.state
    
    # inverse of self.population()
    def update_state(self, pop):
        pop = np.clip(pop, 0, np.Inf) # enforce non-negative population first
        self.state = np.array([2 * pop[0] / self.bound - 1], dtype=np.float32)
        return self.state
    
    def population(self):
        pop = np.array(
          [(self.state[0] + 1) * self.bound / 2],dtype=np.float32)
        return np.clip(pop, 0, np.Inf)
    
    
    
